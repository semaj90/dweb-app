//go:build legacy
// +build legacy

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// UpstreamServer represents a backend server
type UpstreamServer struct {
	URL          *url.URL      `json:"url"`
	Alive        bool          `json:"alive"`
	Connections  int64         `json:"connections"`
	ResponseTime time.Duration `json:"response_time"`
	LatencyEMA   float64       `json:"latency_ema_ms"`
	Failures     int64         `json:"failures"`
	Successes    int64         `json:"successes"`
	LastFailure  time.Time     `json:"last_failure"`
	QuarantineUntil time.Time  `json:"quarantine_until"`
	mutex        sync.RWMutex
}

// LoadBalancer manages upstream servers
type LoadBalancer struct {
	servers  []*UpstreamServer
	current  uint64
	strategy string
	mutex    sync.RWMutex
	totalRequests    int64
	activeConnections int64
	quarantineBase   time.Duration
	adminToken       string
}

// HealthChecker monitors server health
type HealthChecker struct {
	servers  []*UpstreamServer
	interval time.Duration
	timeout  time.Duration
}

// Metrics for performance monitoring
type Metrics struct {
	TotalRequests    int64            `json:"total_requests"`
	ActiveConnections int64           `json:"active_connections"`
	ServerStats      []*UpstreamServer `json:"server_stats"`
	GPUUtilization   float64          `json:"gpu_utilization"`
	MemoryUsage      float64          `json:"memory_usage"`
	Timestamp        string           `json:"timestamp"`
	SimulatedGPU     bool             `json:"simulated_gpu"`
}

func NewUpstreamServer(serverURL string) *UpstreamServer {
	url, err := url.Parse(serverURL)
	if err != nil {
		log.Printf("Error parsing server URL %s: %v", serverURL, err)
		return nil
	}

	return &UpstreamServer{
		URL:   url,
		Alive: true,
	}
}

func NewLoadBalancer(strategy string) *LoadBalancer {
	return &LoadBalancer{
		servers:  make([]*UpstreamServer, 0),
		strategy: strategy,
	}
}

func (lb *LoadBalancer) AddServer(server *UpstreamServer) {
	if server == nil {
		return
	}

	lb.mutex.Lock()
	defer lb.mutex.Unlock()

	lb.servers = append(lb.servers, server)
	log.Printf("Added upstream server: %s", server.URL.String())
}

// GetNextServer returns the next available server based on strategy
func (lb *LoadBalancer) GetNextServer() *UpstreamServer {
	lb.mutex.RLock()
	defer lb.mutex.RUnlock()
	if len(lb.servers) == 0 {
		return nil
	}

	// Order: strategy primary -> least connections fallback -> round robin -> random
	var candidate *UpstreamServer
	switch lb.strategy {
	case "gpu_aware":
		candidate = lb.gpuAwareSelection()
	case "least_connections":
		candidate = lb.leastConnections()
	case "random":
		candidate = lb.randomSelection()
	case "round_robin":
		candidate = lb.roundRobin()
	default:
		candidate = lb.roundRobin()
	}
	if candidate == nil {
		candidate = lb.leastConnections()
	}
	if candidate == nil {
		candidate = lb.roundRobin()
	}
	if candidate == nil {
		candidate = lb.randomSelection()
	}
	return candidate
}

func (lb *LoadBalancer) roundRobin() *UpstreamServer {
	// skip unhealthy / quarantined
	for i := 0; i < len(lb.servers); i++ {
		next := atomic.AddUint64(&lb.current, 1)
		srv := lb.servers[(int(next)-1)%len(lb.servers)]
		srv.mutex.RLock()
		alive := srv.Alive && time.Now().After(srv.QuarantineUntil)
		srv.mutex.RUnlock()
		if alive {
			return srv
		}
	}
	return nil
}

func (lb *LoadBalancer) leastConnections() *UpstreamServer {
	var selected *UpstreamServer
	minConnections := int64(^uint64(0) >> 1)
	now := time.Now()
	for _, server := range lb.servers {
		server.mutex.RLock()
		alive := server.Alive && now.After(server.QuarantineUntil)
		connections := server.Connections
		server.mutex.RUnlock()
		if alive && connections < minConnections {
			minConnections = connections
			selected = server
		}
	}
	return selected
}

func (lb *LoadBalancer) randomSelection() *UpstreamServer {
	alive := make([]*UpstreamServer, 0)
	now := time.Now()
	for _, server := range lb.servers {
		server.mutex.RLock()
		ok := server.Alive && now.After(server.QuarantineUntil)
		server.mutex.RUnlock()
		if ok {
			alive = append(alive, server)
		}
	}
	if len(alive) == 0 { return nil }
	return alive[rand.Intn(len(alive))]
}

func (lb *LoadBalancer) gpuAwareSelection() *UpstreamServer {
	// Prefer lowest EMA latency then lowest current connections
	var selected *UpstreamServer
	bestScore := math.MaxFloat64
	now := time.Now()
	for _, server := range lb.servers {
		server.mutex.RLock()
		alive := server.Alive && now.After(server.QuarantineUntil)
		ema := server.LatencyEMA
		conns := float64(server.Connections)
		server.mutex.RUnlock()
		if !alive { continue }
		score := ema + conns*0.5 // weight connections lightly
		if score < bestScore {
			bestScore = score
			selected = server
		}
	}
	return selected
}

// ProxyHandler handles incoming requests and forwards to upstream servers
func (lb *LoadBalancer) ProxyHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	atomic.AddInt64(&lb.totalRequests, 1)
	atomic.AddInt64(&lb.activeConnections, 1)
	defer atomic.AddInt64(&lb.activeConnections, -1)

	server := lb.GetNextServer()
	if server == nil {
		http.Error(w, "No available servers", http.StatusServiceUnavailable)
		return
	}

	// Increment connection count
	atomic.AddInt64(&server.Connections, 1)
	defer atomic.AddInt64(&server.Connections, -1)

	// Reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(server.URL)
	requestID := fmt.Sprintf("req-%d", time.Now().UnixNano())
	proxy.Director = func(req *http.Request) {
		req.Header = r.Header.Clone()
		req.URL.Scheme = server.URL.Scheme
		req.URL.Host = server.URL.Host
		req.URL.Path = r.URL.Path
		req.URL.RawQuery = r.URL.RawQuery
		req.Header.Set("X-Forwarded-For", r.RemoteAddr)
		req.Header.Set("X-Load-Balancer", "go-cuda-lb")
		req.Header.Set("X-Upstream-Server", server.URL.String())
		req.Header.Set("X-Request-ID", requestID)
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Proxy error for %s: %v", server.URL.String(), err)
		server.mutex.Lock()
		server.Alive = false
		server.Failures++
		server.LastFailure = time.Now()
		// exponential backoff quarantine
		backoff := lb.quarantineBase * time.Duration(server.Failures)
		if backoff > time.Minute*5 { backoff = time.Minute * 5 }
		server.QuarantineUntil = time.Now().Add(backoff)
		server.mutex.Unlock()
		http.Error(w, "Upstream server error", http.StatusBadGateway)
	}
	proxy.ServeHTTP(w, r)
	duration := time.Since(start)
	server.mutex.Lock()
	server.ResponseTime = duration
	// Update EMA (convert to ms)
	ms := float64(duration.Milliseconds())
	if server.LatencyEMA == 0 {
		server.LatencyEMA = ms
	} else {
		alpha := 0.2
		server.LatencyEMA = alpha*ms + (1-alpha)*server.LatencyEMA
	}
	server.Successes++
	// reset failures gradually
	if server.Failures > 0 { server.Failures-- }
	server.mutex.Unlock()
	log.Printf("[%s] %s -> %s in %v (EMA %.1fms)", requestID, r.Method, server.URL.String(), duration, server.LatencyEMA)
}

// HealthCheck implementation
func NewHealthChecker(servers []*UpstreamServer, interval time.Duration) *HealthChecker {
	return &HealthChecker{
		servers:  servers,
		interval: interval,
		timeout:  time.Second * 5,
	}
}

func (hc *HealthChecker) Start(ctx context.Context) {
	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()

	log.Printf("Starting health checker with %d servers, interval: %v", len(hc.servers), hc.interval)

	for {
		select {
		case <-ctx.Done():
			log.Println("Health checker stopped")
			return
		case <-ticker.C:
			hc.checkAllServers()
		}
	}
}

func (hc *HealthChecker) checkAllServers() {
	var wg sync.WaitGroup

	for _, server := range hc.servers {
		wg.Add(1)
		go func(srv *UpstreamServer) {
			defer wg.Done()
			hc.checkServer(srv)
		}(server)
	}

	wg.Wait()
}

func (hc *HealthChecker) checkServer(server *UpstreamServer) {
	client := &http.Client{
		Timeout: hc.timeout,
	}

	healthURL := fmt.Sprintf("%s/health", server.URL.String())
	resp, err := client.Get(healthURL)

	server.mutex.Lock()
	defer server.mutex.Unlock()

	if err != nil || resp.StatusCode != http.StatusOK {
		if server.Alive {
			log.Printf("Server %s marked as unhealthy", server.URL.String())
		}
		server.Alive = false
	} else {
		if !server.Alive {
			log.Printf("Server %s back online", server.URL.String())
		}
		server.Alive = true
	}

	if resp != nil {
		resp.Body.Close()
	}
}

// Metrics endpoint
func (lb *LoadBalancer) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	lb.mutex.RLock()
	servers := make([]*UpstreamServer, len(lb.servers))
	copy(servers, lb.servers)
	lb.mutex.RUnlock()
	// produce snapshot copies to avoid races while encoding
	snap := make([]*UpstreamServer, 0, len(servers))
	for _, s := range servers {
		s.mutex.RLock()
		clone := &UpstreamServer{URL: s.URL, Alive: s.Alive, Connections: s.Connections, ResponseTime: s.ResponseTime, LatencyEMA: s.LatencyEMA, Failures: s.Failures, Successes: s.Successes, LastFailure: s.LastFailure, QuarantineUntil: s.QuarantineUntil}
		s.mutex.RUnlock()
		snap = append(snap, clone)
	}
	metrics := &Metrics{
		TotalRequests: atomic.LoadInt64(&lb.totalRequests),
		ActiveConnections: atomic.LoadInt64(&lb.activeConnections),
		ServerStats: snap,
		Timestamp: time.Now().Format(time.RFC3339),
	}
	if os.Getenv("CUDA_ENABLED") == "true" {
		metrics.GPUUtilization = getGPUUtilization()
		metrics.MemoryUsage = getGPUMemoryUsage()
	}
	metrics.SimulatedGPU = true // currently simulated
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// Prometheus metrics endpoint (optional)
func (lb *LoadBalancer) PrometheusHandler(w http.ResponseWriter, r *http.Request) {
	if os.Getenv("ENABLE_PROMETHEUS") != "true" {
		http.Error(w, "Prometheus disabled", http.StatusNotFound)
		return
	}
	lb.mutex.RLock()
	servers := make([]*UpstreamServer, len(lb.servers))
	copy(servers, lb.servers)
	lb.mutex.RUnlock()
	var b strings.Builder
	b.WriteString("# HELP lb_total_requests Total requests handled\n")
	b.WriteString("# TYPE lb_total_requests counter\n")
	b.WriteString(fmt.Sprintf("lb_total_requests %d\n", atomic.LoadInt64(&lb.totalRequests)))
	b.WriteString("# HELP lb_active_connections Active downstream connections\n# TYPE lb_active_connections gauge\n")
	b.WriteString(fmt.Sprintf("lb_active_connections %d\n", atomic.LoadInt64(&lb.activeConnections)))
	for i, s := range servers {
		s.mutex.RLock()
		alive := 0
		if s.Alive && time.Now().After(s.QuarantineUntil) { alive = 1 }
		b.WriteString(fmt.Sprintf("lb_upstream_alive{index=\"%d\",url=\"%s\"} %d\n", i, s.URL.String(), alive))
		b.WriteString(fmt.Sprintf("lb_upstream_latency_ema_ms{index=\"%d\",url=\"%s\"} %.2f\n", i, s.URL.String(), s.LatencyEMA))
		b.WriteString(fmt.Sprintf("lb_upstream_connections{index=\"%d\",url=\"%s\"} %d\n", i, s.URL.String(), s.Connections))
		b.WriteString(fmt.Sprintf("lb_upstream_failures_total{index=\"%d\",url=\"%s\"} %d\n", i, s.URL.String(), s.Failures))
		s.mutex.RUnlock()
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	w.Write([]byte(b.String()))
}

// GPU utility functions (placeholder for CUDA integration)
func getGPUUtilization() float64 {
	// TODO: Integrate with CUDA runtime for real GPU metrics
	// For now, return simulated value
	return 75.5
}

func getGPUMemoryUsage() float64 {
	// TODO: Get actual GPU memory usage
	return 6.2 // GB
}

// Status endpoint
func (lb *LoadBalancer) StatusHandler(w http.ResponseWriter, r *http.Request) {
	lb.mutex.RLock()
	total := len(lb.servers)
	alive := lb.countAliveServers()
	strategy := lb.strategy
	lb.mutex.RUnlock()
	status := map[string]interface{}{
		"service":        "go-load-balancer",
		"strategy":       strategy,
		"total_servers":  total,
		"alive_servers":  alive,
		"cuda_enabled":   os.Getenv("CUDA_ENABLED") == "true",
		"gpu_memory":     os.Getenv("GPU_MEMORY_LIMIT"),
		"timestamp":      time.Now().Format(time.RFC3339),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (lb *LoadBalancer) countAliveServers() int {
	count := 0
	for _, server := range lb.servers {
		server.mutex.RLock()
		alive := server.Alive && time.Now().After(server.QuarantineUntil)
		server.mutex.RUnlock()
		if alive {
			count++
		}
	}
	return count
}

// Admin endpoints
func (lb *LoadBalancer) adminUpstreamsHandler(w http.ResponseWriter, r *http.Request) {
	token := lb.adminToken
	if token == "" || r.Header.Get("X-LB-Admin-Token") != token {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	switch r.Method {
	case http.MethodGet:
		lb.mutex.RLock()
		list := make([]string, 0, len(lb.servers))
		for _, s := range lb.servers { list = append(list, s.URL.String()) }
		lb.mutex.RUnlock()
		json.NewEncoder(w).Encode(map[string]interface{}{ "upstreams": list })
	case http.MethodPost:
		var body struct { Add []string `json:"add"`; Remove []string `json:"remove"` }
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil { http.Error(w, err.Error(), 400); return }
		lb.mutex.Lock()
		// removals
		if len(body.Remove) > 0 {
			filtered := lb.servers[:0]
			for _, s := range lb.servers {
				remove := false
				for _, rem := range body.Remove { if s.URL.String() == rem { remove = true; break } }
				if !remove { filtered = append(filtered, s) }
			}
			lb.servers = filtered
		}
		// additions
		for _, a := range body.Add {
			if a == "" { continue }
			if u := NewUpstreamServer(a); u != nil { lb.servers = append(lb.servers, u) }
		}
		lb.mutex.Unlock()
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status":"updated"})
	default:
		http.Error(w, "method not allowed", 405)
	}
}

func main() {
	// Configuration with fallbacks & validation
	port := firstNonEmpty(os.Getenv("LB_PORT"), os.Getenv("PORT"), "8099")
	strategy := firstNonEmpty(os.Getenv("LB_STRATEGY"), os.Getenv("LOAD_BALANCER_STRATEGY"), "round_robin")
	allowed := map[string]struct{}{ "round_robin":{}, "least_connections":{}, "random":{}, "gpu_aware":{} }
	if _, ok := allowed[strategy]; !ok { log.Fatalf("invalid strategy %s", strategy) }
	upstreamServers := firstNonEmpty(os.Getenv("UPSTREAM_SERVICES"), "http://localhost:8094,http://localhost:8095")
	healthInterval := parseDurationOr(os.Getenv("HEALTH_CHECK_INTERVAL"), 30*time.Second)
	quarantineBase := parseDurationOr(os.Getenv("QUARANTINE_BASE"), 30*time.Second)
	adminToken := os.Getenv("LB_ADMIN_TOKEN")

	lb := NewLoadBalancer(strategy)
	lb.quarantineBase = quarantineBase
	lb.adminToken = adminToken

	for _, serverURL := range strings.Split(upstreamServers, ",") {
		if server := NewUpstreamServer(strings.TrimSpace(serverURL)); server != nil {
			lb.AddServer(server)
		}
	}
	if len(lb.servers) == 0 { log.Fatal("No valid upstream servers configured") }

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	healthChecker := NewHealthChecker(lb.servers, healthInterval)
	go healthChecker.Start(ctx)

	mux := http.NewServeMux()
	mux.HandleFunc("/", lb.ProxyHandler)
	mux.HandleFunc("/metrics", lb.MetricsHandler)
	mux.HandleFunc("/prometheus", lb.PrometheusHandler)
	mux.HandleFunc("/status", lb.StatusHandler)
	mux.HandleFunc("/admin/upstreams", lb.adminUpstreamsHandler)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	})

	srv := &http.Server{ Addr: ":"+port, Handler: mux }

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		sig := <-sigCh
		log.Printf("Received signal %v, shutting down...", sig)
		cancel()
		ctxShutdown, cancelFn := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelFn()
		if err := srv.Shutdown(ctxShutdown); err != nil { log.Printf("Graceful shutdown error: %v", err) }
	}()

	log.Printf("🚀 Go Load Balancer starting on port %s", port)
	log.Printf("🔄 Strategy: %s", strategy)
	log.Printf("🎯 Upstream servers: %d", len(lb.servers))
	log.Printf("🏥 Health check interval: %v", healthInterval)
	log.Printf("🛡️ Quarantine base: %v", quarantineBase)
	if adminToken != "" { log.Printf("🔐 Admin upstream reconfig enabled") }
	if os.Getenv("ENABLE_PROMETHEUS") == "true" { log.Printf("📈 Prometheus endpoint enabled at /prometheus") }
	if os.Getenv("CUDA_ENABLED") == "true" {
		log.Printf("⚡ CUDA acceleration enabled")
		log.Printf("🔧 GPU memory limit: %s", os.Getenv("GPU_MEMORY_LIMIT"))
	}
	if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("server error: %v", err)
	}
	log.Printf("Shutdown complete")
}

// helper utilities
func firstNonEmpty(vals ...string) string { for _, v := range vals { if strings.TrimSpace(v) != "" { return v } }; return "" }
func parseDurationOr(s string, def time.Duration) time.Duration { if s == "" { return def }; if d, err := time.ParseDuration(s); err == nil { return d }; return def }