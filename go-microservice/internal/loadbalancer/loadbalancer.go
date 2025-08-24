package loadbalancer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net"
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

	fastjson "legal-ai-production/internal/fastjson"
)

type UpstreamServer struct {
	URL            *url.URL      `json:"url"`
	Alive          bool          `json:"alive"`
	Connections    int64         `json:"connections"`
	ResponseTime   time.Duration `json:"response_time"`
	LatencyEMA     float64       `json:"latency_ema_ms"`
	Failures       int64         `json:"failures"`
	Successes      int64         `json:"successes"`
	LastFailure    time.Time     `json:"last_failure"`
	QuarantineUntil time.Time    `json:"quarantine_until"`
	mutex          sync.RWMutex
}

type LoadBalancer struct {
	servers          []*UpstreamServer
	current          uint64
	strategy         string
	mutex            sync.RWMutex
	totalRequests    int64
	activeConnections int64
	quarantineBase   time.Duration
	adminToken       string
}

type HealthChecker struct {
	servers  []*UpstreamServer
	interval time.Duration
	timeout  time.Duration
}

type Metrics struct {
	TotalRequests     int64            `json:"total_requests"`
	ActiveConnections int64            `json:"active_connections"`
	ServerStats       []*UpstreamServer `json:"server_stats"`
	GPUUtilization    float64          `json:"gpu_utilization"`
	MemoryUsage       float64          `json:"memory_usage"`
	Timestamp         string           `json:"timestamp"`
	SimulatedGPU      bool             `json:"simulated_gpu"`
	JSON              fastjson.Stats   `json:"json_stats"`
}

func NewUpstreamServer(serverURL string) *UpstreamServer { url, err := url.Parse(serverURL); if err != nil { log.Printf("Error parsing server URL %s: %v", serverURL, err); return nil }; return &UpstreamServer{URL: url, Alive: true} }
func NewLoadBalancer(strategy string) *LoadBalancer { return &LoadBalancer{servers: make([]*UpstreamServer,0), strategy: strategy} }
func (lb *LoadBalancer) AddServer(server *UpstreamServer) {
	if server == nil { return }
	lb.mutex.Lock()
	lb.servers = append(lb.servers, server)
	lb.mutex.Unlock()
	log.Printf("Added upstream server: %s", server.URL.String())
}
func (lb *LoadBalancer) GetNextServer() *UpstreamServer {
	lb.mutex.RLock(); defer lb.mutex.RUnlock()
	if len(lb.servers) == 0 { return nil }
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
	if candidate == nil { candidate = lb.leastConnections() }
	if candidate == nil { candidate = lb.roundRobin() }
	if candidate == nil { candidate = lb.randomSelection() }
	return candidate
}
func (lb *LoadBalancer) roundRobin() *UpstreamServer {
	for i := 0; i < len(lb.servers); i++ {
		next := atomic.AddUint64(&lb.current, 1)
		srv := lb.servers[(int(next)-1)%len(lb.servers)]
		srv.mutex.RLock()
		alive := srv.Alive && time.Now().After(srv.QuarantineUntil)
		srv.mutex.RUnlock()
		if alive { return srv }
	}
	return nil
}
func (lb *LoadBalancer) leastConnections() *UpstreamServer {
	var sel *UpstreamServer
	min := int64(^uint64(0) >> 1)
	now := time.Now()
	for _, s := range lb.servers {
		s.mutex.RLock()
		alive := s.Alive && now.After(s.QuarantineUntil)
		conns := s.Connections
		s.mutex.RUnlock()
		if alive && conns < min { min = conns; sel = s }
	}
	return sel
}
func (lb *LoadBalancer) randomSelection() *UpstreamServer {
	alive := make([]*UpstreamServer, 0)
	now := time.Now()
	for _, s := range lb.servers {
		s.mutex.RLock(); ok := s.Alive && now.After(s.QuarantineUntil); s.mutex.RUnlock()
		if ok { alive = append(alive, s) }
	}
	if len(alive) == 0 { return nil }
	return alive[rand.Intn(len(alive))]
}
func (lb *LoadBalancer) gpuAwareSelection() *UpstreamServer {
	var sel *UpstreamServer
	best := math.MaxFloat64
	now := time.Now()
	for _, s := range lb.servers {
		s.mutex.RLock()
		alive := s.Alive && now.After(s.QuarantineUntil)
		ema := s.LatencyEMA
		conns := float64(s.Connections)
		s.mutex.RUnlock()
		if !alive { continue }
		score := ema + conns*0.5
		if score < best { best = score; sel = s }
	}
	return sel
}
func (lb *LoadBalancer) ProxyHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	atomic.AddInt64(&lb.totalRequests, 1)
	atomic.AddInt64(&lb.activeConnections, 1)
	defer atomic.AddInt64(&lb.activeConnections, -1)

	srv := lb.GetNextServer()
	if srv == nil { http.Error(w, "No available servers", http.StatusServiceUnavailable); return }
	atomic.AddInt64(&srv.Connections, 1)
	defer atomic.AddInt64(&srv.Connections, -1)

	proxy := httputil.NewSingleHostReverseProxy(srv.URL)
	reqID := fmt.Sprintf("req-%d", time.Now().UnixNano())
	proxy.Director = func(req *http.Request) {
		req.Header = r.Header.Clone()
		req.URL.Scheme = srv.URL.Scheme
		req.URL.Host = srv.URL.Host
		req.URL.Path = r.URL.Path
		req.URL.RawQuery = r.URL.RawQuery
		req.Header.Set("X-Forwarded-For", r.RemoteAddr)
		req.Header.Set("X-Load-Balancer", "go-cuda-lb")
		req.Header.Set("X-Upstream-Server", srv.URL.String())
		req.Header.Set("X-Request-ID", reqID)
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Proxy error for %s: %v", srv.URL.String(), err)
		srv.mutex.Lock()
		srv.Alive = false
		srv.Failures++
		srv.LastFailure = time.Now()
		backoff := lb.quarantineBase * time.Duration(srv.Failures)
		if backoff > 5*time.Minute { backoff = 5 * time.Minute }
		srv.QuarantineUntil = time.Now().Add(backoff)
		srv.mutex.Unlock()
		http.Error(w, "Upstream server error", http.StatusBadGateway)
	}
	proxy.ServeHTTP(w, r)
	dur := time.Since(start)
	srv.mutex.Lock()
	srv.ResponseTime = dur
	ms := float64(dur.Milliseconds())
	if srv.LatencyEMA == 0 { srv.LatencyEMA = ms } else { alpha := 0.2; srv.LatencyEMA = alpha*ms + (1-alpha)*srv.LatencyEMA }
	srv.Successes++
	if srv.Failures > 0 { srv.Failures-- }
	srv.mutex.Unlock()
	log.Printf("[%s] %s -> %s in %v (EMA %.1fms)", reqID, r.Method, srv.URL.String(), dur, srv.LatencyEMA)
}
func NewHealthChecker(servers []*UpstreamServer, interval time.Duration) *HealthChecker {
	return &HealthChecker{servers: servers, interval: interval, timeout: 5 * time.Second}
}
func (hc *HealthChecker) Start(ctx context.Context) {
	tick := time.NewTicker(hc.interval)
	defer tick.Stop()
	log.Printf("Starting health checker with %d servers, interval: %v", len(hc.servers), hc.interval)
	for {
		select {
		case <-ctx.Done():
			log.Println("Health checker stopped")
			return
		case <-tick.C:
			hc.checkAllServers()
		}
	}
}
func (hc *HealthChecker) checkAllServers() {
	var wg sync.WaitGroup
	for _, s := range hc.servers {
		wg.Add(1)
		go func(srv *UpstreamServer) { defer wg.Done(); hc.checkServer(srv) }(s)
	}
	wg.Wait()
}
func (hc *HealthChecker) checkServer(srv *UpstreamServer) {
	client := &http.Client{Timeout: hc.timeout}
	resp, err := client.Get(fmt.Sprintf("%s/health", srv.URL.String()))
	srv.mutex.Lock(); defer srv.mutex.Unlock()
	if err != nil || resp.StatusCode != http.StatusOK {
		if srv.Alive { log.Printf("Server %s marked as unhealthy", srv.URL.String()) }
		srv.Alive = false
	} else {
		if !srv.Alive { log.Printf("Server %s back online", srv.URL.String()) }
		srv.Alive = true
	}
	if resp != nil { resp.Body.Close() }
}
func (lb *LoadBalancer) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	lb.mutex.RLock(); servers := make([]*UpstreamServer, len(lb.servers)); copy(servers, lb.servers); lb.mutex.RUnlock()
	snap := make([]*UpstreamServer, 0, len(servers))
	for _, s := range servers {
		s.mutex.RLock()
		clone := &UpstreamServer{URL: s.URL, Alive: s.Alive, Connections: s.Connections, ResponseTime: s.ResponseTime, LatencyEMA: s.LatencyEMA, Failures: s.Failures, Successes: s.Successes, LastFailure: s.LastFailure, QuarantineUntil: s.QuarantineUntil}
		s.mutex.RUnlock()
		snap = append(snap, clone)
	}
	m := &Metrics{TotalRequests: atomic.LoadInt64(&lb.totalRequests), ActiveConnections: atomic.LoadInt64(&lb.activeConnections), ServerStats: snap, Timestamp: time.Now().Format(time.RFC3339), JSON: fastjson.GetStats()}
	if os.Getenv("CUDA_ENABLED") == "true" { m.GPUUtilization = getGPUUtilization(); m.MemoryUsage = getGPUMemoryUsage() }
	m.SimulatedGPU = true
	w.Header().Set("Content-Type", "application/json")
	if buf, err := fastjson.EncodeToBuffer(m); err == nil {
		w.Write(buf.Bytes())
		fastjson.ReleaseBuffer(buf)
		return
	}
	// Fallback
	json.NewEncoder(w).Encode(m)
}
func (lb *LoadBalancer) PrometheusHandler(w http.ResponseWriter, r *http.Request) {
	if os.Getenv("ENABLE_PROMETHEUS") != "true" { http.Error(w, "Prometheus disabled", http.StatusNotFound); return }
	lb.mutex.RLock(); servers := make([]*UpstreamServer, len(lb.servers)); copy(servers, lb.servers); lb.mutex.RUnlock()
	var b strings.Builder
	b.WriteString("# HELP lb_total_requests Total requests handled\n# TYPE lb_total_requests counter\n")
	b.WriteString(fmt.Sprintf("lb_total_requests %d\n", atomic.LoadInt64(&lb.totalRequests)))
	b.WriteString("# HELP lb_active_connections Active downstream connections\n# TYPE lb_active_connections gauge\n")
	b.WriteString(fmt.Sprintf("lb_active_connections %d\n", atomic.LoadInt64(&lb.activeConnections)))
	for i, s := range servers {
		s.mutex.RLock(); alive := 0; if s.Alive && time.Now().After(s.QuarantineUntil) { alive = 1 }
		b.WriteString(fmt.Sprintf("lb_upstream_alive{index=\"%d\",url=\"%s\"} %d\n", i, s.URL.String(), alive))
		b.WriteString(fmt.Sprintf("lb_upstream_latency_ema_ms{index=\"%d\",url=\"%s\"} %.2f\n", i, s.URL.String(), s.LatencyEMA))
		b.WriteString(fmt.Sprintf("lb_upstream_connections{index=\"%d\",url=\"%s\"} %d\n", i, s.URL.String(), s.Connections))
		b.WriteString(fmt.Sprintf("lb_upstream_failures_total{index=\"%d\",url=\"%s\"} %d\n", i, s.URL.String(), s.Failures))
		s.mutex.RUnlock()
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	w.Write([]byte(b.String()))
}
func (lb *LoadBalancer) StatusHandler(w http.ResponseWriter, r *http.Request) {
	lb.mutex.RLock(); total := len(lb.servers); alive := lb.countAliveServers(); strategy := lb.strategy; lb.mutex.RUnlock()
	status := map[string]any{"service": "go-load-balancer", "strategy": strategy, "total_servers": total, "alive_servers": alive, "cuda_enabled": os.Getenv("CUDA_ENABLED") == "true", "gpu_memory": os.Getenv("GPU_MEMORY_LIMIT"), "timestamp": time.Now().Format(time.RFC3339)}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}
func (lb *LoadBalancer) countAliveServers() int {
	c := 0
	for _, s := range lb.servers {
		s.mutex.RLock(); alive := s.Alive && time.Now().After(s.QuarantineUntil); s.mutex.RUnlock(); if alive { c++ }
	}
	return c
}
func (lb *LoadBalancer) adminUpstreamsHandler(w http.ResponseWriter, r *http.Request) {
	token := lb.adminToken
	if token == "" || r.Header.Get("X-LB-Admin-Token") != token { http.Error(w, "unauthorized", http.StatusUnauthorized); return }
	switch r.Method {
	case http.MethodGet:
		lb.mutex.RLock(); list := make([]string, 0, len(lb.servers)); for _, s := range lb.servers { list = append(list, s.URL.String()) }; lb.mutex.RUnlock(); json.NewEncoder(w).Encode(map[string]any{"upstreams": list})
	case http.MethodPost:
		var bdy struct { Add []string `json:"add"`; Remove []string `json:"remove"` }
		if err := json.NewDecoder(r.Body).Decode(&bdy); err != nil { http.Error(w, err.Error(), 400); return }
		lb.mutex.Lock()
		if len(bdy.Remove) > 0 {
			filtered := lb.servers[:0]
			for _, s := range lb.servers {
				rm := false
				for _, rem := range bdy.Remove { if s.URL.String() == rem { rm = true; break } }
				if !rm { filtered = append(filtered, s) }
			}
			lb.servers = filtered
		}
		for _, a := range bdy.Add {
			if a == "" { continue }
			if u := NewUpstreamServer(strings.TrimSpace(a)); u != nil { lb.servers = append(lb.servers, u) }
		}
		lb.mutex.Unlock()
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(map[string]string{"status": "updated"})
	default:
		http.Error(w, "method not allowed", 405)
	}
}
func getGPUUtilization() float64 { return 75.5 }
func getGPUMemoryUsage() float64 { return 6.2 }
func firstNonEmpty(vals ...string) string { for _, v := range vals { if strings.TrimSpace(v) != "" { return v } }; return "" }
func parseDurationOr(s string, def time.Duration) time.Duration { if s == "" { return def }; if d, err := time.ParseDuration(s); err == nil { return d }; return def }
func Start() {
	port := firstNonEmpty(os.Getenv("LB_PORT"), os.Getenv("PORT"), "8099")
	strategy := firstNonEmpty(os.Getenv("LB_STRATEGY"), os.Getenv("LOAD_BALANCER_STRATEGY"), "round_robin")
	allowed := map[string]struct{}{
		"round_robin":      {},
		"least_connections": {},
		"random":           {},
		"gpu_aware":        {},
	}
	if _, ok := allowed[strategy]; !ok { log.Fatalf("invalid strategy %s", strategy) }
	upstream := firstNonEmpty(os.Getenv("UPSTREAM_SERVICES"), "http://localhost:8094,http://localhost:8095")
	healthInterval := parseDurationOr(os.Getenv("HEALTH_CHECK_INTERVAL"), 30*time.Second)
	quarantineBase := parseDurationOr(os.Getenv("QUARANTINE_BASE"), 30*time.Second)
	adminToken := os.Getenv("LB_ADMIN_TOKEN")
	lb := NewLoadBalancer(strategy)
	lb.quarantineBase = quarantineBase
	lb.adminToken = adminToken
	for _, u := range strings.Split(upstream, ",") { if s := NewUpstreamServer(strings.TrimSpace(u)); s != nil { lb.AddServer(s) } }
	if len(lb.servers) == 0 { log.Fatal("No valid upstream servers configured") }
	ctx, cancel := context.WithCancel(context.Background()); defer cancel()
	hc := NewHealthChecker(lb.servers, healthInterval); go hc.Start(ctx)
	mux := http.NewServeMux()
	mux.HandleFunc("/", lb.ProxyHandler)
	mux.HandleFunc("/metrics", lb.MetricsHandler)
	mux.HandleFunc("/metrics/json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(fastjson.GetStats())
	})
	mux.HandleFunc("/prometheus", lb.PrometheusHandler)
	mux.HandleFunc("/status", lb.StatusHandler)
	mux.HandleFunc("/admin/upstreams", lb.adminUpstreamsHandler)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.Header().Set("Content-Type", "application/json"); json.NewEncoder(w).Encode(map[string]string{"status": "healthy"}) })
	srv := &http.Server{Addr: ":" + port, Handler: mux}
	go func() { sigCh := make(chan os.Signal, 1); signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM); sig := <-sigCh; log.Printf("Received signal %v, shutting down...", sig); cancel(); ctxS, cf := context.WithTimeout(context.Background(), 10*time.Second); defer cf(); if err := srv.Shutdown(ctxS); err != nil { log.Printf("Graceful shutdown error: %v", err) } }()
	singleton := firstNonEmpty(os.Getenv("LB_SINGLETON")); if singleton == "" { singleton = "1" }
	var ln net.Listener; var err error
	if singleton == "1" { ln, err = net.Listen("tcp", ":"+port); if err != nil { log.Printf("❌ Port %s already in use – another load-balancer instance is running.", port); return } } else { log.Printf("⚠️  LB_SINGLETON disabled – relying on ListenAndServe error handling") }
	log.Printf("🚀 Go Load Balancer starting on port %s", port)
	log.Printf("🔄 Strategy: %s", strategy)
	log.Printf("🎯 Upstream servers: %d", len(lb.servers))
	log.Printf("🏥 Health check interval: %v", healthInterval)
	log.Printf("🛡️ Quarantine base: %v", quarantineBase)
	if adminToken != "" { log.Printf("🔐 Admin upstream reconfig enabled") }
	if os.Getenv("ENABLE_PROMETHEUS") == "true" { log.Printf("📈 Prometheus endpoint enabled at /prometheus") }
	if os.Getenv("CUDA_ENABLED") == "true" { log.Printf("⚡ CUDA acceleration enabled"); log.Printf("🔧 GPU memory limit: %s", os.Getenv("GPU_MEMORY_LIMIT")) }
	if ln != nil { if serveErr := srv.Serve(ln); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) { log.Fatalf("server error: %v", serveErr) } } else { if err := srv.ListenAndServe(); !errors.Is(err, http.ErrServerClosed) { log.Fatalf("server error: %v", err) } }
	log.Printf("Shutdown complete")
}
