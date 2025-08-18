package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// UpstreamServer represents a backend server
type UpstreamServer struct {
	URL          *url.URL      `json:"url"`
	Alive        bool          `json:"alive"`
	Connections  int64         `json:"connections"`
	ResponseTime time.Duration `json:"response_time"`
	mutex        sync.RWMutex
}

// LoadBalancer manages upstream servers
type LoadBalancer struct {
	servers  []*UpstreamServer
	current  uint64
	strategy string
	mutex    sync.RWMutex
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
	
	switch lb.strategy {
	case "round_robin":
		return lb.roundRobin()
	case "least_connections":
		return lb.leastConnections()
	case "random":
		return lb.randomSelection()
	case "gpu_aware":
		return lb.gpuAwareSelection()
	default:
		return lb.roundRobin()
	}
}

func (lb *LoadBalancer) roundRobin() *UpstreamServer {
	next := atomic.AddUint64(&lb.current, 1)
	return lb.servers[(int(next)-1)%len(lb.servers)]
}

func (lb *LoadBalancer) leastConnections() *UpstreamServer {
	var selected *UpstreamServer
	minConnections := int64(^uint64(0) >> 1) // Max int64
	
	for _, server := range lb.servers {
		if server.Alive {
			server.mutex.RLock()
			connections := server.Connections
			server.mutex.RUnlock()
			
			if connections < minConnections {
				minConnections = connections
				selected = server
			}
		}
	}
	
	return selected
}

func (lb *LoadBalancer) randomSelection() *UpstreamServer {
	alive := make([]*UpstreamServer, 0)
	for _, server := range lb.servers {
		if server.Alive {
			alive = append(alive, server)
		}
	}
	
	if len(alive) == 0 {
		return nil
	}
	
	return alive[rand.Intn(len(alive))]
}

func (lb *LoadBalancer) gpuAwareSelection() *UpstreamServer {
	// GPU-aware load balancing: prefer servers with lower GPU utilization
	var selected *UpstreamServer
	minResponseTime := time.Hour
	
	for _, server := range lb.servers {
		if server.Alive {
			server.mutex.RLock()
			responseTime := server.ResponseTime
			server.mutex.RUnlock()
			
			if responseTime < minResponseTime {
				minResponseTime = responseTime
				selected = server
			}
		}
	}
	
	return selected
}

// ProxyHandler handles incoming requests and forwards to upstream servers
func (lb *LoadBalancer) ProxyHandler(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	
	server := lb.GetNextServer()
	if server == nil {
		http.Error(w, "No available servers", http.StatusServiceUnavailable)
		return
	}
	
	// Increment connection count
	atomic.AddInt64(&server.Connections, 1)
	defer atomic.AddInt64(&server.Connections, -1)
	
	// Create reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(server.URL)
	
	// Custom director for request modification
	proxy.Director = func(req *http.Request) {
		req.Header = r.Header
		req.URL.Scheme = server.URL.Scheme
		req.URL.Host = server.URL.Host
		req.URL.Path = r.URL.Path
		req.URL.RawQuery = r.URL.RawQuery
		
		// Add load balancer headers
		req.Header.Set("X-Forwarded-For", r.RemoteAddr)
		req.Header.Set("X-Load-Balancer", "go-cuda-lb")
		req.Header.Set("X-Upstream-Server", server.URL.String())
	}
	
	// Custom error handler
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("Proxy error for %s: %v", server.URL.String(), err)
		
		// Mark server as unhealthy on connection errors
		server.mutex.Lock()
		server.Alive = false
		server.mutex.Unlock()
		
		http.Error(w, "Upstream server error", http.StatusBadGateway)
	}
	
	// Forward request
	proxy.ServeHTTP(w, r)
	
	// Update response time metrics
	duration := time.Since(start)
	server.mutex.Lock()
	server.ResponseTime = duration
	server.mutex.Unlock()
	
	log.Printf("Request to %s completed in %v", server.URL.String(), duration)
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
	defer lb.mutex.RUnlock()
	
	metrics := &Metrics{
		ServerStats: make([]*UpstreamServer, len(lb.servers)),
	}
	
	copy(metrics.ServerStats, lb.servers)
	
	// Add GPU metrics if CUDA is enabled
	if os.Getenv("CUDA_ENABLED") == "true" {
		metrics.GPUUtilization = getGPUUtilization()
		metrics.MemoryUsage = getGPUMemoryUsage()
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
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
	defer lb.mutex.RUnlock()
	
	status := map[string]interface{}{
		"service":        "go-load-balancer",
		"strategy":       lb.strategy,
		"total_servers":  len(lb.servers),
		"alive_servers":  lb.countAliveServers(),
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
		if server.Alive {
			count++
		}
	}
	return count
}

func main() {
	// Configuration
	port := os.Getenv("PORT")
	if port == "" {
		port = "8099"
	}
	
	strategy := os.Getenv("LOAD_BALANCER_STRATEGY")
	if strategy == "" {
		strategy = "round_robin"
	}
	
	upstreamServers := os.Getenv("UPSTREAM_SERVICES")
	if upstreamServers == "" {
		upstreamServers = "http://localhost:8094,http://localhost:8095,http://localhost:8096"
	}
	
	healthInterval := time.Second * 30
	if intervalStr := os.Getenv("HEALTH_CHECK_INTERVAL"); intervalStr != "" {
		if duration, err := time.ParseDuration(intervalStr); err == nil {
			healthInterval = duration
		}
	}
	
	// Initialize load balancer
	lb := NewLoadBalancer(strategy)
	
	// Add upstream servers
	serverURLs := strings.Split(upstreamServers, ",")
	for _, serverURL := range serverURLs {
		if server := NewUpstreamServer(strings.TrimSpace(serverURL)); server != nil {
			lb.AddServer(server)
		}
	}
	
	if len(lb.servers) == 0 {
		log.Fatal("No valid upstream servers configured")
	}
	
	// Start health checker
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	healthChecker := NewHealthChecker(lb.servers, healthInterval)
	go healthChecker.Start(ctx)
	
	// Setup HTTP routes
	http.HandleFunc("/", lb.ProxyHandler)
	http.HandleFunc("/metrics", lb.MetricsHandler)
	http.HandleFunc("/status", lb.StatusHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
	})
	
	log.Printf("🚀 Go Load Balancer starting on port %s", port)
	log.Printf("🔄 Strategy: %s", strategy) 
	log.Printf("🎯 Upstream servers: %d", len(lb.servers))
	log.Printf("🏥 Health check interval: %v", healthInterval)
	if os.Getenv("CUDA_ENABLED") == "true" {
		log.Printf("⚡ CUDA acceleration enabled")
		log.Printf("🔧 GPU memory limit: %s", os.Getenv("GPU_MEMORY_LIMIT"))
	}
	
	log.Fatal(http.ListenAndServe(":"+port, nil))
}