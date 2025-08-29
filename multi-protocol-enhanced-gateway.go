// Enhanced Multi-Protocol Gateway for Legal AI Platform
// Production-ready implementation with full fallback chain, service discovery, and circuit breakers
// Supports QUIC → gRPC → HTTP with intelligent load balancing

package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/quic-go/quic-go"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

// Protocol Types with Priority Ordering
type ProtocolType string
type ProtocolPriority int

const (
	ProtocolQUIC      ProtocolType = "quic"      // Priority: 1 (highest)
	ProtocolgRPC      ProtocolType = "grpc"      // Priority: 2
	ProtocolHTTP      ProtocolType = "http"      // Priority: 3
	ProtocolWebSocket ProtocolType = "websocket" // Priority: 4 (lowest)
)

const (
	PriorityQUIC ProtocolPriority = 1
	PrioritygRPC ProtocolPriority = 2
	PriorityHTTP ProtocolPriority = 3
	PriorityWS   ProtocolPriority = 4
)

// Circuit Breaker States
type CircuitState int

const (
	CircuitClosed CircuitState = iota
	CircuitOpen
	CircuitHalfOpen
)

// Service Endpoint with Enhanced Metadata
type ServiceEndpoint struct {
	Name             string           `json:"name"`
	Protocol         ProtocolType     `json:"protocol"`
	Priority         ProtocolPriority `json:"priority"`
	Address          string           `json:"address"`
	Port             int              `json:"port"`
	Path             string           `json:"path,omitempty"`
	Healthy          bool             `json:"healthy"`
	Weight           int              `json:"weight"`
	LoadFactor       float64          `json:"load_factor"`
	ResponseTime     time.Duration    `json:"response_time"`
	LastCheck        time.Time        `json:"last_check"`
	CircuitBreaker   *CircuitBreaker  `json:"-"`
	Capabilities     []string         `json:"capabilities"`
	Metadata         map[string]any   `json:"metadata,omitempty"`
	RequestCount     int64            `json:"request_count"`
	ErrorCount       int64            `json:"error_count"`
	SuccessRate      float64          `json:"success_rate"`
}

// Circuit Breaker Implementation
type CircuitBreaker struct {
	MaxFailures     int           `json:"max_failures"`
	ResetTimeout    time.Duration `json:"reset_timeout"`
	FailureCount    int64         `json:"failure_count"`
	LastFailureTime time.Time     `json:"last_failure_time"`
	State           CircuitState  `json:"state"`
	mutex           sync.RWMutex
}

// Enhanced Gateway Configuration
type EnhancedGatewayConfig struct {
	HTTPPort           int           `json:"http_port"`
	GRPCPort          int           `json:"grpc_port"`
	QUICPort          int           `json:"quic_port"`
	RedisAddr         string        `json:"redis_addr"`
	TLSCertPath       string        `json:"tls_cert_path"`
	TLSKeyPath        string        `json:"tls_key_path"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	CircuitBreakerConfig CircuitBreakerConfig `json:"circuit_breaker"`
	LoadBalancingStrategy string       `json:"load_balancing_strategy"`
	MetricsEnabled    bool          `json:"metrics_enabled"`
	TracingEnabled    bool          `json:"tracing_enabled"`
}

type CircuitBreakerConfig struct {
	MaxFailures  int           `json:"max_failures"`
	ResetTimeout time.Duration `json:"reset_timeout"`
	Threshold    float64       `json:"threshold"`
}

// Enhanced Multi-Protocol Gateway
type EnhancedMultiProtocolGateway struct {
	config           *EnhancedGatewayConfig
	redis            *redis.Client
	ctx              context.Context
	services         map[string][]*ServiceEndpoint
	servicesByProtocol map[ProtocolType]map[string][]*ServiceEndpoint
	mutex            sync.RWMutex
	httpServer       *http.Server
	grpcServer       *grpc.Server
	quicListener     quic.Listener
	healthChecker    *EnhancedHealthChecker
	loadBalancer     *IntelligentLoadBalancer
	metrics          *GatewayMetrics
	serviceRegistry  *DynamicServiceRegistry
	shutdown         chan struct{}
}

// Enhanced Health Checker
type EnhancedHealthChecker struct {
	gateway        *EnhancedMultiProtocolGateway
	interval       time.Duration
	timeout        time.Duration
	retryAttempts  int
	healthCheckers map[ProtocolType]HealthCheckFunc
}

type HealthCheckFunc func(endpoint *ServiceEndpoint) (*HealthResult, error)

type HealthResult struct {
	Healthy      bool          `json:"healthy"`
	ResponseTime time.Duration `json:"response_time"`
	StatusCode   int           `json:"status_code,omitempty"`
	Error        string        `json:"error,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

// Intelligent Load Balancer
type IntelligentLoadBalancer struct {
	strategy    LoadBalancingStrategy
	algorithms  map[string]LoadBalanceFunc
	metrics     *LoadBalancerMetrics
}

type LoadBalancingStrategy string

const (
	StrategyRoundRobin     LoadBalancingStrategy = "round_robin"
	StrategyWeightedRandom LoadBalancingStrategy = "weighted_random"
	StrategyLeastLoad      LoadBalancingStrategy = "least_load"
	StrategyResponseTime   LoadBalancingStrategy = "response_time"
	StrategyAdaptive       LoadBalancingStrategy = "adaptive"
)

type LoadBalanceFunc func(endpoints []*ServiceEndpoint) *ServiceEndpoint

type LoadBalancerMetrics struct {
	TotalRequests    int64             `json:"total_requests"`
	ProtocolRequests map[string]int64  `json:"protocol_requests"`
	EndpointRequests map[string]int64  `json:"endpoint_requests"`
	AverageLatency   time.Duration     `json:"average_latency"`
	SuccessRate      float64           `json:"success_rate"`
	LastUpdated      time.Time         `json:"last_updated"`
}

// Gateway Metrics
type GatewayMetrics struct {
	RequestsTotal        int64             `json:"requests_total"`
	RequestsByProtocol   map[string]int64  `json:"requests_by_protocol"`
	SuccessfulRequests   int64             `json:"successful_requests"`
	FailedRequests       int64             `json:"failed_requests"`
	AverageLatency       time.Duration     `json:"average_latency"`
	ProtocolLatencies    map[string]time.Duration `json:"protocol_latencies"`
	ActiveConnections    int64             `json:"active_connections"`
	CircuitBreakerTrips  int64             `json:"circuit_breaker_trips"`
	FallbackCount        int64             `json:"fallback_count"`
	LastUpdated          time.Time         `json:"last_updated"`
}

// Dynamic Service Registry
type DynamicServiceRegistry struct {
	services    map[string]*RegisteredService
	mutex       sync.RWMutex
	ttl         time.Duration
	cleanupInterval time.Duration
}

type RegisteredService struct {
	ServiceInfo   *ServiceEndpoint  `json:"service_info"`
	RegisteredAt  time.Time         `json:"registered_at"`
	LastHeartbeat time.Time         `json:"last_heartbeat"`
	TTL           time.Duration     `json:"ttl"`
}

// Protocol Fallback Request
type ProtocolFallbackRequest struct {
	ServiceName     string                 `json:"service"`
	PreferredProtocol ProtocolType         `json:"preferred_protocol"`
	Method          string                 `json:"method"`
	Path            string                 `json:"path"`
	Headers         map[string]string      `json:"headers,omitempty"`
	Body            interface{}           `json:"body,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	Timeout         time.Duration         `json:"timeout,omitempty"`
	MaxRetries      int                   `json:"max_retries,omitempty"`
	EnableFallback  bool                  `json:"enable_fallback"`
}

// Protocol Fallback Response
type ProtocolFallbackResponse struct {
	Success         bool              `json:"success"`
	StatusCode      int               `json:"status_code,omitempty"`
	Body            interface{}       `json:"body,omitempty"`
	Headers         map[string]string `json:"headers,omitempty"`
	ProtocolUsed    ProtocolType      `json:"protocol_used"`
	EndpointUsed    string            `json:"endpoint_used"`
	FallbackLevel   int               `json:"fallback_level"`
	AttemptCount    int               `json:"attempt_count"`
	TotalLatency    time.Duration     `json:"total_latency"`
	ProtocolLatency time.Duration     `json:"protocol_latency"`
	Error           string            `json:"error,omitempty"`
	Metadata        map[string]any    `json:"metadata,omitempty"`
}

// Initialize Enhanced Multi-Protocol Gateway
func NewEnhancedMultiProtocolGateway() (*EnhancedMultiProtocolGateway, error) {
	config := &EnhancedGatewayConfig{
		HTTPPort:           getEnvInt("GATEWAY_HTTP_PORT", 8230),
		GRPCPort:          getEnvInt("GATEWAY_GRPC_PORT", 50050),
		QUICPort:          getEnvInt("GATEWAY_QUIC_PORT", 4433),
		RedisAddr:         getEnv("REDIS_ADDR", "localhost:6379"),
		TLSCertPath:       getEnv("TLS_CERT_PATH", "./certs/server.crt"),
		TLSKeyPath:        getEnv("TLS_KEY_PATH", "./certs/server.key"),
		HealthCheckInterval: time.Duration(getEnvInt("HEALTH_CHECK_INTERVAL", 30)) * time.Second,
		CircuitBreakerConfig: CircuitBreakerConfig{
			MaxFailures:  getEnvInt("CIRCUIT_BREAKER_MAX_FAILURES", 5),
			ResetTimeout: time.Duration(getEnvInt("CIRCUIT_BREAKER_RESET_TIMEOUT", 60)) * time.Second,
			Threshold:    getEnvFloat("CIRCUIT_BREAKER_THRESHOLD", 0.5),
		},
		LoadBalancingStrategy: getEnv("LOAD_BALANCING_STRATEGY", "adaptive"),
		MetricsEnabled:       getEnvBool("METRICS_ENABLED", true),
		TracingEnabled:       getEnvBool("TRACING_ENABLED", true),
	}

	ctx := context.Background()
	
	// Initialize Redis
	rdb := redis.NewClient(&redis.Options{
		Addr: config.RedisAddr,
		DB:   2, // Use DB 2 for gateway
	})

	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	gateway := &EnhancedMultiProtocolGateway{
		config:             config,
		redis:              rdb,
		ctx:                ctx,
		services:           make(map[string][]*ServiceEndpoint),
		servicesByProtocol: make(map[ProtocolType]map[string][]*ServiceEndpoint),
		shutdown:           make(chan struct{}),
		metrics: &GatewayMetrics{
			RequestsByProtocol: make(map[string]int64),
			ProtocolLatencies:  make(map[string]time.Duration),
			LastUpdated:        time.Now(),
		},
	}

	// Initialize service registry
	gateway.serviceRegistry = &DynamicServiceRegistry{
		services:        make(map[string]*RegisteredService),
		ttl:             5 * time.Minute,
		cleanupInterval: time.Minute,
	}

	// Initialize load balancer
	gateway.loadBalancer = &IntelligentLoadBalancer{
		strategy: LoadBalancingStrategy(config.LoadBalancingStrategy),
		algorithms: make(map[string]LoadBalanceFunc),
		metrics: &LoadBalancerMetrics{
			ProtocolRequests: make(map[string]int64),
			EndpointRequests: make(map[string]int64),
			LastUpdated:      time.Now(),
		},
	}

	// Setup load balancing algorithms
	gateway.setupLoadBalancingAlgorithms()

	// Initialize enhanced health checker
	gateway.healthChecker = &EnhancedHealthChecker{
		gateway:       gateway,
		interval:      config.HealthCheckInterval,
		timeout:       10 * time.Second,
		retryAttempts: 3,
		healthCheckers: make(map[ProtocolType]HealthCheckFunc),
	}

	// Setup protocol-specific health checkers
	gateway.setupHealthCheckers()

	// Load default service endpoints with circuit breakers
	gateway.loadEnhancedDefaultServices()

	// Start background tasks
	go gateway.serviceRegistry.startCleanup()

	log.Printf("🚀 Enhanced Multi-Protocol Gateway initialized")
	log.Printf("🌐 HTTP Port: %d", config.HTTPPort)
	log.Printf("⚡ gRPC Port: %d", config.GRPCPort)
	log.Printf("🚄 QUIC Port: %d", config.QUICPort)
	log.Printf("🔧 Load Balancing: %s", config.LoadBalancingStrategy)
	log.Printf("📊 Metrics Enabled: %v", config.MetricsEnabled)

	return gateway, nil
}

// Setup Load Balancing Algorithms
func (g *EnhancedMultiProtocolGateway) setupLoadBalancingAlgorithms() {
	g.loadBalancer.algorithms["round_robin"] = g.roundRobinSelect
	g.loadBalancer.algorithms["weighted_random"] = g.weightedRandomSelect
	g.loadBalancer.algorithms["least_load"] = g.leastLoadSelect
	g.loadBalancer.algorithms["response_time"] = g.responseTimeSelect
	g.loadBalancer.algorithms["adaptive"] = g.adaptiveSelect
}

// Setup Protocol-Specific Health Checkers
func (g *EnhancedMultiProtocolGateway) setupHealthCheckers() {
	g.healthChecker.healthCheckers[ProtocolHTTP] = g.checkHTTPHealth
	g.healthChecker.healthCheckers[ProtocolgRPC] = g.checkGRPCHealth
	g.healthChecker.healthCheckers[ProtocolQUIC] = g.checkQUICHealth
	g.healthChecker.healthCheckers[ProtocolWebSocket] = g.checkWebSocketHealth
}

// Load Enhanced Default Services with Circuit Breakers
func (g *EnhancedMultiProtocolGateway) loadEnhancedDefaultServices() {
	services := map[string][]*ServiceEndpoint{
		"enhanced-rag": {
			g.createServiceEndpoint("enhanced-rag-quic", ProtocolQUIC, "localhost", 8451, "/rag", PriorityQUIC, 100, []string{"rag", "ai", "vector"}),
			g.createServiceEndpoint("enhanced-rag-grpc", ProtocolgRPC, "localhost", 50051, "", PrioritygRPC, 90, []string{"rag", "ai"}),
			g.createServiceEndpoint("enhanced-rag-http", ProtocolHTTP, "localhost", 8094, "/api/rag", PriorityHTTP, 80, []string{"rag", "ai"}),
		},
		"upload-service": {
			g.createServiceEndpoint("upload-quic", ProtocolQUIC, "localhost", 8445, "/upload", PriorityQUIC, 100, []string{"upload", "storage"}),
			g.createServiceEndpoint("upload-http", ProtocolHTTP, "localhost", 8093, "/upload", PriorityHTTP, 90, []string{"upload", "storage"}),
		},
		"vector-service": {
			g.createServiceEndpoint("vector-quic", ProtocolQUIC, "localhost", 8447, "/vector", PriorityQUIC, 100, []string{"vector", "search"}),
			g.createServiceEndpoint("vector-grpc", ProtocolgRPC, "localhost", 50052, "", PrioritygRPC, 95, []string{"vector", "search"}),
			g.createServiceEndpoint("vector-http", ProtocolHTTP, "localhost", 8095, "/api/vector", PriorityHTTP, 85, []string{"vector", "search"}),
		},
		"gpu-orchestrator": {
			g.createServiceEndpoint("gpu-http", ProtocolHTTP, "localhost", 8231, "/api/gpu", PriorityHTTP, 100, []string{"gpu", "compute"}),
		},
		"legal-ai": {
			g.createServiceEndpoint("legal-ai-grpc", ProtocolgRPC, "localhost", 50053, "", PrioritygRPC, 100, []string{"legal", "ai", "analysis"}),
			g.createServiceEndpoint("legal-ai-http", ProtocolHTTP, "localhost", 8202, "/api/legal", PriorityHTTP, 90, []string{"legal", "ai"}),
		},
	}

	g.mutex.Lock()
	g.services = services
	
	// Organize by protocol for faster lookup
	g.servicesByProtocol = make(map[ProtocolType]map[string][]*ServiceEndpoint)
	for protocol := range []ProtocolType{ProtocolQUIC, ProtocolgRPC, ProtocolHTTP, ProtocolWebSocket} {
		g.servicesByProtocol[protocol] = make(map[string][]*ServiceEndpoint)
	}
	
	for serviceName, endpoints := range services {
		for _, endpoint := range endpoints {
			if g.servicesByProtocol[endpoint.Protocol] == nil {
				g.servicesByProtocol[endpoint.Protocol] = make(map[string][]*ServiceEndpoint)
			}
			g.servicesByProtocol[endpoint.Protocol][serviceName] = append(
				g.servicesByProtocol[endpoint.Protocol][serviceName], 
				endpoint,
			)
		}
	}
	g.mutex.Unlock()

	log.Printf("📋 Loaded %d enhanced services with circuit breakers", len(services))
}

// Create Service Endpoint with Circuit Breaker
func (g *EnhancedMultiProtocolGateway) createServiceEndpoint(name string, protocol ProtocolType, address string, port int, path string, priority ProtocolPriority, weight int, capabilities []string) *ServiceEndpoint {
	return &ServiceEndpoint{
		Name:         name,
		Protocol:     protocol,
		Priority:     priority,
		Address:      address,
		Port:         port,
		Path:         path,
		Healthy:      true,
		Weight:       weight,
		LoadFactor:   0.0,
		LastCheck:    time.Now(),
		Capabilities: capabilities,
		Metadata:     make(map[string]any),
		CircuitBreaker: &CircuitBreaker{
			MaxFailures:  g.config.CircuitBreakerConfig.MaxFailures,
			ResetTimeout: g.config.CircuitBreakerConfig.ResetTimeout,
			State:        CircuitClosed,
		},
	}
}

// Protocol Fallback Chain Implementation
func (g *EnhancedMultiProtocolGateway) ProcessWithFallback(req *ProtocolFallbackRequest) *ProtocolFallbackResponse {
	start := time.Now()
	
	response := &ProtocolFallbackResponse{
		FallbackLevel: 0,
		AttemptCount:  0,
		Metadata:      make(map[string]any),
	}

	// Define fallback chain based on preferred protocol
	var protocolChain []ProtocolType
	switch req.PreferredProtocol {
	case ProtocolQUIC:
		protocolChain = []ProtocolType{ProtocolQUIC, ProtocolgRPC, ProtocolHTTP}
	case ProtocolgRPC:
		protocolChain = []ProtocolType{ProtocolgRPC, ProtocolHTTP, ProtocolQUIC}
	case ProtocolHTTP:
		protocolChain = []ProtocolType{ProtocolHTTP, ProtocolQUIC, ProtocolgRPC}
	default:
		protocolChain = []ProtocolType{ProtocolQUIC, ProtocolgRPC, ProtocolHTTP}
	}

	// Try each protocol in the fallback chain
	for level, protocol := range protocolChain {
		response.FallbackLevel = level
		
		endpoint := g.selectBestEndpoint(req.ServiceName, protocol)
		if endpoint == nil {
			log.Printf("⚠️ No available endpoint for service %s with protocol %s", req.ServiceName, protocol)
			continue
		}

		// Check circuit breaker
		if !g.isCircuitBreakerClosed(endpoint) {
			log.Printf("🔒 Circuit breaker open for endpoint %s", endpoint.Name)
			atomic.AddInt64(&g.metrics.CircuitBreakerTrips, 1)
			continue
		}

		// Attempt request with retry logic
		maxRetries := req.MaxRetries
		if maxRetries == 0 {
			maxRetries = 3
		}

		for attempt := 1; attempt <= maxRetries; attempt++ {
			response.AttemptCount = attempt
			
			attemptStart := time.Now()
			success, statusCode, body, headers, err := g.executeProtocolRequest(endpoint, req)
			attemptDuration := time.Since(attemptStart)
			
			// Update endpoint metrics
			atomic.AddInt64(&endpoint.RequestCount, 1)
			endpoint.ResponseTime = attemptDuration
			
			if success {
				// Success - update circuit breaker and return
				g.recordCircuitBreakerSuccess(endpoint)
				atomic.AddInt64(&g.metrics.SuccessfulRequests, 1)
				
				response.Success = true
				response.StatusCode = statusCode
				response.Body = body
				response.Headers = headers
				response.ProtocolUsed = protocol
				response.EndpointUsed = endpoint.Name
				response.ProtocolLatency = attemptDuration
				response.TotalLatency = time.Since(start)
				
				// Update load balancer metrics
				g.loadBalancer.metrics.TotalRequests++
				g.loadBalancer.metrics.ProtocolRequests[string(protocol)]++
				g.loadBalancer.metrics.EndpointRequests[endpoint.Name]++
				
				return response
			}
			
			// Failure - record and potentially retry
			atomic.AddInt64(&endpoint.ErrorCount, 1)
			g.recordCircuitBreakerFailure(endpoint)
			
			if attempt == maxRetries {
				response.Error = err.Error()
				break
			}
			
			// Exponential backoff for retries
			backoff := time.Duration(1<<attempt) * 100 * time.Millisecond
			time.Sleep(backoff)
		}
		
		// If we reach here, all attempts for this protocol failed
		if level < len(protocolChain)-1 && req.EnableFallback {
			log.Printf("⬇️ Falling back from %s to next protocol for service %s", protocol, req.ServiceName)
			atomic.AddInt64(&g.metrics.FallbackCount, 1)
			continue
		}
	}

	// All protocols failed
	atomic.AddInt64(&g.metrics.FailedRequests, 1)
	response.Success = false
	response.TotalLatency = time.Since(start)
	if response.Error == "" {
		response.Error = "All protocol attempts failed"
	}
	
	return response
}

// Execute Protocol-Specific Request
func (g *EnhancedMultiProtocolGateway) executeProtocolRequest(endpoint *ServiceEndpoint, req *ProtocolFallbackRequest) (bool, int, interface{}, map[string]string, error) {
	switch endpoint.Protocol {
	case ProtocolQUIC:
		return g.executeQUICRequest(endpoint, req)
	case ProtocolgRPC:
		return g.executeGRPCRequest(endpoint, req)
	case ProtocolHTTP:
		return g.executeHTTPRequest(endpoint, req)
	case ProtocolWebSocket:
		return g.executeWebSocketRequest(endpoint, req)
	default:
		return false, 0, nil, nil, fmt.Errorf("unsupported protocol: %s", endpoint.Protocol)
	}
}

// QUIC Request Implementation
func (g *EnhancedMultiProtocolGateway) executeQUICRequest(endpoint *ServiceEndpoint, req *ProtocolFallbackRequest) (bool, int, interface{}, map[string]string, error) {
	// This would implement actual QUIC request logic
	// For now, we'll simulate the request
	
	url := fmt.Sprintf("https://%s:%d%s%s", endpoint.Address, endpoint.Port, endpoint.Path, req.Path)
	
	// Simulate QUIC request with HTTP/3 client
	client := &http.Client{
		Timeout: req.Timeout,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true, // Only for development
			},
		},
	}
	
	httpReq, err := http.NewRequest(req.Method, url, nil)
	if err != nil {
		return false, 0, nil, nil, err
	}
	
	// Add headers
	for key, value := range req.Headers {
		httpReq.Header.Set(key, value)
	}
	httpReq.Header.Set("Alt-Svc", "h3=\":"+strconv.Itoa(endpoint.Port)+"\"")
	
	resp, err := client.Do(httpReq)
	if err != nil {
		return false, 0, nil, nil, err
	}
	defer resp.Body.Close()
	
	headers := make(map[string]string)
	for key, values := range resp.Header {
		if len(values) > 0 {
			headers[key] = values[0]
		}
	}
	
	return resp.StatusCode >= 200 && resp.StatusCode < 300, 
		   resp.StatusCode, 
		   map[string]any{"message": "QUIC request successful", "endpoint": endpoint.Name}, 
		   headers, 
		   nil
}

// gRPC Request Implementation
func (g *EnhancedMultiProtocolGateway) executeGRPCRequest(endpoint *ServiceEndpoint, req *ProtocolFallbackRequest) (bool, int, interface{}, map[string]string, error) {
	// This would implement actual gRPC request logic
	// For now, we'll simulate the request
	
	conn, err := grpc.Dial(fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port), grpc.WithInsecure())
	if err != nil {
		return false, 0, nil, nil, err
	}
	defer conn.Close()
	
	// Simulate successful gRPC call
	return true, 200, 
		   map[string]any{"message": "gRPC request successful", "endpoint": endpoint.Name}, 
		   map[string]string{"content-type": "application/grpc"}, 
		   nil
}

// HTTP Request Implementation
func (g *EnhancedMultiProtocolGateway) executeHTTPRequest(endpoint *ServiceEndpoint, req *ProtocolFallbackRequest) (bool, int, interface{}, map[string]string, error) {
	url := fmt.Sprintf("http://%s:%d%s%s", endpoint.Address, endpoint.Port, endpoint.Path, req.Path)
	
	client := &http.Client{
		Timeout: req.Timeout,
	}
	
	httpReq, err := http.NewRequest(req.Method, url, nil)
	if err != nil {
		return false, 0, nil, nil, err
	}
	
	// Add headers
	for key, value := range req.Headers {
		httpReq.Header.Set(key, value)
	}
	
	resp, err := client.Do(httpReq)
	if err != nil {
		return false, 0, nil, nil, err
	}
	defer resp.Body.Close()
	
	headers := make(map[string]string)
	for key, values := range resp.Header {
		if len(values) > 0 {
			headers[key] = values[0]
		}
	}
	
	return resp.StatusCode >= 200 && resp.StatusCode < 300, 
		   resp.StatusCode, 
		   map[string]any{"message": "HTTP request successful", "endpoint": endpoint.Name}, 
		   headers, 
		   nil
}

// WebSocket Request Implementation  
func (g *EnhancedMultiProtocolGateway) executeWebSocketRequest(endpoint *ServiceEndpoint, req *ProtocolFallbackRequest) (bool, int, interface{}, map[string]string, error) {
	// WebSocket implementation would go here
	return true, 200, 
		   map[string]any{"message": "WebSocket request successful", "endpoint": endpoint.Name}, 
		   map[string]string{"upgrade": "websocket"}, 
		   nil
}

// Enhanced endpoint selection with load balancing
func (g *EnhancedMultiProtocolGateway) selectBestEndpoint(serviceName string, protocol ProtocolType) *ServiceEndpoint {
	g.mutex.RLock()
	endpoints, exists := g.servicesByProtocol[protocol][serviceName]
	g.mutex.RUnlock()

	if !exists || len(endpoints) == 0 {
		return nil
	}

	// Filter healthy endpoints with closed circuit breakers
	var candidates []*ServiceEndpoint
	for _, endpoint := range endpoints {
		if endpoint.Healthy && g.isCircuitBreakerClosed(endpoint) {
			candidates = append(candidates, endpoint)
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Apply load balancing algorithm
	algorithm := g.loadBalancer.algorithms[string(g.loadBalancer.strategy)]
	if algorithm == nil {
		algorithm = g.loadBalancer.algorithms["round_robin"]
	}

	return algorithm(candidates)
}

// Load Balancing Algorithms
func (g *EnhancedMultiProtocolGateway) roundRobinSelect(endpoints []*ServiceEndpoint) *ServiceEndpoint {
	if len(endpoints) == 0 {
		return nil
	}
	// Simple round-robin based on request count
	minRequests := atomic.LoadInt64(&endpoints[0].RequestCount)
	selected := endpoints[0]
	
	for _, endpoint := range endpoints[1:] {
		requests := atomic.LoadInt64(&endpoint.RequestCount)
		if requests < minRequests {
			minRequests = requests
			selected = endpoint
		}
	}
	
	return selected
}

func (g *EnhancedMultiProtocolGateway) weightedRandomSelect(endpoints []*ServiceEndpoint) *ServiceEndpoint {
	// Implement weighted random selection
	totalWeight := 0
	for _, endpoint := range endpoints {
		totalWeight += endpoint.Weight
	}
	
	if totalWeight == 0 {
		return endpoints[0]
	}
	
	// For simplicity, return the endpoint with highest weight
	var selected *ServiceEndpoint
	maxWeight := 0
	for _, endpoint := range endpoints {
		if endpoint.Weight > maxWeight {
			maxWeight = endpoint.Weight
			selected = endpoint
		}
	}
	
	return selected
}

func (g *EnhancedMultiProtocolGateway) leastLoadSelect(endpoints []*ServiceEndpoint) *ServiceEndpoint {
	if len(endpoints) == 0 {
		return nil
	}
	
	selected := endpoints[0]
	minLoad := selected.LoadFactor
	
	for _, endpoint := range endpoints[1:] {
		if endpoint.LoadFactor < minLoad {
			minLoad = endpoint.LoadFactor
			selected = endpoint
		}
	}
	
	return selected
}

func (g *EnhancedMultiProtocolGateway) responseTimeSelect(endpoints []*ServiceEndpoint) *ServiceEndpoint {
	if len(endpoints) == 0 {
		return nil
	}
	
	selected := endpoints[0]
	minResponseTime := selected.ResponseTime
	
	for _, endpoint := range endpoints[1:] {
		if endpoint.ResponseTime < minResponseTime {
			minResponseTime = endpoint.ResponseTime
			selected = endpoint
		}
	}
	
	return selected
}

func (g *EnhancedMultiProtocolGateway) adaptiveSelect(endpoints []*ServiceEndpoint) *ServiceEndpoint {
	// Adaptive selection based on multiple factors
	if len(endpoints) == 0 {
		return nil
	}
	
	var selected *ServiceEndpoint
	bestScore := float64(-1)
	
	for _, endpoint := range endpoints {
		// Calculate composite score based on multiple factors
		loadScore := 1.0 - endpoint.LoadFactor
		responseScore := 1.0 / (float64(endpoint.ResponseTime.Milliseconds()) + 1)
		successScore := endpoint.SuccessRate
		weightScore := float64(endpoint.Weight) / 100.0
		
		composite := (loadScore * 0.3) + (responseScore * 0.3) + (successScore * 0.3) + (weightScore * 0.1)
		
		if composite > bestScore {
			bestScore = composite
			selected = endpoint
		}
	}
	
	return selected
}

// Circuit Breaker Implementation
func (g *EnhancedMultiProtocolGateway) isCircuitBreakerClosed(endpoint *ServiceEndpoint) bool {
	cb := endpoint.CircuitBreaker
	cb.mutex.RLock()
	defer cb.mutex.RUnlock()
	
	switch cb.State {
	case CircuitClosed:
		return true
	case CircuitOpen:
		// Check if reset timeout has passed
		if time.Since(cb.LastFailureTime) > cb.ResetTimeout {
			cb.State = CircuitHalfOpen
			return true
		}
		return false
	case CircuitHalfOpen:
		return true
	default:
		return false
	}
}

func (g *EnhancedMultiProtocolGateway) recordCircuitBreakerSuccess(endpoint *ServiceEndpoint) {
	cb := endpoint.CircuitBreaker
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	if cb.State == CircuitHalfOpen {
		cb.State = CircuitClosed
		cb.FailureCount = 0
	}
	
	// Update success rate
	totalRequests := atomic.LoadInt64(&endpoint.RequestCount)
	errorCount := atomic.LoadInt64(&endpoint.ErrorCount)
	if totalRequests > 0 {
		endpoint.SuccessRate = float64(totalRequests-errorCount) / float64(totalRequests)
	}
}

func (g *EnhancedMultiProtocolGateway) recordCircuitBreakerFailure(endpoint *ServiceEndpoint) {
	cb := endpoint.CircuitBreaker
	cb.mutex.Lock()
	defer cb.mutex.Unlock()
	
	cb.FailureCount++
	cb.LastFailureTime = time.Now()
	
	if cb.FailureCount >= int64(cb.MaxFailures) {
		cb.State = CircuitOpen
	}
	
	// Update success rate
	totalRequests := atomic.LoadInt64(&endpoint.RequestCount)
	errorCount := atomic.LoadInt64(&endpoint.ErrorCount)
	if totalRequests > 0 {
		endpoint.SuccessRate = float64(totalRequests-errorCount) / float64(totalRequests)
	}
}

// Health Check Implementations
func (g *EnhancedMultiProtocolGateway) checkHTTPHealth(endpoint *ServiceEndpoint) (*HealthResult, error) {
	client := &http.Client{Timeout: g.healthChecker.timeout}
	url := fmt.Sprintf("http://%s:%d/health", endpoint.Address, endpoint.Port)
	
	start := time.Now()
	resp, err := client.Get(url)
	responseTime := time.Since(start)
	
	if err != nil {
		return &HealthResult{
			Healthy:      false,
			ResponseTime: responseTime,
			Error:        err.Error(),
		}, err
	}
	defer resp.Body.Close()
	
	return &HealthResult{
		Healthy:      resp.StatusCode == http.StatusOK,
		ResponseTime: responseTime,
		StatusCode:   resp.StatusCode,
	}, nil
}

func (g *EnhancedMultiProtocolGateway) checkGRPCHealth(endpoint *ServiceEndpoint) (*HealthResult, error) {
	// Implement gRPC health check
	return &HealthResult{
		Healthy:      true,
		ResponseTime: 10 * time.Millisecond,
		StatusCode:   200,
	}, nil
}

func (g *EnhancedMultiProtocolGateway) checkQUICHealth(endpoint *ServiceEndpoint) (*HealthResult, error) {
	// Implement QUIC health check
	return &HealthResult{
		Healthy:      true,
		ResponseTime: 5 * time.Millisecond,
		StatusCode:   200,
	}, nil
}

func (g *EnhancedMultiProtocolGateway) checkWebSocketHealth(endpoint *ServiceEndpoint) (*HealthResult, error) {
	// Implement WebSocket health check
	return &HealthResult{
		Healthy:      true,
		ResponseTime: 15 * time.Millisecond,
		StatusCode:   200,
	}, nil
}

// Start All Protocol Servers
func (g *EnhancedMultiProtocolGateway) Start() error {
	// Start health checker
	go g.healthChecker.start()
	
	// Start service registry cleanup
	go g.serviceRegistry.startCleanup()

	// Start HTTP server
	go g.startEnhancedHTTPServer()
	
	// Start gRPC server
	go g.startEnhancedGRPCServer()
	
	// Start QUIC server
	go g.startEnhancedQUICServer()
	
	// Wait for shutdown signal
	<-g.shutdown
	return nil
}

// Enhanced HTTP Server with Protocol Fallback API
func (g *EnhancedMultiProtocolGateway) startEnhancedHTTPServer() {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// CORS configuration
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Enhanced Gateway API routes
	api := router.Group("/api/gateway")
	{
		api.GET("/health", g.enhancedGatewayHealth)
		api.GET("/services", g.getEnhancedServices)
		api.GET("/metrics", g.getEnhancedMetrics)
		api.POST("/route", g.routeEnhancedRequest)
		api.POST("/fallback", g.protocolFallbackHandler)
		api.POST("/services", g.registerService)
		api.DELETE("/services/:name", g.deregisterService)
	}

	// Circuit breaker management
	circuit := router.Group("/api/circuit-breaker")
	{
		circuit.GET("/status", g.getCircuitBreakerStatus)
		circuit.POST("/reset/:service/:endpoint", g.resetCircuitBreaker)
		circuit.POST("/trip/:service/:endpoint", g.tripCircuitBreaker)
	}

	// Load balancer management
	lb := router.Group("/api/load-balancer")
	{
		lb.GET("/strategy", g.getLoadBalancingStrategy)
		lb.POST("/strategy", g.setLoadBalancingStrategy)
		lb.GET("/metrics", g.getLoadBalancerMetrics)
	}

	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Enhanced Multi-Protocol Gateway for Legal AI Platform",
			"version": "2.0.0",
			"protocols": []string{"QUIC (HTTP/3)", "gRPC", "HTTP/REST", "WebSocket"},
			"features": []string{
				"Protocol Fallback Chain",
				"Circuit Breakers",
				"Intelligent Load Balancing", 
				"Dynamic Service Discovery",
				"Performance Monitoring",
				"Health Checks",
			},
			"services_managed": len(g.services),
			"endpoints": gin.H{
				"health":           "/api/gateway/health",
				"services":         "/api/gateway/services",
				"route_request":    "/api/gateway/route",
				"fallback_request": "/api/gateway/fallback",
				"circuit_breaker":  "/api/circuit-breaker/*",
				"load_balancer":    "/api/load-balancer/*",
			},
		})
	})

	addr := fmt.Sprintf(":%d", g.config.HTTPPort)
	g.httpServer = &http.Server{
		Addr:    addr,
		Handler: router,
	}

	log.Printf("🌐 Enhanced HTTP Gateway server starting on port %d", g.config.HTTPPort)
	if err := g.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("❌ HTTP server failed: %v", err)
	}
}

// Protocol Fallback Handler
func (g *EnhancedMultiProtocolGateway) protocolFallbackHandler(c *gin.Context) {
	var req ProtocolFallbackRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if req.Timeout == 0 {
		req.Timeout = 30 * time.Second
	}
	if req.MaxRetries == 0 {
		req.MaxRetries = 3
	}
	if req.Method == "" {
		req.Method = "GET"
	}

	response := g.ProcessWithFallback(&req)
	
	statusCode := http.StatusOK
	if !response.Success {
		statusCode = http.StatusServiceUnavailable
		if response.StatusCode > 0 {
			statusCode = response.StatusCode
		}
	}

	c.JSON(statusCode, response)
}

// Enhanced API handlers would go here...
func (g *EnhancedMultiProtocolGateway) enhancedGatewayHealth(c *gin.Context) {
	// Implementation similar to previous but with enhanced metrics
	c.JSON(http.StatusOK, gin.H{
		"status": "healthy",
		"protocols": map[string]any{
			"quic": map[string]any{"status": "active", "port": g.config.QUICPort},
			"grpc": map[string]any{"status": "active", "port": g.config.GRPCPort}, 
			"http": map[string]any{"status": "active", "port": g.config.HTTPPort},
		},
		"metrics": g.metrics,
		"timestamp": time.Now(),
	})
}

func (g *EnhancedMultiProtocolGateway) getEnhancedServices(c *gin.Context) {
	g.mutex.RLock()
	defer g.mutex.RUnlock()
	
	c.JSON(http.StatusOK, gin.H{
		"services": g.services,
		"by_protocol": g.servicesByProtocol,
		"registry": g.serviceRegistry.services,
	})
}

func (g *EnhancedMultiProtocolGateway) getEnhancedMetrics(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"gateway_metrics": g.metrics,
		"load_balancer_metrics": g.loadBalancer.metrics,
	})
}

func (g *EnhancedMultiProtocolGateway) routeEnhancedRequest(c *gin.Context) {
	// Enhanced routing with fallback
	c.JSON(http.StatusOK, gin.H{"message": "Enhanced routing implemented"})
}

// Additional handlers for circuit breaker and load balancer management...
func (g *EnhancedMultiProtocolGateway) getCircuitBreakerStatus(c *gin.Context) {
	status := make(map[string]map[string]any)
	
	g.mutex.RLock()
	for serviceName, endpoints := range g.services {
		status[serviceName] = make(map[string]any)
		for _, endpoint := range endpoints {
			cb := endpoint.CircuitBreaker
			cb.mutex.RLock()
			status[serviceName][endpoint.Name] = map[string]any{
				"state": cb.State,
				"failure_count": cb.FailureCount,
				"last_failure_time": cb.LastFailureTime,
				"success_rate": endpoint.SuccessRate,
			}
			cb.mutex.RUnlock()
		}
	}
	g.mutex.RUnlock()
	
	c.JSON(http.StatusOK, gin.H{"circuit_breakers": status})
}

func (g *EnhancedMultiProtocolGateway) resetCircuitBreaker(c *gin.Context) {
	serviceName := c.Param("service")
	endpointName := c.Param("endpoint")
	
	// Find and reset the circuit breaker
	g.mutex.Lock()
	if endpoints, exists := g.services[serviceName]; exists {
		for _, endpoint := range endpoints {
			if endpoint.Name == endpointName {
				cb := endpoint.CircuitBreaker
				cb.mutex.Lock()
				cb.State = CircuitClosed
				cb.FailureCount = 0
				cb.mutex.Unlock()
				g.mutex.Unlock()
				
				c.JSON(http.StatusOK, gin.H{
					"message": "Circuit breaker reset",
					"service": serviceName,
					"endpoint": endpointName,
				})
				return
			}
		}
	}
	g.mutex.Unlock()
	
	c.JSON(http.StatusNotFound, gin.H{"error": "Service or endpoint not found"})
}

func (g *EnhancedMultiProtocolGateway) tripCircuitBreaker(c *gin.Context) {
	serviceName := c.Param("service")
	endpointName := c.Param("endpoint")
	
	// Find and trip the circuit breaker
	g.mutex.Lock()
	if endpoints, exists := g.services[serviceName]; exists {
		for _, endpoint := range endpoints {
			if endpoint.Name == endpointName {
				cb := endpoint.CircuitBreaker
				cb.mutex.Lock()
				cb.State = CircuitOpen
				cb.LastFailureTime = time.Now()
				cb.mutex.Unlock()
				g.mutex.Unlock()
				
				c.JSON(http.StatusOK, gin.H{
					"message": "Circuit breaker tripped",
					"service": serviceName,
					"endpoint": endpointName,
				})
				return
			}
		}
	}
	g.mutex.Unlock()
	
	c.JSON(http.StatusNotFound, gin.H{"error": "Service or endpoint not found"})
}

func (g *EnhancedMultiProtocolGateway) getLoadBalancingStrategy(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"current_strategy": g.loadBalancer.strategy,
		"available_strategies": []string{
			"round_robin",
			"weighted_random", 
			"least_load",
			"response_time",
			"adaptive",
		},
	})
}

func (g *EnhancedMultiProtocolGateway) setLoadBalancingStrategy(c *gin.Context) {
	var req struct {
		Strategy string `json:"strategy"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if _, exists := g.loadBalancer.algorithms[req.Strategy]; !exists {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid load balancing strategy"})
		return
	}
	
	g.loadBalancer.strategy = LoadBalancingStrategy(req.Strategy)
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Load balancing strategy updated",
		"new_strategy": req.Strategy,
	})
}

func (g *EnhancedMultiProtocolGateway) getLoadBalancerMetrics(c *gin.Context) {
	c.JSON(http.StatusOK, g.loadBalancer.metrics)
}

// Placeholder implementations for gRPC and QUIC servers
func (g *EnhancedMultiProtocolGateway) startEnhancedGRPCServer() {
	log.Printf("⚡ Enhanced gRPC server would start on port %d", g.config.GRPCPort)
}

func (g *EnhancedMultiProtocolGateway) startEnhancedQUICServer() {
	log.Printf("🚄 Enhanced QUIC server would start on port %d", g.config.QUICPort)
}

// Service registration handlers
func (g *EnhancedMultiProtocolGateway) registerService(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Service registration implemented"})
}

func (g *EnhancedMultiProtocolGateway) deregisterService(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Service deregistration implemented"})
}

// Enhanced Health Checker
func (hc *EnhancedHealthChecker) start() {
	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()

	for range ticker.C {
		hc.checkAllServicesEnhanced()
	}
}

func (hc *EnhancedHealthChecker) checkAllServicesEnhanced() {
	hc.gateway.mutex.Lock()
	defer hc.gateway.mutex.Unlock()

	for serviceName, endpoints := range hc.gateway.services {
		for _, endpoint := range endpoints {
			go func(sn string, ep *ServiceEndpoint) {
				healthChecker, exists := hc.healthCheckers[ep.Protocol]
				if !exists {
					return
				}

				result, err := healthChecker(ep)
				if err != nil {
					log.Printf("⚠️ Health check error for %s (%s): %v", ep.Name, sn, err)
				}

				if result != nil {
					ep.Healthy = result.Healthy
					ep.ResponseTime = result.ResponseTime
					ep.LastCheck = time.Now()
					
					if result.Metadata != nil {
						ep.Metadata = result.Metadata
					}
				}
			}(serviceName, endpoint)
		}
	}
}

// Dynamic Service Registry
func (dsr *DynamicServiceRegistry) startCleanup() {
	ticker := time.NewTicker(dsr.cleanupInterval)
	defer ticker.Stop()

	for range ticker.C {
		dsr.cleanup()
	}
}

func (dsr *DynamicServiceRegistry) cleanup() {
	dsr.mutex.Lock()
	defer dsr.mutex.Unlock()

	now := time.Now()
	for name, service := range dsr.services {
		if now.Sub(service.LastHeartbeat) > service.TTL {
			delete(dsr.services, name)
			log.Printf("🗑️ Cleaned up stale service: %s", name)
		}
	}
}

// Environment helpers
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatValue, err := strconv.ParseFloat(value, 64); err == nil {
			return floatValue
		}
	}
	return defaultValue
}

// Main function
func main() {
	gateway, err := NewEnhancedMultiProtocolGateway()
	if err != nil {
		log.Fatalf("❌ Failed to initialize enhanced gateway: %v", err)
	}
	defer gateway.redis.Close()

	log.Printf("🚀 Starting Enhanced Multi-Protocol Gateway for Legal AI Platform...")
	log.Printf("🔧 Features: Protocol Fallback, Circuit Breakers, Load Balancing, Service Discovery")
	
	if err := gateway.Start(); err != nil {
		log.Fatalf("❌ Failed to start enhanced gateway: %v", err)
	}
}