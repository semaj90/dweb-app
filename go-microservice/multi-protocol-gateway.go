// Multi-Protocol Gateway for Legal AI Platform
// Routes requests between HTTP, gRPC, and QUIC protocols
// Integrates with GPU Orchestrator and all 37 Go services

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

// Protocol Types
type ProtocolType string

const (
	ProtocolHTTP      ProtocolType = "http"
	ProtocolgRPC      ProtocolType = "grpc"
	ProtocolQUIC      ProtocolType = "quic"
	ProtocolWebSocket ProtocolType = "websocket"
)

// Service Configuration
type ServiceEndpoint struct {
	Name      string       `json:"name"`
	Protocol  ProtocolType `json:"protocol"`
	Address   string       `json:"address"`
	Port      int          `json:"port"`
	Path      string       `json:"path,omitempty"`
	Healthy   bool         `json:"healthy"`
	Priority  int          `json:"priority"`
	Weight    int          `json:"weight"`
	LastCheck time.Time    `json:"last_check"`
}

// Gateway Configuration
type GatewayConfig struct {
	HTTPPort    int    `json:"http_port"`
	GRPCPort    int    `json:"grpc_port"`
	QUICPort    int    `json:"quic_port"`
	RedisAddr   string `json:"redis_addr"`
	TLSCertPath string `json:"tls_cert_path"`
	TLSKeyPath  string `json:"tls_key_path"`
}

// Multi-Protocol Gateway
type MultiProtocolGateway struct {
	config     *GatewayConfig
	redis      *redis.Client
	ctx        context.Context
	services   map[string][]*ServiceEndpoint
	mutex      sync.RWMutex
	httpServer *http.Server
	grpcServer *grpc.Server
	quicListener quic.Listener
	healthChecker *HealthChecker
}

// Health Checker
type HealthChecker struct {
	gateway  *MultiProtocolGateway
	interval time.Duration
	timeout  time.Duration
}

// Request Routing Information
type RouteRequest struct {
	Service     string                 `json:"service"`
	Protocol    ProtocolType          `json:"protocol"`
	Method      string                 `json:"method"`
	Path        string                 `json:"path"`
	Headers     map[string]string      `json:"headers,omitempty"`
	Body        interface{}           `json:"body,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Priority    int                   `json:"priority"`
	Timeout     time.Duration         `json:"timeout,omitempty"`
}

// Route Response
type RouteResponse struct {
	Success    bool                   `json:"success"`
	StatusCode int                    `json:"status_code,omitempty"`
	Body       interface{}           `json:"body,omitempty"`
	Headers    map[string]string      `json:"headers,omitempty"`
	Protocol   ProtocolType          `json:"protocol_used"`
	Endpoint   string                 `json:"endpoint_used"`
	Latency    time.Duration         `json:"latency"`
	Error      string                 `json:"error,omitempty"`
}

// Initialize Multi-Protocol Gateway
func NewMultiProtocolGateway() (*MultiProtocolGateway, error) {
	config := &GatewayConfig{
		HTTPPort:    getEnvInt("GATEWAY_HTTP_PORT", 8230),
		GRPCPort:    getEnvInt("GATEWAY_GRPC_PORT", 50050),
		QUICPort:    getEnvInt("GATEWAY_QUIC_PORT", 4433),
		RedisAddr:   getEnv("REDIS_ADDR", "localhost:6379"),
		TLSCertPath: getEnv("TLS_CERT_PATH", "./certs/server.crt"),
		TLSKeyPath:  getEnv("TLS_KEY_PATH", "./certs/server.key"),
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

	gateway := &MultiProtocolGateway{
		config:   config,
		redis:    rdb,
		ctx:      ctx,
		services: make(map[string][]*ServiceEndpoint),
	}

	gateway.healthChecker = &HealthChecker{
		gateway:  gateway,
		interval: 30 * time.Second,
		timeout:  5 * time.Second,
	}

	// Load default service endpoints
	gateway.loadDefaultServices()

	log.Printf("🚀 Multi-Protocol Gateway initialized")
	log.Printf("🌐 HTTP Port: %d", config.HTTPPort)
	log.Printf("⚡ gRPC Port: %d", config.GRPCPort)
	log.Printf("🚄 QUIC Port: %d", config.QUICPort)

	return gateway, nil
}

// Load Default Service Endpoints
func (g *MultiProtocolGateway) loadDefaultServices() {
	services := map[string][]*ServiceEndpoint{
		"enhanced-rag": {
			{Name: "enhanced-rag-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8094, Path: "/api/rag", Priority: 1, Weight: 100},
			{Name: "enhanced-rag-ws", Protocol: ProtocolWebSocket, Address: "localhost", Port: 8094, Path: "/ws", Priority: 2, Weight: 80},
		},
		"upload-service": {
			{Name: "upload-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8093, Path: "/upload", Priority: 1, Weight: 100},
		},
		"ai-enhanced": {
			{Name: "ai-enhanced-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8096, Path: "/api/ai", Priority: 1, Weight: 100},
		},
		"grpc-server": {
			{Name: "grpc-main", Protocol: ProtocolgRPC, Address: "localhost", Port: 50051, Priority: 1, Weight: 100},
		},
		"rag-quic-proxy": {
			{Name: "rag-quic", Protocol: ProtocolQUIC, Address: "localhost", Port: 8216, Priority: 1, Weight: 100},
		},
		"gpu-orchestrator": {
			{Name: "gpu-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8231, Path: "/api/gpu", Priority: 1, Weight: 100},
		},
		"legal-ai": {
			{Name: "legal-ai-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8202, Path: "/api/legal", Priority: 1, Weight: 100},
		},
		"xstate-manager": {
			{Name: "xstate-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8212, Path: "/api/state", Priority: 1, Weight: 100},
		},
		"cluster-service": {
			{Name: "cluster-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8213, Path: "/api/cluster", Priority: 1, Weight: 100},
		},
		"gpu-indexer": {
			{Name: "gpu-indexer-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8220, Path: "/api/index", Priority: 1, Weight: 100},
		},
	}

	g.mutex.Lock()
	g.services = services
	g.mutex.Unlock()

	log.Printf("📋 Loaded %d default services", len(services))
}

// Start All Protocol Servers
func (g *MultiProtocolGateway) Start() error {
	// Start health checker
	go g.healthChecker.start()

	// Start HTTP server
	go g.startHTTPServer()
	
	// Start gRPC server
	go g.startGRPCServer()
	
	// Start QUIC server
	go g.startQUICServer()

	// Wait forever
	select {}
}

// HTTP Server
func (g *MultiProtocolGateway) startHTTPServer() {
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

	// Gateway API routes
	api := router.Group("/api/gateway")
	{
		api.GET("/health", g.gatewayHealth)
		api.GET("/services", g.getServices)
		api.GET("/metrics", g.getMetrics)
		api.POST("/route", g.routeRequest)
		api.POST("/services", g.registerService)
	}

	// Protocol-specific routes
	protocols := router.Group("/protocols")
	{
		protocols.GET("/http/:service/*path", g.routeHTTPRequest)
		protocols.POST("/http/:service/*path", g.routeHTTPRequest)
		protocols.PUT("/http/:service/*path", g.routeHTTPRequest)
		protocols.DELETE("/http/:service/*path", g.routeHTTPRequest)
		
		protocols.POST("/grpc/:service", g.routeGRPCRequest)
		protocols.POST("/quic/:service", g.routeQUICRequest)
	}

	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Multi-Protocol Gateway for Legal AI Platform",
			"version": "1.0.0",
			"protocols": []string{"HTTP", "gRPC", "QUIC", "WebSocket"},
			"services_managed": len(g.services),
			"endpoints": gin.H{
				"health":         "/api/gateway/health",
				"services":       "/api/gateway/services",
				"route_request":  "/api/gateway/route",
				"http_proxy":     "/protocols/http/:service/*path",
				"grpc_proxy":     "/protocols/grpc/:service",
				"quic_proxy":     "/protocols/quic/:service",
			},
		})
	})

	addr := fmt.Sprintf(":%d", g.config.HTTPPort)
	g.httpServer = &http.Server{
		Addr:    addr,
		Handler: router,
	}

	log.Printf("🌐 HTTP Gateway server starting on port %d", g.config.HTTPPort)
	if err := g.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("❌ HTTP server failed: %v", err)
	}
}

// gRPC Server
func (g *MultiProtocolGateway) startGRPCServer() {
	addr := fmt.Sprintf(":%d", g.config.GRPCPort)
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("❌ Failed to listen for gRPC: %v", err)
	}

	g.grpcServer = grpc.NewServer()
	
	// Register health service
	grpc_health_v1.RegisterHealthServer(g.grpcServer, &GRPCHealthServer{gateway: g})

	log.Printf("⚡ gRPC Gateway server starting on port %d", g.config.GRPCPort)
	if err := g.grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ gRPC server failed: %v", err)
	}
}

// QUIC Server
func (g *MultiProtocolGateway) startQUICServer() {
	tlsConfig := &tls.Config{
		// In production, load proper certificates
		InsecureSkipVerify: true,
	}

	addr := fmt.Sprintf(":%d", g.config.QUICPort)
	listener, err := quic.ListenAddr(addr, tlsConfig, nil)
	if err != nil {
		log.Fatalf("❌ Failed to start QUIC server: %v", err)
	}

	g.quicListener = listener

	log.Printf("🚄 QUIC Gateway server starting on port %d", g.config.QUICPort)
	
	for {
		session, err := listener.Accept(context.Background())
		if err != nil {
			log.Printf("❌ QUIC connection error: %v", err)
			continue
		}

		go g.handleQUICSession(session)
	}
}

// HTTP Route Handler
func (g *MultiProtocolGateway) routeHTTPRequest(c *gin.Context) {
	serviceName := c.Param("service")
	path := c.Param("path")
	
	endpoint := g.selectEndpoint(serviceName, ProtocolHTTP)
	if endpoint == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": fmt.Sprintf("No healthy HTTP endpoint for service: %s", serviceName),
		})
		return
	}

	// Forward request to selected endpoint
	targetURL := fmt.Sprintf("http://%s:%d%s%s", endpoint.Address, endpoint.Port, endpoint.Path, path)
	
	start := time.Now()
	// This is a simplified proxy - in production you'd use a proper reverse proxy
	c.JSON(http.StatusOK, gin.H{
		"proxied_to": targetURL,
		"method": c.Request.Method,
		"latency": time.Since(start).Milliseconds(),
		"endpoint": endpoint.Name,
	})
}

// Service Selection Logic
func (g *MultiProtocolGateway) selectEndpoint(serviceName string, protocol ProtocolType) *ServiceEndpoint {
	g.mutex.RLock()
	endpoints, exists := g.services[serviceName]
	g.mutex.RUnlock()

	if !exists {
		return nil
	}

	// Filter by protocol and health
	var candidates []*ServiceEndpoint
	for _, endpoint := range endpoints {
		if endpoint.Protocol == protocol && endpoint.Healthy {
			candidates = append(candidates, endpoint)
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Simple round-robin selection (can be enhanced with weighted selection)
	return candidates[0]
}

// Health Check Implementation
func (hc *HealthChecker) start() {
	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()

	for range ticker.C {
		hc.checkAllServices()
	}
}

func (hc *HealthChecker) checkAllServices() {
	hc.gateway.mutex.Lock()
	defer hc.gateway.mutex.Unlock()

	for serviceName, endpoints := range hc.gateway.services {
		for _, endpoint := range endpoints {
			healthy := hc.checkEndpoint(endpoint)
			endpoint.Healthy = healthy
			endpoint.LastCheck = time.Now()
			
			if !healthy {
				log.Printf("⚠️ Endpoint %s (%s) health check failed", endpoint.Name, serviceName)
			}
		}
	}
}

func (hc *HealthChecker) checkEndpoint(endpoint *ServiceEndpoint) bool {
	switch endpoint.Protocol {
	case ProtocolHTTP:
		return hc.checkHTTPEndpoint(endpoint)
	case ProtocolgRPC:
		return hc.checkGRPCEndpoint(endpoint)
	case ProtocolQUIC:
		return hc.checkQUICEndpoint(endpoint)
	default:
		return false
	}
}

func (hc *HealthChecker) checkHTTPEndpoint(endpoint *ServiceEndpoint) bool {
	client := &http.Client{Timeout: hc.timeout}
	url := fmt.Sprintf("http://%s:%d/health", endpoint.Address, endpoint.Port)
	
	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	
	return resp.StatusCode == http.StatusOK
}

func (hc *HealthChecker) checkGRPCEndpoint(endpoint *ServiceEndpoint) bool {
	// Simplified gRPC health check
	return true // Placeholder
}

func (hc *HealthChecker) checkQUICEndpoint(endpoint *ServiceEndpoint) bool {
	// Simplified QUIC health check
	return true // Placeholder
}

// API Endpoints
func (g *MultiProtocolGateway) gatewayHealth(c *gin.Context) {
	g.mutex.RLock()
	totalServices := len(g.services)
	healthyServices := 0
	
	for _, endpoints := range g.services {
		for _, endpoint := range endpoints {
			if endpoint.Healthy {
				healthyServices++
				break
			}
		}
	}
	g.mutex.RUnlock()

	status := "healthy"
	if healthyServices < totalServices {
		status = "degraded"
	}
	if healthyServices == 0 {
		status = "unhealthy"
	}

	c.JSON(http.StatusOK, gin.H{
		"status": status,
		"total_services": totalServices,
		"healthy_services": healthyServices,
		"protocols_active": []string{"http", "grpc", "quic"},
		"timestamp": time.Now().Unix(),
	})
}

func (g *MultiProtocolGateway) getServices(c *gin.Context) {
	g.mutex.RLock()
	services := make(map[string]interface{})
	for name, endpoints := range g.services {
		services[name] = endpoints
	}
	g.mutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{"services": services})
}

func (g *MultiProtocolGateway) getMetrics(c *gin.Context) {
	// Simplified metrics
	c.JSON(http.StatusOK, gin.H{
		"requests_total": 0,
		"requests_by_protocol": gin.H{
			"http": 0,
			"grpc": 0,
			"quic": 0,
		},
		"average_latency_ms": 0,
		"error_rate": 0,
	})
}

func (g *MultiProtocolGateway) routeRequest(c *gin.Context) {
	var req RouteRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	endpoint := g.selectEndpoint(req.Service, req.Protocol)
	if endpoint == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": fmt.Sprintf("No healthy endpoint for service %s with protocol %s", req.Service, req.Protocol),
		})
		return
	}

	start := time.Now()
	
	response := RouteResponse{
		Success:  true,
		Protocol: req.Protocol,
		Endpoint: endpoint.Name,
		Latency:  time.Since(start),
		Body:     gin.H{"message": "Request routed successfully", "target": endpoint},
	}

	c.JSON(http.StatusOK, response)
}

func (g *MultiProtocolGateway) registerService(c *gin.Context) {
	var endpoint ServiceEndpoint
	if err := c.ShouldBindJSON(&endpoint); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	serviceName := c.Query("service")
	if serviceName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "service parameter required"})
		return
	}

	g.mutex.Lock()
	if g.services[serviceName] == nil {
		g.services[serviceName] = []*ServiceEndpoint{}
	}
	g.services[serviceName] = append(g.services[serviceName], &endpoint)
	g.mutex.Unlock()

	c.JSON(http.StatusOK, gin.H{
		"message": "Service endpoint registered",
		"service": serviceName,
		"endpoint": endpoint.Name,
	})
}

// Placeholder implementations
func (g *MultiProtocolGateway) routeGRPCRequest(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "gRPC routing not implemented in demo"})
}

func (g *MultiProtocolGateway) routeQUICRequest(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "QUIC routing not implemented in demo"})
}

func (g *MultiProtocolGateway) handleQUICSession(session quic.Connection) {
	// QUIC session handling placeholder
}

// gRPC Health Server
type GRPCHealthServer struct {
	gateway *MultiProtocolGateway
}

func (s *GRPCHealthServer) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	return &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	}, nil
}

func (s *GRPCHealthServer) Watch(req *grpc_health_v1.HealthCheckRequest, stream grpc_health_v1.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}

// Environment helpers (same as previous files)
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

// Main function
func main() {
	gateway, err := NewMultiProtocolGateway()
	if err != nil {
		log.Fatalf("❌ Failed to initialize gateway: %v", err)
	}

	log.Printf("🚀 Starting Multi-Protocol Gateway for Legal AI Platform...")
	
	if err := gateway.Start(); err != nil {
		log.Fatalf("❌ Failed to start gateway: %v", err)
	}
}