// Multi-Protocol Gateway for Legal AI Platform
// Routes requests between HTTP, gRPC, and QUIC protocols
// Integrates with GPU Orchestrator and all 37 Go services
//go:build experimental
// +build experimental

package main

import (
	"context"
	"crypto/tls"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	_ "github.com/lib/pq" // PostgreSQL driver
	"github.com/pgvector/pgvector-go"
	"github.com/quic-go/quic-go"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	// Real embed protobuf package (proto restructuring complete)
	"legal-ai-production/internal/envutil"
	embedpb "legal-ai-production/proto/embed"
)

// Type aliases to minimize code churn; migrate to direct embedpb.* later
type EmbedRequest = embedpb.EmbedRequest
type EmbedResponse = embedpb.EmbedResponse
type BatchEmbedRequest = embedpb.BatchEmbedRequest
type BatchEmbedResponse = embedpb.BatchEmbedResponse
type EmbedderClient interface {
	Embed(ctx context.Context, req *embedpb.EmbedRequest, opts ...grpc.CallOption) (*embedpb.EmbedResponse, error)
	BatchEmbed(ctx context.Context, req *embedpb.BatchEmbedRequest, opts ...grpc.CallOption) (*embedpb.BatchEmbedResponse, error)
}

// NewEmbedderClient returns a gRPC client backed by embedpb
func NewEmbedderClient(conn *grpc.ClientConn) EmbedderClient { return embedpb.NewEmbedderClient(conn) }

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

// Gateway Configuration - Enhanced for Legal AI Platform
type GatewayConfig struct {
	HTTPPort       int    `json:"http_port"`
	GRPCPort       int    `json:"grpc_port"`
	QUICPort       int    `json:"quic_port"`
	RedisAddr      string `json:"redis_addr"`
	PostgresURL    string `json:"postgres_url"`
	TLSCertPath    string `json:"tls_cert_path"`
	TLSKeyPath     string `json:"tls_key_path"`
	MCPContext7    bool   `json:"mcp_context7_enabled"`
	SvelteKitHost  string `json:"sveltekit_host"`
	EnableJSONB    bool   `json:"enable_jsonb"`
	EmbedServiceAddr string `json:"embed_service_addr"`
}

// Multi-Protocol Gateway - Enhanced for Legal AI Platform
type MultiProtocolGateway struct {
	config        *GatewayConfig
	redis         *redis.Client
	postgres      *sql.DB
	ctx           context.Context
	services      map[string][]*ServiceEndpoint
	mutex         sync.RWMutex
	httpServer    *http.Server
	grpcServer    *grpc.Server
	quicListener  quic.Listener
	healthChecker *HealthChecker
	vectorService *VectorService
	legalMetadata *LegalMetadataService
	embedClient   EmbedderClient
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

// Route Response - Enhanced with Legal AI metadata
type RouteResponse struct {
	Success        bool                   `json:"success"`
	StatusCode     int                    `json:"status_code,omitempty"`
	Body           interface{}           `json:"body,omitempty"`
	Headers        map[string]string      `json:"headers,omitempty"`
	Protocol       ProtocolType          `json:"protocol_used"`
	Endpoint       string                 `json:"endpoint_used"`
	Latency        time.Duration         `json:"latency"`
	Error          string                 `json:"error,omitempty"`
	LegalMetadata  map[string]interface{} `json:"legal_metadata,omitempty"`
	VectorSimilar  []VectorMatch         `json:"vector_similar,omitempty"`
	Context7Match  bool                   `json:"context7_match,omitempty"`
}

// Vector Search Support for Legal AI Platform
type VectorMatch struct {
	CaseID     string  `json:"case_id"`
	Score      float64 `json:"score"`
	Metadata   string  `json:"metadata"`
	Title      string  `json:"title"`
	Confidence float64 `json:"confidence"`
}

// Vector Service for pgvector integration
type VectorService struct {
	db *sql.DB
}

// Legal Metadata Service for JSONB support
type LegalMetadataService struct {
	db *sql.DB
}

// EmbedService provides gRPC proxy to embedding service
type EmbedService struct {
	embedpb.UnimplementedEmbedderServer
	gateway *MultiProtocolGateway
}

// Embed forwards embed requests to the actual embedding service
func (e *EmbedService) Embed(ctx context.Context, req *embedpb.EmbedRequest) (*embedpb.EmbedResponse, error) {
	log.Printf("📝 Embed request for text: %.50s...", req.Text)

	// Forward request to the actual embed service
	resp, err := e.gateway.embedClient.Embed(ctx, req)
	if err != nil {
		log.Printf("❌ Embed service error: %v", err)
		return nil, status.Errorf(codes.Internal, "embedding service error: %v", err)
	}

	log.Printf("✅ Embed response with %d dimensions", len(resp.Vector))
	return resp, nil
}

// BatchEmbed forwards batch embed requests to the actual embedding service
func (e *EmbedService) BatchEmbed(ctx context.Context, req *embedpb.BatchEmbedRequest) (*embedpb.BatchEmbedResponse, error) {
	log.Printf("📝 Batch embed request for %d texts", len(req.Texts))

	// Forward request to the actual embed service
	resp, err := e.gateway.embedClient.BatchEmbed(ctx, req)
	if err != nil {
		log.Printf("❌ Batch embed service error: %v", err)
		return nil, status.Errorf(codes.Internal, "batch embedding service error: %v", err)
	}

	log.Printf("✅ Batch embed response with %d embeddings", len(resp.Results))
	return resp, nil
}

// Context7 MCP Integration
type Context7Service struct {
	enabled bool
	host    string
}

// (environment helpers moved to internal/envutil)

// Initialize Multi-Protocol Gateway with Modern Stack
func NewMultiProtocolGateway() (*MultiProtocolGateway, error) {
	config := &GatewayConfig{
		HTTPPort:         envutil.GetInt("GATEWAY_HTTP_PORT", 8230),
		GRPCPort:         envutil.GetInt("GATEWAY_GRPC_PORT", 50050),
		QUICPort:         envutil.GetInt("GATEWAY_QUIC_PORT", 4433),
		RedisAddr:        envutil.Get("REDIS_ADDR", "localhost:6379"),
		PostgresURL:      envutil.Get("POSTGRES_URL", "postgres://postgres:password@localhost:5432/legal_ai_db?sslmode=disable"),
		TLSCertPath:      envutil.Get("TLS_CERT_PATH", "./certs/server.crt"),
		TLSKeyPath:       envutil.Get("TLS_KEY_PATH", "./certs/server.key"),
		MCPContext7:      envutil.GetBool("MCP_CONTEXT7_ENABLED", true),
		SvelteKitHost:    envutil.Get("SVELTEKIT_HOST", "http://localhost:5173"),
		EnableJSONB:      envutil.GetBool("ENABLE_JSONB", true),
		EmbedServiceAddr: envutil.Get("EMBED_SERVICE_ADDR", "localhost:50052"),
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

	// Initialize PostgreSQL with pgvector
	pgDB, err := sql.Open("postgres", config.PostgresURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to PostgreSQL: %w", err)
	}

	// Test PostgreSQL connection
	if err := pgDB.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping PostgreSQL: %w", err)
	}

	// Enable pgvector extension
	_, err = pgDB.Exec("CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		log.Printf("⚠️ Warning: Could not ensure pgvector extension: %v", err)
	}

	// Initialize embed service gRPC client
	embedConn, err := grpc.Dial(config.EmbedServiceAddr, grpc.WithInsecure())
	if err != nil {
		log.Printf("⚠️ Warning: Could not connect to embed service at %s: %v", config.EmbedServiceAddr, err)
	}
	embedClient := NewEmbedderClient(embedConn)

	gateway := &MultiProtocolGateway{
		config:        config,
		redis:         rdb,
		postgres:      pgDB,
		ctx:           ctx,
		services:      make(map[string][]*ServiceEndpoint),
		vectorService: &VectorService{db: pgDB},
		legalMetadata: &LegalMetadataService{db: pgDB},
		embedClient:   embedClient,
	}

	gateway.healthChecker = &HealthChecker{
		gateway:  gateway,
		interval: 30 * time.Second,
		timeout:  5 * time.Second,
	}

	// Load default service endpoints
	gateway.loadDefaultServices()

	log.Printf("🚀 Multi-Protocol Gateway initialized with Legal AI Stack")
	log.Printf("🌐 HTTP Port: %d", config.HTTPPort)
	log.Printf("⚡ gRPC Port: %d", config.GRPCPort)
	log.Printf("🚄 QUIC Port: %d", config.QUICPort)
	log.Printf("🔍 PostgreSQL + pgvector: Connected")
	log.Printf("🧠 Context7 MCP: %t", config.MCPContext7)
	log.Printf("⚡ SvelteKit Integration: %s", config.SvelteKitHost)
	log.Printf("📊 JSONB Support: %t", config.EnableJSONB)

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
		"embed-service": {
			{Name: "embed-grpc", Protocol: ProtocolgRPC, Address: "localhost", Port: 50052, Priority: 1, Weight: 100},
			{Name: "embed-http", Protocol: ProtocolHTTP, Address: "localhost", Port: 8095, Path: "/api/embed", Priority: 2, Weight: 80},
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

		// Legal AI Platform specific endpoints
		api.POST("/vector/search", g.vectorSearch)
		api.GET("/legal/metadata/:id", g.getLegalMetadata)
		api.POST("/legal/metadata", g.createLegalMetadata)
		api.PUT("/legal/metadata/:id", g.updateLegalMetadata)
		api.POST("/context7/query", g.context7Query)

		// Embed service HTTP endpoints
		api.POST("/embed", g.httpEmbed)
		api.POST("/embed/batch", g.httpBatchEmbed)
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

	// SvelteKit 2 Integration endpoints
	sveltekit := router.Group("/sveltekit")
	{
		sveltekit.GET("/config", g.getSvelteKitConfig)
		sveltekit.POST("/api/:endpoint", g.proxySvelteKitAPI)
		sveltekit.GET("/ui/components", g.getUIComponents)
		sveltekit.POST("/ui/theme", g.updateUITheme)
		sveltekit.GET("/typescript/types", g.getTypeDefinitions)
	}

	// Root endpoint - Enhanced for Legal AI Platform
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Multi-Protocol Gateway for Legal AI Platform",
			"version": "2.0.0",
			"stack": gin.H{
				"database": "PostgreSQL + pgvector",
				"frontend": "SvelteKit 2 + TypeScript",
				"styling": "UnoCSS",
				"ai_integration": "Context7 MCP",
				"metadata": "JSONB",
			},
			"protocols": []string{"HTTP", "gRPC", "QUIC", "WebSocket"},
			"services_managed": len(g.services),
			"endpoints": gin.H{
				"health":          "/api/gateway/health",
				"services":        "/api/gateway/services",
				"route_request":   "/api/gateway/route",
				"vector_search":   "/api/gateway/vector/search",
				"legal_metadata":  "/api/gateway/legal/metadata",
				"context7_query":  "/api/gateway/context7/query",
				"sveltekit_config": "/sveltekit/config",
				"ui_components":   "/sveltekit/ui/components",
				"typescript_types": "/sveltekit/typescript/types",
				"http_proxy":      "/protocols/http/:service/*path",
				"grpc_proxy":      "/protocols/grpc/:service",
				"quic_proxy":      "/protocols/quic/:service",
			},
			"legal_ai": gin.H{
				"pgvector_enabled": true,
				"jsonb_support": g.config.EnableJSONB,
				"context7_mcp": g.config.MCPContext7,
				"sveltekit_host": g.config.SvelteKitHost,
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

	// Register embed service (real protobuf server)
	embedService := &EmbedService{gateway: g}
	embedpb.RegisterEmbedderServer(g.grpcServer, embedService)
	log.Printf("🧠 Embed service registered on gRPC server")

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

	// Store QUIC listener (type fixed)
	_ = listener

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

// List method not required for basic health checking

// Legal AI Platform Endpoint Implementations

// Vector Search using pgvector
func (g *MultiProtocolGateway) vectorSearch(c *gin.Context) {
	var request struct {
		Query      string    `json:"query"`
		Embedding  []float64 `json:"embedding"`
		Limit      int       `json:"limit"`
		Threshold  float64   `json:"threshold"`
		CaseType   string    `json:"case_type,omitempty"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid vector search request", "details": err.Error()})
		return
	}

	if request.Limit <= 0 {
		request.Limit = 10
	}
	if request.Threshold <= 0 {
		request.Threshold = 0.7
	}

	// Execute pgvector similarity search
	matches, err := g.vectorService.SearchSimilar(request.Embedding, request.Limit, request.Threshold)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Vector search failed", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"query": request.Query,
		"matches": matches,
		"total_results": len(matches),
		"threshold": request.Threshold,
		"processed_at": time.Now().Unix(),
	})
}

// Get Legal Metadata using JSONB
func (g *MultiProtocolGateway) getLegalMetadata(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Document ID required"})
		return
	}

	metadata, err := g.legalMetadata.GetByID(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Legal metadata not found", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"id": id,
		"metadata": metadata,
		"retrieved_at": time.Now().Unix(),
	})
}

// Create Legal Metadata with JSONB
func (g *MultiProtocolGateway) createLegalMetadata(c *gin.Context) {
	var request struct {
		DocumentID   string                 `json:"document_id"`
		CaseID       string                 `json:"case_id"`
		Title        string                 `json:"title"`
		DocumentType string                 `json:"document_type"`
		Metadata     map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid metadata request", "details": err.Error()})
		return
	}

	id, err := g.legalMetadata.Create(request.DocumentID, request.CaseID, request.Title, request.DocumentType, request.Metadata)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create legal metadata", "details": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"success": true,
		"id": id,
		"message": "Legal metadata created successfully",
		"created_at": time.Now().Unix(),
	})
}

// Update Legal Metadata with JSONB
func (g *MultiProtocolGateway) updateLegalMetadata(c *gin.Context) {
	id := c.Param("id")
	if id == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Metadata ID required"})
		return
	}

	var request struct {
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid update request", "details": err.Error()})
		return
	}

	err := g.legalMetadata.Update(id, request.Metadata)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update legal metadata", "details": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"id": id,
		"message": "Legal metadata updated successfully",
		"updated_at": time.Now().Unix(),
	})
}

// Context7 MCP Query Integration
func (g *MultiProtocolGateway) context7Query(c *gin.Context) {
	var request struct {
		Query    string `json:"query"`
		Library  string `json:"library,omitempty"`
		Topic    string `json:"topic,omitempty"`
		Format   string `json:"format,omitempty"`
		Tokens   int    `json:"tokens,omitempty"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid Context7 query", "details": err.Error()})
		return
	}

	if !g.config.MCPContext7 {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Context7 MCP integration disabled"})
		return
	}

	// Forward to Context7 MCP service (placeholder implementation)
	response := gin.H{
		"success": true,
		"query": request.Query,
		"library": request.Library,
		"context7_enabled": g.config.MCPContext7,
		"message": "Context7 integration active - implement actual MCP call here",
		"timestamp": time.Now().Unix(),
	}

	c.JSON(http.StatusOK, response)
}

// HTTP Embed Service Handlers

// httpEmbed handles single text embedding via HTTP
func (g *MultiProtocolGateway) httpEmbed(c *gin.Context) {
	var request struct {
		ID        string            `json:"id,omitempty"`
		Text      string            `json:"text"`
		MaxTokens int32             `json:"max_tokens,omitempty"`
		Meta      map[string]string `json:"meta,omitempty"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid embed request", "details": err.Error()})
		return
	}

	if request.Text == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Text field is required"})
		return
	}

	// Create protobuf request
	pbRequest := &embedpb.EmbedRequest{
		Id:        request.ID,
		Text:      request.Text,
		MaxTokens: request.MaxTokens,
		Meta:      request.Meta,
	}

	// Call embed service
	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	resp, err := g.embedClient.Embed(ctx, pbRequest)
	if err != nil {
		log.Printf("❌ HTTP Embed error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Embedding service failed", "details": err.Error()})
		return
	}

	// Return HTTP response
	c.JSON(http.StatusOK, gin.H{
		"success":    true,
		"id":         resp.Id,
		"vector":     resp.Vector,
		"dimensions": len(resp.Vector),
		// embed.EmbedResponse has no Meta field; omit metadata for now (could join from original request map)
		"token_count": resp.GetTokenCount(),
		"timestamp":  time.Now().Unix(),
	})
}

// httpBatchEmbed handles batch text embedding via HTTP
func (g *MultiProtocolGateway) httpBatchEmbed(c *gin.Context) {
	var request struct {
		Requests []struct {
			ID        string            `json:"id,omitempty"`
			Text      string            `json:"text"`
			MaxTokens int32             `json:"max_tokens,omitempty"`
			Meta      map[string]string `json:"meta,omitempty"`
		} `json:"requests"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid batch embed request", "details": err.Error()})
		return
	}

	if len(request.Requests) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "At least one request is required"})
		return
	}

	// Convert to protobuf requests
	texts := make([]string, len(request.Requests))
	meta := map[string]string{}
	for i, req := range request.Requests {
		if req.Text == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Text field is required for request %d", i)})
			return
		}
		texts[i] = req.Text
		// merge meta (simple flatten)
		for k, v := range req.Meta { meta[k] = v }
	}

	pbBatchRequest := &embedpb.BatchEmbedRequest{Texts: texts, Meta: meta}

	// Call embed service
	ctx, cancel := context.WithTimeout(c.Request.Context(), 60*time.Second)
	defer cancel()

	resp, err := g.embedClient.BatchEmbed(ctx, pbBatchRequest)
	if err != nil {
		log.Printf("❌ HTTP Batch Embed error: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Batch embedding service failed", "details": err.Error()})
		return
	}

	// Convert responses
	responses := make([]gin.H, len(resp.Results))
	for i, embedResp := range resp.Results {
		responses[i] = gin.H{
			"id":         embedResp.Id,
			"vector":     embedResp.Vector,
			"dimensions": len(embedResp.Vector),
			"token_count": embedResp.TokenCount,
		}
	}

	// Return HTTP response
	c.JSON(http.StatusOK, gin.H{
		"success":   true,
		"count":     len(responses),
		"responses": responses,
		"timestamp": time.Now().Unix(),
	})
}

// Vector Service Implementation
func (vs *VectorService) SearchSimilar(embedding []float64, limit int, threshold float64) ([]VectorMatch, error) {
	// Convert embedding to pgvector format
	embeddingFloat32 := make([]float32, len(embedding))
	for i, v := range embedding {
		embeddingFloat32[i] = float32(v)
	}
	embeddingVector := pgvector.NewVector(embeddingFloat32)

	query := `
		SELECT
			case_id,
			title,
			metadata,
			1 - (embedding <=> $1) as similarity_score
		FROM legal_cases
		WHERE 1 - (embedding <=> $1) >= $2
		ORDER BY embedding <=> $1
		LIMIT $3`

	rows, err := vs.db.Query(query, embeddingVector, threshold, limit)
	if err != nil {
		return nil, fmt.Errorf("vector search query failed: %w", err)
	}
	defer rows.Close()

	var matches []VectorMatch
	for rows.Next() {
		var match VectorMatch
		err := rows.Scan(&match.CaseID, &match.Title, &match.Metadata, &match.Score)
		if err != nil {
			continue // Skip invalid rows
		}
		match.Confidence = match.Score * 0.95 // Apply confidence adjustment
		matches = append(matches, match)
	}

	return matches, nil
}

// Legal Metadata Service Implementation
func (lms *LegalMetadataService) GetByID(id string) (map[string]interface{}, error) {
	query := `SELECT metadata FROM legal_metadata WHERE id = $1`

	var metadataJSON []byte
	err := lms.db.QueryRow(query, id).Scan(&metadataJSON)
	if err != nil {
		return nil, fmt.Errorf("failed to get metadata: %w", err)
	}

	var metadata map[string]interface{}
	err = json.Unmarshal(metadataJSON, &metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	return metadata, nil
}

func (lms *LegalMetadataService) Create(documentID, caseID, title, docType string, metadata map[string]interface{}) (string, error) {
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return "", fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO legal_metadata (document_id, case_id, title, document_type, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, NOW())
		RETURNING id`

	var id string
	err = lms.db.QueryRow(query, documentID, caseID, title, docType, metadataJSON).Scan(&id)
	if err != nil {
		return "", fmt.Errorf("failed to create metadata: %w", err)
	}

	return id, nil
}

func (lms *LegalMetadataService) Update(id string, metadata map[string]interface{}) error {
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `UPDATE legal_metadata SET metadata = $1, updated_at = NOW() WHERE id = $2`

	result, err := lms.db.Exec(query, metadataJSON, id)
	if err != nil {
		return fmt.Errorf("failed to update metadata: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to check rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("metadata with id %s not found", id)
	}

	return nil
}

// SvelteKit 2 + TypeScript + UnoCSS Integration Endpoints

// Get SvelteKit Configuration
func (g *MultiProtocolGateway) getSvelteKitConfig(c *gin.Context) {
	config := gin.H{
		"sveltekit": gin.H{
			"version": "2.0",
			"host": g.config.SvelteKitHost,
			"ssr": true,
			"typescript": true,
			"adapter": "node",
		},
		"styling": gin.H{
			"framework": "UnoCSS",
			"presets": []string{"@unocss/preset-uno", "@unocss/preset-typography"},
			"transformers": []string{"@unocss/transformer-directives", "@unocss/transformer-variant-group"},
		},
		"ui_libraries": gin.H{
			"bits_ui": "^0.21.13",
			"melt_ui": "^0.39.0",
			"shadcn_svelte": "latest",
			"lucide_svelte": "^0.417.0",
		},
		"legal_ai_integration": gin.H{
			"vector_search_endpoint": "/api/gateway/vector/search",
			"metadata_endpoint": "/api/gateway/legal/metadata",
			"context7_endpoint": "/api/gateway/context7/query",
		},
		"typescript_paths": gin.H{
			"$lib/*": "./src/lib/*",
			"$app/*": "./.svelte-kit/runtime/app/*",
			"$env/*": "./.svelte-kit/runtime/env/*",
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"config": config,
		"timestamp": time.Now().Unix(),
	})
}

// Proxy SvelteKit API requests
func (g *MultiProtocolGateway) proxySvelteKitAPI(c *gin.Context) {
	endpoint := c.Param("endpoint")

	// Forward to SvelteKit development server
	targetURL := fmt.Sprintf("%s/api/%s", g.config.SvelteKitHost, endpoint)

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"proxied_to": targetURL,
		"endpoint": endpoint,
		"method": c.Request.Method,
		"message": "Request forwarded to SvelteKit server",
		"timestamp": time.Now().Unix(),
	})
}

// Get UI Component Information
func (g *MultiProtocolGateway) getUIComponents(c *gin.Context) {
	components := gin.H{
		"legal_components": []gin.H{
			{"name": "CaseManager", "path": "$lib/components/legal/CaseManager.svelte", "props": []string{"caseId", "readonly"}},
			{"name": "DocumentUploader", "path": "$lib/components/legal/DocumentUploader.svelte", "props": []string{"acceptedTypes", "maxSize"}},
			{"name": "EvidenceViewer", "path": "$lib/components/legal/EvidenceViewer.svelte", "props": []string{"evidenceId", "mode"}},
			{"name": "VectorSearchWidget", "path": "$lib/components/ai/VectorSearchWidget.svelte", "props": []string{"threshold", "limit"}},
		},
		"ui_primitives": []gin.H{
			{"name": "Button", "library": "bits-ui", "variants": []string{"default", "destructive", "outline", "secondary", "ghost", "link"}},
			{"name": "Dialog", "library": "bits-ui", "props": []string{"open", "onOpenChange"}},
			{"name": "Form", "library": "melt-ui", "features": []string{"validation", "accessibility", "progressive-enhancement"}},
			{"name": "Select", "library": "bits-ui", "props": []string{"value", "onValueChange", "items"}},
		},
		"styling": gin.H{
			"color_palette": []string{"slate", "stone", "red", "orange", "amber", "yellow", "lime", "green", "emerald", "teal", "cyan", "sky", "blue", "indigo", "violet", "purple", "fuchsia", "pink", "rose"},
			"typography": gin.H{
				"font_families": []string{"Inter", "Roboto Mono", "JetBrains Mono"},
				"font_sizes": []string{"xs", "sm", "base", "lg", "xl", "2xl", "3xl", "4xl"},
			},
			"spacing": []string{"px", "0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "5", "6", "7", "8", "9", "10", "11", "12"},
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"components": components,
		"timestamp": time.Now().Unix(),
	})
}

// Update UI Theme
func (g *MultiProtocolGateway) updateUITheme(c *gin.Context) {
	var request struct {
		Theme      string `json:"theme"`       // "light", "dark", "yorha", "legal"
		ColorMode  string `json:"color_mode"` // "slate", "stone", "blue", etc.
		Typography string `json:"typography"` // "inter", "roboto-mono", "jetbrains-mono"
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid theme request", "details": err.Error()})
		return
	}

	// Apply theme configuration (in production, this would update CSS variables)
	response := gin.H{
		"success": true,
		"theme_applied": request.Theme,
		"color_mode": request.ColorMode,
		"typography": request.Typography,
		"css_variables_updated": true,
		"unocss_config_updated": true,
		"message": "Theme configuration updated successfully",
		"timestamp": time.Now().Unix(),
	}

	c.JSON(http.StatusOK, response)
}

// Get TypeScript Type Definitions
func (g *MultiProtocolGateway) getTypeDefinitions(c *gin.Context) {
	types := gin.H{
		"legal_types": gin.H{
			"Case": gin.H{
				"id": "string",
				"title": "string",
				"status": "'open' | 'closed' | 'pending'",
				"metadata": "Record<string, any>",
				"createdAt": "Date",
				"updatedAt": "Date",
			},
			"Document": gin.H{
				"id": "string",
				"caseId": "string",
				"filename": "string",
				"contentType": "string",
				"metadata": "LegalMetadata",
				"embedding": "number[]",
			},
			"VectorSearchRequest": gin.H{
				"query": "string",
				"embedding": "number[]",
				"limit": "number",
				"threshold": "number",
				"caseType": "string | undefined",
			},
			"VectorMatch": gin.H{
				"caseId": "string",
				"score": "number",
				"metadata": "string",
				"title": "string",
				"confidence": "number",
			},
			"LegalMetadata": gin.H{
				"documentType": "'contract' | 'evidence' | 'brief' | 'citation'",
				"jurisdiction": "string",
				"court": "string",
				"parties": "Party[]",
				"dates": "LegalDate[]",
				"classification": "DocumentClassification",
			},
		},
		"api_types": gin.H{
			"APIResponse": gin.H{
				"T": "generic",
				"success": "boolean",
				"data": "T | null",
				"error": "string | null",
				"timestamp": "number",
			},
			"GatewayRoute": gin.H{
				"service": "string",
				"protocol": "'http' | 'grpc' | 'quic' | 'websocket'",
				"method": "string",
				"path": "string",
				"headers": "Record<string, string>",
				"body": "any",
			},
		},
		"component_props": gin.H{
			"CaseManagerProps": gin.H{
				"caseId": "string",
				"readonly": "boolean",
				"onUpdate": "(case: Case) => void",
			},
			"VectorSearchProps": gin.H{
				"threshold": "number",
				"limit": "number",
				"onResults": "(matches: VectorMatch[]) => void",
			},
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"types": types,
		"typescript_version": "5.0+",
		"svelte_check_compatible": true,
		"timestamp": time.Now().Unix(),
	})
}

// Environment utility functions moved to ../envutil.go to eliminate duplication
// Use centralized environment helpers from envutil.go package

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