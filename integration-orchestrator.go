// Legal AI Platform Integration Orchestrator
// Connects new vector processing pipeline with existing 37 Go microservices
// Native Windows, Multi-Protocol (HTTP/gRPC/QUIC/WebSocket/NATS)

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go"
)

// Integration Orchestrator - Links vector pipeline with existing services
type IntegrationOrchestrator struct {
	// Existing services (from GO_BINARIES_CATALOG.md)
	enhancedRAG         *ServiceClient // Port 8094
	uploadService       *ServiceClient // Port 8093
	clusterManager      *ServiceClient // Port 8213
	xstateManager       *ServiceClient // Port 8212
	loadBalancer        *ServiceClient // Port 8222
	liveAgent          *ServiceClient // Port 8200

	// New vector processing services
	vectorWorker        *ServiceClient // Our new Go worker
	embeddingService    *ServiceClient // Python FastAPI service

	// Infrastructure
	postgres    *pgxpool.Pool
	redis       *redis.Client
	nats        *nats.Conn

	// Configuration from existing system
	config      *PlatformConfig

	// Service registry
	services    map[string]*ServiceInfo
	healthMutex sync.RWMutex
}

type ServiceInfo struct {
	Name     string    `json:"name"`
	Port     int       `json:"port"`
	Status   string    `json:"status"` // healthy, degraded, failed
	Protocol string    `json:"protocol"` // http, grpc, quic, ws, nats
	Latency  time.Duration `json:"latency"`
	LastCheck time.Time   `json:"lastCheck"`
}

type ServiceClient struct {
	Name     string
	BaseURL  string
	Protocol string
	Client   *http.Client
}

type PlatformConfig struct {
	// Existing service ports (from GO_BINARIES_CATALOG.md)
	Services map[string]int `json:"services"`

	// Vector processing configuration
	VectorConfig struct {
		RedisStream      string `json:"redisStream"`
		QdrantURL        string `json:"qdrantURL"`
		EmbeddingService string `json:"embeddingService"`
		CUDAWorker       string `json:"cudaWorker"`
	} `json:"vectorConfig"`

	// NATS subjects (from FULL_STACK_INTEGRATION_COMPLETE.md)
	NATSSubjects struct {
		CaseEvents     []string `json:"caseEvents"`
		DocumentEvents []string `json:"documentEvents"`
		AIEvents       []string `json:"aiEvents"`
		SystemEvents   []string `json:"systemEvents"`
	} `json:"natsSubjects"`
}

func NewIntegrationOrchestrator() (*IntegrationOrchestrator, error) {
	config := &PlatformConfig{
		Services: map[string]int{
			// AI/RAG Services (from GO_BINARIES_CATALOG.md)
			"enhanced-rag":                    8094,
			"upload-service":                  8093,
			"ai-enhanced":                     8096,
			"enhanced-legal-ai":               8202,
			"live-agent-enhanced":             8200,

			// Orchestration Services
			"xstate-manager":                  8212,
			"cluster-http":                    8213,
			"load-balancer":                   8222,

			// Protocol Services
			"grpc-server":                     50051,
			"rag-kratos":                      50052,
			"rag-quic-proxy":                  8216,

			// New Vector Services
			"vector-worker":                   8095,
			"embedding-service":               8096,
		},
	}

	// Vector processing configuration
	config.VectorConfig.RedisStream = "vec:requests"
	config.VectorConfig.QdrantURL = "http://localhost:6333"
	config.VectorConfig.EmbeddingService = "http://localhost:8096"
	config.VectorConfig.CUDAWorker = "./cuda-rotate-worker.exe"

	// NATS subjects (from FULL_STACK_INTEGRATION_COMPLETE.md)
	config.NATSSubjects.CaseEvents = []string{
		"legal.case.created", "legal.case.updated", "legal.case.closed",
	}
	config.NATSSubjects.DocumentEvents = []string{
		"legal.document.uploaded", "legal.document.processed",
		"legal.document.analyzed", "legal.document.indexed",
	}
	config.NATSSubjects.AIEvents = []string{
		"legal.ai.analysis.started", "legal.ai.analysis.completed",
		"legal.ai.analysis.failed", "legal.search.query", "legal.search.results",
	}
	config.NATSSubjects.SystemEvents = []string{
		"system.health", "system.metrics", "system.alerts",
	}

	orchestrator := &IntegrationOrchestrator{
		config:   config,
		services: make(map[string]*ServiceInfo),
	}

	// Initialize infrastructure connections
	if err := orchestrator.initializeInfrastructure(); err != nil {
		return nil, fmt.Errorf("failed to initialize infrastructure: %w", err)
	}

	// Initialize service clients
	orchestrator.initializeServiceClients()

	return orchestrator, nil
}

func (o *IntegrationOrchestrator) initializeInfrastructure() error {
	// PostgreSQL connection (with pgvector support)
	ctx := context.Background()
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgresql://postgres:postgres@localhost:5432/legal_ai_db"
	}

	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		return fmt.Errorf("failed to connect to PostgreSQL: %w", err)
	}
	o.postgres = pool

	// Redis connection
	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "redis://localhost:6379"
	}

	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return fmt.Errorf("failed to parse Redis URL: %w", err)
	}
	o.redis = redis.NewClient(opt)

	// NATS connection (from FULL_STACK_INTEGRATION_COMPLETE.md)
	natsURL := os.Getenv("NATS_URL")
	if natsURL == "" {
		natsURL = "nats://localhost:4222"
	}

	nc, err := nats.Connect(natsURL,
		nats.Name("Legal AI Integration Orchestrator"),
		nats.UserInfo("legal_ai_client", "legal_ai_2024"),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to NATS: %w", err)
	}
	o.nats = nc

	return nil
}

func (o *IntegrationOrchestrator) initializeServiceClients() {
	httpClient := &http.Client{Timeout: 10 * time.Second}

	// Initialize existing service clients
	o.enhancedRAG = &ServiceClient{
		Name:     "enhanced-rag",
		BaseURL:  fmt.Sprintf("http://localhost:%d", o.config.Services["enhanced-rag"]),
		Protocol: "http",
		Client:   httpClient,
	}

	o.uploadService = &ServiceClient{
		Name:     "upload-service",
		BaseURL:  fmt.Sprintf("http://localhost:%d", o.config.Services["upload-service"]),
		Protocol: "http",
		Client:   httpClient,
	}

	o.clusterManager = &ServiceClient{
		Name:     "cluster-http",
		BaseURL:  fmt.Sprintf("http://localhost:%d", o.config.Services["cluster-http"]),
		Protocol: "http",
		Client:   httpClient,
	}

	o.xstateManager = &ServiceClient{
		Name:     "xstate-manager",
		BaseURL:  fmt.Sprintf("http://localhost:%d", o.config.Services["xstate-manager"]),
		Protocol: "http",
		Client:   httpClient,
	}

	o.liveAgent = &ServiceClient{
		Name:     "live-agent-enhanced",
		BaseURL:  fmt.Sprintf("http://localhost:%d", o.config.Services["live-agent-enhanced"]),
		Protocol: "http",
		Client:   httpClient,
	}

	// Initialize new vector processing clients
	o.vectorWorker = &ServiceClient{
		Name:     "vector-worker",
		BaseURL:  fmt.Sprintf("http://localhost:%d", o.config.Services["vector-worker"]),
		Protocol: "http",
		Client:   httpClient,
	}

	o.embeddingService = &ServiceClient{
		Name:     "embedding-service",
		BaseURL:  fmt.Sprintf("http://localhost:%d", o.config.Services["embedding-service"]),
		Protocol: "http",
		Client:   httpClient,
	}
}

// Start the integration orchestrator with health monitoring
func (o *IntegrationOrchestrator) Start() error {
	// Start health monitoring
	go o.startHealthMonitoring()

	// Start NATS event processing
	go o.startNATSProcessing()

	// Start API server for orchestration
	return o.startAPIServer()
}

func (o *IntegrationOrchestrator) startHealthMonitoring() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.performHealthChecks()
		}
	}
}

func (o *IntegrationOrchestrator) performHealthChecks() {
	o.healthMutex.Lock()
	defer o.healthMutex.Unlock()

	services := []*ServiceClient{
		o.enhancedRAG, o.uploadService, o.clusterManager,
		o.xstateManager, o.liveAgent, o.vectorWorker, o.embeddingService,
	}

	for _, service := range services {
		start := time.Now()
		resp, err := service.Client.Get(service.BaseURL + "/health")
		latency := time.Since(start)

		info := &ServiceInfo{
			Name:      service.Name,
			Port:      extractPortFromURL(service.BaseURL),
			Protocol:  service.Protocol,
			Latency:   latency,
			LastCheck: time.Now(),
		}

		if err != nil || resp.StatusCode != 200 {
			info.Status = "failed"
		} else if latency > 100*time.Millisecond {
			info.Status = "degraded"
		} else {
			info.Status = "healthy"
		}

		if resp != nil {
			resp.Body.Close()
		}

		o.services[service.Name] = info

		// Publish service health to NATS
		o.publishServiceHealth(info)
	}
}

func (o *IntegrationOrchestrator) publishServiceHealth(info *ServiceInfo) {
	healthData, _ := json.Marshal(map[string]interface{}{
		"service":   info.Name,
		"status":    info.Status,
		"latency":   info.Latency.Milliseconds(),
		"timestamp": time.Now().Unix(),
	})

	o.nats.Publish("system.health", healthData)
}

func (o *IntegrationOrchestrator) startNATSProcessing() {
	// Subscribe to all legal AI subjects
	allSubjects := append(o.config.NATSSubjects.CaseEvents,
		append(o.config.NATSSubjects.DocumentEvents,
			append(o.config.NATSSubjects.AIEvents, o.config.NATSSubjects.SystemEvents...)...)...)

	for _, subject := range allSubjects {
		o.nats.Subscribe(subject, o.handleNATSMessage)
	}

	// Special handling for vector processing events
	o.nats.Subscribe("legal.document.uploaded", o.handleDocumentUpload)
	o.nats.Subscribe("legal.ai.analysis.started", o.handleAIAnalysis)
}

func (o *IntegrationOrchestrator) handleNATSMessage(msg *nats.Msg) {
	log.Printf("📨 NATS Message: %s -> %s", msg.Subject, string(msg.Data))

	// Forward to appropriate service based on subject
	switch {
	case contains(o.config.NATSSubjects.CaseEvents, msg.Subject):
		o.forwardToCaseService(msg)
	case contains(o.config.NATSSubjects.DocumentEvents, msg.Subject):
		o.forwardToDocumentService(msg)
	case contains(o.config.NATSSubjects.AIEvents, msg.Subject):
		o.forwardToAIService(msg)
	case contains(o.config.NATSSubjects.SystemEvents, msg.Subject):
		o.forwardToSystemService(msg)
	}
}

func (o *IntegrationOrchestrator) handleDocumentUpload(msg *nats.Msg) {
	// Parse document upload event
	var event map[string]interface{}
	if err := json.Unmarshal(msg.Data, &event); err != nil {
		log.Printf("❌ Failed to parse document upload event: %v", err)
		return
	}

	// Trigger vector processing pipeline
	o.triggerVectorProcessing(event)
}

func (o *IntegrationOrchestrator) triggerVectorProcessing(event map[string]interface{}) {
	// Create vector processing job
	job := map[string]interface{}{
		"owner_type": "document",
		"owner_id":   event["documentId"],
		"event":      "upsert",
		"payload": map[string]interface{}{
			"filename": event["filename"],
			"content":  event["content"],
		},
		"timestamp": time.Now().Unix(),
	}

	// Publish to Redis Stream for vector worker
	jobData, _ := json.Marshal(job)
	o.redis.XAdd(context.Background(), &redis.XAddArgs{
		Stream: o.config.VectorConfig.RedisStream,
		Values: map[string]interface{}{
			"payload": string(jobData),
		},
	})

	log.Printf("✅ Triggered vector processing for document: %v", event["documentId"])
}

func (o *IntegrationOrchestrator) handleAIAnalysis(msg *nats.Msg) {
	// Forward to enhanced RAG service for processing
	o.forwardToService(o.enhancedRAG, msg.Data)
}

func (o *IntegrationOrchestrator) forwardToCaseService(msg *nats.Msg) {
	// Forward to appropriate case management service
	o.forwardToService(o.enhancedRAG, msg.Data) // Enhanced RAG handles case analysis
}

func (o *IntegrationOrchestrator) forwardToDocumentService(msg *nats.Msg) {
	// Forward to upload service for document processing
	o.forwardToService(o.uploadService, msg.Data)
}

func (o *IntegrationOrchestrator) forwardToAIService(msg *nats.Msg) {
	// Route to appropriate AI service based on workload
	o.forwardToService(o.enhancedRAG, msg.Data)
}

func (o *IntegrationOrchestrator) forwardToSystemService(msg *nats.Msg) {
	// Forward to cluster manager for system events
	o.forwardToService(o.clusterManager, msg.Data)
}

func (o *IntegrationOrchestrator) forwardToService(service *ServiceClient, data []byte) {
	// Forward message to appropriate service
	_, err := service.Client.Post(service.BaseURL+"/nats/message", "application/json",
		bytes.NewReader(data))
	if err != nil {
		log.Printf("❌ Failed to forward to %s: %v", service.Name, err)
	}
}

func (o *IntegrationOrchestrator) startAPIServer() error {
	r := gin.Default()

	// Health endpoint for orchestrator
	r.GET("/health", o.handleHealth)

	// Service registry endpoint
	r.GET("/services", o.handleServices)

	// Vector processing status
	r.GET("/vector/status", o.handleVectorStatus)

	// Integration metrics
	r.GET("/integration/metrics", o.handleIntegrationMetrics)

	// Start server
	port := os.Getenv("ORCHESTRATOR_PORT")
	if port == "" {
		port = "8099"
	}

	log.Printf("🚀 Integration Orchestrator starting on port %s", port)
	return r.Run(":" + port)
}

func (o *IntegrationOrchestrator) handleHealth(c *gin.Context) {
	o.healthMutex.RLock()
	defer o.healthMutex.RUnlock()

	healthy := 0
	total := len(o.services)

	for _, service := range o.services {
		if service.Status == "healthy" {
			healthy++
		}
	}

	status := "healthy"
	if healthy < total/2 {
		status = "degraded"
	}
	if healthy == 0 {
		status = "failed"
	}

	c.JSON(200, gin.H{
		"status":           status,
		"servicesHealthy":  healthy,
		"servicesTotal":    total,
		"services":         o.services,
		"vectorPipeline":   o.getVectorPipelineStatus(),
		"timestamp":        time.Now().Unix(),
	})
}

func (o *IntegrationOrchestrator) handleServices(c *gin.Context) {
	o.healthMutex.RLock()
	defer o.healthMutex.RUnlock()

	c.JSON(200, gin.H{
		"services": o.services,
		"serviceMap": o.config.Services,
	})
}

func (o *IntegrationOrchestrator) handleVectorStatus(c *gin.Context) {
	// Get Redis stream info
	ctx := context.Background()
	streamInfo, err := o.redis.XInfoStream(ctx, o.config.VectorConfig.RedisStream).Result()

	vectorStatus := gin.H{
		"redisStream": o.config.VectorConfig.RedisStream,
		"qdrantURL":   o.config.VectorConfig.QdrantURL,
	}

	if err == nil {
		vectorStatus["streamLength"] = streamInfo.Length
		vectorStatus["streamGroups"] = streamInfo.Groups
	}

	c.JSON(200, vectorStatus)
}

func (o *IntegrationOrchestrator) handleIntegrationMetrics(c *gin.Context) {
	c.JSON(200, gin.H{
		"platform": "Legal AI Platform",
		"integration": gin.H{
			"orchestrator":     "active",
			"services":         len(o.services),
			"vectorPipeline":   o.getVectorPipelineStatus(),
			"natsConnection":   o.nats.Status() == nats.CONNECTED,
			"redisConnection":  o.redis.Ping(context.Background()).Err() == nil,
		},
		"protocols": []string{"HTTP", "gRPC", "QUIC", "WebSocket", "NATS"},
		"timestamp": time.Now().Unix(),
	})
}

func (o *IntegrationOrchestrator) getVectorPipelineStatus() string {
	// Check if vector services are healthy
	if o.services["vector-worker"] != nil && o.services["vector-worker"].Status == "healthy" &&
	   o.services["embedding-service"] != nil && o.services["embedding-service"].Status == "healthy" {
		return "active"
	}
	return "degraded"
}

// Helper functions
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func extractPortFromURL(url string) int {
	// Extract port from URL string
	// Simplified implementation
	return 8000 // Placeholder
}

func runIntegrationOrchestrator() {
	log.Printf("🎯 Legal AI Platform Integration Orchestrator")
	log.Printf("🏗️ Linking Vector Pipeline with 37 Go Microservices")

	orchestrator, err := NewIntegrationOrchestrator()
	if err != nil {
		log.Fatalf("❌ Failed to create orchestrator: %v", err)
	}
	defer orchestrator.postgres.Close()
	defer orchestrator.redis.Close()
	defer orchestrator.nats.Close()

	if err := orchestrator.Start(); err != nil {
		log.Fatalf("❌ Failed to start orchestrator: %v", err)
	}
}