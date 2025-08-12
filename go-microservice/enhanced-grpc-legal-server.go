//go:build legacy
// +build legacy

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
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/net/http2"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

// =====================================
// Configuration & Environment
// =====================================

type ServerConfig struct {
	HTTPPort      string            `json:"http_port"`
	GRPCPort      string            `json:"grpc_port"`
	MetricsPort   string            `json:"metrics_port"`
	Environment   string            `json:"environment"`
	TLSEnabled    bool              `json:"tls_enabled"`
	CertFile      string            `json:"cert_file"`
	KeyFile       string            `json:"key_file"`
	OllamaHost    string            `json:"ollama_host"`
	RedisURL      string            `json:"redis_url"`
	PostgresURL   string            `json:"postgres_url"`
	QdrantURL     string            `json:"qdrant_url"`
	GPUEnabled    bool              `json:"gpu_enabled"`
	MaxWorkers    int               `json:"max_workers"`
	RateLimits    map[string]int    `json:"rate_limits"`
	CORSOrigins   []string          `json:"cors_origins"`
	Monitoring    MonitoringConfig  `json:"monitoring"`
}

type MonitoringConfig struct {
	PrometheusEnabled bool   `json:"prometheus_enabled"`
	TracingEnabled    bool   `json:"tracing_enabled"`
	LogLevel          string `json:"log_level"`
	HealthChecks      bool   `json:"health_checks"`
}

// =====================================
// gRPC Service Definitions
// =====================================

type LegalAIServer struct {
	config          *ServerConfig
	ollamaClient    *OllamaClient
	vectorService   *VectorService
	documentService *DocumentService
	metrics         *MetricsCollector
	grpc_health_v1.UnimplementedHealthServer
}

type DocumentProcessingRequest struct {
	DocumentId   string            `json:"document_id"`
	Content      []byte            `json:"content"`
	DocumentType string            `json:"document_type"`
	Jurisdiction string            `json:"jurisdiction"`
	Metadata     map[string]string `json:"metadata"`
	Options      ProcessingOptions `json:"options"`
}

type ProcessingOptions struct {
	ExtractText       bool    `json:"extract_text"`
	GenerateEmbedding bool    `json:"generate_embedding"`
	AnalyzeEntities   bool    `json:"analyze_entities"`
	SummarizeContent  bool    `json:"summarize_content"`
	ConfidenceLevel   float64 `json:"confidence_level"`
	Language          string  `json:"language"`
}

type DocumentProcessingResponse struct {
	DocumentId      string                 `json:"document_id"`
	ExtractedText   string                 `json:"extracted_text"`
	Embeddings      []float64              `json:"embeddings,omitempty"`
	Entities        []Entity               `json:"entities,omitempty"`
	Summary         string                 `json:"summary,omitempty"`
	ProcessingTime  int64                  `json:"processing_time_ms"`
	Confidence      float64                `json:"confidence"`
	Status          string                 `json:"status"`
	Error           string                 `json:"error,omitempty"`
	Metadata        map[string]interface{} `json:"metadata"`
}

type Entity struct {
	Text       string  `json:"text"`
	Type       string  `json:"type"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
}

type SemanticSearchRequest struct {
	Query            string            `json:"query"`
	Collection       string            `json:"collection"`
	Limit            int               `json:"limit"`
	Threshold        float64           `json:"threshold"`
	Filters          map[string]string `json:"filters"`
	IncludePayload   bool              `json:"include_payload"`
	IncludeVectors   bool              `json:"include_vectors"`
	RerankResults    bool              `json:"rerank_results"`
	ContextWindow    int               `json:"context_window"`
}

type SemanticSearchResponse struct {
	Results        []SearchResult `json:"results"`
	TotalFound     int            `json:"total_found"`
	QueryTime      int64          `json:"query_time_ms"`
	RerankTime     int64          `json:"rerank_time_ms,omitempty"`
	Status         string         `json:"status"`
	Error          string         `json:"error,omitempty"`
}

type SearchResult struct {
	ID         string                 `json:"id"`
	Score      float64                `json:"score"`
	Document   map[string]interface{} `json:"document,omitempty"`
	Snippet    string                 `json:"snippet,omitempty"`
	Highlights []string               `json:"highlights,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// =====================================
// Metrics & Monitoring
// =====================================

type MetricsCollector struct {
	documentProcessingDuration prometheus.HistogramVec
	searchQueryDuration        prometheus.HistogramVec
	ollamaRequestDuration      prometheus.HistogramVec
	grpcRequestsTotal          prometheus.CounterVec
	httpRequestsTotal          prometheus.CounterVec
	activeConnections          prometheus.GaugeVec
	gpuUtilization            prometheus.GaugeVec
	errorRate                 prometheus.CounterVec
}

func NewMetricsCollector() *MetricsCollector {
	mc := &MetricsCollector{
		documentProcessingDuration: *prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "legal_ai_document_processing_duration_seconds",
				Help:    "Time taken to process legal documents",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"document_type", "jurisdiction", "processing_stage"},
		),
		searchQueryDuration: *prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "legal_ai_search_query_duration_seconds",
				Help:    "Time taken for semantic search queries",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
			},
			[]string{"collection", "query_type", "rerank_enabled"},
		),
		ollamaRequestDuration: *prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "legal_ai_ollama_request_duration_seconds",
				Help:    "Time taken for Ollama LLM requests",
				Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
			},
			[]string{"model", "request_type", "gpu_enabled"},
		),
		grpcRequestsTotal: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_ai_grpc_requests_total",
				Help: "Total number of gRPC requests",
			},
			[]string{"method", "status"},
		),
		httpRequestsTotal: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_ai_http_requests_total",
				Help: "Total number of HTTP requests",
			},
			[]string{"method", "endpoint", "status"},
		),
		activeConnections: *prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "legal_ai_active_connections",
				Help: "Current number of active connections",
			},
			[]string{"connection_type", "protocol"},
		),
		gpuUtilization: *prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "legal_ai_gpu_utilization_percent",
				Help: "GPU utilization percentage",
			},
			[]string{"gpu_id", "model"},
		),
		errorRate: *prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_ai_errors_total",
				Help: "Total number of errors by type",
			},
			[]string{"error_type", "service", "severity"},
		),
	}

	// Register all metrics
	prometheus.MustRegister(
		mc.documentProcessingDuration,
		mc.searchQueryDuration,
		mc.ollamaRequestDuration,
		mc.grpcRequestsTotal,
		mc.httpRequestsTotal,
		mc.activeConnections,
		mc.gpuUtilization,
		mc.errorRate,
	)

	return mc
}

// =====================================
// Ollama Client Integration
// =====================================

type OllamaClient struct {
	BaseURL    string
	HTTPClient *http.Client
	Models     map[string]ModelInfo
	mutex      sync.RWMutex
}

type ModelInfo struct {
	Name         string    `json:"name"`
	Size         int64     `json:"size"`
	Digest       string    `json:"digest"`
	ModifiedAt   time.Time `json:"modified_at"`
	Capabilities []string  `json:"capabilities"`
}

type OllamaRequest struct {
	Model       string                 `json:"model"`
	Prompt      string                 `json:"prompt,omitempty"`
	Messages    []ChatMessage          `json:"messages,omitempty"`
	Stream      bool                   `json:"stream"`
	Temperature float64                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	System      string                 `json:"system,omitempty"`
	Context     []int                  `json:"context,omitempty"`
	Options     map[string]interface{} `json:"options,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaResponse struct {
	Model              string                 `json:"model"`
	Response           string                 `json:"response"`
	Done               bool                   `json:"done"`
	Context            []int                  `json:"context,omitempty"`
	TotalDuration      int64                  `json:"total_duration,omitempty"`
	LoadDuration       int64                  `json:"load_duration,omitempty"`
	PromptEvalCount    int                    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64                  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int                    `json:"eval_count,omitempty"`
	EvalDuration       int64                  `json:"eval_duration,omitempty"`
	Error              string                 `json:"error,omitempty"`
	CreatedAt          time.Time              `json:"created_at,omitempty"`
}

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 300 * time.Second,
			Transport: &http2.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: true,
				},
			},
		},
		Models: make(map[string]ModelInfo),
	}
}

// =====================================
// Vector Service (Qdrant Integration)
// =====================================

type VectorService struct {
	client     *http.Client
	baseURL    string
	apiKey     string
	collections map[string]CollectionInfo
	mutex      sync.RWMutex
}

type CollectionInfo struct {
	Name        string `json:"name"`
	VectorSize  int    `json:"vector_size"`
	Distance    string `json:"distance"`
	IndexCount  int    `json:"index_count"`
	Status      string `json:"status"`
}

type UpsertRequest struct {
	Points []VectorPoint `json:"points"`
}

type VectorPoint struct {
	ID      string                 `json:"id"`
	Vector  []float64              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

type VectorSearchRequest struct {
	Vector      []float64              `json:"vector"`
	Limit       int                    `json:"limit"`
	Filter      map[string]interface{} `json:"filter,omitempty"`
	WithPayload bool                   `json:"with_payload"`
	WithVector  bool                   `json:"with_vector"`
	ScoreThreshold float64             `json:"score_threshold,omitempty"`
}

func NewVectorService(baseURL, apiKey string) *VectorService {
	return &VectorService{
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http2.Transport{},
		},
		baseURL:     baseURL,
		apiKey:      apiKey,
		collections: make(map[string]CollectionInfo),
	}
}

// =====================================
// Document Processing Service
// =====================================

type DocumentService struct {
	processor    *TextExtractor
	embedder     *EmbeddingService
	analyzer     *EntityAnalyzer
	summarizer   *SummarizationService
	tempDir      string
	maxFileSize  int64
	allowedTypes []string
}

type TextExtractor struct {
	ocrEnabled bool
	languages  []string
}

type EmbeddingService struct {
	model      string
	dimensions int
	client     *OllamaClient
}

type EntityAnalyzer struct {
	models    []string
	threshold float64
}

type SummarizationService struct {
	model     string
	maxLength int
	minLength int
}

func NewDocumentService(config *ServerConfig, ollamaClient *OllamaClient) *DocumentService {
	return &DocumentService{
		processor: &TextExtractor{
			ocrEnabled: true,
			languages:  []string{"en", "es", "fr"},
		},
		embedder: &EmbeddingService{
			model:      "nomic-embed-text",
			dimensions: 768,
			client:     ollamaClient,
		},
		analyzer: &EntityAnalyzer{
			models:    []string{"en_core_web_sm", "legal_ner_model"},
			threshold: 0.8,
		},
		summarizer: &SummarizationService{
			model:     "llama3.1:8b",
			maxLength: 500,
			minLength: 100,
		},
		tempDir:      "/tmp/legal-ai-docs",
		maxFileSize:  100 * 1024 * 1024, // 100MB
		allowedTypes: []string{"pdf", "docx", "txt", "rtf", "html"},
	}
}

// =====================================
// HTTP/2 + gRPC Server Implementation
// =====================================

func (s *LegalAIServer) ProcessDocument(ctx context.Context, req *DocumentProcessingRequest) (*DocumentProcessingResponse, error) {
	startTime := time.Now()
	
	// Update metrics
	defer func() {
		duration := time.Since(startTime).Seconds()
		s.metrics.documentProcessingDuration.WithLabelValues(
			req.DocumentType,
			req.Jurisdiction,
			"complete",
		).Observe(duration)
	}()

	// Validate request
	if len(req.Content) == 0 {
		s.metrics.errorRate.WithLabelValues("validation", "document_processing", "error").Inc()
		return nil, fmt.Errorf("document content cannot be empty")
	}

	if len(req.Content) > s.documentService.maxFileSize {
		s.metrics.errorRate.WithLabelValues("validation", "document_processing", "error").Inc()
		return nil, fmt.Errorf("document size exceeds maximum allowed size")
	}

	response := &DocumentProcessingResponse{
		DocumentId: req.DocumentId,
		Status:     "processing",
		Metadata:   make(map[string]interface{}),
	}

	// Text extraction
	if req.Options.ExtractText {
		extractedText, err := s.documentService.processor.ExtractText(req.Content, req.DocumentType)
		if err != nil {
			s.metrics.errorRate.WithLabelValues("text_extraction", "document_processing", "error").Inc()
			response.Error = fmt.Sprintf("Text extraction failed: %v", err)
			response.Status = "error"
			return response, nil
		}
		response.ExtractedText = extractedText
		response.Metadata["word_count"] = len(strings.Fields(extractedText))
	}

	// Generate embeddings
	if req.Options.GenerateEmbedding && response.ExtractedText != "" {
		embeddings, err := s.documentService.embedder.GenerateEmbedding(response.ExtractedText)
		if err != nil {
			s.metrics.errorRate.WithLabelValues("embedding", "document_processing", "warning").Inc()
			log.Printf("Warning: Failed to generate embeddings: %v", err)
		} else {
			response.Embeddings = embeddings
		}
	}

	// Entity analysis
	if req.Options.AnalyzeEntities && response.ExtractedText != "" {
		entities, err := s.documentService.analyzer.ExtractEntities(response.ExtractedText)
		if err != nil {
			s.metrics.errorRate.WithLabelValues("entity_analysis", "document_processing", "warning").Inc()
			log.Printf("Warning: Failed to extract entities: %v", err)
		} else {
			response.Entities = entities
			response.Metadata["entity_count"] = len(entities)
		}
	}

	// Content summarization
	if req.Options.SummarizeContent && response.ExtractedText != "" {
		summary, err := s.documentService.summarizer.Summarize(response.ExtractedText)
		if err != nil {
			s.metrics.errorRate.WithLabelValues("summarization", "document_processing", "warning").Inc()
			log.Printf("Warning: Failed to generate summary: %v", err)
		} else {
			response.Summary = summary
		}
	}

	response.ProcessingTime = time.Since(startTime).Milliseconds()
	response.Confidence = calculateConfidence(response)
	response.Status = "completed"

	return response, nil
}

func (s *LegalAIServer) SemanticSearch(ctx context.Context, req *SemanticSearchRequest) (*SemanticSearchResponse, error) {
	startTime := time.Now()
	
	// Update metrics
	defer func() {
		duration := time.Since(startTime).Seconds()
		s.metrics.searchQueryDuration.WithLabelValues(
			req.Collection,
			"semantic",
			fmt.Sprintf("%v", req.RerankResults),
		).Observe(duration)
	}()

	// Generate query embedding
	queryEmbedding, err := s.documentService.embedder.GenerateEmbedding(req.Query)
	if err != nil {
		s.metrics.errorRate.WithLabelValues("query_embedding", "search", "error").Inc()
		return &SemanticSearchResponse{
			Status: "error",
			Error:  fmt.Sprintf("Failed to generate query embedding: %v", err),
		}, nil
	}

	// Perform vector search
	searchResults, err := s.vectorService.Search(ctx, &VectorSearchRequest{
		Vector:         queryEmbedding,
		Limit:          req.Limit,
		WithPayload:    req.IncludePayload,
		WithVector:     req.IncludeVectors,
		ScoreThreshold: req.Threshold,
	})
	if err != nil {
		s.metrics.errorRate.WithLabelValues("vector_search", "search", "error").Inc()
		return &SemanticSearchResponse{
			Status: "error",
			Error:  fmt.Sprintf("Vector search failed: %v", err),
		}, nil
	}

	response := &SemanticSearchResponse{
		Results:    convertSearchResults(searchResults),
		TotalFound: len(searchResults),
		QueryTime:  time.Since(startTime).Milliseconds(),
		Status:     "success",
	}

	// Re-ranking (optional)
	if req.RerankResults && len(response.Results) > 1 {
		rerankStart := time.Now()
		response.Results = s.rerankResults(req.Query, response.Results, req.ContextWindow)
		response.RerankTime = time.Since(rerankStart).Milliseconds()
	}

	return response, nil
}

// =====================================
// HTTP Handlers (REST API)
// =====================================

func (s *LegalAIServer) setupHTTPRoutes() *gin.Engine {
	if s.config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// Middleware
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	router.Use(s.corsMiddleware())
	router.Use(s.rateLimitMiddleware())
	router.Use(s.metricsMiddleware())

	// Health checks
	router.GET("/health", s.healthCheck)
	router.GET("/health/ready", s.readinessCheck)
	router.GET("/health/live", s.livenessCheck)

	// API v1 routes
	v1 := router.Group("/api/v1")
	{
		// Document processing
		v1.POST("/documents/process", s.handleDocumentProcessing)
		v1.POST("/documents/upload", s.handleDocumentUpload)
		v1.GET("/documents/:id", s.handleGetDocument)

		// Search endpoints
		v1.POST("/search/semantic", s.handleSemanticSearch)
		v1.POST("/search/hybrid", s.handleHybridSearch)
		v1.GET("/search/suggest", s.handleSearchSuggestions)

		// Chat and streaming
		v1.POST("/chat", s.handleChatCompletion)
		v1.GET("/chat/stream", s.handleChatStream)
		v1.WebSocket("/chat/ws", s.handleChatWebSocket)

		// Vector operations
		v1.POST("/vectors/embed", s.handleGenerateEmbeddings)
		v1.POST("/vectors/search", s.handleVectorSearch)
		v1.POST("/vectors/upsert", s.handleVectorUpsert)

		// System endpoints
		v1.GET("/status", s.handleSystemStatus)
		v1.GET("/models", s.handleListModels)
		v1.GET("/collections", s.handleListCollections)
	}

	// Metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	return router
}

func (s *LegalAIServer) handleDocumentProcessing(c *gin.Context) {
	var req DocumentProcessingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.metrics.httpRequestsTotal.WithLabelValues("POST", "/documents/process", "400").Inc()
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Minute)
	defer cancel()

	response, err := s.ProcessDocument(ctx, &req)
	if err != nil {
		s.metrics.httpRequestsTotal.WithLabelValues("POST", "/documents/process", "500").Inc()
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	s.metrics.httpRequestsTotal.WithLabelValues("POST", "/documents/process", "200").Inc()
	c.JSON(http.StatusOK, response)
}

func (s *LegalAIServer) handleSemanticSearch(c *gin.Context) {
	var req SemanticSearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		s.metrics.httpRequestsTotal.WithLabelValues("POST", "/search/semantic", "400").Inc()
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if req.Limit <= 0 {
		req.Limit = 10
	}
	if req.Threshold <= 0 {
		req.Threshold = 0.7
	}
	if req.Collection == "" {
		req.Collection = "legal_documents"
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	response, err := s.SemanticSearch(ctx, &req)
	if err != nil {
		s.metrics.httpRequestsTotal.WithLabelValues("POST", "/search/semantic", "500").Inc()
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	s.metrics.httpRequestsTotal.WithLabelValues("POST", "/search/semantic", "200").Inc()
	c.JSON(http.StatusOK, response)
}

func (s *LegalAIServer) handleChatStream(c *gin.Context) {
	// Upgrade to WebSocket for real-time streaming
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			origin := r.Header.Get("Origin")
			for _, allowedOrigin := range s.config.CORSOrigins {
				if origin == allowedOrigin || allowedOrigin == "*" {
					return true
				}
			}
			return false
		},
	}

	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		s.metrics.errorRate.WithLabelValues("websocket_upgrade", "chat", "error").Inc()
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	s.metrics.activeConnections.WithLabelValues("websocket", "chat").Inc()
	defer s.metrics.activeConnections.WithLabelValues("websocket", "chat").Dec()

	// Handle WebSocket chat session
	s.handleWebSocketChat(conn)
}

// =====================================
// Server Initialization & Main
// =====================================

func loadConfig() *ServerConfig {
	config := &ServerConfig{
		HTTPPort:    getEnv("HTTP_PORT", "8080"),
		GRPCPort:    getEnv("GRPC_PORT", "50051"),
		MetricsPort: getEnv("METRICS_PORT", "9090"),
		Environment: getEnv("ENVIRONMENT", "development"),
		TLSEnabled:  getEnv("TLS_ENABLED", "false") == "true",
		CertFile:    getEnv("TLS_CERT_FILE", ""),
		KeyFile:     getEnv("TLS_KEY_FILE", ""),
		OllamaHost:  getEnv("OLLAMA_HOST", "http://localhost:11434"),
		RedisURL:    getEnv("REDIS_URL", "redis://localhost:6379"),
		PostgresURL: getEnv("POSTGRES_URL", "postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db"),
		QdrantURL:   getEnv("QDRANT_URL", "http://localhost:6333"),
		GPUEnabled:  getEnv("GPU_ENABLED", "true") == "true",
		MaxWorkers:  getIntEnv("MAX_WORKERS", runtime.NumCPU()),
		RateLimits: map[string]int{
			"documents": 10,
			"search":    50,
			"chat":      20,
		},
		CORSOrigins: strings.Split(getEnv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"), ","),
		Monitoring: MonitoringConfig{
			PrometheusEnabled: true,
			TracingEnabled:    getEnv("TRACING_ENABLED", "false") == "true",
			LogLevel:          getEnv("LOG_LEVEL", "info"),
			HealthChecks:      true,
		},
	}

	return config
}

func main() {
	log.Printf("🚀 Starting YoRHa Legal AI Go Microservice")
	log.Printf("📊 Runtime: Go %s on %s/%s", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	
	// Load configuration
	config := loadConfig()
	log.Printf("⚙️  Configuration loaded - Environment: %s", config.Environment)

	// Initialize metrics
	metrics := NewMetricsCollector()
	log.Printf("📈 Metrics collector initialized")

	// Initialize Ollama client
	ollamaClient := NewOllamaClient(config.OllamaHost)
	log.Printf("🤖 Ollama client initialized: %s", config.OllamaHost)

	// Initialize vector service
	vectorService := NewVectorService(config.QdrantURL, "")
	log.Printf("🔍 Vector service initialized: %s", config.QdrantURL)

	// Initialize document service
	documentService := NewDocumentService(config, ollamaClient)
	log.Printf("📄 Document service initialized")

	// Create server instance
	server := &LegalAIServer{
		config:          config,
		ollamaClient:    ollamaClient,
		vectorService:   vectorService,
		documentService: documentService,
		metrics:         metrics,
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start gRPC server
	go func() {
		if err := server.startGRPCServer(ctx); err != nil {
			log.Fatalf("❌ gRPC server failed: %v", err)
		}
	}()

	// Start HTTP server
	go func() {
		if err := server.startHTTPServer(ctx); err != nil {
			log.Fatalf("❌ HTTP server failed: %v", err)
		}
	}()

	// Start metrics server
	go func() {
		if err := server.startMetricsServer(ctx); err != nil {
			log.Fatalf("❌ Metrics server failed: %v", err)
		}
	}()

	log.Printf("✅ All servers started successfully")
	log.Printf("🌐 HTTP Server: http://localhost:%s", config.HTTPPort)
	log.Printf("🔧 gRPC Server: localhost:%s", config.GRPCPort)
	log.Printf("📊 Metrics: http://localhost:%s/metrics", config.MetricsPort)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Printf("🛑 Shutdown signal received, gracefully shutting down...")
	cancel()

	// Allow time for cleanup
	time.Sleep(5 * time.Second)
	log.Printf("✅ YoRHa Legal AI Go Microservice shutdown complete")
}

// =====================================
// Helper Functions
// =====================================

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getIntEnv(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := time.ParseDuration(value); err == nil {
			return int(intValue)
		}
	}
	return defaultValue
}

func calculateConfidence(response *DocumentProcessingResponse) float64 {
	confidence := 0.8 // Base confidence
	
	if response.ExtractedText != "" {
		confidence += 0.1
	}
	if len(response.Embeddings) > 0 {
		confidence += 0.05
	}
	if len(response.Entities) > 0 {
		confidence += 0.05
	}
	
	if confidence > 1.0 {
		confidence = 1.0
	}
	
	return confidence
}

func convertSearchResults(results []VectorPoint) []SearchResult {
	converted := make([]SearchResult, len(results))
	for i, result := range results {
		converted[i] = SearchResult{
			ID:       result.ID,
			Score:    0.95, // Placeholder - calculate from vector distance
			Document: result.Payload,
			Metadata: result.Payload,
		}
	}
	return converted
}

// Placeholder implementations for methods referenced but not fully implemented
func (s *LegalAIServer) startGRPCServer(ctx context.Context) error {
	lis, err := net.Listen("tcp", ":"+s.config.GRPCPort)
	if err != nil {
		return fmt.Errorf("failed to listen on gRPC port: %v", err)
	}

	var opts []grpc.ServerOption
	if s.config.TLSEnabled {
		creds, err := credentials.NewServerTLSFromFile(s.config.CertFile, s.config.KeyFile)
		if err != nil {
			return fmt.Errorf("failed to create TLS credentials: %v", err)
		}
		opts = append(opts, grpc.Creds(creds))
	}

	grpcServer := grpc.NewServer(opts...)
	
	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	
	// Enable reflection for development
	if s.config.Environment != "production" {
		reflection.Register(grpcServer)
	}

	log.Printf("🔧 gRPC server listening on port %s", s.config.GRPCPort)
	return grpcServer.Serve(lis)
}

func (s *LegalAIServer) startHTTPServer(ctx context.Context) error {
	router := s.setupHTTPRoutes()
	
	server := &http.Server{
		Addr:    ":" + s.config.HTTPPort,
		Handler: router,
	}

	// Enable HTTP/2
	if err := http2.ConfigureServer(server, nil); err != nil {
		return fmt.Errorf("failed to configure HTTP/2: %v", err)
	}

	log.Printf("🌐 HTTP server listening on port %s", s.config.HTTPPort)
	return server.ListenAndServe()
}

func (s *LegalAIServer) startMetricsServer(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	
	server := &http.Server{
		Addr:    ":" + s.config.MetricsPort,
		Handler: mux,
	}

	log.Printf("📊 Metrics server listening on port %s", s.config.MetricsPort)
	return server.ListenAndServe()
}

// Middleware implementations
func (s *LegalAIServer) corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		for _, allowedOrigin := range s.config.CORSOrigins {
			if origin == allowedOrigin || allowedOrigin == "*" {
				c.Header("Access-Control-Allow-Origin", origin)
				break
			}
		}
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	}
}

func (s *LegalAIServer) rateLimitMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Simplified rate limiting - implement with Redis in production
		c.Next()
	}
}

func (s *LegalAIServer) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		c.Next()
		
		duration := time.Since(start).Seconds()
		status := fmt.Sprintf("%d", c.Writer.Status())
		
		s.metrics.httpRequestsTotal.WithLabelValues(
			c.Request.Method,
			c.Request.URL.Path,
			status,
		).Inc()
	}
}

// Health check handlers
func (s *LegalAIServer) healthCheck(c *gin.Context) {
	health := gin.H{
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   "2.0.0",
		"services": gin.H{
			"ollama":     s.checkOllamaHealth(),
			"vector_db":  s.checkVectorDBHealth(),
			"postgres":   s.checkPostgresHealth(),
			"redis":      s.checkRedisHealth(),
		},
	}
	
	c.JSON(http.StatusOK, health)
}

func (s *LegalAIServer) readinessCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ready"})
}

func (s *LegalAIServer) livenessCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "alive"})
}

// Service health check methods (simplified implementations)
func (s *LegalAIServer) checkOllamaHealth() string {
	// TODO: Implement actual health check
	return "healthy"
}

func (s *LegalAIServer) checkVectorDBHealth() string {
	// TODO: Implement actual health check
	return "healthy"
}

func (s *LegalAIServer) checkPostgresHealth() string {
	// TODO: Implement actual health check
	return "healthy"
}

func (s *LegalAIServer) checkRedisHealth() string {
	// TODO: Implement actual health check
	return "healthy"
}

// Placeholder method implementations
func (te *TextExtractor) ExtractText(content []byte, docType string) (string, error) {
	// TODO: Implement text extraction logic
	return string(content), nil
}

func (es *EmbeddingService) GenerateEmbedding(text string) ([]float64, error) {
	// TODO: Implement embedding generation
	return make([]float64, es.dimensions), nil
}

func (ea *EntityAnalyzer) ExtractEntities(text string) ([]Entity, error) {
	// TODO: Implement entity extraction
	return []Entity{}, nil
}

func (ss *SummarizationService) Summarize(text string) (string, error) {
	// TODO: Implement summarization
	return "Summary placeholder", nil
}

func (vs *VectorService) Search(ctx context.Context, req *VectorSearchRequest) ([]VectorPoint, error) {
	// TODO: Implement vector search
	return []VectorPoint{}, nil
}

func (s *LegalAIServer) rerankResults(query string, results []SearchResult, contextWindow int) []SearchResult {
	// TODO: Implement result re-ranking
	return results
}

func (s *LegalAIServer) handleWebSocketChat(conn *websocket.Conn) {
	// TODO: Implement WebSocket chat handler
}

// Additional handler placeholders
func (s *LegalAIServer) handleDocumentUpload(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleGetDocument(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleHybridSearch(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleSearchSuggestions(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleChatCompletion(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleChatWebSocket(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleGenerateEmbeddings(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleVectorSearch(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleVectorUpsert(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleSystemStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"system": "YoRHa Legal AI",
		"version": "2.0.0",
		"status": "operational",
		"uptime": time.Now().Format(time.RFC3339),
	})
}

func (s *LegalAIServer) handleListModels(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}

func (s *LegalAIServer) handleListCollections(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{"message": "Not implemented yet"})
}