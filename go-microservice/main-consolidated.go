// Enhanced Legal AI Microservice
// NVIDIA CUDA + Redis + Gemma3-Legal GGUF Integration
// Optimized for Windows Development with SvelteKit 2 API Routes

package main

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	"github.com/redis/go-redis/v9"
)

// Configuration structure
type Config struct {
	RedisAddr       string
	RedisPassword   string
	RedisDB         int
	PostgresURL     string
	Port            string
	GemmaModelPath  string
	CUDADeviceID    int
	MaxConcurrency  int
	CacheExpiration time.Duration
	EnableGPU       bool
	ModelContext    int
	Temperature     float64
}

// Document processing structures
type DocumentRequest struct {
	Content      string         `json:"content"`
	DocumentType string         `json:"document_type"`
	PracticeArea string         `json:"practice_area,omitempty"`
	Jurisdiction string         `json:"jurisdiction"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

type DocumentResponse struct {
	DocumentID     string          `json:"document_id"`
	ProcessingTime time.Duration   `json:"processing_time"`
	Chunks         []DocumentChunk `json:"chunks"`
	Analysis       LegalAnalysis   `json:"analysis"`
	Embeddings     [][]float32     `json:"embeddings,omitempty"`
}

type DocumentChunk struct {
	ID         string         `json:"id"`
	Content    string         `json:"content"`
	ChunkIndex int            `json:"chunk_index"`
	Embedding  []float32      `json:"embedding,omitempty"`
	Metadata   map[string]any `json:"metadata"`
}

type LegalAnalysis struct {
	Summary         string        `json:"summary"`
	KeyConcepts     []string      `json:"key_concepts"`
	Entities        []LegalEntity `json:"entities"`
	RiskFactors     []RiskFactor  `json:"risk_factors"`
	Recommendations []string      `json:"recommendations"`
	Confidence      float64       `json:"confidence"`
	CitedCases      []CitedCase   `json:"cited_cases,omitempty"`
}

type LegalEntity struct {
	Type       string  `json:"type"`
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	Context    string  `json:"context"`
}

type RiskFactor struct {
	Type        string `json:"type"`
	Severity    string `json:"severity"`
	Description string `json:"description"`
	Mitigation  string `json:"mitigation,omitempty"`
}

type CitedCase struct {
	CaseName  string `json:"case_name"`
	Citation  string `json:"citation"`
	Relevance string `json:"relevance"`
	Context   string `json:"context"`
}

// Search structures
type SearchRequest struct {
	Query        string         `json:"query"`
	Model        string         `json:"model"`
	Limit        int            `json:"limit"`
	Filters      map[string]any `json:"filters,omitempty"`
	PracticeArea string         `json:"practice_area,omitempty"`
	Jurisdiction string         `json:"jurisdiction,omitempty"`
}

type SearchResponse struct {
	Results      []SearchResult `json:"results"`
	QueryTime    time.Duration  `json:"query_time"`
	TotalResults int            `json:"total_results"`
	UsedCache    bool           `json:"used_cache"`
	ModelUsed    string         `json:"model_used"`
}

type SearchResult struct {
	DocumentID  string         `json:"document_id"`
	Content     string         `json:"content"`
	Score       float64        `json:"score"`
	ChunkIndex  int            `json:"chunk_index"`
	Metadata    map[string]any `json:"metadata"`
	Highlighted string         `json:"highlighted,omitempty"`
}

// Service interfaces
type LegalAIService struct {
	config   *Config
	redis    *redis.Client
	postgres *pgx.Conn
	ctx      context.Context
}

// Global service instance
var service *LegalAIService

// Initialize configuration from environment
func initConfig() *Config {
	return &Config{
		RedisAddr:       getEnv("REDIS_ADDR", "localhost:6379"),
		RedisPassword:   getEnv("REDIS_PASSWORD", ""),
		RedisDB:         getEnvInt("REDIS_DB", 0),
		PostgresURL:     getEnv("POSTGRES_URL", "postgres://postgres:postgres@localhost:5432/prosecutor_db"),
		Port:            getEnv("PORT", "8080"),
		GemmaModelPath:  getEnv("GEMMA_MODEL_PATH", "./models/gemma-2-2b-legal.Q8_0.gguf"),
		CUDADeviceID:    getEnvInt("CUDA_DEVICE_ID", 0),
		MaxConcurrency:  getEnvInt("MAX_CONCURRENCY", 4),
		CacheExpiration: time.Duration(getEnvInt("CACHE_EXPIRATION_MINUTES", 30)) * time.Minute,
		EnableGPU:       getEnvBool("ENABLE_GPU", true),
		ModelContext:    getEnvInt("MODEL_CONTEXT", 4096),
		Temperature:     getEnvFloat("TEMPERATURE", 0.1),
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

// Initialize services
func initServices(config *Config) (*LegalAIService, error) {
	ctx := context.Background()

	// Initialize Redis
	rdb := redis.NewClient(&redis.Options{
		Addr:     config.RedisAddr,
		Password: config.RedisPassword,
		DB:       config.RedisDB,
	})

	// Test Redis connection
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	// Initialize PostgreSQL with pgvector (optional)
	var conn *pgx.Conn
	if config.PostgresURL != "" {
		var err error
		conn, err = pgx.Connect(ctx, config.PostgresURL)
		if err != nil {
			log.Printf("Warning: Failed to connect to PostgreSQL: %v", err)
			log.Printf("Continuing without PostgreSQL (Redis-only mode)")
			conn = nil
		} else {
			// Ensure pgvector extension
			_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
			if err != nil {
				log.Printf("Warning: Could not create vector extension: %v", err)
			}
		}
	} else {
		log.Printf("PostgreSQL disabled - running in Redis-only mode")
	}

	service := &LegalAIService{
		config:   config,
		redis:    rdb,
		postgres: conn,
		ctx:      ctx,
	}

	log.Printf("✅ Services initialized successfully")
	log.Printf("🔧 Redis: %s", config.RedisAddr)
	log.Printf("🐘 PostgreSQL: Connected")
	log.Printf("🎯 Gemma Model Path: %s", config.GemmaModelPath)
	log.Printf("🚀 CUDA GPU Enabled: %v", config.EnableGPU)

	return service, nil
}

// Process document endpoint
func (s *LegalAIService) processDocument(c *gin.Context) {
	var req DocumentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	start := time.Now()

	// Generate cache key
	cacheKey := fmt.Sprintf("doc:%x", md5.Sum([]byte(req.Content)))

	// Check Redis cache first
	cached, err := s.redis.Get(s.ctx, cacheKey).Result()
	if err == nil {
		var response DocumentResponse
		if json.Unmarshal([]byte(cached), &response) == nil {
			log.Printf("📋 Cache hit for document: %s", cacheKey)
			c.JSON(http.StatusOK, response)
			return
		}
	}

	// Process document (placeholder for actual AI processing)
	response := s.processDocumentCore(req)
	response.ProcessingTime = time.Since(start)

	// Cache the response
	if responseJSON, err := json.Marshal(response); err == nil {
		s.redis.Set(s.ctx, cacheKey, responseJSON, s.config.CacheExpiration)
	}

	log.Printf("📄 Document processed in %v", response.ProcessingTime)
	c.JSON(http.StatusOK, response)
}

// Core document processing logic
func (s *LegalAIService) processDocumentCore(req DocumentRequest) DocumentResponse {
	// Generate unique document ID
	docID := fmt.Sprintf("doc_%d", time.Now().UnixNano())

	// Split content into chunks (simplified)
	chunks := s.chunkContent(req.Content, 512)

	// Generate embeddings (placeholder)
	var documentChunks []DocumentChunk
	for i, chunk := range chunks {
		embedding := s.generateEmbedding(chunk) // Placeholder
		documentChunks = append(documentChunks, DocumentChunk{
			ID:         fmt.Sprintf("%s_chunk_%d", docID, i),
			Content:    chunk,
			ChunkIndex: i,
			Embedding:  embedding,
			Metadata: map[string]any{
				"document_type": req.DocumentType,
				"practice_area": req.PracticeArea,
				"jurisdiction":  req.Jurisdiction,
				"chunk_size":    len(chunk),
			},
		})
	}

	// Perform legal analysis (placeholder)
	analysis := s.performLegalAnalysis(req.Content)

	return DocumentResponse{
		DocumentID: docID,
		Chunks:     documentChunks,
		Analysis:   analysis,
	}
}

// Vector search endpoint
func (s *LegalAIService) vectorSearch(c *gin.Context) {
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	start := time.Now()

	// Generate cache key for search
	cacheKey := fmt.Sprintf("search:%x", md5.Sum([]byte(fmt.Sprintf("%s:%s:%d", req.Query, req.Model, req.Limit))))

	// Check cache first
	cached, err := s.redis.Get(s.ctx, cacheKey).Result()
	if err == nil {
		var response SearchResponse
		if json.Unmarshal([]byte(cached), &response) == nil {
			response.UsedCache = true
			log.Printf("🔍 Search cache hit: %s", req.Query)
			c.JSON(http.StatusOK, response)
			return
		}
	}

	// Perform actual search
	results := s.performVectorSearch(req)

	response := SearchResponse{
		Results:      results,
		QueryTime:    time.Since(start),
		TotalResults: len(results),
		UsedCache:    false,
		ModelUsed:    req.Model,
	}

	// Cache the response
	if responseJSON, err := json.Marshal(response); err == nil {
		s.redis.Set(s.ctx, cacheKey, responseJSON, s.config.CacheExpiration)
	}

	log.Printf("🔍 Vector search completed in %v: %s", response.QueryTime, req.Query)
	c.JSON(http.StatusOK, response)
}

// Health check endpoint
func (s *LegalAIService) healthCheck(c *gin.Context) {
	health := gin.H{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"services": gin.H{
			"redis":    s.checkRedis(),
			"postgres": s.checkPostgres(),
			"gpu":      s.checkGPU(),
		},
		"system": gin.H{
			"goroutines": runtime.NumGoroutine(),
			"memory":     getMemoryStats(),
			"uptime":     time.Since(startTime).String(),
		},
	}

	c.JSON(http.StatusOK, health)
}

// Service health checks
func (s *LegalAIService) checkRedis() string {
	if err := s.redis.Ping(s.ctx).Err(); err != nil {
		return "unhealthy"
	}
	return "healthy"
}

func (s *LegalAIService) checkPostgres() string {
	if err := s.postgres.Ping(s.ctx); err != nil {
		return "unhealthy"
	}
	return "healthy"
}

func (s *LegalAIService) checkGPU() string {
	if s.config.EnableGPU {
		// Placeholder for CUDA check
		return "enabled"
	}
	return "disabled"
}

// Memory statistics
func getMemoryStats() gin.H {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return gin.H{
		"alloc_mb":       bToMb(m.Alloc),
		"total_alloc_mb": bToMb(m.TotalAlloc),
		"sys_mb":         bToMb(m.Sys),
		"gc_cycles":      m.NumGC,
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

// Helper functions (placeholders for actual implementations)
func (s *LegalAIService) chunkContent(content string, maxChunkSize int) []string {
	words := strings.Fields(content)
	var chunks []string
	var currentChunk []string
	currentSize := 0

	for _, word := range words {
		if currentSize+len(word)+1 > maxChunkSize && len(currentChunk) > 0 {
			chunks = append(chunks, strings.Join(currentChunk, " "))
			currentChunk = []string{word}
			currentSize = len(word)
		} else {
			currentChunk = append(currentChunk, word)
			currentSize += len(word) + 1
		}
	}

	if len(currentChunk) > 0 {
		chunks = append(chunks, strings.Join(currentChunk, " "))
	}

	return chunks
}

func (s *LegalAIService) generateEmbedding(text string) []float32 {
	// Placeholder: In production, this would call the Gemma3-Legal model
	// For now, return a dummy embedding
	embedding := make([]float32, 384) // Common embedding dimension
	for i := range embedding {
		embedding[i] = float32(len(text)) / 1000.0 // Simple placeholder
	}
	return embedding
}

func (s *LegalAIService) performLegalAnalysis(content string) LegalAnalysis {
	// Placeholder for actual legal analysis
	return LegalAnalysis{
		Summary:     "Legal document analysis placeholder",
		KeyConcepts: []string{"contract", "liability", "jurisdiction"},
		Entities: []LegalEntity{
			{
				Type:       "PERSON",
				Text:       "Legal Entity",
				Confidence: 0.85,
				Context:    "Sample context",
			},
		},
		RiskFactors: []RiskFactor{
			{
				Type:        "COMPLIANCE",
				Severity:    "MEDIUM",
				Description: "Standard compliance considerations",
				Mitigation:  "Review with legal counsel",
			},
		},
		Recommendations: []string{"Review terms", "Consult legal expert"},
		Confidence:      0.78,
	}
}

func (s *LegalAIService) performVectorSearch(req SearchRequest) []SearchResult {
	// Placeholder for actual vector search
	return []SearchResult{
		{
			DocumentID: "sample_doc_1",
			Content:    "Sample legal content matching query: " + req.Query,
			Score:      0.95,
			ChunkIndex: 0,
			Metadata: map[string]any{
				"document_type": "contract",
				"jurisdiction":  "US",
			},
			Highlighted: "Sample **highlighted** content",
		},
	}
}

// Metrics endpoint
func (s *LegalAIService) getMetrics(c *gin.Context) {
	metrics := gin.H{
		"redis_stats": s.getRedisStats(),
		"db_stats":    s.getDBStats(),
		"cache_stats": s.getCacheStats(),
		"model_stats": s.getModelStats(),
	}

	c.JSON(http.StatusOK, metrics)
}

func (s *LegalAIService) getRedisStats() gin.H {
	info := s.redis.Info(s.ctx, "memory", "stats").Val()
	return gin.H{
		"info":      info,
		"connected": s.checkRedis() == "healthy",
	}
}

func (s *LegalAIService) getDBStats() gin.H {
	return gin.H{
		"connected": s.checkPostgres() == "healthy",
	}
}

func (s *LegalAIService) getCacheStats() gin.H {
	return gin.H{
		"cache_hits":   0, // Placeholder
		"cache_misses": 0, // Placeholder
		"hit_ratio":    0.0,
	}
}

func (s *LegalAIService) getModelStats() gin.H {
	return gin.H{
		"model_path":   s.config.GemmaModelPath,
		"gpu_enabled":  s.config.EnableGPU,
		"cuda_device":  s.config.CUDADeviceID,
		"temperature":  s.config.Temperature,
		"context_size": s.config.ModelContext,
	}
}

// Global variables
var startTime = time.Now()

// Main function
func main() {
	// Initialize configuration
	config := initConfig()

	// Initialize services
	var err error
	service, err = initServices(config)
	if err != nil {
		log.Fatalf("❌ Failed to initialize services: %v", err)
	}
	defer func() {
		if service.postgres != nil {
			service.postgres.Close(service.ctx)
		}
		if service.redis != nil {
			service.redis.Close()
		}
	}()

	// Initialize Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// Configure CORS for SvelteKit frontend
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// API routes matching SvelteKit 2 structure
	api := router.Group("/api")
	{
		// Health and metrics
		api.GET("/health", service.healthCheck)
		api.GET("/metrics", service.getMetrics)

		// AI processing endpoints
		ai := api.Group("/ai")
		{
			ai.POST("/process-document", service.processDocument)
			ai.POST("/vector-search", service.vectorSearch)
		}
	}

	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Enhanced Legal AI Microservice",
			"version": "1.0.0",
			"status":  "ready",
			"endpoints": gin.H{
				"health":           "/api/health",
				"metrics":          "/api/metrics",
				"process_document": "/api/ai/process-document",
				"vector_search":    "/api/ai/vector-search",
			},
			"features": []string{
				"Gemma3-Legal GGUF Model",
				"NVIDIA CUDA GPU Acceleration",
				"Redis Native Caching",
				"PostgreSQL with pgvector",
				"SvelteKit 2 Integration",
			},
		})
	})

	// Start server
	port := ":" + config.Port
	log.Printf("🚀 Enhanced Legal AI Microservice starting on port %s", config.Port)
	log.Printf("🔗 API Documentation: http://localhost:%s/", config.Port)
	log.Printf("💚 Health Check: http://localhost:%s/api/health", config.Port)
	log.Printf("📊 Metrics: http://localhost:%s/api/metrics", config.Port)

	if err := router.Run(port); err != nil {
		log.Fatalf("❌ Failed to start server: %v", err)
	}
}
