// Search Embedder Service - High-Performance Embedding Generation for Neo4j Tricubic Search
// Integrates with Ollama nomic-embed-text model for 384-dimensional legal document embeddings
// Provides batch processing and real-time embedding generation with Redis caching

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

// Ollama API configuration
type OllamaConfig struct {
	BaseURL    string `json:"base_url"`
	Model      string `json:"model"`
	Dimensions int    `json:"dimensions"`
	Timeout    int    `json:"timeout_seconds"`
}

// Embedding request for legal documents
type EmbeddingRequest struct {
	Text         string                 `json:"text"`
	DocumentID   string                 `json:"document_id,omitempty"`
	DocumentType string                 `json:"document_type,omitempty"`
	PracticeArea string                 `json:"practice_area,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	CacheKey     string                 `json:"cache_key,omitempty"`
}

// Batch embedding request
type BatchEmbeddingRequest struct {
	Documents []EmbeddingRequest `json:"documents"`
	BatchSize int                `json:"batch_size,omitempty"`
	Parallel  bool               `json:"parallel,omitempty"`
}

// Embedding response
type EmbeddingResponse struct {
	DocumentID  string    `json:"document_id,omitempty"`
	Embedding   []float32 `json:"embedding"`
	Dimensions  int       `json:"dimensions"`
	Model       string    `json:"model"`
	ProcessTime float64   `json:"process_time_ms"`
	FromCache   bool      `json:"from_cache"`
	SpatialPos  [3]float64 `json:"spatial_position"` // For Neo4j spatial indexing
}

// Batch embedding response
type BatchEmbeddingResponse struct {
	Results     []EmbeddingResponse `json:"results"`
	TotalTime   float64             `json:"total_time_ms"`
	BatchSize   int                 `json:"batch_size"`
	CacheHits   int                 `json:"cache_hits"`
	CacheMisses int                 `json:"cache_misses"`
}

// Ollama API request format
type OllamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

// Ollama API response format
type OllamaEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	Model     string    `json:"model"`
}

// Search Embedder Service
type SearchEmbedderService struct {
	config      OllamaConfig
	redisClient *redis.Client
	httpClient  *http.Client
	cache       map[string][]float32 // In-memory cache for hot embeddings
	cacheMutex  sync.RWMutex
	
	// Performance metrics
	totalRequests    int64
	totalCacheHits   int64
	totalProcessTime float64
	mutex           sync.Mutex
}

// Initialize Search Embedder Service
func NewSearchEmbedderService(config OllamaConfig, redisClient *redis.Client) *SearchEmbedderService {
	return &SearchEmbedderService{
		config:      config,
		redisClient: redisClient,
		httpClient: &http.Client{
			Timeout: time.Duration(config.Timeout) * time.Second,
		},
		cache: make(map[string][]float32),
	}
}

// Generate embedding for single document
func (s *SearchEmbedderService) GenerateEmbedding(req EmbeddingRequest) (*EmbeddingResponse, error) {
	startTime := time.Now()
	
	// Check cache first
	cacheKey := s.generateCacheKey(req)
	if embedding, found := s.getFromCache(cacheKey); found {
		s.incrementCacheHits()
		return &EmbeddingResponse{
			DocumentID:  req.DocumentID,
			Embedding:   embedding,
			Dimensions:  len(embedding),
			Model:       s.config.Model,
			ProcessTime: float64(time.Since(startTime).Nanoseconds()) / 1e6,
			FromCache:   true,
			SpatialPos:  s.embeddingToSpatial(embedding),
		}, nil
	}

	// Generate embedding via Ollama
	embedding, err := s.callOllamaEmbedding(req.Text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Cache the result
	s.cacheEmbedding(cacheKey, embedding)

	s.incrementTotalRequests()
	processTime := float64(time.Since(startTime).Nanoseconds()) / 1e6
	s.addProcessTime(processTime)

	return &EmbeddingResponse{
		DocumentID:  req.DocumentID,
		Embedding:   embedding,
		Dimensions:  len(embedding),
		Model:       s.config.Model,
		ProcessTime: processTime,
		FromCache:   false,
		SpatialPos:  s.embeddingToSpatial(embedding),
	}, nil
}

// Generate embeddings for batch of documents
func (s *SearchEmbedderService) GenerateBatchEmbeddings(req BatchEmbeddingRequest) (*BatchEmbeddingResponse, error) {
	startTime := time.Now()
	batchSize := req.BatchSize
	if batchSize <= 0 {
		batchSize = 10 // Default batch size
	}

	var results []EmbeddingResponse
	var cacheHits, cacheMisses int

	if req.Parallel && len(req.Documents) > 5 {
		// Parallel processing for large batches
		results, cacheHits, cacheMisses = s.processBatchParallel(req.Documents, batchSize)
	} else {
		// Sequential processing for small batches
		results, cacheHits, cacheMisses = s.processBatchSequential(req.Documents)
	}

	totalTime := float64(time.Since(startTime).Nanoseconds()) / 1e6

	return &BatchEmbeddingResponse{
		Results:     results,
		TotalTime:   totalTime,
		BatchSize:   len(req.Documents),
		CacheHits:   cacheHits,
		CacheMisses: cacheMisses,
	}, nil
}

// Process batch in parallel
func (s *SearchEmbedderService) processBatchParallel(documents []EmbeddingRequest, batchSize int) ([]EmbeddingResponse, int, int) {
	var results []EmbeddingResponse
	var resultsMutex sync.Mutex
	var wg sync.WaitGroup
	var cacheHits, cacheMisses int
	var metricsMutex sync.Mutex

	// Process in chunks
	for i := 0; i < len(documents); i += batchSize {
		end := i + batchSize
		if end > len(documents) {
			end = len(documents)
		}

		chunk := documents[i:end]
		wg.Add(1)

		go func(docs []EmbeddingRequest) {
			defer wg.Done()

			for _, doc := range docs {
				embedding, err := s.GenerateEmbedding(doc)
				if err != nil {
					log.Printf("Failed to generate embedding for doc %s: %v", doc.DocumentID, err)
					continue
				}

				resultsMutex.Lock()
				results = append(results, *embedding)
				resultsMutex.Unlock()

				metricsMutex.Lock()
				if embedding.FromCache {
					cacheHits++
				} else {
					cacheMisses++
				}
				metricsMutex.Unlock()
			}
		}(chunk)
	}

	wg.Wait()
	return results, cacheHits, cacheMisses
}

// Process batch sequentially
func (s *SearchEmbedderService) processBatchSequential(documents []EmbeddingRequest) ([]EmbeddingResponse, int, int) {
	var results []EmbeddingResponse
	var cacheHits, cacheMisses int

	for _, doc := range documents {
		embedding, err := s.GenerateEmbedding(doc)
		if err != nil {
			log.Printf("Failed to generate embedding for doc %s: %v", doc.DocumentID, err)
			continue
		}

		results = append(results, *embedding)

		if embedding.FromCache {
			cacheHits++
		} else {
			cacheMisses++
		}
	}

	return results, cacheHits, cacheMisses
}

// Call Ollama API for embedding generation
func (s *SearchEmbedderService) callOllamaEmbedding(text string) ([]float32, error) {
	ollamaReq := OllamaEmbedRequest{
		Model:  s.config.Model,
		Prompt: text,
		Stream: false,
	}

	jsonData, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, err
	}

	resp, err := s.httpClient.Post(
		s.config.BaseURL+"/api/embeddings",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error: %s", string(body))
	}

	var ollamaResp OllamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, err
	}

	// Convert float64 to float32
	embedding := make([]float32, len(ollamaResp.Embedding))
	for i, v := range ollamaResp.Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}

// Convert embedding to 3D spatial coordinates for Neo4j
func (s *SearchEmbedderService) embeddingToSpatial(embedding []float32) [3]float64 {
	if len(embedding) < 3 {
		return [3]float64{0, 0, 0}
	}

	// Use PCA-like projection to 3D space
	// This is a simplified mapping - in production, use proper dimensionality reduction
	x := float64(embedding[0]) * 100
	y := float64(embedding[1]) * 100
	z := float64(embedding[2]) * 100

	// Apply some spatial transformation for better distribution
	x = math.Sin(x/10) * 50
	y = math.Cos(y/10) * 50  
	z = math.Tanh(z/10) * 25

	return [3]float64{x, y, z}
}

// Cache management
func (s *SearchEmbedderService) generateCacheKey(req EmbeddingRequest) string {
	if req.CacheKey != "" {
		return req.CacheKey
	}
	return fmt.Sprintf("embed:%s:%s", s.config.Model, hashText(req.Text))
}

func (s *SearchEmbedderService) getFromCache(key string) ([]float32, bool) {
	// Check in-memory cache first
	s.cacheMutex.RLock()
	if embedding, found := s.cache[key]; found {
		s.cacheMutex.RUnlock()
		return embedding, true
	}
	s.cacheMutex.RUnlock()

	// Check Redis cache
	ctx := context.Background()
	data, err := s.redisClient.Get(ctx, key).Result()
	if err != nil {
		return nil, false
	}

	var embedding []float32
	if err := json.Unmarshal([]byte(data), &embedding); err != nil {
		return nil, false
	}

	// Store in memory cache for hot access
	s.cacheMutex.Lock()
	s.cache[key] = embedding
	s.cacheMutex.Unlock()

	return embedding, true
}

func (s *SearchEmbedderService) cacheEmbedding(key string, embedding []float32) {
	// Store in memory cache
	s.cacheMutex.Lock()
	s.cache[key] = embedding
	s.cacheMutex.Unlock()

	// Store in Redis with TTL
	ctx := context.Background()
	data, _ := json.Marshal(embedding)
	s.redisClient.Set(ctx, key, data, 24*time.Hour) // 24 hour TTL
}

// Metrics
func (s *SearchEmbedderService) incrementTotalRequests() {
	s.mutex.Lock()
	s.totalRequests++
	s.mutex.Unlock()
}

func (s *SearchEmbedderService) incrementCacheHits() {
	s.mutex.Lock()
	s.totalCacheHits++
	s.mutex.Unlock()
}

func (s *SearchEmbedderService) addProcessTime(time float64) {
	s.mutex.Lock()
	s.totalProcessTime += time
	s.mutex.Unlock()
}

func (s *SearchEmbedderService) getMetrics() map[string]interface{} {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	var avgProcessTime float64
	if s.totalRequests > 0 {
		avgProcessTime = s.totalProcessTime / float64(s.totalRequests)
	}

	var cacheHitRatio float64
	if s.totalRequests > 0 {
		cacheHitRatio = float64(s.totalCacheHits) / float64(s.totalRequests)
	}

	return map[string]interface{}{
		"total_requests":     s.totalRequests,
		"cache_hits":         s.totalCacheHits,
		"cache_hit_ratio":    cacheHitRatio,
		"avg_process_time":   avgProcessTime,
		"memory_cache_size":  len(s.cache),
	}
}

// HTTP API endpoints
func (s *SearchEmbedderService) setupRoutes(router *gin.Engine) {
	api := router.Group("/api/embedder")
	{
		api.POST("/generate", s.handleGenerateEmbedding)
		api.POST("/batch", s.handleBatchEmbedding)
		api.GET("/health", s.handleHealthCheck)
		api.GET("/metrics", s.handleMetrics)
		api.DELETE("/cache", s.handleClearCache)
	}
}

func (s *SearchEmbedderService) handleGenerateEmbedding(c *gin.Context) {
	var req EmbeddingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := s.GenerateEmbedding(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (s *SearchEmbedderService) handleBatchEmbedding(c *gin.Context) {
	var req BatchEmbeddingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := s.GenerateBatchEmbeddings(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (s *SearchEmbedderService) handleHealthCheck(c *gin.Context) {
	// Test Ollama connectivity
	testReq := EmbeddingRequest{
		Text: "test connectivity",
	}

	_, err := s.callOllamaEmbedding(testReq.Text)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":     "healthy",
		"model":      s.config.Model,
		"dimensions": s.config.Dimensions,
		"timestamp":  time.Now(),
	})
}

func (s *SearchEmbedderService) handleMetrics(c *gin.Context) {
	c.JSON(http.StatusOK, s.getMetrics())
}

func (s *SearchEmbedderService) handleClearCache(c *gin.Context) {
	// Clear in-memory cache
	s.cacheMutex.Lock()
	s.cache = make(map[string][]float32)
	s.cacheMutex.Unlock()

	// Clear Redis cache (optional - could be selective)
	ctx := context.Background()
	pattern := "embed:*"
	keys, err := s.redisClient.Keys(ctx, pattern).Result()
	if err == nil && len(keys) > 0 {
		s.redisClient.Del(ctx, keys...)
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "Cache cleared successfully",
		"keys_deleted": len(keys),
	})
}

// Utility function for text hashing
func hashText(text string) string {
	// Simple hash for demonstration - use proper hash function in production
	hash := uint32(0)
	for _, r := range text {
		hash = hash*31 + uint32(r)
	}
	return fmt.Sprintf("%x", hash)
}

// Main function
func main() {
	// Configuration
	config := OllamaConfig{
		BaseURL:    "http://localhost:11434",
		Model:      "nomic-embed-text",
		Dimensions: 384,
		Timeout:    30,
	}

	// Redis client
	redisClient := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   1, // Use different DB for embeddings
	})

	// Initialize service
	service := NewSearchEmbedderService(config, redisClient)

	// Setup Gin router
	router := gin.Default()
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	service.setupRoutes(router)

	log.Println("🚀 Search Embedder Service starting on :8088")
	log.Println("🤖 Using Ollama model:", config.Model)
	log.Println("📐 Embedding dimensions:", config.Dimensions)
	log.Println("🔍 Endpoints:")
	log.Println("   POST /api/embedder/generate - Generate single embedding")
	log.Println("   POST /api/embedder/batch - Generate batch embeddings")
	log.Println("   GET  /api/embedder/health - Health check")
	log.Println("   GET  /api/embedder/metrics - Service metrics")
	log.Println("   DELETE /api/embedder/cache - Clear cache")

	if err := router.Run(":8088"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}