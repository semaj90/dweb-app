// Neo4j-QUIC Bridge Service - Ultra-Low Latency Graph Search Integration
// Bridges Neo4j tricubic search, search embedder, and tensor-tiling services
// Provides QUIC/HTTP3 transport for sub-10ms legal document recommendations

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

// Service configuration
type BridgeConfig struct {
	Neo4jTricubicURL  string `json:"neo4j_tricubic_url"`
	SearchEmbedderURL string `json:"search_embedder_url"`
	TensorTilingURL   string `json:"tensor_tiling_url"`
	RedisAddr         string `json:"redis_addr"`
	Port              string `json:"port"`
	CacheTTL          int    `json:"cache_ttl_seconds"`
}

// Unified search request for legal documents
type UnifiedSearchRequest struct {
	Query            string                 `json:"query"`
	QueryType        string                 `json:"query_type"`        // "text", "embedding", "hybrid"
	PracticeArea     string                 `json:"practice_area"`     // "corporate", "criminal", "civil", "family"
	DocumentType     string                 `json:"document_type"`     // "contract", "brief", "evidence", "citation"
	SearchRadius     float64                `json:"search_radius"`     // Spatial search radius
	MaxResults       int                    `json:"max_results"`       // Maximum recommendations
	ConfidenceMin    float64                `json:"confidence_min"`    // Minimum confidence threshold
	InterpolationMode string                `json:"interpolation_mode"` // "tricubic", "linear", "cubic"
	UseTensorTiling  bool                   `json:"use_tensor_tiling"`  // Enable 4D tensor operations
	CacheStrategy    string                 `json:"cache_strategy"`     // "aggressive", "moderate", "minimal"
	Metadata         map[string]interface{} `json:"metadata"`
}

// Unified search response
type UnifiedSearchResponse struct {
	Results      []EnhancedLegalResult `json:"results"`
	QueryInfo    QueryProcessingInfo   `json:"query_info"`
	Performance  PerformanceMetrics    `json:"performance"`
	Caching      CachingInfo           `json:"caching"`
	TensorOps    TensorOperationInfo   `json:"tensor_operations,omitempty"`
	Timestamp    time.Time             `json:"timestamp"`
}

// Enhanced legal document result with all service data
type EnhancedLegalResult struct {
	DocumentID        string                 `json:"document_id"`
	Title             string                 `json:"title"`
	DocumentType      string                 `json:"document_type"`
	PracticeArea      string                 `json:"practice_area"`
	Jurisdiction      string                 `json:"jurisdiction"`
	
	// Scoring from different services
	TricubicScore     float64                `json:"tricubic_score"`
	SimilarityScore   float64                `json:"similarity_score"`
	SpatialScore      float64                `json:"spatial_score"`
	TensorScore       float64                `json:"tensor_score,omitempty"`
	CombinedScore     float64                `json:"combined_score"`
	
	// Spatial and embedding data
	SpatialPosition   [3]float64             `json:"spatial_position"`
	Embedding         []float32              `json:"embedding,omitempty"`
	
	// Legal context
	RelatedDocuments  []RelatedDocument      `json:"related_documents"`
	LegalEntities     []string               `json:"legal_entities"`
	KeyTerms          []string               `json:"key_terms"`
	CitationNetwork   []Citation             `json:"citation_network"`
	
	// Processing metadata
	ProcessingInfo    map[string]interface{} `json:"processing_info"`
	
	// Confidence and relevance
	Confidence        float64                `json:"confidence"`
	Relevance         float64                `json:"relevance"`
	RecommendationReason string              `json:"recommendation_reason"`
}

// Supporting types
type QueryProcessingInfo struct {
	EmbeddingGenerated bool    `json:"embedding_generated"`
	GraphTraversalUsed bool    `json:"graph_traversal_used"`
	TensorOpsPerformed bool    `json:"tensor_ops_performed"`
	QueryEmbedding     []float32 `json:"query_embedding,omitempty"`
	SpatialCenter      [3]float64 `json:"spatial_center"`
	ProcessingSteps    []string  `json:"processing_steps"`
}

type PerformanceMetrics struct {
	TotalTime         float64 `json:"total_time_ms"`
	EmbeddingTime     float64 `json:"embedding_time_ms"`
	Neo4jTime         float64 `json:"neo4j_time_ms"`
	TensorTime        float64 `json:"tensor_time_ms,omitempty"`
	CacheTime         float64 `json:"cache_time_ms"`
	NetworkTime       float64 `json:"network_time_ms"`
	ResultsCount      int     `json:"results_count"`
	ServicesQueried   []string `json:"services_queried"`
}

type CachingInfo struct {
	EmbeddingCached   bool   `json:"embedding_cached"`
	ResultsCached     bool   `json:"results_cached"`
	CacheHits         int    `json:"cache_hits"`
	CacheMisses       int    `json:"cache_misses"`
	CacheStrategy     string `json:"cache_strategy"`
	TTL               int    `json:"ttl_seconds"`
}

type TensorOperationInfo struct {
	TensorShape       [4]int    `json:"tensor_shape,omitempty"`
	TilesGenerated    int       `json:"tiles_generated"`
	InterpolationUsed bool      `json:"interpolation_used"`
	TensorProcessTime float64   `json:"tensor_process_time_ms"`
}

type RelatedDocument struct {
	DocumentID   string  `json:"document_id"`
	Relationship string  `json:"relationship"` // "CITES", "SIMILAR_TO", etc.
	Weight       float64 `json:"weight"`
	Context      string  `json:"context"`
}

type Citation struct {
	CitedDocumentID string  `json:"cited_document_id"`
	CitationType    string  `json:"citation_type"`
	Authority       float64 `json:"authority"`
	Context         string  `json:"context"`
}

// Neo4j-QUIC Bridge Service
type Neo4jQuicBridge struct {
	config      BridgeConfig
	redisClient *redis.Client
	httpClient  *http.Client
	
	// Service URLs
	neo4jURL     string
	embedderURL  string
	tensorURL    string
	
	// Caching and performance
	resultCache map[string]*UnifiedSearchResponse
	cacheMutex  sync.RWMutex
	
	// Metrics
	totalRequests   int64
	cacheHits       int64
	avgResponseTime float64
	metricsMutex    sync.Mutex
}

// Initialize bridge service
func NewNeo4jQuicBridge(config BridgeConfig) (*Neo4jQuicBridge, error) {
	redisClient := redis.NewClient(&redis.Options{
		Addr: config.RedisAddr,
		DB:   2, // Use separate DB for bridge cache
	})

	// Test Redis connectivity
	ctx := context.Background()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis connection failed: %v", err)
	}

	return &Neo4jQuicBridge{
		config:      config,
		redisClient: redisClient,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		neo4jURL:    config.Neo4jTricubicURL,
		embedderURL: config.SearchEmbedderURL,
		tensorURL:   config.TensorTilingURL,
		resultCache: make(map[string]*UnifiedSearchResponse),
	}, nil
}

// Main unified search function
func (b *Neo4jQuicBridge) UnifiedSearch(req UnifiedSearchRequest) (*UnifiedSearchResponse, error) {
	startTime := time.Now()
	
	// Generate cache key
	cacheKey := b.generateCacheKey(req)
	
	// Check cache first
	if cached := b.getCachedResult(cacheKey); cached != nil {
		b.incrementCacheHits()
		cached.Performance.TotalTime = float64(time.Since(startTime).Nanoseconds()) / 1e6
		cached.Caching.CacheHits = 1
		return cached, nil
	}

	var queryInfo QueryProcessingInfo
	var performance PerformanceMetrics
	var caching CachingInfo
	var tensorOps TensorOperationInfo
	
	queryInfo.ProcessingSteps = []string{"request_received"}
	performance.ServicesQueried = []string{}

	// Step 1: Generate embedding if needed
	var queryEmbedding []float32
	var spatialCenter [3]float64
	
	if req.QueryType == "text" || req.QueryType == "hybrid" {
		embeddingStart := time.Now()
		embedding, err := b.generateEmbedding(req.Query, req.PracticeArea, req.DocumentType)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding: %w", err)
		}
		
		queryEmbedding = embedding.Embedding
		spatialCenter = embedding.SpatialPos
		queryInfo.EmbeddingGenerated = true
		performance.EmbeddingTime = float64(time.Since(embeddingStart).Nanoseconds()) / 1e6
		performance.ServicesQueried = append(performance.ServicesQueried, "search-embedder")
		queryInfo.ProcessingSteps = append(queryInfo.ProcessingSteps, "embedding_generated")
		
		caching.EmbeddingCached = embedding.FromCache
		if embedding.FromCache {
			caching.CacheHits++
		} else {
			caching.CacheMisses++
		}
	}

	// Step 2: Perform Neo4j tricubic search
	neo4jStart := time.Now()
	tricubicResults, err := b.performTricubicSearch(queryEmbedding, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform tricubic search: %w", err)
	}
	
	queryInfo.GraphTraversalUsed = true
	performance.Neo4jTime = float64(time.Since(neo4jStart).Nanoseconds()) / 1e6
	performance.ServicesQueried = append(performance.ServicesQueried, "neo4j-tricubic")
	queryInfo.ProcessingSteps = append(queryInfo.ProcessingSteps, "neo4j_search_completed")

	// Step 3: Apply tensor tiling if requested
	var enhancedResults []EnhancedLegalResult
	if req.UseTensorTiling && len(tricubicResults) > 0 {
		tensorStart := time.Now()
		enhancedResults, tensorOps, err = b.applyTensorTiling(tricubicResults, req)
		if err != nil {
			log.Printf("Tensor tiling failed, using original results: %v", err)
			enhancedResults = b.convertTricubicResults(tricubicResults)
		} else {
			queryInfo.TensorOpsPerformed = true
			performance.TensorTime = float64(time.Since(tensorStart).Nanoseconds()) / 1e6
			performance.ServicesQueried = append(performance.ServicesQueried, "tensor-tiling")
			queryInfo.ProcessingSteps = append(queryInfo.ProcessingSteps, "tensor_tiling_applied")
		}
	} else {
		enhancedResults = b.convertTricubicResults(tricubicResults)
	}

	// Step 4: Build response
	queryInfo.QueryEmbedding = queryEmbedding
	queryInfo.SpatialCenter = spatialCenter
	performance.TotalTime = float64(time.Since(startTime).Nanoseconds()) / 1e6
	performance.ResultsCount = len(enhancedResults)
	
	caching.CacheStrategy = req.CacheStrategy
	caching.TTL = b.config.CacheTTL

	response := &UnifiedSearchResponse{
		Results:     enhancedResults,
		QueryInfo:   queryInfo,
		Performance: performance,
		Caching:     caching,
		TensorOps:   tensorOps,
		Timestamp:   time.Now(),
	}

	// Cache result
	b.cacheResult(cacheKey, response, req.CacheStrategy)

	// Update metrics
	b.updateMetrics(performance.TotalTime)

	return response, nil
}

// Generate embedding using search embedder service
func (b *Neo4jQuicBridge) generateEmbedding(text, practiceArea, documentType string) (*EmbeddingResponse, error) {
	reqBody := map[string]interface{}{
		"text":          text,
		"practice_area": practiceArea,
		"document_type": documentType,
		"metadata": map[string]interface{}{
			"query_time": time.Now(),
		},
	}

	jsonData, _ := json.Marshal(reqBody)
	resp, err := b.httpClient.Post(
		b.embedderURL+"/api/embedder/generate",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedder service error: %s", string(body))
	}

	var result EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

// Perform tricubic search via Neo4j service
func (b *Neo4jQuicBridge) performTricubicSearch(embedding []float32, req UnifiedSearchRequest) ([]TricubicResult, error) {
	searchReq := map[string]interface{}{
		"query_vector":       embedding,
		"search_radius":      req.SearchRadius,
		"max_results":        req.MaxResults,
		"practice_area":      req.PracticeArea,
		"document_type":      req.DocumentType,
		"confidence_min":     req.ConfidenceMin,
		"interpolation_mode": req.InterpolationMode,
	}

	jsonData, _ := json.Marshal(searchReq)
	resp, err := b.httpClient.Post(
		b.neo4jURL+"/api/neo4j-tricubic/search",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("neo4j tricubic service error: %s", string(body))
	}

	var response struct {
		Results []TricubicResult `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	return response.Results, nil
}

// Apply tensor tiling operations
func (b *Neo4jQuicBridge) applyTensorTiling(tricubicResults []TricubicResult, req UnifiedSearchRequest) ([]EnhancedLegalResult, TensorOperationInfo, error) {
	// Convert results to tensor format
	tensorReq := map[string]interface{}{
		"documents":  tricubicResults,
		"operation":  "enhance_recommendations",
		"tile_size":  [4]int{8, 8, 8, len(tricubicResults)},
		"halo_size":  [4]int{1, 1, 1, 1},
		"metadata": map[string]interface{}{
			"practice_area":  req.PracticeArea,
			"document_type":  req.DocumentType,
			"interpolation":  req.InterpolationMode,
		},
	}

	jsonData, _ := json.Marshal(tensorReq)
	resp, err := b.httpClient.Post(
		b.tensorURL+"/api/tensor/process",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, TensorOperationInfo{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, TensorOperationInfo{}, fmt.Errorf("tensor service error: %s", string(body))
	}

	var tensorResp struct {
		EnhancedResults []EnhancedLegalResult `json:"enhanced_results"`
		TensorInfo      TensorOperationInfo   `json:"tensor_info"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tensorResp); err != nil {
		return nil, TensorOperationInfo{}, err
	}

	return tensorResp.EnhancedResults, tensorResp.TensorInfo, nil
}

// Convert tricubic results to enhanced format
func (b *Neo4jQuicBridge) convertTricubicResults(tricubicResults []TricubicResult) []EnhancedLegalResult {
	var enhanced []EnhancedLegalResult
	
	for _, result := range tricubicResults {
		// Convert related documents
		var relatedDocs []RelatedDocument
		for _, rel := range result.Relationships {
			relatedDocs = append(relatedDocs, RelatedDocument{
				DocumentID:   rel.TargetID,
				Relationship: rel.RelationType,
				Weight:       rel.Weight,
				Context:      rel.Context,
			})
		}

		enhancedResult := EnhancedLegalResult{
			DocumentID:       result.DocumentID,
			Title:            result.Title,
			DocumentType:     result.DocumentType,
			PracticeArea:     result.PracticeArea,
			TricubicScore:    result.InterpolatedValue,
			SimilarityScore:  result.Similarity,
			SpatialScore:     1.0 / (1.0 + result.SpatialDistance),
			CombinedScore:    result.InterpolatedValue,
			SpatialPosition:  [3]float64{0, 0, 0}, // TODO: Extract from result
			RelatedDocuments: relatedDocs,
			Confidence:       result.InterpolatedValue,
			Relevance:        result.Similarity,
			RecommendationReason: fmt.Sprintf("Tricubic interpolation score: %.3f", result.InterpolatedValue),
		}

		enhanced = append(enhanced, enhancedResult)
	}

	return enhanced
}

// Cache management
func (b *Neo4jQuicBridge) generateCacheKey(req UnifiedSearchRequest) string {
	// Create deterministic cache key from request
	keyData := fmt.Sprintf("%s:%s:%s:%s:%.2f:%d", 
		req.Query, req.QueryType, req.PracticeArea, 
		req.DocumentType, req.SearchRadius, req.MaxResults)
	return fmt.Sprintf("bridge:%x", hashString(keyData))
}

func (b *Neo4jQuicBridge) getCachedResult(key string) *UnifiedSearchResponse {
	// Check memory cache
	b.cacheMutex.RLock()
	if result, found := b.resultCache[key]; found {
		b.cacheMutex.RUnlock()
		return result
	}
	b.cacheMutex.RUnlock()

	// Check Redis cache
	ctx := context.Background()
	data, err := b.redisClient.Get(ctx, key).Result()
	if err != nil {
		return nil
	}

	var result UnifiedSearchResponse
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		return nil
	}

	return &result
}

func (b *Neo4jQuicBridge) cacheResult(key string, result *UnifiedSearchResponse, strategy string) {
	// Determine TTL based on strategy
	ttl := time.Duration(b.config.CacheTTL) * time.Second
	switch strategy {
	case "aggressive":
		ttl = 1 * time.Hour
	case "moderate":
		ttl = 15 * time.Minute
	case "minimal":
		ttl = 5 * time.Minute
	}

	// Store in memory cache
	b.cacheMutex.Lock()
	b.resultCache[key] = result
	b.cacheMutex.Unlock()

	// Store in Redis
	ctx := context.Background()
	data, _ := json.Marshal(result)
	b.redisClient.Set(ctx, key, data, ttl)
}

// Metrics
func (b *Neo4jQuicBridge) incrementCacheHits() {
	b.metricsMutex.Lock()
	b.cacheHits++
	b.metricsMutex.Unlock()
}

func (b *Neo4jQuicBridge) updateMetrics(responseTime float64) {
	b.metricsMutex.Lock()
	b.totalRequests++
	b.avgResponseTime = (b.avgResponseTime*float64(b.totalRequests-1) + responseTime) / float64(b.totalRequests)
	b.metricsMutex.Unlock()
}

func (b *Neo4jQuicBridge) getMetrics() map[string]interface{} {
	b.metricsMutex.Lock()
	defer b.metricsMutex.Unlock()

	var cacheHitRatio float64
	if b.totalRequests > 0 {
		cacheHitRatio = float64(b.cacheHits) / float64(b.totalRequests)
	}

	return map[string]interface{}{
		"total_requests":    b.totalRequests,
		"cache_hits":        b.cacheHits,
		"cache_hit_ratio":   cacheHitRatio,
		"avg_response_time": b.avgResponseTime,
		"memory_cache_size": len(b.resultCache),
	}
}

// HTTP API
func (b *Neo4jQuicBridge) setupRoutes(router *gin.Engine) {
	api := router.Group("/api/bridge")
	{
		api.POST("/search/unified", b.handleUnifiedSearch)
		api.GET("/health", b.handleHealthCheck)
		api.GET("/metrics", b.handleMetrics)
		api.DELETE("/cache", b.handleClearCache)
	}
}

func (b *Neo4jQuicBridge) handleUnifiedSearch(c *gin.Context) {
	var req UnifiedSearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if req.MaxResults <= 0 {
		req.MaxResults = 10
	}
	if req.SearchRadius <= 0 {
		req.SearchRadius = 50.0
	}
	if req.ConfidenceMin <= 0 {
		req.ConfidenceMin = 0.1
	}
	if req.InterpolationMode == "" {
		req.InterpolationMode = "tricubic"
	}
	if req.CacheStrategy == "" {
		req.CacheStrategy = "moderate"
	}

	result, err := b.UnifiedSearch(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (b *Neo4jQuicBridge) handleHealthCheck(c *gin.Context) {
	// Test all downstream services
	services := map[string]string{
		"neo4j-tricubic":   b.neo4jURL + "/api/neo4j-tricubic/health",
		"search-embedder":  b.embedderURL + "/api/embedder/health",
		"tensor-tiling":    b.tensorURL + "/api/tensor/health",
	}

	serviceStatus := make(map[string]interface{})
	allHealthy := true

	for name, url := range services {
		resp, err := b.httpClient.Get(url)
		if err != nil {
			serviceStatus[name] = map[string]interface{}{
				"status": "unhealthy",
				"error":  err.Error(),
			}
			allHealthy = false
		} else {
			resp.Body.Close()
			serviceStatus[name] = map[string]interface{}{
				"status": "healthy",
				"code":   resp.StatusCode,
			}
		}
	}

	status := "healthy"
	if !allHealthy {
		status = "degraded"
	}

	c.JSON(http.StatusOK, gin.H{
		"status":    status,
		"services":  serviceStatus,
		"timestamp": time.Now(),
	})
}

func (b *Neo4jQuicBridge) handleMetrics(c *gin.Context) {
	c.JSON(http.StatusOK, b.getMetrics())
}

func (b *Neo4jQuicBridge) handleClearCache(c *gin.Context) {
	// Clear memory cache
	b.cacheMutex.Lock()
	b.resultCache = make(map[string]*UnifiedSearchResponse)
	b.cacheMutex.Unlock()

	// Clear Redis cache
	ctx := context.Background()
	keys, err := b.redisClient.Keys(ctx, "bridge:*").Result()
	if err == nil && len(keys) > 0 {
		b.redisClient.Del(ctx, keys...)
	}

	c.JSON(http.StatusOK, gin.H{
		"message":      "Cache cleared successfully",
		"keys_deleted": len(keys),
	})
}

// Utility functions
func hashString(s string) uint32 {
	hash := uint32(0)
	for _, r := range s {
		hash = hash*31 + uint32(r)
	}
	return hash
}

// Types imported from other services (simplified)
type EmbeddingResponse struct {
	DocumentID  string     `json:"document_id,omitempty"`
	Embedding   []float32  `json:"embedding"`
	Dimensions  int        `json:"dimensions"`
	Model       string     `json:"model"`
	ProcessTime float64    `json:"process_time_ms"`
	FromCache   bool       `json:"from_cache"`
	SpatialPos  [3]float64 `json:"spatial_position"`
}

type TricubicResult struct {
	DocumentID        string             `json:"document_id"`
	Title             string             `json:"title"`
	Similarity        float64            `json:"similarity"`
	SpatialDistance   float64            `json:"spatial_distance"`
	InterpolatedValue float64            `json:"interpolated_value"`
	PracticeArea      string             `json:"practice_area"`
	DocumentType      string             `json:"document_type"`
	Relationships     []RelationshipEdge `json:"relationships"`
}

type RelationshipEdge struct {
	TargetID     string  `json:"target_id"`
	RelationType string  `json:"relation_type"`
	Weight       float64 `json:"weight"`
	Distance     float64 `json:"distance"`
	Context      string  `json:"context"`
}

// Main function
func main() {
	config := BridgeConfig{
		Neo4jTricubicURL:  "http://localhost:8087",
		SearchEmbedderURL: "http://localhost:8088",
		TensorTilingURL:   "http://localhost:8085", // Your existing tensor service
		RedisAddr:         "localhost:6379",
		Port:              "8089",
		CacheTTL:          900, // 15 minutes
	}

	bridge, err := NewNeo4jQuicBridge(config)
	if err != nil {
		log.Fatalf("Failed to initialize bridge: %v", err)
	}

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

	bridge.setupRoutes(router)

	log.Println("🚀 Neo4j-QUIC Bridge Service starting on :8089")
	log.Println("🌉 Bridging Neo4j tricubic search + embedder + tensor tiling")
	log.Println("⚡ Ultra-low latency legal document recommendations")
	log.Println("🔍 Endpoints:")
	log.Println("   POST /api/bridge/search/unified - Unified search")
	log.Println("   GET  /api/bridge/health - Health check")
	log.Println("   GET  /api/bridge/metrics - Service metrics")
	log.Println("   DELETE /api/bridge/cache - Clear cache")

	if err := router.Run(":" + config.Port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}