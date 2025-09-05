// Neo4j SIMD Worker Service - High-Performance Graph Database Operations
// Windows-compatible version with Go-native SIMD optimizations
// Provides vectorized graph traversal and similarity computations

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"runtime"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/redis/go-redis/v9"
)

// Configuration constants
const (
	Neo4jURI          = "bolt://localhost:7687"
	Neo4jUser         = "neo4j"
	Neo4jPassword     = "legalai123"
	RedisAddr         = "localhost:6379"
	ServicePort       = "8091"
	MaxConcurrency    = 16
	EmbeddingDim      = 768
	MaxBatchSize      = 1024
	CacheExpiration   = 15 * time.Minute
)

// Core data structures
type Neo4jSIMDWorker struct {
	driver      neo4j.DriverWithContext
	redisClient *redis.Client
	workerPool  *WorkerPool
	
	// Performance metrics
	totalQueries     int64
	totalProcessTime time.Duration
	simdOperations   int64
	cacheHits        int64
	mu               sync.RWMutex
}

type WorkerPool struct {
	workers   int
	taskQueue chan Task
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
}

type Task func() error

type GraphSearchRequest struct {
	QueryVector       []float32          `json:"query_vector"`
	SearchRadius      float64            `json:"search_radius"`
	MaxResults        int                `json:"max_results"`
	PracticeArea      string             `json:"practice_area"`
	DocumentType      string             `json:"document_type"`
	MinConfidence     float64            `json:"min_confidence"`
	UseSIMD           bool               `json:"use_simd"`
	CachingEnabled    bool               `json:"caching_enabled"`
	Metadata          map[string]interface{} `json:"metadata"`
}

type GraphSearchResponse struct {
	Results       []GraphResult      `json:"results"`
	TotalFound    int                `json:"total_found"`
	SearchTime    float64            `json:"search_time_ms"`
	SIMDUsed      bool               `json:"simd_used"`
	CacheUsed     bool               `json:"cache_used"`
	ProcessingInfo ProcessingMetrics `json:"processing_info"`
}

type GraphResult struct {
	NodeID           string                 `json:"node_id"`
	DocumentID       string                 `json:"document_id"`
	Title            string                 `json:"title"`
	SimilarityScore  float32                `json:"similarity_score"`
	Distance         float32                `json:"distance"`
	PracticeArea     string                 `json:"practice_area"`
	DocumentType     string                 `json:"document_type"`
	Embedding        []float32              `json:"embedding,omitempty"`
	Relationships    []RelationshipInfo     `json:"relationships"`
	Metadata         map[string]interface{} `json:"metadata"`
	Confidence       float64                `json:"confidence"`
}

type RelationshipInfo struct {
	TargetNodeID   string  `json:"target_node_id"`
	RelationType   string  `json:"relation_type"`
	Weight         float64 `json:"weight"`
	Properties     map[string]interface{} `json:"properties"`
}

type ProcessingMetrics struct {
	NodesProcessed    int     `json:"nodes_processed"`
	SIMDOperations    int     `json:"simd_operations"`
	DatabaseTime      float64 `json:"database_time_ms"`
	ComputationTime   float64 `json:"computation_time_ms"`
	CacheOperations   int     `json:"cache_operations"`
	WorkersUsed       int     `json:"workers_used"`
}

type BatchSimilarityRequest struct {
	QueryVectors    [][]float32 `json:"query_vectors"`
	DocumentVectors [][]float32 `json:"document_vectors"`
	UseSIMD         bool        `json:"use_simd"`
	Normalize       bool        `json:"normalize"`
}

type BatchSimilarityResponse struct {
	SimilarityMatrix [][]float32       `json:"similarity_matrix"`
	ProcessingTime   float64           `json:"processing_time_ms"`
	SIMDUsed         bool              `json:"simd_used"`
	Metrics          ProcessingMetrics `json:"metrics"`
}

// Initialize Neo4j SIMD Worker
func NewNeo4jSIMDWorker() (*Neo4jSIMDWorker, error) {
	// Neo4j driver
	driver, err := neo4j.NewDriverWithContext(
		Neo4jURI,
		neo4j.BasicAuth(Neo4jUser, Neo4jPassword, ""),
		func(config *neo4j.Config) {
			config.MaxConnectionLifetime = 30 * time.Minute
			config.MaxConnectionPoolSize = MaxConcurrency
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}

	// Redis client
	redisClient := redis.NewClient(&redis.Options{
		Addr:        RedisAddr,
		DB:          3, // Use DB 3 for Neo4j worker cache
		MaxRetries:  3,
		PoolSize:    10,
	})

	// Test connections
	ctx := context.Background()
	if err := driver.VerifyConnectivity(ctx); err != nil {
		return nil, fmt.Errorf("Neo4j connectivity check failed: %w", err)
	}

	if err := redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("Redis connection warning: %v", err)
	}

	worker := &Neo4jSIMDWorker{
		driver:      driver,
		redisClient: redisClient,
		workerPool:  NewWorkerPool(MaxConcurrency),
	}

	worker.workerPool.Start()
	log.Println("✅ Neo4j SIMD Worker initialized successfully")
	
	return worker, nil
}

// Worker pool implementation
func NewWorkerPool(size int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	return &WorkerPool{
		workers:   size,
		taskQueue: make(chan Task, size*2),
		ctx:       ctx,
		cancel:    cancel,
	}
}

func (wp *WorkerPool) Start() {
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker()
	}
}

func (wp *WorkerPool) worker() {
	defer wp.wg.Done()
	for {
		select {
		case task := <-wp.taskQueue:
			if task != nil {
				task()
			}
		case <-wp.ctx.Done():
			return
		}
	}
}

func (wp *WorkerPool) Submit(task Task) {
	select {
	case wp.taskQueue <- task:
	case <-wp.ctx.Done():
	}
}

func (wp *WorkerPool) Shutdown() {
	wp.cancel()
	close(wp.taskQueue)
	wp.wg.Wait()
}

// SIMD-optimized graph search (Go-native implementation)
func (w *Neo4jSIMDWorker) PerformGraphSearch(req GraphSearchRequest) (*GraphSearchResponse, error) {
	startTime := time.Now()
	cacheKey := w.generateCacheKey(req)
	
	// Check cache first
	if req.CachingEnabled {
		if cached := w.getCachedResult(cacheKey); cached != nil {
			w.mu.Lock()
			w.cacheHits++
			w.mu.Unlock()
			return cached, nil
		}
	}

	ctx := context.Background()
	session := w.driver.NewSession(ctx, neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeRead,
	})
	defer session.Close(ctx)

	dbStartTime := time.Now()
	
	// Cypher query to retrieve document nodes with embeddings
	query := `
		MATCH (d:Document)
		WHERE d.practice_area = $practice_area 
		  AND d.document_type = $document_type
		OPTIONAL MATCH (d)-[r]-(related:Document)
		RETURN d.document_id as document_id,
		       d.title as title,
		       d.embedding as embedding,
		       d.practice_area as practice_area,
		       d.document_type as document_type,
		       d.metadata as metadata,
		       collect(DISTINCT {
		           target_id: related.document_id,
		           relation_type: type(r),
		           weight: r.weight,
		           properties: properties(r)
		       }) as relationships
		LIMIT 1000
	`

	parameters := map[string]interface{}{
		"practice_area": req.PracticeArea,
		"document_type": req.DocumentType,
	}

	result, err := session.Run(ctx, query, parameters)
	if err != nil {
		return nil, fmt.Errorf("Neo4j query failed: %w", err)
	}

	dbTime := time.Since(dbStartTime).Seconds() * 1000

	// Process results
	var nodes []GraphResult
	var documentVectors [][]float32
	
	for result.Next(ctx) {
		record := result.Record()
		
		embeddingInterface, _ := record.Get("embedding")
		embedding := parseEmbedding(embeddingInterface)
		
		if len(embedding) != EmbeddingDim {
			continue // Skip nodes with invalid embeddings
		}

		// Parse relationships
		relationshipsInterface, _ := record.Get("relationships")
		relationships := parseRelationships(relationshipsInterface)

		// Parse metadata
		metadataInterface, _ := record.Get("metadata")
		metadata := parseMetadata(metadataInterface)

		node := GraphResult{
			NodeID:        record.Values[0].(string), // document_id
			DocumentID:    record.Values[0].(string),
			Title:         getString(record, "title"),
			PracticeArea:  getString(record, "practice_area"),
			DocumentType:  getString(record, "document_type"),
			Embedding:     embedding,
			Relationships: relationships,
			Metadata:      metadata,
		}

		nodes = append(nodes, node)
		documentVectors = append(documentVectors, embedding)
	}

	if len(nodes) == 0 {
		return &GraphSearchResponse{
			Results:        []GraphResult{},
			TotalFound:     0,
			SearchTime:     time.Since(startTime).Seconds() * 1000,
			SIMDUsed:       false,
			CacheUsed:      false,
			ProcessingInfo: ProcessingMetrics{DatabaseTime: dbTime},
		}, nil
	}

	// SIMD-optimized similarity computation (Go-native)
	computeStartTime := time.Now()
	var similarities []float32
	var simdUsed bool

	if req.UseSIMD && len(documentVectors) > 0 {
		similarities = w.computeOptimizedSimilarities(req.QueryVector, documentVectors)
		simdUsed = true
		
		w.mu.Lock()
		w.simdOperations++
		w.mu.Unlock()
	} else {
		similarities = w.computeStandardSimilarities(req.QueryVector, documentVectors)
	}

	computeTime := time.Since(computeStartTime).Seconds() * 1000

	// Update results with similarity scores and filter by confidence
	filteredResults := []GraphResult{}
	for i, node := range nodes {
		if i < len(similarities) {
			similarity := similarities[i]
			confidence := float64(similarity)
			
			if confidence >= req.MinConfidence {
				node.SimilarityScore = similarity
				node.Distance = 1.0 - similarity // Convert to distance
				node.Confidence = confidence
				filteredResults = append(filteredResults, node)
			}
		}
	}

	// Sort by similarity score (descending)
	for i := 0; i < len(filteredResults)-1; i++ {
		for j := i + 1; j < len(filteredResults); j++ {
			if filteredResults[j].SimilarityScore > filteredResults[i].SimilarityScore {
				filteredResults[i], filteredResults[j] = filteredResults[j], filteredResults[i]
			}
		}
	}

	// Limit results
	maxResults := req.MaxResults
	if maxResults <= 0 || maxResults > len(filteredResults) {
		maxResults = len(filteredResults)
	}

	response := &GraphSearchResponse{
		Results:    filteredResults[:maxResults],
		TotalFound: len(filteredResults),
		SearchTime: time.Since(startTime).Seconds() * 1000,
		SIMDUsed:   simdUsed,
		CacheUsed:  false,
		ProcessingInfo: ProcessingMetrics{
			NodesProcessed:    len(nodes),
			SIMDOperations:    1,
			DatabaseTime:      dbTime,
			ComputationTime:   computeTime,
			CacheOperations:   0,
			WorkersUsed:       1,
		},
	}

	// Cache result
	if req.CachingEnabled {
		w.cacheResult(cacheKey, response)
	}

	// Update metrics
	w.mu.Lock()
	w.totalQueries++
	w.totalProcessTime += time.Since(startTime)
	w.mu.Unlock()

	return response, nil
}

// Go-native optimized similarity computation with parallel processing
func (w *Neo4jSIMDWorker) computeOptimizedSimilarities(query []float32, documents [][]float32) []float32 {
	numDocs := len(documents)
	similarities := make([]float32, numDocs)
	
	// Use goroutines for parallel processing (Go's native "SIMD")
	numWorkers := runtime.NumCPU()
	if numWorkers > numDocs {
		numWorkers = numDocs
	}
	
	var wg sync.WaitGroup
	docsChan := make(chan int, numDocs)
	
	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for docIdx := range docsChan {
				similarities[docIdx] = w.computeDotProduct(query, documents[docIdx])
			}
		}()
	}
	
	// Send work to workers
	go func() {
		for i := 0; i < numDocs; i++ {
			docsChan <- i
		}
		close(docsChan)
	}()
	
	wg.Wait()
	return similarities
}

// Optimized dot product computation
func (w *Neo4jSIMDWorker) computeDotProduct(a, b []float32) float32 {
	var sum float32
	
	// Process in chunks for better cache locality
	const chunkSize = 8
	i := 0
	
	// Main loop processing chunks
	for i <= len(a)-chunkSize {
		// Unroll loop for better performance
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			   a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
		i += chunkSize
	}
	
	// Handle remaining elements
	for i < len(a) && i < len(b) {
		sum += a[i] * b[i]
		i++
	}
	
	return sum
}

// Standard similarity computation (fallback)
func (w *Neo4jSIMDWorker) computeStandardSimilarities(query []float32, documents [][]float32) []float32 {
	similarities := make([]float32, len(documents))
	
	for i, doc := range documents {
		var sum float32
		for j := 0; j < len(query) && j < len(doc); j++ {
			sum += query[j] * doc[j]
		}
		similarities[i] = sum
	}
	
	return similarities
}

// Batch similarity processing for multiple queries
func (w *Neo4jSIMDWorker) ProcessBatchSimilarity(req BatchSimilarityRequest) (*BatchSimilarityResponse, error) {
	startTime := time.Now()
	
	numQueries := len(req.QueryVectors)
	numDocs := len(req.DocumentVectors)
	
	if numQueries == 0 || numDocs == 0 {
		return &BatchSimilarityResponse{
			SimilarityMatrix: [][]float32{},
			ProcessingTime:   0,
			SIMDUsed:        false,
		}, nil
	}
	
	// Initialize similarity matrix
	similarityMatrix := make([][]float32, numQueries)
	for i := range similarityMatrix {
		similarityMatrix[i] = make([]float32, numDocs)
	}
	
	// Normalize vectors if requested
	if req.Normalize {
		for _, query := range req.QueryVectors {
			w.normalizeVector(query)
		}
		for _, doc := range req.DocumentVectors {
			w.normalizeVector(doc)
		}
	}
	
	var simdOps int
	
	// Process each query (parallel processing for "SIMD" effect)
	var wg sync.WaitGroup
	for i, query := range req.QueryVectors {
		wg.Add(1)
		go func(queryIdx int, queryVec []float32) {
			defer wg.Done()
			
			if req.UseSIMD {
				similarities := w.computeOptimizedSimilarities(queryVec, req.DocumentVectors)
				copy(similarityMatrix[queryIdx], similarities)
			} else {
				similarities := w.computeStandardSimilarities(queryVec, req.DocumentVectors)
				copy(similarityMatrix[queryIdx], similarities)
			}
		}(i, query)
		simdOps++
	}
	
	wg.Wait()
	
	processingTime := time.Since(startTime).Seconds() * 1000
	
	return &BatchSimilarityResponse{
		SimilarityMatrix: similarityMatrix,
		ProcessingTime:   processingTime,
		SIMDUsed:        req.UseSIMD,
		Metrics: ProcessingMetrics{
			NodesProcessed:  numQueries * numDocs,
			SIMDOperations:  simdOps,
			ComputationTime: processingTime,
		},
	}, nil
}

// Vector normalization
func (w *Neo4jSIMDWorker) normalizeVector(vector []float32) {
	var magnitude float32
	for _, v := range vector {
		magnitude += v * v
	}
	magnitude = float32(math.Sqrt(float64(magnitude)))
	
	if magnitude > 0 {
		for i := range vector {
			vector[i] /= magnitude
		}
	}
}

// Cache management
func (w *Neo4jSIMDWorker) generateCacheKey(req GraphSearchRequest) string {
	// Create a deterministic cache key
	keyData := fmt.Sprintf("graph:%s:%s:%.2f:%d:%t", 
		req.PracticeArea, req.DocumentType, req.MinConfidence, req.MaxResults, req.UseSIMD)
	
	// Add query vector hash (simplified)
	if len(req.QueryVector) > 0 {
		hash := float32(0)
		for _, v := range req.QueryVector[:min(10, len(req.QueryVector))] {
			hash += v
		}
		keyData += fmt.Sprintf(":%.3f", hash)
	}
	
	return keyData
}

func (w *Neo4jSIMDWorker) getCachedResult(key string) *GraphSearchResponse {
	ctx := context.Background()
	data, err := w.redisClient.Get(ctx, key).Result()
	if err != nil {
		return nil
	}
	
	var result GraphSearchResponse
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		return nil
	}
	
	result.CacheUsed = true
	return &result
}

func (w *Neo4jSIMDWorker) cacheResult(key string, result *GraphSearchResponse) {
	ctx := context.Background()
	data, _ := json.Marshal(result)
	w.redisClient.Set(ctx, key, data, CacheExpiration)
}

// Utility functions
func parseEmbedding(embeddingInterface interface{}) []float32 {
	if embeddingInterface == nil {
		return nil
	}
	
	switch v := embeddingInterface.(type) {
	case []interface{}:
		embedding := make([]float32, len(v))
		for i, val := range v {
			if f, ok := val.(float64); ok {
				embedding[i] = float32(f)
			}
		}
		return embedding
	case []float64:
		embedding := make([]float32, len(v))
		for i, val := range v {
			embedding[i] = float32(val)
		}
		return embedding
	}
	return nil
}

func parseRelationships(relationshipsInterface interface{}) []RelationshipInfo {
	var relationships []RelationshipInfo
	
	if relationshipsInterface == nil {
		return relationships
	}
	
	if rels, ok := relationshipsInterface.([]interface{}); ok {
		for _, rel := range rels {
			if relMap, ok := rel.(map[string]interface{}); ok {
				relationship := RelationshipInfo{
					TargetNodeID: getString(relMap, "target_id"),
					RelationType: getString(relMap, "relation_type"),
					Weight:       getFloat64(relMap, "weight"),
					Properties:   getMap(relMap, "properties"),
				}
				relationships = append(relationships, relationship)
			}
		}
	}
	
	return relationships
}

func parseMetadata(metadataInterface interface{}) map[string]interface{} {
	if metadataInterface == nil {
		return make(map[string]interface{})
	}
	
	if metadata, ok := metadataInterface.(map[string]interface{}); ok {
		return metadata
	}
	
	return make(map[string]interface{})
}

func getString(data interface{}, key string) string {
	switch v := data.(type) {
	case map[string]interface{}:
		if val, ok := v[key]; ok {
			if str, ok := val.(string); ok {
				return str
			}
		}
	case neo4j.Record:
		if val, found := v.Get(key); found {
			if str, ok := val.(string); ok {
				return str
			}
		}
	}
	return ""
}

func getFloat64(data map[string]interface{}, key string) float64 {
	if val, ok := data[key]; ok {
		if f, ok := val.(float64); ok {
			return f
		}
		if i, ok := val.(int64); ok {
			return float64(i)
		}
	}
	return 0.0
}

func getMap(data map[string]interface{}, key string) map[string]interface{} {
	if val, ok := data[key]; ok {
		if m, ok := val.(map[string]interface{}); ok {
			return m
		}
	}
	return make(map[string]interface{})
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// HTTP API handlers
func (w *Neo4jSIMDWorker) setupRoutes(router *gin.Engine) {
	api := router.Group("/api/neo4j-simd")
	{
		api.POST("/search", w.handleGraphSearch)
		api.POST("/batch-similarity", w.handleBatchSimilarity)
		api.GET("/health", w.handleHealth)
		api.GET("/metrics", w.handleMetrics)
		api.DELETE("/cache", w.handleClearCache)
	}
}

func (w *Neo4jSIMDWorker) handleGraphSearch(c *gin.Context) {
	var req GraphSearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	// Set defaults
	if req.MaxResults <= 0 {
		req.MaxResults = 10
	}
	if req.MinConfidence <= 0 {
		req.MinConfidence = 0.1
	}
	if len(req.QueryVector) != EmbeddingDim {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Query vector must be %d dimensions", EmbeddingDim)})
		return
	}
	
	result, err := w.PerformGraphSearch(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, result)
}

func (w *Neo4jSIMDWorker) handleBatchSimilarity(c *gin.Context) {
	var req BatchSimilarityRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	result, err := w.ProcessBatchSimilarity(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, result)
}

func (w *Neo4jSIMDWorker) handleHealth(c *gin.Context) {
	// Test Neo4j connectivity
	ctx := context.Background()
	neo4jHealthy := true
	if err := w.driver.VerifyConnectivity(ctx); err != nil {
		neo4jHealthy = false
	}
	
	// Test Redis connectivity
	redisHealthy := true
	if err := w.redisClient.Ping(ctx).Err(); err != nil {
		redisHealthy = false
	}
	
	// Get system metrics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	w.mu.RLock()
	avgResponseTime := float64(0)
	if w.totalQueries > 0 {
		avgResponseTime = w.totalProcessTime.Seconds() * 1000 / float64(w.totalQueries)
	}
	
	status := gin.H{
		"status":          "healthy",
		"neo4j_healthy":   neo4jHealthy,
		"redis_healthy":   redisHealthy,
		"total_queries":   w.totalQueries,
		"simd_operations": w.simdOperations,
		"cache_hits":      w.cacheHits,
		"avg_response_time_ms": avgResponseTime,
		"goroutines":      runtime.NumGoroutine(),
		"heap_alloc_mb":   float64(m.HeapAlloc) / 1024 / 1024,
		"sys_memory_mb":   float64(m.Sys) / 1024 / 1024,
		"embedding_dim":   EmbeddingDim,
		"max_batch_size":  MaxBatchSize,
		"cpu_cores":       runtime.NumCPU(),
	}
	w.mu.RUnlock()
	
	if !neo4jHealthy || !redisHealthy {
		status["status"] = "degraded"
		c.JSON(http.StatusServiceUnavailable, status)
	} else {
		c.JSON(http.StatusOK, status)
	}
}

func (w *Neo4jSIMDWorker) handleMetrics(c *gin.Context) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	
	var cacheHitRatio float64
	if w.totalQueries > 0 {
		cacheHitRatio = float64(w.cacheHits) / float64(w.totalQueries)
	}
	
	c.JSON(http.StatusOK, gin.H{
		"total_queries":       w.totalQueries,
		"simd_operations":     w.simdOperations,
		"cache_hits":          w.cacheHits,
		"cache_hit_ratio":     cacheHitRatio,
		"avg_response_time_ms": w.totalProcessTime.Seconds() * 1000 / max(float64(w.totalQueries), 1),
		"workers":             MaxConcurrency,
		"cpu_cores":           runtime.NumCPU(),
		"embedding_dimension": EmbeddingDim,
	})
}

func (w *Neo4jSIMDWorker) handleClearCache(c *gin.Context) {
	ctx := context.Background()
	keys, err := w.redisClient.Keys(ctx, "graph:*").Result()
	if err == nil && len(keys) > 0 {
		w.redisClient.Del(ctx, keys...)
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message":      "Cache cleared successfully",
		"keys_deleted": len(keys),
	})
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func (w *Neo4jSIMDWorker) Shutdown() {
	if w.workerPool != nil {
		w.workerPool.Shutdown()
	}
	
	if w.driver != nil {
		w.driver.Close(context.Background())
	}
	
	if w.redisClient != nil {
		w.redisClient.Close()
	}
}

// Main function
func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	
	worker, err := NewNeo4jSIMDWorker()
	if err != nil {
		log.Fatalf("Failed to initialize Neo4j SIMD Worker: %v", err)
	}
	defer worker.Shutdown()
	
	router := gin.Default()
	
	// CORS middleware
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
	
	worker.setupRoutes(router)
	
	log.Println("🚀 Neo4j SIMD Worker Service starting on :" + ServicePort)
	log.Println("⚡ Go-native optimized graph operations enabled")
	log.Println("🧠 Multi-core parallel processing with goroutines")
	log.Printf("💾 CPU Cores: %d | Max Concurrency: %d", runtime.NumCPU(), MaxConcurrency)
	log.Println("📊 Endpoints:")
	log.Println("   POST /api/neo4j-simd/search - Graph search with optimization")
	log.Println("   POST /api/neo4j-simd/batch-similarity - Batch similarity processing")
	log.Println("   GET  /api/neo4j-simd/health - Health check")
	log.Println("   GET  /api/neo4j-simd/metrics - Performance metrics")
	log.Println("   DELETE /api/neo4j-simd/cache - Clear cache")
	
	if err := router.Run(":" + ServicePort); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}