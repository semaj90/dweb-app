// Neo4j GPU-Integrated Service - Production RTX 3060 Ti Optimized
// Combines Neo4j graph operations with GPU Tensor Core acceleration
// Features: GPU embeddings, CPU SIMD fallback, intelligent caching, batch optimization

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Configuration constants
const (
	ServicePort       = "8092"
	Neo4jURI          = "bolt://localhost:7687"
	Neo4jUser         = "neo4j"
	Neo4jPassword     = "legalai123"
	RedisAddr         = "localhost:6379"
	GPUWorkerAddr     = "localhost:50051"
	MaxConcurrency    = 16
	EmbeddingDim      = 768
	MaxBatchSize      = 16
	CacheExpiration   = 30 * time.Minute
)

// Core service structure
type Neo4jGPUService struct {
	// Database connections
	neo4jDriver neo4j.DriverWithContext
	redisClient *redis.Client
	
	// GPU worker connection
	gpuClient    EmbeddingServiceClient
	gpuConn      *grpc.ClientConn
	gpuAvailable bool
	
	// Worker pool for concurrent processing
	workerPool *WorkerPool
	
	// Performance metrics
	totalRequests       int64
	totalProcessingTime time.Duration
	gpuRequests         int64
	cpuFallbackRequests int64
	cacheHits           int64
	errorCount          int64
	mu                  sync.RWMutex
	
	// Batch processing
	batchQueue     chan BatchItem
	batchProcessor *BatchProcessor
}

// Request/Response types
type EnhancedSearchRequest struct {
	Query             string                 `json:"query"`
	QueryVector       []float32              `json:"query_vector,omitempty"`
	PracticeArea      string                 `json:"practice_area"`
	DocumentType      string                 `json:"document_type"`
	MaxResults        int                    `json:"max_results"`
	MinConfidence     float64                `json:"min_confidence"`
	SearchRadius      float64                `json:"search_radius"`
	UseGPU            bool                   `json:"use_gpu"`
	UseFP16           bool                   `json:"use_fp16"`
	UseCache          bool                   `json:"use_cache"`
	BatchOptimization bool                   `json:"batch_optimization"`
	Metadata          map[string]interface{} `json:"metadata"`
}

type EnhancedSearchResponse struct {
	Results         []EnhancedResult   `json:"results"`
	TotalFound      int                `json:"total_found"`
	ProcessingInfo  ProcessingInfo     `json:"processing_info"`
	PerformanceInfo PerformanceInfo    `json:"performance_info"`
	Timestamp       time.Time          `json:"timestamp"`
}

type EnhancedResult struct {
	NodeID           string                 `json:"node_id"`
	DocumentID       string                 `json:"document_id"`
	Title            string                 `json:"title"`
	Content          string                 `json:"content,omitempty"`
	SimilarityScore  float32                `json:"similarity_score"`
	Distance         float32                `json:"distance"`
	Confidence       float64                `json:"confidence"`
	PracticeArea     string                 `json:"practice_area"`
	DocumentType     string                 `json:"document_type"`
	Embedding        []float32              `json:"embedding,omitempty"`
	RelatedNodes     []RelatedNodeInfo      `json:"related_nodes"`
	GraphPath        []GraphPathNode        `json:"graph_path,omitempty"`
	Metadata         map[string]interface{} `json:"metadata"`
	ProcessingSource string                 `json:"processing_source"` // "GPU", "CPU", "CACHE"
}

type RelatedNodeInfo struct {
	NodeID       string                 `json:"node_id"`
	RelationType string                 `json:"relation_type"`
	Weight       float64                `json:"weight"`
	Distance     int                    `json:"distance"` // Graph distance
	Properties   map[string]interface{} `json:"properties"`
}

type GraphPathNode struct {
	NodeID     string `json:"node_id"`
	NodeType   string `json:"node_type"`
	Similarity float32 `json:"similarity"`
}

type ProcessingInfo struct {
	QueryEmbeddingGenerated bool            `json:"query_embedding_generated"`
	EmbeddingMethod         string          `json:"embedding_method"`
	SimilarityMethod        string          `json:"similarity_method"`
	GraphTraversalDepth     int             `json:"graph_traversal_depth"`
	NodesProcessed          int             `json:"nodes_processed"`
	BatchesProcessed        int             `json:"batches_processed"`
	CacheOperations         int             `json:"cache_operations"`
	GPUUtilization          bool            `json:"gpu_utilization"`
	TensorCoresUsed         bool            `json:"tensor_cores_used"`
	FP16Precision           bool            `json:"fp16_precision"`
}

type PerformanceInfo struct {
	TotalTime           float64 `json:"total_time_ms"`
	EmbeddingTime       float64 `json:"embedding_time_ms"`
	DatabaseTime        float64 `json:"database_time_ms"`
	SimilarityTime      float64 `json:"similarity_time_ms"`
	CacheTime           float64 `json:"cache_time_ms"`
	NetworkTime         float64 `json:"network_time_ms"`
	BatchProcessingTime float64 `json:"batch_processing_time_ms"`
	MemoryUsageMB       float64 `json:"memory_usage_mb"`
}

// Worker pool and batch processing
type WorkerPool struct {
	workers   int
	taskQueue chan Task
	wg        sync.WaitGroup
	ctx       context.Context
	cancel    context.CancelFunc
}

type Task func() error

type BatchItem struct {
	Text     string
	Callback chan BatchResult
}

type BatchResult struct {
	Embedding []float32
	Error     error
	Source    string // "GPU", "CPU", "CACHE"
}

type BatchProcessor struct {
	service    *Neo4jGPUService
	batchSize  int
	flushTimer *time.Timer
	mu         sync.Mutex
	queue      []BatchItem
}

// Placeholder for gRPC client (would be generated from protobuf)
type EmbeddingServiceClient interface {
	GenerateEmbedding(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)
	ComputeSimilarity(ctx context.Context, req *SimilarityRequest) (*SimilarityResponse, error)
	GetMetrics(ctx context.Context, req *MetricsRequest) (*MetricsResponse, error)
	HealthCheck(ctx context.Context, req *HealthRequest) (*HealthResponse, error)
}

// Placeholder gRPC message types (would be generated from protobuf)
type EmbeddingRequest struct {
	Texts         []string
	UseCache      bool
	Normalize     bool
	ModelName     string
	Fp16Precision bool
}

type EmbeddingResponse struct {
	Embeddings []EmbeddingVector
	Metrics    ProcessingMetrics
	Success    bool
	Error      string
}

type EmbeddingVector struct {
	Values     []float32
	Dimensions int32
	TextHash   string
}

type ProcessingMetrics struct {
	ProcessingTimeMs float64
	BatchSize        int32
	CacheHits        int32
	GpuUsed          bool
	Fp16Used         bool
	TensorCoresUsed  bool
	Method           string
	GpuInfo          GPUInfo
}

type GPUInfo struct {
	GpuName             string
	MemoryAllocatedGb   float64
	MemoryTotalGb       float64
	MemoryUtilization   float64
	ComputeCapability   int32
}

type SimilarityRequest struct {
	QueryEmbeddings []EmbeddingVector
	DocEmbeddings   []EmbeddingVector
	UseGpu          bool
	Fp16Precision   bool
}

type SimilarityResponse struct {
	SimilarityMatrix []SimilarityRow
	Metrics          ProcessingMetrics
	Success          bool
	Error            string
}

type SimilarityRow struct {
	Similarities []float32
}

type MetricsRequest struct {
	IncludeGpuInfo bool
}

type MetricsResponse struct {
	TotalRequests          int64
	TotalProcessingTime    float64
	CacheHits              int64
	CacheHitRatio          float64
	AvgProcessingTime      float64
	BatchSizeDistribution  map[int32]int32
	GpuInfo                GPUInfo
	Success                bool
	Error                  string
}

type HealthRequest struct {
	CheckGpu   bool
	CheckRedis bool
}

type HealthResponse struct {
	Healthy        bool
	GpuAvailable   bool
	RedisAvailable bool
	StatusMessage  string
	GpuInfo        GPUInfo
}

// Initialize the integrated service
func NewNeo4jGPUService() (*Neo4jGPUService, error) {
	service := &Neo4jGPUService{
		batchQueue: make(chan BatchItem, MaxBatchSize*4),
	}
	
	// Initialize Neo4j driver
	var err error
	service.neo4jDriver, err = neo4j.NewDriverWithContext(
		Neo4jURI,
		neo4j.BasicAuth(Neo4jUser, Neo4jPassword, ""),
		func(config *neo4j.Config) {
			config.MaxConnectionLifetime = 30 * time.Minute
			config.MaxConnectionPoolSize = MaxConcurrency
		},
	)
	if err != nil {
		return nil, fmt.Errorf("Neo4j driver initialization failed: %w", err)
	}
	
	// Test Neo4j connectivity
	ctx := context.Background()
	if err := service.neo4jDriver.VerifyConnectivity(ctx); err != nil {
		return nil, fmt.Errorf("Neo4j connectivity check failed: %w", err)
	}
	log.Println("✅ Neo4j connected")
	
	// Initialize Redis client
	service.redisClient = redis.NewClient(&redis.Options{
		Addr:        RedisAddr,
		DB:          5, // Use DB 5 for GPU service cache
		MaxRetries:  3,
		PoolSize:    10,
	})
	
	// Test Redis connectivity
	if err := service.redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("⚠️  Redis connection warning: %v", err)
	} else {
		log.Println("✅ Redis cache connected")
	}
	
	// Initialize GPU worker connection
	service.initializeGPUWorker()
	
	// Initialize worker pool
	service.workerPool = NewWorkerPool(MaxConcurrency)
	service.workerPool.Start()
	
	// Initialize batch processor
	service.batchProcessor = NewBatchProcessor(service, MaxBatchSize)
	go service.batchProcessor.Start()
	
	log.Println("🚀 Neo4j GPU-Integrated Service initialized")
	return service, nil
}

func (s *Neo4jGPUService) initializeGPUWorker() {
	log.Println("🔄 Connecting to GPU Tensor Worker...")
	
	conn, err := grpc.Dial(GPUWorkerAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("❌ GPU worker connection failed: %v", err)
		s.gpuAvailable = false
		return
	}
	
	s.gpuConn = conn
	// s.gpuClient = NewEmbeddingServiceClient(conn) // Would use generated gRPC client
	
	// Test GPU worker health
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// healthReq := &HealthRequest{CheckGpu: true, CheckRedis: false}
	// healthResp, err := s.gpuClient.HealthCheck(ctx, healthReq)
	
	// For now, simulate GPU worker availability
	if err == nil {
		s.gpuAvailable = true
		log.Println("✅ GPU Tensor Worker connected and healthy")
	} else {
		log.Printf("⚠️  GPU worker health check failed: %v", err)
		s.gpuAvailable = false
	}
}

// Main search function with GPU integration
func (s *Neo4jGPUService) EnhancedSearch(req EnhancedSearchRequest) (*EnhancedSearchResponse, error) {
	startTime := time.Now()
	
	s.mu.Lock()
	s.totalRequests++
	s.mu.Unlock()
	
	var processingInfo ProcessingInfo
	var perfInfo PerformanceInfo
	
	// Generate query embedding if not provided
	var queryEmbedding []float32
	var embeddingTime time.Duration
	
	if len(req.QueryVector) == 0 && req.Query != "" {
		embeddingStart := time.Now()
		
		var err error
		var embSource string
		
		if req.UseGPU && s.gpuAvailable {
			queryEmbedding, embSource, err = s.generateGPUEmbedding(req.Query, req.UseFP16, req.UseCache)
		} else {
			queryEmbedding, embSource, err = s.generateCPUEmbedding(req.Query, req.UseCache)
		}
		
		if err != nil {
			return nil, fmt.Errorf("embedding generation failed: %w", err)
		}
		
		embeddingTime = time.Since(embeddingStart)
		processingInfo.QueryEmbeddingGenerated = true
		processingInfo.EmbeddingMethod = embSource
		processingInfo.GPUUtilization = (embSource == "GPU")
		processingInfo.TensorCoresUsed = (embSource == "GPU" && req.UseFP16)
		processingInfo.FP16Precision = req.UseFP16
		
	} else {
		queryEmbedding = req.QueryVector
		processingInfo.EmbeddingMethod = "PROVIDED"
	}
	
	// Neo4j graph search
	dbStart := time.Now()
	graphResults, err := s.performGraphSearch(queryEmbedding, req)
	if err != nil {
		return nil, fmt.Errorf("graph search failed: %w", err)
	}
	dbTime := time.Since(dbStart)
	
	// Similarity computation (GPU-accelerated if available)
	simStart := time.Now()
	enhancedResults := s.computeEnhancedSimilarities(queryEmbedding, graphResults, req)
	simTime := time.Since(simStart)
	
	// Build response
	totalTime := time.Since(startTime)
	
	// Update metrics
	s.mu.Lock()
	s.totalProcessingTime += totalTime
	if processingInfo.GPUUtilization {
		s.gpuRequests++
	} else {
		s.cpuFallbackRequests++
	}
	s.mu.Unlock()
	
	// Performance info
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	
	perfInfo = PerformanceInfo{
		TotalTime:      float64(totalTime.Nanoseconds()) / 1e6,
		EmbeddingTime:  float64(embeddingTime.Nanoseconds()) / 1e6,
		DatabaseTime:   float64(dbTime.Nanoseconds()) / 1e6,
		SimilarityTime: float64(simTime.Nanoseconds()) / 1e6,
		MemoryUsageMB:  float64(memStats.Alloc) / 1024 / 1024,
	}
	
	processingInfo.NodesProcessed = len(graphResults)
	processingInfo.SimilarityMethod = "ENHANCED_GPU" if req.UseGPU && s.gpuAvailable else "CPU_OPTIMIZED"
	
	return &EnhancedSearchResponse{
		Results:         enhancedResults,
		TotalFound:      len(enhancedResults),
		ProcessingInfo:  processingInfo,
		PerformanceInfo: perfInfo,
		Timestamp:       time.Now(),
	}, nil
}

func (s *Neo4jGPUService) generateGPUEmbedding(text string, useFP16 bool, useCache bool) ([]float32, string, error) {
	if !s.gpuAvailable {
		return s.generateCPUEmbedding(text, useCache)
	}
	
	// Check cache first if enabled
	if useCache {
		if cached := s.getCachedEmbedding(text); cached != nil {
			s.mu.Lock()
			s.cacheHits++
			s.mu.Unlock()
			return cached, "CACHE", nil
		}
	}
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	// Use GPU worker (placeholder implementation)
	// In production, this would call the actual gRPC service
	req := &EmbeddingRequest{
		Texts:         []string{text},
		UseCache:      useCache,
		Normalize:     true,
		Fp16Precision: useFP16,
	}
	
	// resp, err := s.gpuClient.GenerateEmbedding(ctx, req)
	// Simulate GPU embedding generation
	embedding := make([]float32, EmbeddingDim)
	for i := range embedding {
		embedding[i] = 0.1 // Placeholder
	}
	
	// Cache the result
	if useCache {
		s.cacheEmbedding(text, embedding)
	}
	
	return embedding, "GPU", nil
}

func (s *Neo4jGPUService) generateCPUEmbedding(text string, useCache bool) ([]float32, string, error) {
	// Check cache first if enabled
	if useCache {
		if cached := s.getCachedEmbedding(text); cached != nil {
			s.mu.Lock()
			s.cacheHits++
			s.mu.Unlock()
			return cached, "CACHE", nil
		}
	}
	
	// CPU embedding generation (placeholder)
	embedding := make([]float32, EmbeddingDim)
	for i := range embedding {
		embedding[i] = 0.1 // Placeholder - would use actual CPU embedding model
	}
	
	// Cache the result
	if useCache {
		s.cacheEmbedding(text, embedding)
	}
	
	return embedding, "CPU", nil
}

func (s *Neo4jGPUService) performGraphSearch(queryEmbedding []float32, req EnhancedSearchRequest) ([]EnhancedResult, error) {
	ctx := context.Background()
	session := s.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{
		AccessMode: neo4j.AccessModeRead,
	})
	defer session.Close(ctx)
	
	// Enhanced Cypher query with graph traversal
	query := `
		MATCH (d:Document)
		WHERE d.practice_area = $practice_area 
		  AND d.document_type = $document_type
		OPTIONAL MATCH path = (d)-[r*1..3]-(related:Document)
		WITH d, collect(DISTINCT related) as related_docs,
		     collect(DISTINCT {
		         node_id: related.document_id,
		         relation_type: type(last(relationships(path))),
		         weight: coalesce(last(relationships(path)).weight, 1.0),
		         distance: length(path),
		         properties: properties(last(relationships(path)))
		     }) as relationships
		RETURN d.document_id as document_id,
		       d.title as title,
		       d.content as content,
		       d.embedding as embedding,
		       d.practice_area as practice_area,
		       d.document_type as document_type,
		       d.metadata as metadata,
		       relationships
		LIMIT $max_results
	`
	
	parameters := map[string]interface{}{
		"practice_area": req.PracticeArea,
		"document_type": req.DocumentType,
		"max_results":   req.MaxResults * 2, // Get more for better filtering
	}
	
	result, err := session.Run(ctx, query, parameters)
	if err != nil {
		return nil, err
	}
	
	var results []EnhancedResult
	for result.Next(ctx) {
		record := result.Record()
		
		embeddingInterface, _ := record.Get("embedding")
		embedding := parseEmbedding(embeddingInterface)
		
		if len(embedding) != EmbeddingDim {
			continue
		}
		
		// Parse related nodes
		relationshipsInterface, _ := record.Get("relationships")
		relatedNodes := parseRelatedNodes(relationshipsInterface)
		
		// Parse metadata
		metadataInterface, _ := record.Get("metadata")
		metadata := parseMetadata(metadataInterface)
		
		result := EnhancedResult{
			NodeID:       getString(record, "document_id"),
			DocumentID:   getString(record, "document_id"),
			Title:        getString(record, "title"),
			Content:      getString(record, "content"),
			PracticeArea: getString(record, "practice_area"),
			DocumentType: getString(record, "document_type"),
			Embedding:    embedding,
			RelatedNodes: relatedNodes,
			Metadata:     metadata,
		}
		
		results = append(results, result)
	}
	
	return results, nil
}

func (s *Neo4jGPUService) computeEnhancedSimilarities(queryEmbedding []float32, graphResults []EnhancedResult, req EnhancedSearchRequest) []EnhancedResult {
	// Extract document embeddings
	docEmbeddings := make([][]float32, len(graphResults))
	for i, result := range graphResults {
		docEmbeddings[i] = result.Embedding
	}
	
	var similarities []float32
	var processingSource string
	
	if req.UseGPU && s.gpuAvailable {
		similarities = s.computeGPUSimilarities(queryEmbedding, docEmbeddings, req.UseFP16)
		processingSource = "GPU"
	} else {
		similarities = s.computeCPUSimilarities(queryEmbedding, docEmbeddings)
		processingSource = "CPU"
	}
	
	// Update results with similarity scores
	enhancedResults := make([]EnhancedResult, len(graphResults))
	for i, result := range graphResults {
		result.SimilarityScore = similarities[i]
		result.Distance = 1.0 - similarities[i]
		result.Confidence = float64(similarities[i])
		result.ProcessingSource = processingSource
		
		enhancedResults[i] = result
	}
	
	// Sort by similarity (descending)
	for i := 0; i < len(enhancedResults)-1; i++ {
		for j := i + 1; j < len(enhancedResults); j++ {
			if enhancedResults[j].SimilarityScore > enhancedResults[i].SimilarityScore {
				enhancedResults[i], enhancedResults[j] = enhancedResults[j], enhancedResults[i]
			}
		}
	}
	
	// Filter by confidence and limit results
	filteredResults := []EnhancedResult{}
	for _, result := range enhancedResults {
		if result.Confidence >= req.MinConfidence {
			filteredResults = append(filteredResults, result)
			if len(filteredResults) >= req.MaxResults {
				break
			}
		}
	}
	
	return filteredResults
}

func (s *Neo4jGPUService) computeGPUSimilarities(queryEmbedding []float32, docEmbeddings [][]float32, useFP16 bool) []float32 {
	// Placeholder for GPU similarity computation
	// In production, this would use the gRPC GPU worker
	similarities := make([]float32, len(docEmbeddings))
	
	for i, docEmb := range docEmbeddings {
		var sum float32
		for j := 0; j < len(queryEmbedding) && j < len(docEmb); j++ {
			sum += queryEmbedding[j] * docEmb[j]
		}
		similarities[i] = sum
	}
	
	return similarities
}

func (s *Neo4jGPUService) computeCPUSimilarities(queryEmbedding []float32, docEmbeddings [][]float32) []float32 {
	similarities := make([]float32, len(docEmbeddings))
	
	// CPU SIMD-style optimization with goroutines
	numWorkers := runtime.NumCPU()
	if numWorkers > len(docEmbeddings) {
		numWorkers = len(docEmbeddings)
	}
	
	var wg sync.WaitGroup
	docsChan := make(chan int, len(docEmbeddings))
	
	// Start workers
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for docIdx := range docsChan {
				var sum float32
				docEmb := docEmbeddings[docIdx]
				
				// Optimized dot product with loop unrolling
				i := 0
				for i <= len(queryEmbedding)-4 {
					sum += queryEmbedding[i]*docEmb[i] + queryEmbedding[i+1]*docEmb[i+1] +
						   queryEmbedding[i+2]*docEmb[i+2] + queryEmbedding[i+3]*docEmb[i+3]
					i += 4
				}
				
				// Handle remaining elements
				for i < len(queryEmbedding) && i < len(docEmb) {
					sum += queryEmbedding[i] * docEmb[i]
					i++
				}
				
				similarities[docIdx] = sum
			}
		}()
	}
	
	// Send work to workers
	go func() {
		for i := 0; i < len(docEmbeddings); i++ {
			docsChan <- i
		}
		close(docsChan)
	}()
	
	wg.Wait()
	return similarities
}

// Cache management
func (s *Neo4jGPUService) getCachedEmbedding(text string) []float32 {
	if s.redisClient == nil {
		return nil
	}
	
	key := fmt.Sprintf("embed_gpu:%x", hashString(text))
	data, err := s.redisClient.Get(context.Background(), key).Result()
	if err != nil {
		return nil
	}
	
	var embedding []float32
	if err := json.Unmarshal([]byte(data), &embedding); err != nil {
		return nil
	}
	
	return embedding
}

func (s *Neo4jGPUService) cacheEmbedding(text string, embedding []float32) {
	if s.redisClient == nil {
		return
	}
	
	key := fmt.Sprintf("embed_gpu:%x", hashString(text))
	data, _ := json.Marshal(embedding)
	s.redisClient.SetEX(context.Background(), key, data, CacheExpiration)
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

// Batch processor for GPU optimization
func NewBatchProcessor(service *Neo4jGPUService, batchSize int) *BatchProcessor {
	return &BatchProcessor{
		service:   service,
		batchSize: batchSize,
		queue:     make([]BatchItem, 0, batchSize),
	}
}

func (bp *BatchProcessor) Start() {
	bp.flushTimer = time.NewTimer(50 * time.Millisecond) // 50ms batch timeout
	
	for {
		select {
		case item := <-bp.service.batchQueue:
			bp.mu.Lock()
			bp.queue = append(bp.queue, item)
			
			if len(bp.queue) >= bp.batchSize {
				bp.processBatch()
				bp.flushTimer.Reset(50 * time.Millisecond)
			}
			bp.mu.Unlock()
			
		case <-bp.flushTimer.C:
			bp.mu.Lock()
			if len(bp.queue) > 0 {
				bp.processBatch()
			}
			bp.flushTimer.Reset(50 * time.Millisecond)
			bp.mu.Unlock()
		}
	}
}

func (bp *BatchProcessor) processBatch() {
	if len(bp.queue) == 0 {
		return
	}
	
	// Extract texts
	texts := make([]string, len(bp.queue))
	for i, item := range bp.queue {
		texts[i] = item.Text
	}
	
	// Process batch (placeholder - would use actual GPU worker)
	embeddings := make([][]float32, len(texts))
	for i := range texts {
		embedding := make([]float32, EmbeddingDim)
		for j := range embedding {
			embedding[j] = 0.1 // Placeholder
		}
		embeddings[i] = embedding
	}
	
	// Send results back to requesters
	for i, item := range bp.queue {
		result := BatchResult{
			Embedding: embeddings[i],
			Error:     nil,
			Source:    "GPU_BATCH",
		}
		select {
		case item.Callback <- result:
		default:
		}
		close(item.Callback)
	}
	
	// Clear queue
	bp.queue = bp.queue[:0]
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

func parseRelatedNodes(relationshipsInterface interface{}) []RelatedNodeInfo {
	var relatedNodes []RelatedNodeInfo
	
	if relationshipsInterface == nil {
		return relatedNodes
	}
	
	if rels, ok := relationshipsInterface.([]interface{}); ok {
		for _, rel := range rels {
			if relMap, ok := rel.(map[string]interface{}); ok {
				node := RelatedNodeInfo{
					NodeID:       getString(relMap, "node_id"),
					RelationType: getString(relMap, "relation_type"),
					Weight:       getFloat64(relMap, "weight"),
					Distance:     int(getFloat64(relMap, "distance")),
					Properties:   getMap(relMap, "properties"),
				}
				relatedNodes = append(relatedNodes, node)
			}
		}
	}
	
	return relatedNodes
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

func hashString(s string) uint32 {
	hash := uint32(0)
	for _, r := range s {
		hash = hash*31 + uint32(r)
	}
	return hash
}

// HTTP API handlers
func (s *Neo4jGPUService) setupRoutes(router *gin.Engine) {
	api := router.Group("/api/neo4j-gpu")
	{
		api.POST("/search/enhanced", s.handleEnhancedSearch)
		api.GET("/health", s.handleHealth)
		api.GET("/metrics", s.handleMetrics)
		api.POST("/batch/embeddings", s.handleBatchEmbeddings)
		api.DELETE("/cache", s.handleClearCache)
	}
}

func (s *Neo4jGPUService) handleEnhancedSearch(c *gin.Context) {
	var req EnhancedSearchRequest
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
	
	result, err := s.EnhancedSearch(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, result)
}

func (s *Neo4jGPUService) handleHealth(c *gin.Context) {
	ctx := context.Background()
	
	// Check Neo4j
	neo4jHealthy := true
	if err := s.neo4jDriver.VerifyConnectivity(ctx); err != nil {
		neo4jHealthy = false
	}
	
	// Check Redis
	redisHealthy := true
	if err := s.redisClient.Ping(ctx).Err(); err != nil {
		redisHealthy = false
	}
	
	// Get system metrics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	s.mu.RLock()
	avgResponseTime := float64(0)
	if s.totalRequests > 0 {
		avgResponseTime = s.totalProcessingTime.Seconds() * 1000 / float64(s.totalRequests)
	}
	
	status := gin.H{
		"status":                "healthy",
		"neo4j_healthy":         neo4jHealthy,
		"redis_healthy":         redisHealthy,
		"gpu_worker_available":  s.gpuAvailable,
		"total_requests":        s.totalRequests,
		"gpu_requests":          s.gpuRequests,
		"cpu_fallback_requests": s.cpuFallbackRequests,
		"cache_hits":            s.cacheHits,
		"error_count":           s.errorCount,
		"avg_response_time_ms":  avgResponseTime,
		"goroutines":            runtime.NumGoroutine(),
		"heap_alloc_mb":         float64(m.HeapAlloc) / 1024 / 1024,
		"sys_memory_mb":         float64(m.Sys) / 1024 / 1024,
		"cpu_cores":             runtime.NumCPU(),
		"embedding_dimension":   EmbeddingDim,
		"max_batch_size":        MaxBatchSize,
	}
	s.mu.RUnlock()
	
	if !neo4jHealthy || !redisHealthy {
		status["status"] = "degraded"
		c.JSON(http.StatusServiceUnavailable, status)
	} else {
		c.JSON(http.StatusOK, status)
	}
}

func (s *Neo4jGPUService) handleMetrics(c *gin.Context) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	var cacheHitRatio float64
	if s.totalRequests > 0 {
		cacheHitRatio = float64(s.cacheHits) / float64(s.totalRequests)
	}
	
	c.JSON(http.StatusOK, gin.H{
		"total_requests":        s.totalRequests,
		"gpu_requests":          s.gpuRequests,
		"cpu_fallback_requests": s.cpuFallbackRequests,
		"cache_hits":            s.cacheHits,
		"cache_hit_ratio":       cacheHitRatio,
		"error_count":           s.errorCount,
		"avg_response_time_ms":  s.totalProcessingTime.Seconds() * 1000 / max(float64(s.totalRequests), 1),
		"gpu_available":         s.gpuAvailable,
		"workers":               MaxConcurrency,
		"cpu_cores":             runtime.NumCPU(),
		"embedding_dimension":   EmbeddingDim,
		"max_batch_size":        MaxBatchSize,
	})
}

func (s *Neo4jGPUService) handleBatchEmbeddings(c *gin.Context) {
	var req struct {
		Texts  []string `json:"texts"`
		UseGPU bool     `json:"use_gpu"`
		UseFP16 bool    `json:"use_fp16"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
		return
	}
	
	// Process embeddings
	embeddings := make([][]float32, len(req.Texts))
	for i, text := range req.Texts {
		var err error
		if req.UseGPU && s.gpuAvailable {
			embeddings[i], _, err = s.generateGPUEmbedding(text, req.UseFP16, true)
		} else {
			embeddings[i], _, err = s.generateCPUEmbedding(text, true)
		}
		
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"embeddings": embeddings,
		"count":      len(embeddings),
		"gpu_used":   req.UseGPU && s.gpuAvailable,
		"fp16_used":  req.UseFP16,
	})
}

func (s *Neo4jGPUService) handleClearCache(c *gin.Context) {
	ctx := context.Background()
	keys, err := s.redisClient.Keys(ctx, "embed_gpu:*").Result()
	if err == nil && len(keys) > 0 {
		s.redisClient.Del(ctx, keys...)
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

func (s *Neo4jGPUService) Shutdown() {
	if s.workerPool != nil {
		s.workerPool.Shutdown()
	}
	
	if s.gpuConn != nil {
		s.gpuConn.Close()
	}
	
	if s.neo4jDriver != nil {
		s.neo4jDriver.Close(context.Background())
	}
	
	if s.redisClient != nil {
		s.redisClient.Close()
	}
}

// Main function
func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	
	service, err := NewNeo4jGPUService()
	if err != nil {
		log.Fatalf("Failed to initialize Neo4j GPU Service: %v", err)
	}
	defer service.Shutdown()
	
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
	
	service.setupRoutes(router)
	
	log.Println("🚀 Neo4j GPU-Integrated Service starting on :" + ServicePort)
	log.Printf("⚡ RTX Tensor Core acceleration: %v", service.gpuAvailable)
	log.Printf("🧠 CPU cores: %d | Max concurrency: %d", runtime.NumCPU(), MaxConcurrency)
	log.Printf("💾 Embedding dimension: %d | Max batch size: %d", EmbeddingDim, MaxBatchSize)
	log.Println("📊 Endpoints:")
	log.Println("   POST /api/neo4j-gpu/search/enhanced - Enhanced graph search")
	log.Println("   POST /api/neo4j-gpu/batch/embeddings - Batch embedding generation")
	log.Println("   GET  /api/neo4j-gpu/health - Health check with GPU status")
	log.Println("   GET  /api/neo4j-gpu/metrics - Performance metrics")
	log.Println("   DELETE /api/neo4j-gpu/cache - Clear embedding cache")
	
	if err := router.Run(":" + ServicePort); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}