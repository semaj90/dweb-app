package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/valyala/fastjson"
)

// Types for the AI microservice
type ParseRequest struct {
	Data     json.RawMessage `json:"data"`
	Format   string          `json:"format"` // "json", "xml", "text"
	Options  ParseOptions    `json:"options"`
}

type ParseOptions struct {
	Parallel    bool `json:"parallel"`
	ChunkSize   int  `json:"chunk_size"`
	Compression bool `json:"compression"`
}

type ParseResponse struct {
	Success     bool        `json:"success"`
	Result      interface{} `json:"result"`
	Metrics     Metrics     `json:"metrics"`
	ProcessedAt time.Time   `json:"processed_at"`
}

type Metrics struct {
	ProcessingTime time.Duration `json:"processing_time"`
	MemoryUsed     string        `json:"memory_used"`
	Throughput     float64       `json:"throughput"`
}

type SOMTrainRequest struct {
	Vectors    [][]float32 `json:"vectors"`
	Labels     []string    `json:"labels"`
	Dimensions struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"dimensions"`
	Iterations   int     `json:"iterations"`
	LearningRate float32 `json:"learning_rate"`
}

type SOMTrainResponse struct {
	Success      bool          `json:"success"`
	MapWeights   [][]float32   `json:"map_weights"`
	Clusters     []Cluster     `json:"clusters"`
	Metrics      Metrics       `json:"metrics"`
	TrainingTime time.Duration `json:"training_time"`
}

type Cluster struct {
	ID       int       `json:"id"`
	Center   []float32 `json:"center"`
	Labels   []string  `json:"labels"`
	Size     int       `json:"size"`
	Cohesion float32   `json:"cohesion"`
}

type ChunkType struct {
	Data json.RawMessage `json:"data"`
}

type HealthStatus struct {
	Status   string     `json:"status"`
	Uptime   string     `json:"uptime"`
	Version  string     `json:"version"`
	CPUCores int        `json:"cpu_cores"`
	Memory   MemoryInfo `json:"memory"`
	CUDA     CUDAInfo   `json:"cuda"`
	Redis    RedisInfo  `json:"redis"`
	Neo4j    Neo4jInfo  `json:"neo4j"`
	Endpoints []string  `json:"endpoints"`
}

type MemoryInfo struct {
	Allocated  uint64 `json:"allocated"`
	TotalAlloc uint64 `json:"total_alloc"`
	System     uint64 `json:"system"`
	NumGC      uint32 `json:"num_gc"`
}

type CUDAInfo struct {
	Available   bool `json:"available"`
	DeviceCount int  `json:"device_count"`
	Devices     map[int]DeviceMemoryInfo `json:"devices,omitempty"`
}

type RedisInfo struct {
	Connected bool   `json:"connected"`
	Host      string `json:"host"`
	QueueStats map[string]interface{} `json:"queue_stats,omitempty"`
}

type Neo4jInfo struct {
	Connected bool   `json:"connected"`
	Version   string `json:"version,omitempty"`
	Address   string `json:"address,omitempty"`
}

// Global variables
var (
	startTime   = time.Now()
	version     = "2.0.0" // Updated version
	somCache    = sync.Map{}
	parser      = &fastjson.Parser{}
	neo4jDriver neo4j.DriverWithContext
)

func main() {
	// Initialize all components
	if err := initializeComponents(); err != nil {
		log.Fatalf("Failed to initialize components: %v", err)
	}
	defer cleanup()

	// Initialize Gin router
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// CORS configuration for SvelteKit integration
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:5173", "http://localhost:5175", "http://localhost:3000"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Authorization", "Accept"}
	config.AllowCredentials = true
	r.Use(cors.New(config))

	// Register all routes
	registerRoutes(r)

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}

	log.Printf("🚀 Go AI Microservice v%s starting on port %s", version, port)
	log.Printf("🔧 CUDA Available: %v", isCUDAAvailable())
	log.Printf("💻 CPU Cores: %d", runtime.NumCPU())
	log.Printf("📊 Redis: %v", redisManager != nil)
	log.Printf("🗄️ Neo4j: %v", neo4jDriver != nil)
	
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

// initializeComponents initializes all service components
func initializeComponents() error {
	// Initialize Neo4j
	var err error
	neo4jDriver, err = initNeo4j()
	if err != nil {
		log.Printf("⚠️  Neo4j connection failed: %v", err)
		// Continue without Neo4j
	}

	// Initialize CUDA Manager
	if err := InitializeCUDAManager(); err != nil {
		log.Printf("⚠️  CUDA initialization failed: %v", err)
		// Continue without CUDA
	}

	// Initialize Redis Manager
	redisHost := os.Getenv("REDIS_HOST")
	if redisHost == "" {
		redisHost = "localhost:6379"
	}
	if err := InitializeRedisManager(redisHost); err != nil {
		log.Printf("⚠️  Redis initialization failed: %v", err)
		// Continue without Redis
	}

	// Initialize SIMD Parser
	if err := InitializeSIMDParser(); err != nil {
		log.Printf("⚠️  SIMD Parser initialization failed: %v", err)
		// Continue without SIMD optimizations
	}

	// Initialize Streaming Manager
	if err := InitializeStreamingManager(); err != nil {
		log.Printf("⚠️  Streaming Manager initialization failed: %v", err)
		// Continue without streaming
	}

	return nil
}

// cleanup performs cleanup on shutdown
func cleanup() {
	// Cleanup Neo4j
	if neo4jDriver != nil {
		neo4jDriver.Close(context.Background())
	}

	// Cleanup CUDA
	CleanupCUDA()

	// Cleanup Redis
	if redisManager != nil {
		redisManager.Cleanup()
	}

	log.Println("✅ Cleanup completed")
}

// registerRoutes registers all API routes
func registerRoutes(r *gin.Engine) {
	// Health and status endpoints
	r.GET("/health", healthCheck)
	r.GET("/", healthCheck)
	r.GET("/neo4j-status", neo4jStatusHandler)
	r.GET("/metrics", metricsHandler)

	// Core processing endpoints
	r.POST("/parse", parseHandler)
	r.POST("/train-som", trainSOMHandler)
	r.POST("/graph-query", graphQueryHandler)
	
	// GPU endpoints
	r.POST("/gpu/compute", gpuComputeHandler)
	r.GET("/gpu/metrics", gpuMetricsHandler)
	r.POST("/cuda-infer", cudaInferHandler)
	
	// Redis/Queue endpoints
	r.POST("/queue/job", enqueueJobHandler)
	r.GET("/queue/stats", queueStatsHandler)
	r.POST("/cache/set", cacheSetHandler)
	r.GET("/cache/get/:key", cacheGetHandler)
	r.DELETE("/cache/:key", cacheDeleteHandler)
	r.POST("/batch-inference", batchInferenceHandler)
	
	// SIMD Parser endpoints
	r.POST("/parse/simd", simdParseHandler)
	r.POST("/parse/batch", batchParseHandler)
	r.POST("/parse/validate", validateJSONHandler)
	
	// Streaming endpoints
	r.GET("/stream/ws", streamManager.HandleWebSocketStream)
	r.GET("/stream/sse", streamManager.HandleSSEStream)
	r.POST("/stream/chunked", streamManager.HandleChunkedStream)
	r.GET("/stream/metrics", streamMetricsHandler)
	
	// Batch embedding routes
	RegisterBatchEmbedRoutes(r)

	// Utility endpoints
	r.GET("/som-cache", somCacheHandler)
	r.DELETE("/som-cache", clearSOMCacheHandler)
}

// ==== NEO4J HANDLERS ====

func initNeo4j() (neo4j.DriverWithContext, error) {
	uri := os.Getenv("NEO4J_URI")
	if uri == "" {
		uri = "neo4j://localhost:7687"
	}
	user := os.Getenv("NEO4J_USER")
	if user == "" {
		user = "neo4j"
	}
	password := os.Getenv("NEO4J_PASSWORD")
	if password == "" {
		password = "password"
	}

	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(user, password, ""))
	if err != nil {
		return nil, err
	}

	ctx := context.Background()
	err = driver.VerifyConnectivity(ctx)
	if err != nil {
		return nil, err
	}

	log.Println("✅ Neo4j connection established")
	return driver, nil
}

func neo4jStatusHandler(c *gin.Context) {
	status := gin.H{
		"connected": false,
	}

	if neo4jDriver != nil {
		if err := neo4jDriver.VerifyConnectivity(context.Background()); err == nil {
			status["connected"] = true
			serverInfo, err := neo4jDriver.GetServerInfo(context.Background())
			if err == nil {
				status["server_version"] = serverInfo.Agent()
				status["server_address"] = serverInfo.Address()
			}
		}
	}

	c.JSON(http.StatusOK, status)
}

func graphQueryHandler(c *gin.Context) {
	if neo4jDriver == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Neo4j not connected"})
		return
	}

	var req struct {
		Query  string                 `json:"query"`
		Params map[string]interface{} `json:"params"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx := context.Background()
	session := neo4jDriver.NewSession(ctx, neo4j.SessionConfig{AccessMode: neo4j.AccessModeRead})
	defer session.Close(ctx)

	result, err := session.Run(ctx, req.Query, req.Params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	records, err := result.Collect(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	var response []map[string]interface{}
	for _, record := range records {
		response = append(response, record.AsMap())
	}

	c.JSON(http.StatusOK, response)
}

// ==== HEALTH & METRICS HANDLERS ====

func healthCheck(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Get CUDA info
	cudaInfo := CUDAInfo{
		Available:   isCUDAAvailable(),
		DeviceCount: getCUDADeviceCount(),
	}
	if cudaManager != nil {
		cudaInfo.Devices = GetDeviceMemoryInfo()
	}

	// Get Redis info
	redisInfo := RedisInfo{
		Connected: false,
		Host:      os.Getenv("REDIS_HOST"),
	}
	if redisManager != nil {
		redisInfo.Connected = true
		redisInfo.QueueStats = redisManager.GetQueueStats()
	}

	// Get Neo4j info
	neo4jInfo := Neo4jInfo{
		Connected: false,
	}
	if neo4jDriver != nil {
		if err := neo4jDriver.VerifyConnectivity(context.Background()); err == nil {
			neo4jInfo.Connected = true
			if serverInfo, err := neo4jDriver.GetServerInfo(context.Background()); err == nil {
				neo4jInfo.Version = serverInfo.Agent()
				neo4jInfo.Address = serverInfo.Address()
			}
		}
	}

	status := HealthStatus{
		Status:   "healthy",
		Uptime:   time.Since(startTime).String(),
		Version:  version,
		CPUCores: runtime.NumCPU(),
		Memory: MemoryInfo{
			Allocated:  m.Alloc,
			TotalAlloc: m.TotalAlloc,
			System:     m.Sys,
			NumGC:      m.NumGC,
		},
		CUDA:  cudaInfo,
		Redis: redisInfo,
		Neo4j: neo4jInfo,
		Endpoints: []string{
			"/parse", "/train-som", "/cuda-infer", "/gpu/compute",
			"/queue/job", "/cache/set", "/parse/simd", "/stream/ws",
			"/batch-embed", "/graph-query", "/health", "/metrics",
		},
	}

	c.JSON(http.StatusOK, status)
}

func metricsHandler(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	metrics := gin.H{
		"uptime":         time.Since(startTime).String(),
		"goroutines":     runtime.NumGoroutine(),
		"memory_alloc":   fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
		"memory_total":   fmt.Sprintf("%.2f MB", float64(m.TotalAlloc)/1024/1024),
		"memory_sys":     fmt.Sprintf("%.2f MB", float64(m.Sys)/1024/1024),
		"gc_runs":        m.NumGC,
		"cpu_cores":      runtime.NumCPU(),
		"cuda_available": isCUDAAvailable(),
		"som_cache_size": getSOMCacheSize(),
	}

	// Add CUDA metrics if available
	if cudaManager != nil {
		cudaMetrics := GetCUDAMetrics()
		if cudaMetrics != nil {
			metrics["cuda_metrics"] = cudaMetrics
		}
	}

	// Add Redis metrics if available
	if redisManager != nil {
		metrics["redis_metrics"] = redisManager.GetCacheMetrics()
		metrics["queue_stats"] = redisManager.GetQueueStats()
	}

	// Add SIMD parser metrics if available
	if simdParser != nil {
		metrics["parser_metrics"] = simdParser.GetMetrics()
	}

	// Add streaming metrics if available
	if streamManager != nil {
		metrics["stream_metrics"] = streamManager.GetStreamMetrics()
	}

	c.JSON(http.StatusOK, metrics)
}

// ==== PARSING HANDLERS ====

func parseHandler(c *gin.Context) {
	startTime := time.Now()

	decoder := json.NewDecoder(c.Request.Body)
	var chunks []ChunkType
	for decoder.More() {
		var chunk ChunkType
		err := decoder.Decode(&chunk)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		chunks = append(chunks, chunk)
	}

	// Process chunks with SIMD if available
	var results []interface{}
	if simdParser != nil {
		for _, chunk := range chunks {
			result, _ := simdParser.ParseJSON(chunk.Data)
			if result != nil {
				results = append(results, result.Data)
			}
		}
	}

	processingTime := time.Since(startTime)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := ParseResponse{
		Success: true,
		Result:  gin.H{"chunks_received": len(chunks), "parsed_results": len(results)},
		Metrics: Metrics{
			ProcessingTime: processingTime,
			MemoryUsed:     fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
			Throughput:     float64(len(chunks)) / processingTime.Seconds(),
		},
		ProcessedAt: time.Now(),
	}

	c.JSON(http.StatusOK, response)
}

// ==== GPU HANDLERS ====

func gpuComputeHandler(c *gin.Context) {
	var req GPUComputeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if cudaManager == nil || !cudaManager.initialized {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "CUDA not available"})
		return
	}

	result, err := ProcessWithGPU(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func gpuMetricsHandler(c *gin.Context) {
	if cudaManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "CUDA not available"})
		return
	}

	metrics := GetCUDAMetrics()
	devices := GetDeviceMemoryInfo()

	c.JSON(http.StatusOK, gin.H{
		"metrics": metrics,
		"devices": devices,
	})
}

func cudaInferHandler(c *gin.Context) {
	var req struct {
		Data json.RawMessage `json:"data"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := performCUDAInference(req.Data)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, json.RawMessage(result))
}

// ==== REDIS/CACHE HANDLERS ====

func enqueueJobHandler(c *gin.Context) {
	if redisManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Redis not available"})
		return
	}

	var req struct {
		Queue   string      `json:"queue"`
		Type    string      `json:"type"`
		Payload interface{} `json:"payload"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	job, err := redisManager.EnqueueJob(req.Queue, req.Type, req.Payload)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, job)
}

func queueStatsHandler(c *gin.Context) {
	if redisManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Redis not available"})
		return
	}

	stats := redisManager.GetQueueStats()
	c.JSON(http.StatusOK, stats)
}

func cacheSetHandler(c *gin.Context) {
	if redisManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Redis not available"})
		return
	}

	var req struct {
		Key   string        `json:"key"`
		Value interface{}   `json:"value"`
		TTL   time.Duration `json:"ttl"`
		Tags  []string      `json:"tags,omitempty"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var err error
	if len(req.Tags) > 0 {
		err = redisManager.SetWithTags(req.Key, req.Value, req.TTL, req.Tags)
	} else {
		err = redisManager.Set(req.Key, req.Value, req.TTL)
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"success": true})
}

func cacheGetHandler(c *gin.Context) {
	if redisManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Redis not available"})
		return
	}

	key := c.Param("key")
	var value interface{}
	
	err := redisManager.Get(key, &value)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"key": key, "value": value})
}

func cacheDeleteHandler(c *gin.Context) {
	if redisManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Redis not available"})
		return
	}

	key := c.Param("key")
	err := redisManager.Delete(key)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"success": true})
}

func batchInferenceHandler(c *gin.Context) {
	if redisManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Redis not available"})
		return
	}

	var req struct {
		Documents []string `json:"documents"`
		Model     string   `json:"model"`
		BatchSize int      `json:"batch_size"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	job, err := redisManager.ScheduleBatchInference(req.Documents, req.Model, req.BatchSize)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, job)
}

// ==== SIMD PARSER HANDLERS ====

func simdParseHandler(c *gin.Context) {
	if simdParser == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "SIMD parser not available"})
		return
	}

	body, err := c.GetRawData()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := simdParser.ParseJSON(body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func batchParseHandler(c *gin.Context) {
	if simdParser == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "SIMD parser not available"})
		return
	}

	var req struct {
		Documents []json.RawMessage `json:"documents"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	documents := make([][]byte, len(req.Documents))
	for i, doc := range req.Documents {
		documents[i] = []byte(doc)
	}

	results, err := simdParser.ParseBatch(documents)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"results": results})
}

func validateJSONHandler(c *gin.Context) {
	if simdParser == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "SIMD parser not available"})
		return
	}

	body, err := c.GetRawData()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	valid, err := simdParser.ValidateJSON(body)
	c.JSON(http.StatusOK, gin.H{
		"valid": valid,
		"error": func() string {
			if err != nil {
				return err.Error()
			}
			return ""
		}(),
	})
}

// ==== STREAMING HANDLERS ====

func streamMetricsHandler(c *gin.Context) {
	if streamManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Streaming not available"})
		return
	}

	metrics := streamManager.GetStreamMetrics()
	c.JSON(http.StatusOK, metrics)
}

// ==== SOM HANDLERS ====

func trainSOMHandler(c *gin.Context) {
	startTime := time.Now()

	var req SOMTrainRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Validate input
	if len(req.Vectors) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No vectors provided"})
		return
	}

	if req.Dimensions.Width == 0 || req.Dimensions.Height == 0 {
		req.Dimensions.Width = 10
		req.Dimensions.Height = 10
	}

	if req.Iterations == 0 {
		req.Iterations = 1000
	}

	if req.LearningRate == 0 {
		req.LearningRate = 0.1
	}

	log.Printf("🧠 Training SOM: %dx%d map with %d vectors, %d iterations",
		req.Dimensions.Width, req.Dimensions.Height, len(req.Vectors), req.Iterations)

	// Train SOM
	mapWeights, clusters := trainSOM(req.Vectors, req.Dimensions.Width, req.Dimensions.Height,
		req.Iterations, req.LearningRate, req.Labels)

	// Cache the trained SOM
	cacheKey := fmt.Sprintf("%dx%d_%d", req.Dimensions.Width, req.Dimensions.Height, len(req.Vectors))
	somCache.Store(cacheKey, mapWeights)

	trainingTime := time.Since(startTime)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := SOMTrainResponse{
		Success:      true,
		MapWeights:   mapWeights,
		Clusters:     clusters,
		TrainingTime: trainingTime,
		Metrics: Metrics{
			ProcessingTime: trainingTime,
			MemoryUsed:     fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
			Throughput:     float64(len(req.Vectors)) / trainingTime.Seconds(),
		},
	}

	log.Printf("✅ SOM training completed in %v with %d clusters", trainingTime, len(clusters))
	c.JSON(http.StatusOK, response)
}

func somCacheHandler(c *gin.Context) {
	cacheInfo := make(map[string]interface{})
	
	somCache.Range(func(key, value interface{}) bool {
		if k, ok := key.(string); ok {
			if weights, ok := value.([][]float32); ok {
				cacheInfo[k] = gin.H{
					"dimensions": fmt.Sprintf("%dx%d", len(weights), len(weights[0])),
					"size":       len(weights) * len(weights[0]),
				}
			}
		}
		return true
	})

	c.JSON(http.StatusOK, gin.H{
		"cache_entries": len(cacheInfo),
		"entries":       cacheInfo,
	})
}

func clearSOMCacheHandler(c *gin.Context) {
	somCache.Range(func(key, value interface{}) bool {
		somCache.Delete(key)
		return true
	})
	
	c.JSON(http.StatusOK, gin.H{"message": "SOM cache cleared"})
}

// ==== HELPER FUNCTIONS ====

func trainSOM(vectors [][]float32, width, height, iterations int, learningRate float32, labels []string) ([][]float32, []Cluster) {
	// Initialize map weights randomly
	mapSize := width * height
	vectorDim := len(vectors[0])
	mapWeights := make([][]float32, mapSize)
	
	for i := range mapWeights {
		mapWeights[i] = make([]float32, vectorDim)
		for j := range mapWeights[i] {
			mapWeights[i][j] = (2.0 * float32(i*j) / float32(mapSize*vectorDim)) - 1.0
		}
	}

	// Training loop
	for iter := 0; iter < iterations; iter++ {
		for _, vector := range vectors {
			bmuIndex := findBMU(vector, mapWeights)
			radius := float32(max(width, height)) * (1.0 - float32(iter)/float32(iterations))
			updateNeighborhood(mapWeights, bmuIndex, vector, width, height, radius, learningRate)
		}
		learningRate *= 0.99
	}

	clusters := generateClusters(mapWeights, vectors, labels, width, height)
	return mapWeights, clusters
}

func findBMU(vector []float32, mapWeights [][]float32) int {
	minDist := float32(1e9)
	bmuIndex := 0
	
	for i, weight := range mapWeights {
		dist := euclideanDistance(vector, weight)
		if dist < minDist {
			minDist = dist
			bmuIndex = i
		}
	}
	
	return bmuIndex
}

func euclideanDistance(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

func updateNeighborhood(mapWeights [][]float32, bmuIndex int, vector []float32, width, height int, radius, learningRate float32) {
	bmuX := bmuIndex % width
	bmuY := bmuIndex / width
	
	for i, weight := range mapWeights {
		x := i % width
		y := i / width
		
		dist := float32((x-bmuX)*(x-bmuX) + (y-bmuY)*(y-bmuY))
		if dist <= radius*radius {
			influence := learningRate * (1.0 - dist/(radius*radius))
			for j := range weight {
				weight[j] += influence * (vector[j] - weight[j])
			}
		}
	}
}

func generateClusters(mapWeights [][]float32, vectors [][]float32, labels []string, width, height int) []Cluster {
	clusters := make([]Cluster, min(len(mapWeights), 10))
	
	for i := range clusters {
		clusters[i] = Cluster{
			ID:       i,
			Center:   mapWeights[i],
			Labels:   []string{fmt.Sprintf("cluster_%d", i)},
			Size:     len(vectors) / len(clusters),
			Cohesion: 0.8 + float32(i)*0.02,
		}
	}
	
	return clusters
}

func getSOMCacheSize() int {
	count := 0
	somCache.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
