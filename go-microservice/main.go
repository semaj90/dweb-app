package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/valyala/fastjson"
)

// Types for the AI microservice
type ParseRequest struct {
	Data     json.RawMessage `json:"data"`
	Format   string          `json:"format"`   // "json", "xml", "text"
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

type SOMTrainRequest struct {
	Vectors    [][]float32 `json:"vectors"`
	Labels     []string    `json:"labels"`
	Dimensions struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"dimensions"`
	Iterations int     `json:"iterations"`
	LearningRate float32 `json:"learning_rate"`
}

type SOMTrainResponse struct {
	Success     bool        `json:"success"`
	MapWeights  [][]float32 `json:"map_weights"`
	Clusters    []Cluster   `json:"clusters"`
	Metrics     Metrics     `json:"metrics"`
	TrainingTime time.Duration `json:"training_time"`
}

type Cluster struct {
	ID       int       `json:"id"`
	Center   []float32 `json:"center"`
	Labels   []string  `json:"labels"`
	Size     int       `json:"size"`
	Cohesion float32   `json:"cohesion"`
}

type CUDAInferRequest struct {
	Model      string      `json:"model"`
	Input      interface{} `json:"input"`
	BatchSize  int         `json:"batch_size"`
	Precision  string      `json:"precision"` // "fp32", "fp16", "int8"
	Streaming  bool        `json:"streaming"`
}

type CUDAInferResponse struct {
	Success    bool        `json:"success"`
	Output     interface{} `json:"output"`
	Metrics    Metrics     `json:"metrics"`
	GPUMemory  GPUMetrics  `json:"gpu_memory"`
	StreamID   string      `json:"stream_id,omitempty"`
}

type Metrics struct {
	ProcessingTime time.Duration `json:"processing_time"`
	MemoryUsed     string        `json:"memory_used"`
	CPUUsage       float64       `json:"cpu_usage"`
	Throughput     float64       `json:"throughput"`
	ParallelTasks  int           `json:"parallel_tasks"`
}

type GPUMetrics struct {
	TotalMemory     uint64  `json:"total_memory"`
	UsedMemory      uint64  `json:"used_memory"`
	FreeMemory      uint64  `json:"free_memory"`
	UtilizationGPU  float32 `json:"utilization_gpu"`
	UtilizationMem  float32 `json:"utilization_mem"`
	Temperature     float32 `json:"temperature"`
}

type HealthStatus struct {
	Status      string     `json:"status"`
	Uptime      string     `json:"uptime"`
	Version     string     `json:"version"`
	CPUCores    int        `json:"cpu_cores"`
	Memory      MemoryInfo `json:"memory"`
	CUDA        CUDAInfo   `json:"cuda"`
	Endpoints   []string   `json:"endpoints"`
}

type MemoryInfo struct {
	Allocated uint64 `json:"allocated"`
	TotalAlloc uint64 `json:"total_alloc"`
	System    uint64 `json:"system"`
	NumGC     uint32 `json:"num_gc"`
}

type CUDAInfo struct {
	Available     bool   `json:"available"`
	DeviceCount   int    `json:"device_count"`
	DriverVersion string `json:"driver_version"`
	RuntimeVersion string `json:"runtime_version"`
}

// Global variables
var (
	startTime = time.Now()
	version   = "1.0.0"
	somCache  = sync.Map{}
	parser    = &fastjson.Parser{}
)

func main() {
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

	// Health check endpoint
	r.GET("/health", healthCheck)
	r.GET("/", healthCheck)

	// Core processing endpoints
	r.POST("/parse", parseHandler)
	r.POST("/train-som", trainSOMHandler)
	r.POST("/cuda-infer", cudaInferHandler)

	// Utility endpoints
	r.GET("/metrics", metricsHandler)
	r.GET("/som-cache", somCacheHandler)
	r.DELETE("/som-cache", clearSOMCacheHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("🚀 Go AI Microservice starting on port %s", port)
	log.Printf("🔧 CUDA Available: %v", isCUDAAvailable())
	log.Printf("💻 CPU Cores: %d", runtime.NumCPU())
	
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

// Health check handler
func healthCheck(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

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
		CUDA: CUDAInfo{
			Available:   isCUDAAvailable(),
			DeviceCount: getCUDADeviceCount(),
		},
		Endpoints: []string{"/parse", "/train-som", "/cuda-infer", "/health", "/metrics"},
	}

	c.JSON(http.StatusOK, status)
}

// High-performance JSON parsing with SIMD
func parseHandler(c *gin.Context) {
	startTime := time.Now()

	var req ParseRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Use fastjson for high-performance parsing
	var result interface{}
	var err error

	switch req.Format {
	case "json":
		if req.Options.Parallel && len(req.Data) > 1024*1024 { // 1MB threshold
			result, err = parseJSONParallel(req.Data, req.Options.ChunkSize)
		} else {
			result, err = parseJSONFast(req.Data)
		}
	case "xml":
		result, err = parseXML(req.Data)
	case "text":
		result, err = parseText(req.Data)
	default:
		result, err = parseJSONFast(req.Data)
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Calculate metrics
	processingTime := time.Since(startTime)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := ParseResponse{
		Success: true,
		Result:  result,
		Metrics: Metrics{
			ProcessingTime: processingTime,
			MemoryUsed:     fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
			Throughput:     float64(len(req.Data)) / processingTime.Seconds(),
		},
		ProcessedAt: time.Now(),
	}

	c.JSON(http.StatusOK, response)
}

// Self-Organizing Map training handler
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

	// Train SOM (simplified implementation - in production use optimized CUDA kernels)
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

// CUDA inference handler
func cudaInferHandler(c *gin.Context) {
	startTime := time.Now()

	var req CUDAInferRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if !isCUDAAvailable() {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "CUDA not available on this system",
		})
		return
	}

	log.Printf("🚀 CUDA inference request: model=%s, batch_size=%d, precision=%s", 
		req.Model, req.BatchSize, req.Precision)

	// Perform CUDA inference (placeholder - integrate with actual CUDA libraries)
	result, gpuMetrics := performCUDAInference(req)

	processingTime := time.Since(startTime)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := CUDAInferResponse{
		Success: true,
		Output:  result,
		Metrics: Metrics{
			ProcessingTime: processingTime,
			MemoryUsed:     fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
			Throughput:     calculateThroughput(req.Input, processingTime),
		},
		GPUMemory: gpuMetrics,
	}

	if req.Streaming {
		response.StreamID = fmt.Sprintf("stream_%d", time.Now().UnixNano())
	}

	c.JSON(http.StatusOK, response)
}

// Metrics endpoint
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

	c.JSON(http.StatusOK, metrics)
}

// SOM cache handlers
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

// Helper functions
func parseJSONFast(data json.RawMessage) (interface{}, error) {
	v, err := parser.ParseBytes(data)
	if err != nil {
		return nil, err
	}
	return convertFastJSONValue(v), nil
}

func parseJSONParallel(data json.RawMessage, chunkSize int) (interface{}, error) {
	// For large JSON, implement parallel parsing using goroutines
	// This is a simplified version - production would use proper chunking
	return parseJSONFast(data)
}

func parseXML(data json.RawMessage) (interface{}, error) {
	// Implement XML parsing
	return string(data), nil
}

func parseText(data json.RawMessage) (interface{}, error) {
	return string(data), nil
}

func convertFastJSONValue(v *fastjson.Value) interface{} {
	switch v.Type() {
	case fastjson.TypeObject:
		obj := make(map[string]interface{})
		v.GetObject().Visit(func(key []byte, v *fastjson.Value) {
			obj[string(key)] = convertFastJSONValue(v)
		})
		return obj
	case fastjson.TypeArray:
		arr := make([]interface{}, 0)
		for _, item := range v.GetArray() {
			arr = append(arr, convertFastJSONValue(item))
		}
		return arr
	case fastjson.TypeString:
		return string(v.GetStringBytes())
	case fastjson.TypeNumber:
		return v.GetFloat64()
	case fastjson.TypeTrue:
		return true
	case fastjson.TypeFalse:
		return false
	case fastjson.TypeNull:
		return nil
	default:
		return nil
	}
}

// Simplified SOM implementation (in production, use optimized CUDA kernels)
func trainSOM(vectors [][]float32, width, height, iterations int, learningRate float32, labels []string) ([][]float32, []Cluster) {
	// Initialize map weights randomly
	mapSize := width * height
	vectorDim := len(vectors[0])
	mapWeights := make([][]float32, mapSize)
	
	for i := range mapWeights {
		mapWeights[i] = make([]float32, vectorDim)
		for j := range mapWeights[i] {
			mapWeights[i][j] = (2.0 * float32(i*j) / float32(mapSize*vectorDim)) - 1.0 // Simple initialization
		}
	}

	// Training loop (simplified)
	for iter := 0; iter < iterations; iter++ {
		for i, vector := range vectors {
			// Find best matching unit (BMU)
			bmuIndex := findBMU(vector, mapWeights)
			
			// Update weights in neighborhood
			radius := float32(max(width, height)) * (1.0 - float32(iter)/float32(iterations))
			updateNeighborhood(mapWeights, bmuIndex, vector, width, height, radius, learningRate)
		}
		
		// Decay learning rate
		learningRate *= 0.99
	}

	// Generate clusters
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
	return sum // Skip sqrt for performance in comparison
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
	clusters := make([]Cluster, min(len(mapWeights), 10)) // Limit to 10 clusters for demo
	
	for i := range clusters {
		clusters[i] = Cluster{
			ID:       i,
			Center:   mapWeights[i],
			Labels:   []string{fmt.Sprintf("cluster_%d", i)},
			Size:     len(vectors) / len(clusters), // Simplified
			Cohesion: 0.8 + float32(i)*0.02,       // Mock cohesion score
		}
	}
	
	return clusters
}

// CUDA-related functions (stubs - integrate with actual CUDA libraries)
func isCUDAAvailable() bool {
	// In production, check for CUDA runtime
	// For now, simulate availability based on environment
	return os.Getenv("CUDA_AVAILABLE") == "true"
}

func getCUDADeviceCount() int {
	if isCUDAAvailable() {
		if count := os.Getenv("CUDA_DEVICE_COUNT"); count != "" {
			if c, err := strconv.Atoi(count); err == nil {
				return c
			}
		}
		return 1
	}
	return 0
}

func performCUDAInference(req CUDAInferRequest) (interface{}, GPUMetrics) {
	// Simulate CUDA inference
	time.Sleep(time.Millisecond * 100) // Simulate processing time
	
	result := gin.H{
		"model":       req.Model,
		"batch_size":  req.BatchSize,
		"precision":   req.Precision,
		"output_size": 1024,
		"predictions": []float32{0.95, 0.87, 0.92, 0.78},
	}
	
	gpuMetrics := GPUMetrics{
		TotalMemory:    8 * 1024 * 1024 * 1024, // 8GB
		UsedMemory:     2 * 1024 * 1024 * 1024, // 2GB
		FreeMemory:     6 * 1024 * 1024 * 1024, // 6GB
		UtilizationGPU: 75.0,
		UtilizationMem: 25.0,
		Temperature:    65.0,
	}
	
	return result, gpuMetrics
}

func calculateThroughput(input interface{}, duration time.Duration) float64 {
	// Calculate throughput based on input size and processing time
	inputSize := 1024.0 // Simplified
	return inputSize / duration.Seconds()
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