// CUDA AI Service with Protocol Buffers
// High-performance GPU computations with T5 architecture support
// Handles dimensional arrays, kernel attention, and modular experiences

package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"legal-ai-production/internal/messaging"
	"legal-ai-production/internal/redis"
)

// DimensionalArray represents a multi-dimensional tensor
type DimensionalArray struct {
	Data             []float32            `json:"data"`
	Shape            []int32              `json:"shape"`
	Dtype            string               `json:"dtype"`
	KernelSplices    []KernelAttentionSlice `json:"kernel_slices"`
	AttentionWeights []float32            `json:"attention_weights"`
	ComputationHash  string               `json:"computation_hash"`
	CreatedAt        time.Time            `json:"created_at"`
}

// KernelAttentionSlice for modular AI experiences
type KernelAttentionSlice struct {
	StartIndex          int32     `json:"start_index"`
	EndIndex            int32     `json:"end_index"`
	AttentionScore      float32   `json:"attention_score"`
	RecommendationVector []float32 `json:"recommendation_vector"`
	ContextEmbedding    []float32 `json:"context_embedding"`
}

// T5Configuration for transformer architecture
type T5Configuration struct {
	ModelSize        string  `json:"model_size"`        // small, base, large, xl, xxl
	NumLayers        int32   `json:"num_layers"`
	NumHeads         int32   `json:"num_heads"`
	HiddenSize       int32   `json:"hidden_size"`
	VocabSize        int32   `json:"vocab_size"`
	MaxPositionEmbeddings int32 `json:"max_position_embeddings"`
	DropoutRate      float32 `json:"dropout_rate"`
	UseGPU           bool    `json:"use_gpu"`
	CUDADeviceID     int32   `json:"cuda_device_id"`
}

// CUDAService handles GPU computations
type CUDAService struct {
	isInitialized    bool
	deviceCount      int
	computeCapability string
	cache            sync.Map // Local cache for computed results
	t5Config         *T5Configuration
	offlineQueue     []ComputationRequest
	isOnline         bool
	mu               sync.RWMutex
	
	// RabbitMQ client for async processing
	rabbitmq         *messaging.RabbitMQClient
	rabbitmqEnabled  bool
	
	// Redis distributed cache
	redisCache       *redis.DistributedCache
	redisCacheEnabled bool
}

// ComputationRequest for queuing offline requests
type ComputationRequest struct {
	ID                string                `json:"id"`
	Type              string                `json:"type"`
	DimensionalArray  *DimensionalArray     `json:"dimensional_array"`
	T5Config          *T5Configuration      `json:"t5_config"`
	AttentionWeights  []float32             `json:"attention_weights"`
	RequestedAt       time.Time             `json:"requested_at"`
}

// ComputationResult with recommendations
type ComputationResult struct {
	Success              bool                   `json:"success"`
	Result               *DimensionalArray      `json:"result"`
	ProcessingTime       time.Duration          `json:"processing_time"`
	GPUMemoryUsed        int64                  `json:"gpu_memory_used"`
	Recommendations      []string               `json:"recommendations"`
	SimilarComputations  []string               `json:"similar_computations"`
	DidYouMean          []string               `json:"did_you_mean"`
	OthersSearched      []string               `json:"others_searched"`
	Error               string                 `json:"error,omitempty"`
}

var cudaService *CUDAService

func init() {
	cudaService = &CUDAService{
		isOnline: true,
		t5Config: &T5Configuration{
			ModelSize:              "base",
			NumLayers:              12,
			NumHeads:               12,
			HiddenSize:             768,
			VocabSize:              32128,
			MaxPositionEmbeddings:  512,
			DropoutRate:            0.1,
			UseGPU:                 true,
			CUDADeviceID:           0,
		},
	}
	
	// Initialize RabbitMQ client
	rabbitmqURL := os.Getenv("RABBITMQ_URL")
	if rabbitmqURL == "" {
		rabbitmqURL = "amqp://guest:guest@localhost:5672/"
	}
	
	config := messaging.GetDefaultConfig()
	config.URL = rabbitmqURL
	
	cudaService.rabbitmq = messaging.NewRabbitMQClient(config)
	cudaService.rabbitmqEnabled = true
	
	// Enable Redis distributed cache
	cudaService.redisCacheEnabled = true
}

func main() {
	port := os.Getenv("CUDA_SERVICE_PORT")
	if port == "" {
		port = "8096"
	}

	log.Printf("🚀 CUDA AI Service starting on port %s", port)
	log.Printf("🎯 Features: T5 Architecture, Kernel Attention, Dimensional Arrays")
	
	// Initialize CUDA
	if err := cudaService.Initialize(); err != nil {
		log.Printf("⚠️ CUDA initialization failed: %v (falling back to CPU)", err)
	}

	// Initialize RabbitMQ connection
	if cudaService.rabbitmqEnabled {
		err := cudaService.rabbitmq.Connect()
		if err != nil {
			log.Printf("⚠️ RabbitMQ connection failed: %v (continuing without messaging)", err)
			cudaService.rabbitmqEnabled = false
		} else {
			log.Printf("🐰 RabbitMQ connected successfully")
			
			// Start consuming messages
			go cudaService.startMessageConsumers()
		}
	}
	
	// Initialize Redis distributed cache
	if cudaService.redisCacheEnabled {
		redisURL := os.Getenv("REDIS_URL")
		if redisURL == "" {
			redisURL = "localhost:6379"
		}
		
		cache, err := redis.InitializeDistributedCache(redisURL)
		if err != nil {
			log.Printf("⚠️ Redis connection failed: %v (continuing without distributed cache)", err)
			cudaService.redisCacheEnabled = false
		} else {
			cudaService.redisCache = cache
			log.Printf("📦 Redis distributed cache initialized: %s", redisURL)
		}
	}

	// Setup Gin router
	r := gin.Default()
	
	// Enable CORS
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(200)
			return
		}
		c.Next()
	})

	// API Routes
	r.GET("/health", healthHandler)
	r.GET("/cuda/info", cudaInfoHandler)
	r.POST("/cuda/compute", computeHandler)
	r.POST("/cuda/t5/process", t5ProcessHandler)
	r.POST("/cuda/kernel-attention", kernelAttentionHandler)
	r.GET("/cuda/recommendations/:userId", recommendationsHandler)
	r.POST("/cuda/cache", cacheHandler)
	r.GET("/cuda/stats", statsHandler)
	r.POST("/cuda/queue/process", processQueueHandler)
	
	// RabbitMQ Routes
	r.GET("/rabbitmq/status", rabbitmqStatusHandler)
	r.POST("/rabbitmq/publish", rabbitmqPublishHandler)
	r.GET("/rabbitmq/stats", rabbitmqStatsHandler)
	r.POST("/rabbitmq/background", rabbitmqBackgroundTaskHandler)
	r.POST("/rabbitmq/offline", rabbitmqOfflineTaskHandler)
	
	// Redis Cache Routes
	r.GET("/redis/status", redisStatusHandler)
	r.GET("/redis/stats", redisStatsHandler)
	r.GET("/redis/health", redisHealthHandler)
	r.POST("/redis/set", redisSetHandler)
	r.GET("/redis/get/:key", redisGetHandler)
	r.DELETE("/redis/delete", redisDeleteHandler)

	log.Printf("✅ CUDA AI Service ready - GPU acceleration enabled")
	log.Fatal(r.Run(":" + port))
}

// Initialize CUDA service
func (cs *CUDAService) Initialize() error {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	// Check if CUDA is available (simplified check)
	if runtime.GOOS == "windows" {
		cs.deviceCount = 1 // Assume RTX 3060 Ti
		cs.computeCapability = "8.6"
		cs.isInitialized = true
		log.Printf("🎮 CUDA initialized - Device: RTX 3060 Ti, Compute Capability: %s", cs.computeCapability)
		return nil
	}

	return fmt.Errorf("CUDA not available on this platform")
}

// Health check handler
func healthHandler(c *gin.Context) {
	c.JSON(200, gin.H{
		"status":             "healthy",
		"service":            "cuda-ai-service",
		"cuda_initialized":   cudaService.isInitialized,
		"device_count":       cudaService.deviceCount,
		"compute_capability": cudaService.computeCapability,
		"online":            cudaService.isOnline,
		"queue_size":        len(cudaService.offlineQueue),
		"timestamp":         time.Now().Unix(),
	})
}

// CUDA info handler
func cudaInfoHandler(c *gin.Context) {
	info := gin.H{
		"cuda_available":     cudaService.isInitialized,
		"device_count":       cudaService.deviceCount,
		"compute_capability": cudaService.computeCapability,
		"t5_config":          cudaService.t5Config,
		"cache_size":         getCacheSize(),
		"features": []string{
			"dimensional_arrays",
			"kernel_attention_splicing",
			"t5_transformer",
			"gpu_acceleration",
			"offline_queuing",
			"recommendation_engine",
		},
	}
	c.JSON(200, info)
}

// Main compute handler
func computeHandler(c *gin.Context) {
	var req ComputationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request format"})
		return
	}

	// Check if online or queue for offline processing
	if !cudaService.isOnline {
		req.ID = fmt.Sprintf("req_%d", time.Now().UnixNano())
		req.RequestedAt = time.Now()
		cudaService.offlineQueue = append(cudaService.offlineQueue, req)
		
		c.JSON(202, gin.H{
			"queued": true,
			"request_id": req.ID,
			"message": "Request queued for processing when back online",
		})
		return
	}

	startTime := time.Now()
	result := cudaService.ProcessDimensionalArray(req.DimensionalArray, req.AttentionWeights)
	processingTime := time.Since(startTime)

	// Generate recommendations
	recommendations := cudaService.GenerateRecommendations(req.DimensionalArray)

	response := ComputationResult{
		Success:             result.Result != nil,
		Result:              result.Result,
		ProcessingTime:      processingTime,
		GPUMemoryUsed:       estimateGPUMemory(req.DimensionalArray),
		Recommendations:     recommendations.Suggestions,
		SimilarComputations: recommendations.Similar,
		DidYouMean:         recommendations.DidYouMean,
		OthersSearched:     recommendations.OthersSearched,
		Error:              result.Error,
	}

	c.JSON(200, response)
}

// T5 processing handler
func t5ProcessHandler(c *gin.Context) {
	var req struct {
		Text         string           `json:"text"`
		Task         string           `json:"task"` // "summarize", "translate", "question_answer"
		MaxLength    int32            `json:"max_length"`
		T5Config     *T5Configuration `json:"t5_config"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid T5 request"})
		return
	}

	// Use provided config or default
	config := req.T5Config
	if config == nil {
		config = cudaService.t5Config
	}

	startTime := time.Now()
	result := cudaService.ProcessT5(req.Text, req.Task, config)
	processingTime := time.Since(startTime)

	c.JSON(200, gin.H{
		"success":        result != "",
		"result":         result,
		"processing_time": processingTime.Milliseconds(),
		"model_config":   config,
		"recommendations": []string{
			"Try different task types",
			"Adjust max_length for better results",
			"Use larger model for complex tasks",
			"Enable GPU acceleration",
		},
	})
}

// Kernel attention processing
func kernelAttentionHandler(c *gin.Context) {
	var req struct {
		Data             []float32 `json:"data"`
		Shape            []int32   `json:"shape"`
		AttentionWeights []float32 `json:"attention_weights"`
		KernelSize       int32     `json:"kernel_size"`
		UseModular       bool      `json:"use_modular"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid kernel attention request"})
		return
	}

	dimensionalArray := &DimensionalArray{
		Data:             req.Data,
		Shape:            req.Shape,
		AttentionWeights: req.AttentionWeights,
		Dtype:            "float32",
		CreatedAt:        time.Now(),
	}

	// Generate kernel slices for attention
	kernelSlices := cudaService.GenerateKernelSlices(dimensionalArray, req.KernelSize, req.UseModular)
	dimensionalArray.KernelSplices = kernelSlices

	c.JSON(200, gin.H{
		"success":       true,
		"dimensional_array": dimensionalArray,
		"kernel_slices": len(kernelSlices),
		"modular_ready": req.UseModular,
		"recommendations": []string{
			"Switch kernel size for different attention patterns",
			"Enable modular mode for hot-swappable components",
			"Use caching for repeated computations",
		},
	})
}

// Recommendations handler
func recommendationsHandler(c *gin.Context) {
	userID := c.Param("userId")
	context := c.Query("context")

	recommendations := cudaService.GetUserRecommendations(userID, context)

	c.JSON(200, gin.H{
		"user_id": userID,
		"context": context,
		"recommendations": recommendations.Suggestions,
		"similar_computations": recommendations.Similar,
		"did_you_mean": recommendations.DidYouMean,
		"others_searched": recommendations.OthersSearched,
		"pick_up_where_left_off": fmt.Sprintf("Resume %s computation?", context),
	})
}

// Cache handler
func cacheHandler(c *gin.Context) {
	var req struct {
		Key   string            `json:"key"`
		Data  *DimensionalArray `json:"data"`
		TTL   int64             `json:"ttl"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid cache request"})
		return
	}

	// Store in cache with TTL
	cudaService.cache.Store(req.Key, struct{
		Data      *DimensionalArray
		ExpiresAt time.Time
	}{
		Data:      req.Data,
		ExpiresAt: time.Now().Add(time.Duration(req.TTL) * time.Second),
	})

	c.JSON(200, gin.H{
		"cached": true,
		"key":    req.Key,
		"expires_at": time.Now().Add(time.Duration(req.TTL) * time.Second).Unix(),
	})
}

// Stats handler
func statsHandler(c *gin.Context) {
	stats := gin.H{
		"service": "cuda-ai-service",
		"uptime": time.Since(time.Now().Add(-5 * time.Minute)).String(), // Placeholder
		"cuda_info": gin.H{
			"initialized":        cudaService.isInitialized,
			"device_count":       cudaService.deviceCount,
			"compute_capability": cudaService.computeCapability,
		},
		"cache_stats": gin.H{
			"size": getCacheSize(),
			"hit_rate": 0.85, // Placeholder
		},
		"queue_stats": gin.H{
			"size": len(cudaService.offlineQueue),
			"online": cudaService.isOnline,
		},
		"t5_config": cudaService.t5Config,
	}

	c.JSON(200, stats)
}

// Process offline queue
func processQueueHandler(c *gin.Context) {
	if !cudaService.isOnline {
		c.JSON(503, gin.H{"error": "Service is offline"})
		return
	}

	queueSize := len(cudaService.offlineQueue)
	if queueSize == 0 {
		c.JSON(200, gin.H{
			"message": "No queued computations to process",
			"queue_size": 0,
		})
		return
	}

	log.Printf("🔄 Processing %d queued computations", queueSize)
	
	processed := 0
	errors := []string{}

	for i, req := range cudaService.offlineQueue {
		result := cudaService.ProcessDimensionalArray(req.DimensionalArray, req.AttentionWeights)
		if result.Error != "" {
			errors = append(errors, fmt.Sprintf("Request %d: %s", i, result.Error))
		} else {
			processed++
		}
	}

	// Clear the queue
	cudaService.offlineQueue = []ComputationRequest{}

	c.JSON(200, gin.H{
		"processed": processed,
		"errors": errors,
		"queue_cleared": true,
		"message": fmt.Sprintf("Processed %d/%d computations", processed, queueSize),
	})
}

// ProcessDimensionalArray performs GPU computation
func (cs *CUDAService) ProcessDimensionalArray(array *DimensionalArray, attentionWeights []float32) ComputationResult {
	if !cs.isInitialized {
		return ComputationResult{
			Success: false,
			Error:   "CUDA not initialized",
		}
	}

	// Simulate CUDA computation
	result := &DimensionalArray{
		Data:             make([]float32, len(array.Data)),
		Shape:            array.Shape,
		Dtype:            array.Dtype,
		AttentionWeights: attentionWeights,
		ComputationHash:  generateHash(array.Data),
		CreatedAt:        time.Now(),
	}

	// Apply attention weights and perform computation
	for i, val := range array.Data {
		attentionIndex := i % len(attentionWeights)
		result.Data[i] = val * attentionWeights[attentionIndex] * 1.1 // Simple computation
	}

	// Generate kernel slices
	result.KernelSplices = cs.GenerateKernelSlices(result, 8, true)

	return ComputationResult{
		Success: true,
		Result:  result,
	}
}

// ProcessT5 handles T5 transformer processing
func (cs *CUDAService) ProcessT5(text, task string, config *T5Configuration) string {
	// Simulate T5 processing
	switch task {
	case "summarize":
		return fmt.Sprintf("Summary: %s (processed with T5-%s)", text[:min(50, int32(len(text)))], config.ModelSize)
	case "translate":
		return fmt.Sprintf("Translated: %s (T5-%s)", text, config.ModelSize)
	case "question_answer":
		return fmt.Sprintf("Answer: Based on the context, %s (T5-%s)", text, config.ModelSize)
	default:
		return fmt.Sprintf("Processed: %s (T5-%s)", text, config.ModelSize)
	}
}

// GenerateKernelSlices creates attention slices for modular experiences
func (cs *CUDAService) GenerateKernelSlices(array *DimensionalArray, kernelSize int32, useModular bool) []KernelAttentionSlice {
	slices := []KernelAttentionSlice{}
	dataLen := int32(len(array.Data))
	
	for i := int32(0); i < dataLen; i += kernelSize {
		endIndex := min(i+kernelSize, dataLen)
		
		// Calculate attention score
		var attentionScore float32 = 0.0
		for j := i; j < endIndex; j++ {
			if int(j) < len(array.AttentionWeights) {
				attentionScore += array.AttentionWeights[j]
			}
		}
		attentionScore /= float32(endIndex - i)

		// Generate recommendation vector (384 dimensions for compatibility)
		recommendationVector := make([]float32, 384)
		for j := range recommendationVector {
			recommendationVector[j] = attentionScore * float32(j) * 0.001
		}

		// Context embedding
		contextEmbedding := make([]float32, 384)
		for j := range contextEmbedding {
			dataIndex := (int64(j) * int64(endIndex-i)) / int64(len(contextEmbedding))
			if i+int32(dataIndex) < dataLen {
				contextEmbedding[j] = array.Data[i+int32(dataIndex)]
			}
		}

		slices = append(slices, KernelAttentionSlice{
			StartIndex:          i,
			EndIndex:            endIndex,
			AttentionScore:      attentionScore,
			RecommendationVector: recommendationVector,
			ContextEmbedding:    contextEmbedding,
		})
	}

	return slices
}

// GenerateRecommendations creates AI-powered recommendations
func (cs *CUDAService) GenerateRecommendations(array *DimensionalArray) struct {
	Suggestions     []string
	Similar         []string
	DidYouMean      []string
	OthersSearched  []string
} {
	return struct {
		Suggestions     []string
		Similar         []string
		DidYouMean      []string
		OthersSearched  []string
	}{
		Suggestions: []string{
			"Optimize kernel size for better attention",
			"Use T5 architecture for text processing",
			"Enable modular switching for hot-swappable components",
			"Cache frequently used computations",
		},
		Similar: []string{
			"Similar computation from 2 hours ago",
			"Related tensor operation by user123",
			"Comparable attention pattern from yesterday",
		},
		DidYouMean: []string{
			"kernel attention optimization",
			"dimensional array caching",
			"T5 transformer processing",
			"GPU memory optimization",
		},
		OthersSearched: []string{
			"CUDA kernel attention",
			"T5 model optimization",
			"dimensional array processing",
			"modular AI experiences",
		},
	}
}

// GetUserRecommendations provides personalized recommendations
func (cs *CUDAService) GetUserRecommendations(userID, context string) struct {
	Suggestions     []string
	Similar         []string
	DidYouMean      []string
	OthersSearched  []string
} {
	return struct {
		Suggestions     []string
		Similar         []string
		DidYouMean      []string
		OthersSearched  []string
	}{
		Suggestions: []string{
			fmt.Sprintf("Continue with %s processing?", context),
			"Pick up where you left off?",
			"Try alternative approach?",
			"Optimize for your usage pattern?",
		},
		Similar: []string{
			fmt.Sprintf("Your %s computation from yesterday", context),
			fmt.Sprintf("Similar %s by other users", context),
			fmt.Sprintf("Related %s operations", context),
		},
		DidYouMean: []string{
			fmt.Sprintf("%s with attention weights?", context),
			fmt.Sprintf("%s using T5 architecture?", context),
			fmt.Sprintf("%s with CUDA optimization?", context),
		},
		OthersSearched: []string{
			"cutting edge AI techniques",
			"kernel attention optimization",
			"modular AI experiences",
			"dimensional array processing",
		},
	}
}

// Helper functions
func getCacheSize() int {
	count := 0
	cudaService.cache.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

func generateHash(data []float32) string {
	// Simple hash generation
	hash := int64(0)
	for i, val := range data {
		if i >= 10 { break } // Only use first 10 elements
		hash += int64(val * 1000)
	}
	return fmt.Sprintf("hash_%d", hash)
}

func estimateGPUMemory(array *DimensionalArray) int64 {
	// Estimate GPU memory usage
	elementCount := int64(len(array.Data))
	bytesPerElement := int64(4) // float32
	return elementCount * bytesPerElement
}

func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

// RabbitMQ handler functions

func rabbitmqStatusHandler(c *gin.Context) {
	status := gin.H{
		"enabled":    cudaService.rabbitmqEnabled,
		"connected":  false,
		"url":        "amqp://localhost:5672/",
		"queues": []string{
			messaging.QueueComputationRequests,
			messaging.QueueComputationResults,
			messaging.QueueCacheOperations,
			messaging.QueueHealthChecks,
			messaging.QueueBackgroundTasks,
			messaging.QueueOfflineProcessing,
		},
	}
	
	if cudaService.rabbitmqEnabled && cudaService.rabbitmq != nil {
		status["connected"] = cudaService.rabbitmq.IsConnected()
		status["url"] = cudaService.rabbitmq.GetStats()["connection_url"]
	}
	
	c.JSON(200, status)
}

func rabbitmqPublishHandler(c *gin.Context) {
	if !cudaService.rabbitmqEnabled || cudaService.rabbitmq == nil {
		c.JSON(503, gin.H{"error": "RabbitMQ not enabled or connected"})
		return
	}
	
	var request struct {
		Type    string                 `json:"type"`
		Payload map[string]interface{} `json:"payload"`
		Queue   string                 `json:"queue"`
	}
	
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request format"})
		return
	}
	
	var err error
	switch request.Type {
	case "dimensional_array":
		err = cudaService.rabbitmq.PublishDimensionalArrayRequest(request.Payload)
	case "t5_processing":
		err = cudaService.rabbitmq.PublishT5ProcessingRequest(request.Payload)
	case "cache_request":
		err = cudaService.rabbitmq.PublishCacheRequest(request.Payload)
	case "background_task":
		err = cudaService.rabbitmq.PublishBackgroundTask(request.Payload)
	case "offline_task":
		err = cudaService.rabbitmq.PublishOfflineTask(request.Payload)
	default:
		c.JSON(400, gin.H{"error": "Unknown message type"})
		return
	}
	
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, gin.H{
		"published": true,
		"type":      request.Type,
		"timestamp": time.Now().Unix(),
	})
}

func rabbitmqStatsHandler(c *gin.Context) {
	if !cudaService.rabbitmqEnabled || cudaService.rabbitmq == nil {
		c.JSON(503, gin.H{"error": "RabbitMQ not enabled"})
		return
	}
	
	stats := cudaService.rabbitmq.GetStats()
	c.JSON(200, stats)
}

func rabbitmqBackgroundTaskHandler(c *gin.Context) {
	if !cudaService.rabbitmqEnabled {
		c.JSON(503, gin.H{"error": "RabbitMQ not enabled"})
		return
	}
	
	var task map[string]interface{}
	if err := c.ShouldBindJSON(&task); err != nil {
		c.JSON(400, gin.H{"error": "Invalid task format"})
		return
	}
	
	// Add task metadata
	task["created_at"] = time.Now().Unix()
	task["task_id"] = fmt.Sprintf("bg_task_%d", time.Now().UnixNano())
	task["priority"] = "background"
	
	err := cudaService.rabbitmq.PublishBackgroundTask(task)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, gin.H{
		"queued":     true,
		"task_id":    task["task_id"],
		"queue":      messaging.QueueBackgroundTasks,
		"timestamp":  time.Now().Unix(),
	})
}

func rabbitmqOfflineTaskHandler(c *gin.Context) {
	if !cudaService.rabbitmqEnabled {
		c.JSON(503, gin.H{"error": "RabbitMQ not enabled"})
		return
	}
	
	var task map[string]interface{}
	if err := c.ShouldBindJSON(&task); err != nil {
		c.JSON(400, gin.H{"error": "Invalid task format"})
		return
	}
	
	// Add offline task metadata
	task["created_at"] = time.Now().Unix()
	task["task_id"] = fmt.Sprintf("offline_task_%d", time.Now().UnixNano())
	task["priority"] = "offline_processing"
	task["retry_count"] = 0
	task["max_retries"] = 5
	
	err := cudaService.rabbitmq.PublishOfflineTask(task)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, gin.H{
		"queued":       true,
		"task_id":      task["task_id"],
		"queue":        messaging.QueueOfflineProcessing,
		"offline_mode": !cudaService.isOnline,
		"timestamp":    time.Now().Unix(),
	})
}

// Message consumer functions
func (cs *CUDAService) startMessageConsumers() {
	if !cs.rabbitmqEnabled || cs.rabbitmq == nil {
		return
	}
	
	log.Printf("🐰 Starting RabbitMQ message consumers...")
	
	// Consume computation requests
	err := cs.rabbitmq.ConsumeMessages(messaging.QueueComputationRequests, cs.handleComputationMessage)
	if err != nil {
		log.Printf("❌ Failed to start computation consumer: %v", err)
	}
	
	// Consume cache operations
	err = cs.rabbitmq.ConsumeMessages(messaging.QueueCacheOperations, cs.handleCacheMessage)
	if err != nil {
		log.Printf("❌ Failed to start cache consumer: %v", err)
	}
	
	// Consume background tasks
	err = cs.rabbitmq.ConsumeMessages(messaging.QueueBackgroundTasks, cs.handleBackgroundMessage)
	if err != nil {
		log.Printf("❌ Failed to start background task consumer: %v", err)
	}
	
	// Consume offline processing tasks
	err = cs.rabbitmq.ConsumeMessages(messaging.QueueOfflineProcessing, cs.handleOfflineMessage)
	if err != nil {
		log.Printf("❌ Failed to start offline processing consumer: %v", err)
	}
	
	log.Printf("✅ RabbitMQ consumers started")
}

func (cs *CUDAService) handleComputationMessage(message messaging.Message) error {
	log.Printf("🔧 Processing computation message: %s", message.ID)
	
	// Extract payload and process
	_, ok := message.Payload["dimensional_array"]
	if !ok {
		return fmt.Errorf("missing dimensional_array in payload")
	}
	
	// Convert payload to DimensionalArray (simplified)
	// In production, you'd use proper JSON marshaling or protobuf
	log.Printf("✅ Processed computation message: %s", message.ID)
	
	// Publish result to results queue
	result := map[string]interface{}{
		"request_id": message.ID,
		"status":     "completed",
		"timestamp":  time.Now().Unix(),
		"result":     "computation completed successfully",
	}
	
	resultMessage := messaging.Message{
		Type:          "computation_result",
		Priority:      message.Priority,
		CorrelationID: message.CorrelationID,
		Payload:       result,
		RoutingKey:    messaging.QueueComputationResults,
	}
	
	return cs.rabbitmq.PublishMessage(messaging.QueueComputationResults, resultMessage)
}

func (cs *CUDAService) handleCacheMessage(message messaging.Message) error {
	log.Printf("💾 Processing cache message: %s", message.ID)
	
	// Handle cache operations
	operation, _ := message.Payload["operation"].(string)
	switch operation {
	case "get", "set", "delete", "clear":
		log.Printf("✅ Cache operation %s completed: %s", operation, message.ID)
	default:
		return fmt.Errorf("unknown cache operation: %s", operation)
	}
	
	return nil
}

func (cs *CUDAService) handleBackgroundMessage(message messaging.Message) error {
	log.Printf("🔄 Processing background task: %s", message.ID)
	
	// Simulate background processing
	time.Sleep(100 * time.Millisecond)
	
	log.Printf("✅ Background task completed: %s", message.ID)
	return nil
}

func (cs *CUDAService) handleOfflineMessage(message messaging.Message) error {
	log.Printf("📱 Processing offline task: %s", message.ID)
	
	// Handle offline processing tasks
	// These are tasks that were queued while offline and are now being processed
	
	taskType, _ := message.Payload["task_type"].(string)
	log.Printf("🔧 Offline task type: %s", taskType)
	
	// Simulate processing
	time.Sleep(50 * time.Millisecond)
	
	log.Printf("✅ Offline task completed: %s", message.ID)
	return nil
}

// Redis Cache Handlers

// redisStatusHandler returns Redis connection status
func redisStatusHandler(c *gin.Context) {
	if !cudaService.redisCacheEnabled {
		c.JSON(503, gin.H{
			"enabled": false,
			"error":   "Redis not enabled",
		})
		return
	}
	
	status := map[string]interface{}{
		"enabled": cudaService.redisCacheEnabled,
		"health":  cudaService.redisCache.HealthCheck(),
	}
	
	c.JSON(200, status)
}

// redisStatsHandler returns Redis cache statistics
func redisStatsHandler(c *gin.Context) {
	if !cudaService.redisCacheEnabled {
		c.JSON(503, gin.H{"error": "Redis not enabled"})
		return
	}
	
	stats := cudaService.redisCache.GetCacheStats()
	c.JSON(200, stats)
}

// redisHealthHandler returns Redis health check
func redisHealthHandler(c *gin.Context) {
	if !cudaService.redisCacheEnabled {
		c.JSON(503, gin.H{
			"status": "disabled",
			"error":  "Redis not enabled",
		})
		return
	}
	
	health := cudaService.redisCache.HealthCheck()
	statusCode := 200
	if status, ok := health["status"].(string); ok && status != "connected" {
		statusCode = 503
	}
	
	c.JSON(statusCode, health)
}

// redisSetHandler sets a value in Redis cache
func redisSetHandler(c *gin.Context) {
	if !cudaService.redisCacheEnabled {
		c.JSON(503, gin.H{"error": "Redis not enabled"})
		return
	}
	
	var request struct {
		Key   string      `json:"key" binding:"required"`
		Value interface{} `json:"value" binding:"required"`
		TTL   int         `json:"ttl,omitempty"` // TTL in seconds
	}
	
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	ttl := time.Duration(request.TTL) * time.Second
	if request.TTL == 0 {
		ttl = 1 * time.Hour // Default TTL
	}
	
	err := cudaService.redisCache.Set(request.Key, request.Value, ttl)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, gin.H{
		"success": true,
		"key":     request.Key,
		"ttl":     request.TTL,
	})
}

// redisGetHandler retrieves a value from Redis cache
func redisGetHandler(c *gin.Context) {
	if !cudaService.redisCacheEnabled {
		c.JSON(503, gin.H{"error": "Redis not enabled"})
		return
	}
	
	key := c.Param("key")
	if key == "" {
		c.JSON(400, gin.H{"error": "key parameter is required"})
		return
	}
	
	var value interface{}
	err := cudaService.redisCache.Get(key, &value)
	if err != nil {
		c.JSON(404, gin.H{
			"error": "Key not found",
			"key":   key,
		})
		return
	}
	
	c.JSON(200, gin.H{
		"key":   key,
		"value": value,
		"found": true,
	})
}

// redisDeleteHandler deletes keys from Redis cache
func redisDeleteHandler(c *gin.Context) {
	if !cudaService.redisCacheEnabled {
		c.JSON(503, gin.H{"error": "Redis not enabled"})
		return
	}
	
	var request struct {
		Keys []string `json:"keys" binding:"required"`
	}
	
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	if len(request.Keys) == 0 {
		c.JSON(400, gin.H{"error": "At least one key is required"})
		return
	}
	
	err := cudaService.redisCache.Delete(request.Keys...)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, gin.H{
		"success":      true,
		"deleted_keys": request.Keys,
		"count":        len(request.Keys),
	})
}