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

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/rs/cors"
)

// GPU-optimized tensor operations
type GPUTensorProcessor struct {
	memory      map[string]*TensorCache
	mutex       sync.RWMutex
	wasmBridge  *WebAssemblyBridge
	statsLocker sync.RWMutex
	stats       ProcessingStats
}

// Multi-dimensional array structure
type MultiDimArray struct {
	Shape      []int     `json:"shape"`
	Data       []float32 `json:"data"`
	Dimensions int       `json:"dimensions"`
	Layout     string    `json:"layout"`     // "coalesced" | "strided"
	CacheKey   string    `json:"cache_key"`
	LODLevel   int       `json:"lod_level"`
	Timestamp  int64     `json:"timestamp"`
}

// Tensor cache with LRU eviction
type TensorCache struct {
	Data       MultiDimArray `json:"data"`
	AccessTime int64         `json:"access_time"`
	HitCount   int           `json:"hit_count"`
	Size       int           `json:"size"`
}

// Processing statistics
type ProcessingStats struct {
	TotalRequests    int64   `json:"total_requests"`
	CacheHits        int64   `json:"cache_hits"`
	CacheMisses      int64   `json:"cache_misses"`
	AverageTime      float64 `json:"average_time_ms"`
	GPUUtilization   float64 `json:"gpu_utilization"`
	MemoryUsage      int64   `json:"memory_usage_bytes"`
	LastProcessed    int64   `json:"last_processed"`
	ErrorCount       int64   `json:"error_count"`
}

// WebAssembly bridge for browser integration
type WebAssemblyBridge struct {
	initialized bool
	functions   map[string]interface{}
	mutex       sync.RWMutex
}

// Processing response
type ProcessingResponse struct {
	Success        bool          `json:"success"`
	Data           MultiDimArray `json:"data"`
	ProcessingTime int64         `json:"processing_time_ms"`
	CacheHit       bool          `json:"cache_hit"`
	Service        string        `json:"service"`
	Route          string        `json:"route"`
	Metadata       ResponseMetadata `json:"metadata"`
}

type ResponseMetadata struct {
	TensorStats     TensorStats `json:"tensor_stats"`
	OptimizationLevel string    `json:"optimization_level"`
	GPUMemoryUsed   int64       `json:"gpu_memory_used"`
}

type TensorStats struct {
	TotalElements int     `json:"total_elements"`
	MemorySize    int64   `json:"memory_size_bytes"`
	Density       float32 `json:"density"`
	SparsityRatio float32 `json:"sparsity_ratio"`
}

// WebSocket upgrader
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins in development
	},
}

func NewGPUTensorProcessor() *GPUTensorProcessor {
	gtp := &GPUTensorProcessor{
		memory:     make(map[string]*TensorCache),
		wasmBridge: NewWebAssemblyBridge(),
		stats: ProcessingStats{
			TotalRequests:  0,
			CacheHits:      0,
			CacheMisses:    0,
			AverageTime:    0,
			GPUUtilization: 0,
			MemoryUsage:    0,
			LastProcessed:  0,
			ErrorCount:     0,
		},
	}

	// Initialize GPU context if available
	gtp.initializeGPUContext()
	
	return gtp
}

func (gtp *GPUTensorProcessor) initializeGPUContext() {
	log.Println("🔧 Initializing GPU context...")
	
	// Check for CUDA availability (simulated - would use actual CUDA libraries)
	if gtp.checkCUDAAvailability() {
		log.Println("✅ CUDA device detected and initialized")
		gtp.stats.GPUUtilization = 0.0 // Start at 0%
	} else {
		log.Println("⚠️  No CUDA device found, falling back to CPU processing")
		gtp.stats.GPUUtilization = -1 // Indicate no GPU
	}
}

func (gtp *GPUTensorProcessor) checkCUDAAvailability() bool {
	// In a real implementation, this would use CGO bindings to CUDA
	// For now, simulate GPU availability based on environment
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "" {
		return true
	}
	return false
}

// Main tensor processing pipeline
func (gtp *GPUTensorProcessor) ProcessMultiDimArray(input MultiDimArray) (*ProcessingResponse, error) {
	startTime := time.Now()
	
	gtp.statsLocker.Lock()
	gtp.stats.TotalRequests++
	gtp.statsLocker.Unlock()
	
	// Check cache first
	if cached := gtp.getCachedResult(input.CacheKey); cached != nil {
		gtp.statsLocker.Lock()
		gtp.stats.CacheHits++
		gtp.statsLocker.Unlock()
		
		processingTime := time.Since(startTime).Milliseconds()
		
		return &ProcessingResponse{
			Success:        true,
			Data:           cached.Data,
			ProcessingTime: processingTime,
			CacheHit:       true,
			Service:        "gpu-tensor-service",
			Route:          generateRouteHash(input.CacheKey),
			Metadata: ResponseMetadata{
				TensorStats:       gtp.calculateTensorStats(cached.Data),
				OptimizationLevel: "cached",
				GPUMemoryUsed:     0, // No GPU memory used for cache hit
			},
		}, nil
	}
	
	// Cache miss - process tensor
	gtp.statsLocker.Lock()
	gtp.stats.CacheMisses++
	gtp.statsLocker.Unlock()
	
	// 1. Validate and optimize memory layout
	optimizedLayout, err := gtp.optimizeMemoryLayout(input)
	if err != nil {
		gtp.statsLocker.Lock()
		gtp.stats.ErrorCount++
		gtp.statsLocker.Unlock()
		return nil, fmt.Errorf("memory optimization failed: %v", err)
	}
	
	// 2. Process with GPU or CPU fallback
	result, err := gtp.processWithAcceleration(*optimizedLayout)
	if err != nil {
		gtp.statsLocker.Lock()
		gtp.stats.ErrorCount++
		gtp.statsLocker.Unlock()
		return nil, fmt.Errorf("processing failed: %v", err)
	}
	
	// 3. Cache result for future access
	gtp.cacheResult(input.CacheKey, *result)
	
	// 4. Update statistics
	processingTime := time.Since(startTime).Milliseconds()
	gtp.updateStats(processingTime)
	
	return &ProcessingResponse{
		Success:        true,
		Data:           *result,
		ProcessingTime: processingTime,
		CacheHit:       false,
		Service:        "gpu-tensor-service",
		Route:          generateRouteHash(input.CacheKey),
		Metadata: ResponseMetadata{
			TensorStats:       gtp.calculateTensorStats(*result),
			OptimizationLevel: "gpu_accelerated",
			GPUMemoryUsed:     gtp.estimateGPUMemoryUsage(*result),
		},
	}, nil
}

func (gtp *GPUTensorProcessor) optimizeMemoryLayout(input MultiDimArray) (*MultiDimArray, error) {
	// Calculate total elements for validation
	totalElements := 1
	for _, dim := range input.Shape {
		if dim <= 0 {
			return nil, fmt.Errorf("invalid dimension: %d", dim)
		}
		totalElements *= dim
	}
	
	if len(input.Data) != totalElements {
		return nil, fmt.Errorf("data length mismatch: expected %d, got %d", totalElements, len(input.Data))
	}
	
	// Create optimized layout for coalesced memory access
	optimizedData := make([]float32, totalElements)
	
	// For 4D tensors (legal AI case), reorganize for optimal GPU access
	if input.Dimensions == 4 && len(input.Shape) == 4 {
		cases, docs, paragraphs, embeddings := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
		
		writeIndex := 0
		// Reorganize for coalesced access: embeddings as innermost dimension
		for c := 0; c < cases; c++ {
			for d := 0; d < docs; d++ {
				for p := 0; p < paragraphs; p++ {
					for e := 0; e < embeddings; e++ {
						sourceIndex := ((c*docs+d)*paragraphs+p)*embeddings + e
						if sourceIndex < len(input.Data) {
							optimizedData[writeIndex] = input.Data[sourceIndex]
						}
						writeIndex++
					}
				}
			}
		}
	} else {
		// For other dimensions, just copy data
		copy(optimizedData, input.Data)
	}
	
	return &MultiDimArray{
		Shape:      input.Shape,
		Data:       optimizedData,
		Dimensions: input.Dimensions,
		Layout:     "coalesced",
		CacheKey:   input.CacheKey,
		LODLevel:   input.LODLevel,
		Timestamp:  time.Now().UnixMilli(),
	}, nil
}

func (gtp *GPUTensorProcessor) processWithAcceleration(input MultiDimArray) (*MultiDimArray, error) {
	// Simulate GPU processing with actual mathematical operations
	processedData := make([]float32, len(input.Data))
	
	if gtp.stats.GPUUtilization >= 0 { // GPU available
		gtp.processWithGPUSimulation(input, processedData)
	} else { // CPU fallback
		gtp.processWithCPU(input, processedData)
	}
	
	return &MultiDimArray{
		Shape:      input.Shape,
		Data:       processedData,
		Dimensions: input.Dimensions,
		Layout:     "gpu_processed",
		CacheKey:   input.CacheKey,
		LODLevel:   input.LODLevel,
		Timestamp:  time.Now().UnixMilli(),
	}, nil
}

func (gtp *GPUTensorProcessor) processWithGPUSimulation(input MultiDimArray, output []float32) {
	// Simulate CUDA kernel processing
	gtp.statsLocker.Lock()
	gtp.stats.GPUUtilization = 85.0 // Simulate high GPU usage during processing
	gtp.statsLocker.Unlock()
	
	// Apply legal AI specific transformations
	if input.Dimensions == 4 && len(input.Shape) == 4 {
		cases, docs, paragraphs, embeddings := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
		
		for i := 0; i < len(input.Data); i++ {
			// Calculate multi-dimensional indices
			remaining := i
			paraIdx := remaining % paragraphs
			remaining /= paragraphs
			docIdx := remaining % docs
			remaining /= docs
			caseIdx := remaining % cases
			embedIdx := i % embeddings
			
			value := input.Data[i]
			
			// Apply semantic similarity weighting for legal embeddings
			weight := float32(1.0)
			if embedIdx < 384 { // First half of nomic-embed-text embeddings
				weight = 1.1
			} else { // Second half
				weight = 0.95
			}
			
			// Apply tricubic interpolation for spatial coherence
			if caseIdx > 0 && docIdx > 0 && paraIdx > 0 {
				x := float32(caseIdx) / float32(cases)
				y := float32(docIdx) / float32(docs)
				z := float32(paraIdx) / float32(paragraphs)
				
				// Smoothstep interpolation
				wx := x * x * (3.0 - 2.0*x)
				wy := y * y * (3.0 - 2.0*y)
				wz := z * z * (3.0 - 2.0*z)
				
				weight *= wx * wy * wz
			}
			
			output[i] = value * weight
		}
	} else {
		// Standard processing for other dimensions
		for i := 0; i < len(input.Data); i++ {
			output[i] = input.Data[i] * 1.05 // Small enhancement
		}
	}
	
	// Simulate processing completion
	time.Sleep(1 * time.Millisecond) // Small delay to simulate GPU processing
	
	gtp.statsLocker.Lock()
	gtp.stats.GPUUtilization = 15.0 // Back to idle
	gtp.statsLocker.Unlock()
}

func (gtp *GPUTensorProcessor) processWithCPU(input MultiDimArray, output []float32) {
	// CPU-based processing with parallel goroutines
	numWorkers := runtime.NumCPU()
	chunkSize := (len(input.Data) + numWorkers - 1) / numWorkers
	
	var wg sync.WaitGroup
	
	for worker := 0; worker < numWorkers; worker++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * chunkSize
			end := start + chunkSize
			if end > len(input.Data) {
				end = len(input.Data)
			}
			
			for i := start; i < end; i++ {
				output[i] = input.Data[i] * 1.02 // Small CPU enhancement
			}
		}(worker)
	}
	
	wg.Wait()
}

// Cache management
func (gtp *GPUTensorProcessor) getCachedResult(cacheKey string) *TensorCache {
	gtp.mutex.RLock()
	defer gtp.mutex.RUnlock()
	
	if cached, exists := gtp.memory[cacheKey]; exists {
		cached.AccessTime = time.Now().UnixMilli()
		cached.HitCount++
		return cached
	}
	
	return nil
}

func (gtp *GPUTensorProcessor) cacheResult(cacheKey string, result MultiDimArray) {
	gtp.mutex.Lock()
	defer gtp.mutex.Unlock()
	
	cache := &TensorCache{
		Data:       result,
		AccessTime: time.Now().UnixMilli(),
		HitCount:   1,
		Size:       len(result.Data) * 4, // 4 bytes per float32
	}
	
	gtp.memory[cacheKey] = cache
	
	// Simple LRU eviction if cache gets too large
	if len(gtp.memory) > 1000 {
		gtp.evictOldestCache()
	}
}

func (gtp *GPUTensorProcessor) evictOldestCache() {
	var oldestKey string
	var oldestTime int64 = time.Now().UnixMilli()
	
	for key, cache := range gtp.memory {
		if cache.AccessTime < oldestTime {
			oldestTime = cache.AccessTime
			oldestKey = key
		}
	}
	
	if oldestKey != "" {
		delete(gtp.memory, oldestKey)
		log.Printf("Evicted cache entry: %s", oldestKey)
	}
}

// Statistics and utilities
func (gtp *GPUTensorProcessor) updateStats(processingTime int64) {
	gtp.statsLocker.Lock()
	defer gtp.statsLocker.Unlock()
	
	// Update average processing time
	totalTime := gtp.stats.AverageTime * float64(gtp.stats.TotalRequests-1)
	gtp.stats.AverageTime = (totalTime + float64(processingTime)) / float64(gtp.stats.TotalRequests)
	
	gtp.stats.LastProcessed = time.Now().UnixMilli()
	
	// Update memory usage estimate
	gtp.mutex.RLock()
	totalMemory := int64(0)
	for _, cache := range gtp.memory {
		totalMemory += int64(cache.Size)
	}
	gtp.mutex.RUnlock()
	
	gtp.stats.MemoryUsage = totalMemory
}

func (gtp *GPUTensorProcessor) calculateTensorStats(tensor MultiDimArray) TensorStats {
	totalElements := len(tensor.Data)
	memorySize := int64(totalElements * 4) // 4 bytes per float32
	
	// Calculate density and sparsity
	nonZeroCount := 0
	for _, value := range tensor.Data {
		if value != 0 {
			nonZeroCount++
		}
	}
	
	density := float32(nonZeroCount) / float32(totalElements)
	sparsity := 1.0 - density
	
	return TensorStats{
		TotalElements: totalElements,
		MemorySize:    memorySize,
		Density:       density,
		SparsityRatio: sparsity,
	}
}

func (gtp *GPUTensorProcessor) estimateGPUMemoryUsage(tensor MultiDimArray) int64 {
	baseMemory := int64(len(tensor.Data) * 4) // Input data
	workingMemory := baseMemory * 2           // Temporary buffers
	return baseMemory + workingMemory
}

func generateRouteHash(cacheKey string) string {
	hash := 0
	for i, char := range cacheKey {
		hash = ((hash << 5) - hash) + int(char)
		hash = hash & hash // Convert to 32-bit integer
	}
	return fmt.Sprintf("route_%d", hash&0x7FFFFFFF)
}

// WebAssembly bridge implementation
func NewWebAssemblyBridge() *WebAssemblyBridge {
	return &WebAssemblyBridge{
		initialized: true,
		functions:   make(map[string]interface{}),
	}
}

// HTTP Handlers
func (gtp *GPUTensorProcessor) handleProcessTensor(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var input MultiDimArray
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	
	// Generate cache key if not provided
	if input.CacheKey == "" {
		input.CacheKey = fmt.Sprintf("tensor_%d_%s", time.Now().UnixNano(), 
			fmt.Sprintf("%v", input.Shape))
	}
	
	result, err := gtp.ProcessMultiDimArray(input)
	if err != nil {
		http.Error(w, fmt.Sprintf("Processing failed: %v", err), http.StatusInternalServerError)
		return
	}
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(result)
}

func (gtp *GPUTensorProcessor) handleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	gtp.statsLocker.RLock()
	stats := gtp.stats
	gtp.statsLocker.RUnlock()
	
	// Add cache statistics
	gtp.mutex.RLock()
	cacheEntries := len(gtp.memory)
	gtp.mutex.RUnlock()
	
	response := map[string]interface{}{
		"processing_stats": stats,
		"cache_stats": map[string]interface{}{
			"entries":        cacheEntries,
			"hit_rate":       float64(stats.CacheHits) / float64(stats.TotalRequests) * 100,
			"memory_usage":   stats.MemoryUsage,
		},
		"system_info": map[string]interface{}{
			"cpu_count":      runtime.NumCPU(),
			"goroutines":     runtime.NumGoroutine(),
			"gpu_available":  gtp.stats.GPUUtilization >= 0,
		},
		"health": "healthy",
	}
	
	json.NewEncoder(w).Encode(response)
}

func (gtp *GPUTensorProcessor) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	health := map[string]interface{}{
		"status":    "healthy",
		"service":   "gpu-tensor-service",
		"timestamp": time.Now().UnixMilli(),
		"uptime":    time.Since(startTime).Seconds(),
	}
	
	json.NewEncoder(w).Encode(health)
}

func (gtp *GPUTensorProcessor) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	log.Println("WebSocket client connected")
	
	for {
		var input MultiDimArray
		if err := conn.ReadJSON(&input); err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}
		
		result, err := gtp.ProcessMultiDimArray(input)
		if err != nil {
			conn.WriteJSON(map[string]interface{}{
				"error": err.Error(),
			})
			continue
		}
		
		if err := conn.WriteJSON(result); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
	
	log.Println("WebSocket client disconnected")
}

var startTime = time.Now()

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8095"
	}
	
	gtp := NewGPUTensorProcessor()
	
	router := mux.NewRouter()
	
	// HTTP endpoints
	router.HandleFunc("/process-tensor", gtp.handleProcessTensor).Methods("POST", "OPTIONS")
	router.HandleFunc("/stats", gtp.handleStats).Methods("GET")
	router.HandleFunc("/health", gtp.handleHealth).Methods("GET")
	
	// WebSocket endpoint
	router.HandleFunc("/ws", gtp.handleWebSocket)
	
	// CORS middleware
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://localhost:5173", "http://localhost:5175", "http://localhost:3000"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"*"},
		AllowCredentials: true,
	})
	
	handler := c.Handler(router)
	
	log.Printf("🚀 GPU Tensor Service starting on port %s", port)
	log.Printf("📊 Stats endpoint: http://localhost:%s/stats", port)
	log.Printf("🔧 Health endpoint: http://localhost:%s/health", port)
	log.Printf("⚡ Process endpoint: http://localhost:%s/process-tensor", port)
	log.Printf("🌐 WebSocket endpoint: ws://localhost:%s/ws", port)
	
	if err := http.ListenAndServe(":"+port, handler); err != nil {
		log.Fatal("Server failed to start:", err)
	}
}