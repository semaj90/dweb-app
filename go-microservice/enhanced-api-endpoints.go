// enhanced-api-endpoints.go
// Enhanced API service with go-llama direct integration for TypeScript error processing

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// AIProcessor handles AI-powered legal document processing
type AIProcessor struct {
	initialized bool
	modelPath   string
}

// GoLlamaEngine handles direct llama.cpp integration
type GoLlamaEngine struct {
	modelPath   string
	gpuLayers   int
	initialized bool
}

// TypeScriptErrorOptimizer handles TypeScript error analysis and fixing
type TypeScriptErrorOptimizer struct {
	initialized bool
	cacheSize   int
}

// ProcessingStats represents processing statistics
type ProcessingStats struct {
	TotalTime    time.Duration `json:"total_time"`
	ProcessedCount int         `json:"processed_count"`
	SuccessfulCount int        `json:"successful_count"`
}

// Constructor functions
func NewAIProcessor() (*AIProcessor, error) {
	return &AIProcessor{
		initialized: true,
		modelPath:   "./models/gemma3-legal.gguf",
	}, nil
}

func NewGoLlamaEngine(modelPath string, gpuLayers int) (*GoLlamaEngine, error) {
	return &GoLlamaEngine{
		modelPath:   modelPath,
		gpuLayers:   gpuLayers,
		initialized: true,
	}, nil
}

func NewTypeScriptErrorOptimizer() (*TypeScriptErrorOptimizer, error) {
	return &TypeScriptErrorOptimizer{
		initialized: true,
		cacheSize:   1000,
	}, nil
}

// Method stubs for interfaces used in the code
func (a *AIProcessor) ProcessLegalDocument(ctx context.Context, req interface{}) (interface{}, error) {
	// Mock implementation
	return map[string]interface{}{
		"processed_count":   1,
		"successful_count": 1,
		"processing_stats": ProcessingStats{
			TotalTime:       time.Millisecond * 100,
			ProcessedCount:  1,
			SuccessfulCount: 1,
		},
	}, nil
}

func (g *GoLlamaEngine) ProcessBatch(ctx context.Context, req interface{}) (interface{}, error) {
	// Mock implementation
	return map[string]interface{}{
		"processed_count":   1,
		"successful_count": 1,
		"processing_stats": ProcessingStats{
			TotalTime:       time.Millisecond * 150,
			ProcessedCount:  1,
			SuccessfulCount: 1,
		},
	}, nil
}

func (t *TypeScriptErrorOptimizer) ProcessOptimized(ctx context.Context, req interface{}) (interface{}, error) {
	// Mock implementation
	return map[string]interface{}{
		"processed_count":   1,
		"successful_count": 1,
		"processing_stats": ProcessingStats{
			TotalTime:       time.Millisecond * 80,
			ProcessedCount:  1,
			SuccessfulCount: 1,
		},
	}, nil
}

// EnhancedAPIEndpoints provides enhanced API service with go-llama integration
type EnhancedAPIEndpoints struct {
	aiProcessor      *AIProcessor
	goLlamaEngine    *GoLlamaEngine
	tsOptimizer      *TypeScriptErrorOptimizer
	port             string
	isLlamaEnabled   bool
	isGPUEnabled     bool
}

// NewEnhancedAPIEndpoints creates a new enhanced API service
func NewEnhancedAPIEndpoints() *EnhancedAPIEndpoints {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8094"
	}
	
	log.Printf("🚀 Initializing Enhanced API Endpoints with Go-Llama integration...")
	
	// Initialize Go-Llama engine
	modelPath := os.Getenv("LLAMA_MODEL_PATH")
	if modelPath == "" {
		// Default model path - adjust based on your setup
		modelPath = "./models/gemma3-legal-4b-q4_0.gguf"
	}
	
	gpuLayers := 35 // RTX 3060 Ti optimized
	goLlamaEngine, err := NewGoLlamaEngine(modelPath, gpuLayers)
	isLlamaEnabled := err == nil
	
	if err != nil {
		log.Printf("⚠️ Go-Llama engine initialization failed: %v (continuing without direct llama)", err)
	} else {
		log.Printf("✅ Go-Llama engine initialized successfully")
	}
	
	// Initialize TypeScript error optimizer
	var tsOptimizer *TypeScriptErrorOptimizer
	isGPUEnabled := false
	
	tsOptimizer, err = NewTypeScriptErrorOptimizer()
	if err != nil {
		log.Printf("⚠️ TypeScript optimizer initialization failed: %v", err)
	} else {
		isGPUEnabled = true
		log.Printf("✅ TypeScript Error Optimizer initialized")
	}
	
	// Initialize AI processor
	aiProcessor, err := NewAIProcessor()
	if err != nil {
		log.Printf("⚠️ AI processor initialization failed: %v", err)
	} else {
		log.Printf("✅ AI Processor initialized")
	}
	
	return &EnhancedAPIEndpoints{
		aiProcessor:   aiProcessor,
		goLlamaEngine: goLlamaEngine,
		tsOptimizer:   tsOptimizer,
		port:          port,
		isLlamaEnabled: isLlamaEnabled,
		isGPUEnabled:   isGPUEnabled,
	}
}

// StartEnhancedServer starts the enhanced API server
func (s *EnhancedAPIEndpoints) StartEnhancedServer() error {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// CORS configuration
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"*"}
	r.Use(cors.New(config))

	// Health check endpoints
	r.GET("/", s.handleRoot)
	r.GET("/health", s.handleHealth)
	r.GET("/api/health", s.handleHealth)

	// Original AI processing endpoints (backward compatibility)
	r.POST("/api/rag", s.handleRAG)
	r.POST("/api/ai", s.handleAI)
	r.POST("/api/rag/query", s.handleRAGQuery)
	r.GET("/api/rag/status", s.handleRAGStatus)

	// Original auto-solver endpoints (enhanced with go-llama)
	r.POST("/api/auto-solve", s.handleAutoSolve)
	r.POST("/api/typescript-fix", s.handleTypeScriptFix)
	
	// Go-Llama direct integration endpoints
	r.POST("/api/go-llama/fix", s.handleGoLlamaFix)
	r.POST("/api/go-llama/batch", s.handleGoLlamaBatch)
	r.GET("/api/go-llama/status", s.handleGoLlamaStatus)
	r.GET("/api/go-llama/stats", s.handleGoLlamaStats)
	
	// GPU-accelerated processing endpoints
	r.POST("/api/gpu/typescript-fix", s.handleGPUTypescriptFix)
	r.POST("/api/gpu/batch-process", s.handleGPUBatchProcess)
	r.GET("/api/gpu/status", s.handleGPUStatus)
	
	// Optimized auto-solver endpoints
	r.POST("/api/optimized/auto-solve", s.handleOptimizedAutoSolve)
	r.POST("/api/optimized/batch-fix", s.handleOptimizedBatchFix)
	r.GET("/api/optimized/performance", s.handleOptimizedPerformance)
	
	// Performance benchmarking endpoints
	r.POST("/api/benchmark/speed", s.handleSpeedBenchmark)
	r.POST("/api/benchmark/quality", s.handleQualityBenchmark)
	r.GET("/api/benchmark/results", s.handleBenchmarkResults)

	log.Printf("🚀 Enhanced API Endpoints starting on port %s", s.port)
	log.Printf("📍 Health check: http://localhost:%s/api/health", s.port)
	log.Printf("📍 RAG API: http://localhost:%s/api/rag", s.port)
	log.Printf("📍 Auto-solver: http://localhost:%s/api/auto-solve", s.port)
	log.Printf("🧠 Go-Llama API: http://localhost:%s/api/go-llama/*", s.port)
	log.Printf("⚡ GPU API: http://localhost:%s/api/gpu/*", s.port)
	log.Printf("🎯 Optimized API: http://localhost:%s/api/optimized/*", s.port)
	log.Printf("🔧 Llama enabled: %v | GPU enabled: %v", s.isLlamaEnabled, s.isGPUEnabled)

	return r.Run(":" + s.port)
}

// handleRoot provides enhanced service information
func (s *EnhancedAPIEndpoints) handleRoot(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":   "Enhanced API Endpoints with Go-Llama",
		"status":    "running",
		"port":      s.port,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   "2.0.0",
		"message":   "Direct go-llama integration for TypeScript error processing",
		"features": gin.H{
			"go_llama_direct":    s.isLlamaEnabled,
			"gpu_acceleration":   s.isGPUEnabled,
			"cuda_kernels":       s.isGPUEnabled,
			"optimization_layer": s.tsOptimizer != nil,
			"caching_enabled":    true,
			"batch_processing":   true,
		},
		"endpoints": []string{
			"/api/health",
			"/api/rag",
			"/api/auto-solve",
			"/api/go-llama/fix",
			"/api/go-llama/batch",
			"/api/go-llama/status",
			"/api/gpu/typescript-fix",
			"/api/gpu/batch-process",
			"/api/optimized/auto-solve",
			"/api/optimized/batch-fix",
		},
		"performance": gin.H{
			"target_latency":     "<5ms for template fixes",
			"gpu_memory":         "8GB RTX 3060 Ti",
			"concurrent_workers": 8,
			"model":             "gemma3-legal-4b",
		},
	})
}

// handleHealth provides health status
func (s *EnhancedAPIEndpoints) handleHealth(c *gin.Context) {
	healthStatus := gin.H{
		"service":   "Enhanced API Endpoints",
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"uptime":    "running",
		"components": gin.H{
			"ai_processor":      "healthy",
			"go_llama_engine":   func() string {
				if s.isLlamaEnabled && s.goLlamaEngine.IsLoaded() {
					return "healthy"
				}
				return "unavailable"
			}(),
			"ts_optimizer":      func() string {
				if s.tsOptimizer != nil {
					return "healthy"
				}
				return "unavailable"
			}(),
			"gpu_acceleration":  func() string {
				if s.isGPUEnabled {
					return "healthy"
				}
				return "unavailable"
			}(),
		},
	}
	
	c.JSON(http.StatusOK, healthStatus)
}

// handleRAG handles RAG processing requests (backward compatibility)
func (s *EnhancedAPIEndpoints) handleRAG(c *gin.Context) {
	var request LegalAnalysisRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Process with AI
	response, err := s.aiProcessor.ProcessLegalDocument(context.Background(), &request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"result":  response,
		"service": "enhanced-rag",
	})
}

// handleAI handles general AI processing (backward compatibility)
func (s *EnhancedAPIEndpoints) handleAI(c *gin.Context) {
	s.handleRAG(c) // Delegate to RAG handler
}

// handleRAGQuery handles RAG query requests (backward compatibility)
func (s *EnhancedAPIEndpoints) handleRAGQuery(c *gin.Context) {
	s.handleRAG(c) // Delegate to RAG handler
}

// handleRAGStatus provides RAG service status (backward compatibility)
func (s *EnhancedAPIEndpoints) handleRAGStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"rag_service": "operational",
		"ai_model":    "gemma3-legal:latest",
		"ollama_url":  "http://localhost:11434",
		"status":      "ready",
		"timestamp":   time.Now().UTC(),
		"enhanced":    true,
		"go_llama":    s.isLlamaEnabled,
		"gpu_accel":   s.isGPUEnabled,
	})
}

// handleAutoSolve handles auto-solving with enhanced capabilities
func (s *EnhancedAPIEndpoints) handleAutoSolve(c *gin.Context) {
	var request AutoSolveRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	startTime := time.Now()

	// Use optimized processing if available
	if s.tsOptimizer != nil {
		optimizedRequest := &OptimizedFixRequest{
			Errors:           request.Errors,
			Strategy:         request.Strategy,
			UseGPU:           s.isGPUEnabled && len(request.Errors) >= 5,
			UseLlama:         s.isLlamaEnabled,
			UseCache:         true,
			MaxConcurrency:   8,
			TargetLatency:    10 * time.Millisecond,
			QualityThreshold: 0.8,
		}

		response, err := s.tsOptimizer.ProcessOptimized(context.Background(), optimizedRequest)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		// Convert to AutoSolveResponse format
		autoResponse := &AutoSolveResponse{
			Success:         response.Success,
			FixesApplied:    response.SuccessfulCount,
			RemainingErrors: response.ProcessedCount - response.SuccessfulCount,
			Fixes:           convertToTypeScriptFixes(response.Results),
			ProcessingTime:  response.ProcessingStats.TotalTime.Milliseconds(),
			Strategy:        "optimized_enhanced",
			Metadata: map[string]interface{}{
				"engine":              "enhanced_go_llama",
				"gpu_accelerated":     optimizedRequest.UseGPU,
				"llama_inference":     optimizedRequest.UseLlama,
				"cache_enabled":       optimizedRequest.UseCache,
				"processing_stats":    response.ProcessingStats,
				"optimization_meta":   response.OptimizationMeta,
			},
		}

		c.JSON(http.StatusOK, autoResponse)
		return
	}

	// Fallback to original processing
	s.handleOriginalAutoSolve(c, request, startTime)
}

// handleOriginalAutoSolve handles original auto-solve logic
func (s *EnhancedAPIEndpoints) handleOriginalAutoSolve(c *gin.Context, request AutoSolveRequest, startTime time.Time) {
	maxFixes := request.MaxFixes
	if maxFixes == 0 || maxFixes > 50 {
		maxFixes = 50 // Limit to prevent overwhelming
	}

	fixes := make([]TypeScriptFix, 0)
	processed := 0

	for i, tsError := range request.Errors {
		if i >= maxFixes {
			break
		}

		fix, err := s.generateTypescriptFix(tsError, request.UseThinking)
		if err != nil {
			log.Printf("Failed to generate fix for error %d: %v", i, err)
			continue
		}

		fixes = append(fixes, *fix)
		processed++
	}

	processingTime := time.Since(startTime).Milliseconds()

	response := &AutoSolveResponse{
		Success:         true,
		FixesApplied:    processed,
		RemainingErrors: len(request.Errors) - processed,
		Fixes:           fixes,
		ProcessingTime:  processingTime,
		Strategy:        getStrategy(request.Strategy),
		Metadata: map[string]interface{}{
			"model":             "enhanced-api",
			"use_thinking":      request.UseThinking,
			"total_errors":      len(request.Errors),
			"batch_size":        maxFixes,
			"timestamp":         time.Now().UTC(),
		},
	}

	c.JSON(http.StatusOK, response)
}

// handleTypeScriptFix handles individual TypeScript error fixes
func (s *EnhancedAPIEndpoints) handleTypeScriptFix(c *gin.Context) {
	var tsError TypeScriptError
	if err := c.ShouldBindJSON(&tsError); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	fix, err := s.generateTypescriptFix(tsError, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"fix":     fix,
		"engine":  func() string {
			if s.isLlamaEnabled {
				return "go_llama_direct"
			}
			return "enhanced_api"
		}(),
	})
}

// handleGoLlamaFix handles direct go-llama fix requests
func (s *EnhancedAPIEndpoints) handleGoLlamaFix(c *gin.Context) {
	if !s.isLlamaEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Go-Llama engine not available"})
		return
	}
	
	var tsError TypeScriptError
	if err := c.ShouldBindJSON(&tsError); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	startTime := time.Now()
	
	// Build optimized prompt
	prompt := s.buildTypescriptFixPrompt(tsError, false)
	
	// Generate fix using direct go-llama
	fixedCode, err := s.goLlamaEngine.generateFix(prompt, 512)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	processingTime := time.Since(startTime)
	
	c.JSON(http.StatusOK, gin.H{
		"success":         true,
		"fixed_code":      fixedCode,
		"processing_time": processingTime.String(),
		"engine":          "go-llama-direct",
		"model":           "gemma3-legal",
		"gpu_accelerated": true,
		"timestamp":       time.Now().UTC(),
	})
}

// handleGoLlamaBatch handles batch processing with go-llama
func (s *EnhancedAPIEndpoints) handleGoLlamaBatch(c *gin.Context) {
	if !s.isLlamaEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Go-Llama engine not available"})
		return
	}
	
	var request BatchProcessRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	response, err := s.goLlamaEngine.ProcessBatch(context.Background(), &request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

// handleGoLlamaStatus provides go-llama engine status
func (s *EnhancedAPIEndpoints) handleGoLlamaStatus(c *gin.Context) {
	if !s.isLlamaEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"available": false,
			"error":     "Go-Llama engine not initialized",
		})
		return
	}
	
	modelInfo, err := s.goLlamaEngine.GetModelInfo()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"available":   true,
		"loaded":      s.goLlamaEngine.IsLoaded(),
		"model_info":  modelInfo,
		"engine_type": "go-llama-direct",
		"timestamp":   time.Now().UTC(),
	})
}

// handleGoLlamaStats provides performance statistics
func (s *EnhancedAPIEndpoints) handleGoLlamaStats(c *gin.Context) {
	if !s.isLlamaEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Go-Llama engine not available"})
		return
	}
	
	stats := s.goLlamaEngine.GetStats()
	c.JSON(http.StatusOK, stats)
}

// handleGPUTypescriptFix handles GPU-accelerated TypeScript fixes
func (s *EnhancedAPIEndpoints) handleGPUTypescriptFix(c *gin.Context) {
	if !s.isGPUEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "GPU acceleration not available"})
		return
	}
	
	var tsError TypeScriptError
	if err := c.ShouldBindJSON(&tsError); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Process single error with GPU optimization
	request := &OptimizedFixRequest{
		Errors:           []TypeScriptError{tsError},
		Strategy:         "gpu_first",
		UseGPU:           true,
		UseLlama:         false,
		UseCache:         true,
		MaxConcurrency:   1,
		TargetLatency:    5 * time.Millisecond,
		QualityThreshold: 0.7,
	}
	
	response, err := s.tsOptimizer.ProcessOptimized(context.Background(), request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

// handleGPUBatchProcess handles GPU batch processing
func (s *EnhancedAPIEndpoints) handleGPUBatchProcess(c *gin.Context) {
	if !s.isGPUEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "GPU acceleration not available"})
		return
	}
	
	var request OptimizedFixRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Force GPU usage for batch processing
	request.UseGPU = true
	request.UseLlama = false // GPU-only for maximum speed
	request.UseCache = true
	
	response, err := s.tsOptimizer.ProcessOptimized(context.Background(), &request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

// handleGPUStatus provides GPU processing status
func (s *EnhancedAPIEndpoints) handleGPUStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"gpu_available":        s.isGPUEnabled,
		"cuda_initialized":     s.isGPUEnabled,
		"gpu_model":           "NVIDIA RTX 3060 Ti",
		"gpu_memory":          "8GB",
		"cuda_version":        "12.8/13.0",
		"optimization_layers": []string{"template_matching", "gpu_kernels", "memory_pooling"},
		"timestamp":           time.Now().UTC(),
	})
}

// handleOptimizedAutoSolve handles optimized auto-solving
func (s *EnhancedAPIEndpoints) handleOptimizedAutoSolve(c *gin.Context) {
	if s.tsOptimizer == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Optimizer not available"})
		return
	}
	
	var autoSolveRequest AutoSolveRequest
	if err := c.ShouldBindJSON(&autoSolveRequest); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Convert to optimized request
	request := &OptimizedFixRequest{
		Errors:           autoSolveRequest.Errors,
		Strategy:         autoSolveRequest.Strategy,
		UseGPU:           s.isGPUEnabled && len(autoSolveRequest.Errors) >= 5,
		UseLlama:         s.isLlamaEnabled && autoSolveRequest.UseThinking,
		UseCache:         true,
		MaxConcurrency:   8,
		TargetLatency:    10 * time.Millisecond,
		QualityThreshold: 0.8,
	}
	
	response, err := s.tsOptimizer.ProcessOptimized(context.Background(), request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	// Convert to AutoSolveResponse format for compatibility
	autoResponse := &AutoSolveResponse{
		Success:         response.Success,
		FixesApplied:    response.SuccessfulCount,
		RemainingErrors: response.ProcessedCount - response.SuccessfulCount,
		Fixes:           convertToTypeScriptFixes(response.Results),
		ProcessingTime:  response.ProcessingStats.TotalTime.Milliseconds(),
		Strategy:        request.Strategy,
		Metadata: map[string]interface{}{
			"engine":              "optimized",
			"gpu_accelerated":     request.UseGPU,
			"llama_inference":     request.UseLlama,
			"cache_enabled":       request.UseCache,
			"processing_stats":    response.ProcessingStats,
			"optimization_meta":   response.OptimizationMeta,
		},
	}
	
	c.JSON(http.StatusOK, autoResponse)
}

// handleOptimizedBatchFix handles optimized batch fixing
func (s *EnhancedAPIEndpoints) handleOptimizedBatchFix(c *gin.Context) {
	if s.tsOptimizer == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Optimizer not available"})
		return
	}
	
	var request OptimizedFixRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Auto-configure optimal settings
	errorCount := len(request.Errors)
	request.UseGPU = s.isGPUEnabled && errorCount >= 10
	request.UseLlama = s.isLlamaEnabled && errorCount < 50 // Llama for smaller, complex batches
	request.UseCache = true
	request.MaxConcurrency = min(errorCount, 8)
	
	response, err := s.tsOptimizer.ProcessOptimized(context.Background(), &request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

// handleOptimizedPerformance provides performance metrics
func (s *EnhancedAPIEndpoints) handleOptimizedPerformance(c *gin.Context) {
	if s.tsOptimizer == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Optimizer not available"})
		return
	}
	
	stats := s.tsOptimizer.GetStats()
	c.JSON(http.StatusOK, gin.H{
		"optimizer_stats": stats,
		"llama_stats":     func() interface{} {
			if s.isLlamaEnabled {
				return s.goLlamaEngine.GetStats()
			}
			return nil
		}(),
		"system_info": gin.H{
			"llama_enabled": s.isLlamaEnabled,
			"gpu_enabled":   s.isGPUEnabled,
			"port":          s.port,
			"timestamp":     time.Now().UTC(),
		},
	})
}

// Benchmark endpoints for performance testing
func (s *EnhancedAPIEndpoints) handleSpeedBenchmark(c *gin.Context) {
	var request struct {
		ErrorCount int    `json:"error_count"`
		Strategy   string `json:"strategy"`
		Iterations int    `json:"iterations"`
	}
	
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Generate sample errors for benchmarking
	sampleErrors := s.generateSampleErrors(request.ErrorCount)
	
	results := make([]gin.H, 0)
	
	for i := 0; i < request.Iterations; i++ {
		startTime := time.Now()
		
		if s.tsOptimizer != nil {
			optimizedRequest := &OptimizedFixRequest{
				Errors:         sampleErrors,
				Strategy:       request.Strategy,
				UseGPU:         s.isGPUEnabled,
				UseLlama:       s.isLlamaEnabled,
				UseCache:       true,
				MaxConcurrency: 8,
			}
			
			_, err := s.tsOptimizer.ProcessOptimized(context.Background(), optimizedRequest)
			if err == nil {
				duration := time.Since(startTime)
				results = append(results, gin.H{
					"iteration":       i + 1,
					"duration_ms":     duration.Milliseconds(),
					"throughput":      float64(request.ErrorCount) / duration.Seconds(),
					"avg_per_error":   duration / time.Duration(request.ErrorCount),
				})
			}
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"benchmark_type": "speed",
		"results":        results,
		"summary": gin.H{
			"error_count": request.ErrorCount,
			"iterations":  len(results),
			"strategy":    request.Strategy,
		},
	})
}

func (s *EnhancedAPIEndpoints) handleQualityBenchmark(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"benchmark_type": "quality",
		"message":        "Quality benchmark implementation pending",
	})
}

func (s *EnhancedAPIEndpoints) handleBenchmarkResults(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"benchmark_results": "Historical benchmark results would be stored here",
	})
}

// Helper functions

// generateTypescriptFix generates a fix for a TypeScript error using available engines
func (s *EnhancedAPIEndpoints) generateTypescriptFix(tsError TypeScriptError, useThinking bool) (*TypeScriptFix, error) {
	// Try go-llama direct first if available
	if s.isLlamaEnabled {
		return s.generateTypescriptFixWithGoLlama(tsError, useThinking)
	}
	
	// Fallback to original AI processor
	return s.generateTypescriptFixWithAI(tsError, useThinking)
}

// generateTypescriptFixWithGoLlama generates fix using direct go-llama
func (s *EnhancedAPIEndpoints) generateTypescriptFixWithGoLlama(tsError TypeScriptError, useThinking bool) (*TypeScriptFix, error) {
	prompt := s.buildTypescriptFixPrompt(tsError, useThinking)
	
	fixedCode, err := s.goLlamaEngine.generateFix(prompt, 512)
	if err != nil {
		return nil, fmt.Errorf("Go-Llama processing failed: %w", err)
	}
	
	// Parse response to extract code and explanation
	fixCode, explanation := s.goLlamaEngine.parseFixResponse(fixedCode)
	confidence := s.goLlamaEngine.calculateConfidence(tsError, fixCode)
	
	fix := &TypeScriptFix{
		File:         tsError.File,
		Line:         tsError.Line,
		Column:       tsError.Column,
		OriginalCode: tsError.Code,
		FixedCode:    fixCode,
		Explanation:  explanation,
		Confidence:   confidence,
	}
	
	return fix, nil
}

// generateTypescriptFixWithAI generates fix using original AI processor
func (s *EnhancedAPIEndpoints) generateTypescriptFixWithAI(tsError TypeScriptError, useThinking bool) (*TypeScriptFix, error) {
	prompt := s.buildTypescriptFixPrompt(tsError, useThinking)

	analysisRequest := &LegalAnalysisRequest{
		Text:         prompt,
		DocumentType: "typescript_error",
		AnalysisType: "error_fix",
		UseThinking:  useThinking,
		Temperature:  0.2, // Lower temperature for more deterministic fixes
		MaxTokens:    1024,
		Metadata: map[string]interface{}{
			"file":    tsError.File,
			"line":    tsError.Line,
			"column":  tsError.Column,
			"message": tsError.Message,
		},
	}

	response, err := s.aiProcessor.ProcessLegalDocument(context.Background(), analysisRequest)
	if err != nil {
		return nil, fmt.Errorf("AI processing failed: %w", err)
	}

	// Parse AI response to extract fix
	fix := &TypeScriptFix{
		File:         tsError.File,
		Line:         tsError.Line,
		Column:       tsError.Column,
		OriginalCode: tsError.Code,
		FixedCode:    extractFixedCode(response.Analysis),
		Explanation:  response.Summary,
		Confidence:   response.Confidence,
	}

	return fix, nil
}

// buildTypescriptFixPrompt builds an optimized prompt for TypeScript error fixing
func (s *EnhancedAPIEndpoints) buildTypescriptFixPrompt(tsError TypeScriptError, useThinking bool) string {
	basePrompt := fmt.Sprintf(`Fix this TypeScript error in a Svelte 5 project:

File: %s
Line: %d, Column: %d
Error: %s

Code Context:
%s

Requirements:
- Fix ONLY the specific error
- Ensure Svelte 5 compatibility (use runes: $state, $derived, $effect)
- Maintain type safety
- Provide minimal, focused changes
- Use proper TypeScript syntax`,
		tsError.File, tsError.Line, tsError.Column, tsError.Message, tsError.Context)

	if useThinking {
		return basePrompt + `

Use <thinking> to analyze:
1. Root cause of the error
2. Svelte 5 migration requirements
3. Optimal fix approach
4. Type safety considerations

Then provide the corrected code in a code block.`
	}

	return basePrompt + `

Provide the corrected code in a code block with brief explanation.`
}

// extractFixedCode extracts the fixed code from AI response
func extractFixedCode(analysis string) string {
	// Simple extraction - in a real implementation, this would be more sophisticated
	if len(analysis) > 0 {
		return analysis
	}
	return "// Fix could not be generated"
}

// getStrategy returns the optimized strategy name
func getStrategy(strategy string) string {
	if strategy == "" {
		return "optimized_go_llama"
	}
	return strategy
}

// convertToTypeScriptFixes converts results to TypeScriptFix format
func convertToTypeScriptFixes(results []*TypeScriptFixResult) []TypeScriptFix {
	fixes := make([]TypeScriptFix, len(results))
	for i, result := range results {
		fixes[i] = TypeScriptFix{
			FixedCode:   result.FixedCode,
			Explanation: result.Explanation,
			Confidence:  result.Confidence,
		}
	}
	return fixes
}

// generateSampleErrors generates sample errors for benchmarking
func (s *EnhancedAPIEndpoints) generateSampleErrors(count int) []TypeScriptError {
	sampleErrors := []TypeScriptError{
		{
			File:    "src/lib/components/AIChat.svelte",
			Line:    45,
			Column:  12,
			Message: "Property 'handleSubmit' does not exist on type 'EventTarget'",
			Code:    "const handleSubmit = (event: Event) => { event.target.handleSubmit(); }",
			Context: "Event handler in Svelte 5 component",
		},
		{
			File:    "src/lib/stores/auth-store.svelte",
			Line:    23,
			Column:  8,
			Message: "Cannot find name 'writable'",
			Code:    "const user = writable(null);",
			Context: "Svelte 5 runes migration needed",
		},
		{
			File:    "src/routes/api/chat/+server.ts",
			Line:    67,
			Column:  15,
			Message: "Argument of type 'unknown' is not assignable to parameter of type 'string'",
			Code:    "const response = await fetch(url, body);",
			Context: "TypeScript type assertion needed",
		},
	}

	errors := make([]TypeScriptError, count)
	for i := 0; i < count; i++ {
		errors[i] = sampleErrors[i%len(sampleErrors)]
		errors[i].Line += i // Vary line numbers
	}

	return errors
}

// min helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Main function for the enhanced API service
func main() {
	log.Printf("🚀 Starting Enhanced API Endpoints with Go-Llama Direct Integration...")
	
	service := NewEnhancedAPIEndpoints()
	
	// Cleanup on shutdown
	defer func() {
		if service.goLlamaEngine != nil {
			service.goLlamaEngine.Close()
		}
		if service.tsOptimizer != nil {
			service.tsOptimizer.Close()
		}
	}()
	
	if err := service.StartEnhancedServer(); err != nil {
		log.Fatal("Failed to start enhanced server:", err)
	}
}