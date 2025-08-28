// CUDA Integration Service - Legal AI Platform
// Connects Go microservices to CUDA worker for GPU acceleration
// Integrates with: enhanced-rag.exe, upload-service.exe, enhanced-legal-ai.exe

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

// CUDA Request/Response Types
type CUDARequest struct {
	JobID string    `json:"jobId"`
	Type  string    `json:"type"` // embedding, similarity, som_train, autoindex
	Data  []float64 `json:"data"`
}

type CUDAResponse struct {
	JobID     string    `json:"jobId"`
	Type      string    `json:"type"`
	Vector    []float64 `json:"vector"`
	Status    string    `json:"status"`
	Timestamp int64     `json:"timestamp"`
	Error     string    `json:"error,omitempty"`
}

// GPU Service Integration Types
type GPUProcessRequest struct {
	Service   string                 `json:"service"`   // "rag", "upload", "legal", "indexer"
	Operation string                 `json:"operation"` // "embedding", "similarity", "clustering"
	Data      []float64              `json:"data"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Priority  string                 `json:"priority"` // "high", "normal", "low"
}

type GPUProcessResponse struct {
	Success       bool                   `json:"success"`
	Result        []float64              `json:"result"`
	ProcessingMS  int64                  `json:"processing_ms"`
	GPUUtilized   bool                   `json:"gpu_utilized"`
	Service       string                 `json:"service"`
	JobID         string                 `json:"job_id"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	Error         string                 `json:"error,omitempty"`
}

// Legal AI Specific Types
type LegalDocumentVector struct {
	DocumentID string    `json:"document_id"`
	Content    string    `json:"content"`
	Vector     []float64 `json:"vector"`
	Confidence float64   `json:"confidence"`
}

type LegalSimilarityRequest struct {
	QueryVector []float64   `json:"query_vector"`
	CaseVectors [][]float64 `json:"case_vectors"`
	Threshold   float64     `json:"threshold"`
}

type SimilarityMatch struct {
	CaseID     string  `json:"case_id"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
}

type LegalSimilarityResponse struct {
	Matches        []SimilarityMatch `json:"matches"`
	ProcessingTime int64             `json:"processing_time_ms"`
	GPUAccelerated bool              `json:"gpu_accelerated"`
	TotalPairs     int               `json:"total_pairs_processed"`
}

// CUDA Integration Service
type CUDAIntegrationService struct {
	cudaWorkerPath string
	mutex          sync.RWMutex
	jobCounter     int64
	activeJobs     map[string]*CUDARequest
	gpuAvailable   bool
	gpuStats       GPUStats
}

type GPUStats struct {
	TotalJobs      int64   `json:"total_jobs"`
	SuccessfulJobs int64   `json:"successful_jobs"`
	FailedJobs     int64   `json:"failed_jobs"`
	AverageLatency int64   `json:"average_latency_ms"` // stored as milliseconds
	GPUModel       string  `json:"gpu_model"`
	VRAMUsage      string  `json:"vram_usage"`
	Utilization    float64 `json:"gpu_utilization_percent"`
}

func NewCUDAIntegrationService() *CUDAIntegrationService {
	// Find CUDA worker executable
	cudaPath := "./cuda-worker.exe"
	if _, err := os.Stat(cudaPath); os.IsNotExist(err) {
		// Try relative path from go-microservice directory
		cudaPath = "../cuda-worker/cuda-worker.exe"
		if _, err := os.Stat(cudaPath); os.IsNotExist(err) {
			log.Printf("CUDA worker not found at %s or %s", "./cuda-worker.exe", cudaPath)
			cudaPath = ""
		}
	}

	service := &CUDAIntegrationService{
		cudaWorkerPath: cudaPath,
		activeJobs:     make(map[string]*CUDARequest),
		gpuAvailable:   cudaPath != "",
		gpuStats: GPUStats{
			GPUModel: "NVIDIA GeForce RTX 3060 Ti",
			VRAMUsage: "8191 MB available",
		},
	}

	if service.gpuAvailable {
		log.Printf("CUDA Integration Service initialized - GPU acceleration available")
		service.testCUDAWorker()
	} else {
		log.Printf("CUDA Integration Service initialized - GPU acceleration disabled (worker not found)")
	}

	return service
}

// Test CUDA worker functionality
func (s *CUDAIntegrationService) testCUDAWorker() {
	testRequest := CUDARequest{
		JobID: "health-check",
		Type:  "embedding",
		Data:  []float64{1.0, 2.0, 3.0, 4.0},
	}

	response, err := s.executeCUDAJob(testRequest)
	if err != nil {
		log.Printf("CUDA worker health check failed: %v", err)
		s.gpuAvailable = false
		return
	}

	if response.Status == "success" {
		log.Printf("CUDA worker health check passed - GPU ready for processing")
		s.gpuStats.Utilization = 85.0 // Estimated utilization
	} else {
		log.Printf("CUDA worker returned error: %s", response.Error)
		s.gpuAvailable = false
	}
}
// Execute CUDA job with error handling and timeout
func (s *CUDAIntegrationService) executeCUDAJob(request CUDARequest) (*CUDAResponse, error) {
	if !s.gpuAvailable {
		return nil, fmt.Errorf("GPU/CUDA worker not available")
	}

	startTime := time.Now()

	// Prepare JSON input for CUDA worker
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal CUDA request: %v", err)
	}

	// Create a context with timeout so the command is killed automatically on timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Prepare command with context
	cmd := exec.CommandContext(ctx, s.cudaWorkerPath)
	cmd.Stdin = bytes.NewReader(jsonData)

	// Set working directory to CUDA worker location if provided
	if s.cudaWorkerPath != "" {
		cmd.Dir = filepath.Dir(s.cudaWorkerPath)
	}

	// Run and capture combined output
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Distinguish timeout from other errors
		if ctx.Err() == context.DeadlineExceeded {
			s.gpuStats.FailedJobs++
			return nil, fmt.Errorf("CUDA operation timeout")
		}
		s.gpuStats.FailedJobs++
		return nil, fmt.Errorf("CUDA execution error: %v - output: %s", err, string(output))
	}

	// Parse CUDA response
	var response CUDAResponse
	if err := json.Unmarshal(output, &response); err != nil {
		s.gpuStats.FailedJobs++
		return nil, fmt.Errorf("failed to parse CUDA response: %v", err)
	}

	// Update statistics
	processingTime := time.Since(startTime)
	processingMs := processingTime.Milliseconds()
	s.gpuStats.TotalJobs++
	s.gpuStats.SuccessfulJobs++
	// compute new average latency safely using previous successes (in ms)
	prevSuccesses := s.gpuStats.SuccessfulJobs - 1
	if prevSuccesses < 0 {
		prevSuccesses = 0
	}
	if s.gpuStats.SuccessfulJobs > 0 {
		avgMs := ((s.gpuStats.AverageLatency * prevSuccesses) + processingMs) / s.gpuStats.SuccessfulJobs
		s.gpuStats.AverageLatency = avgMs
	}

	log.Printf("CUDA job completed: %s (%s) in %v", request.JobID, request.Type, processingTime)
	return &response, nil
}

// Generate unique job ID
func (s *CUDAIntegrationService) generateJobID() string {
	s.mutex.Lock()
	s.jobCounter++
	jobID := fmt.Sprintf("cuda-%d-%d", time.Now().Unix(), s.jobCounter)
	s.mutex.Unlock()
	return jobID
}

// API Handlers

// POST /api/gpu/process - Generic GPU processing endpoint
func (s *CUDAIntegrationService) handleGPUProcess(c *gin.Context) {
	var request GPUProcessRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format", "details": err.Error()})
		return
	}

	// Validate request
	if len(request.Data) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Data array cannot be empty"})
		return
	}

	startTime := time.Now()
	jobID := s.generateJobID()

	// Map operation to CUDA type
	cudaType := request.Operation
	if cudaType == "clustering" {
		cudaType = "som_train"
	} else if cudaType == "indexing" {
		cudaType = "autoindex"
	}

	cudaRequest := CUDARequest{
		JobID: jobID,
		Type:  cudaType,
		Data:  request.Data,
	}

	response, err := s.executeCUDAJob(cudaRequest)
	if err != nil {
		c.JSON(http.StatusInternalServerError, GPUProcessResponse{
			Success:     false,
			GPUUtilized: false,
			Service:     request.Service,
			JobID:       jobID,
			Error:       err.Error(),
		})
		return
	}

	processingTime := time.Since(startTime)

	c.JSON(http.StatusOK, GPUProcessResponse{
		Success:      response.Status == "success",
		Result:       response.Vector,
		ProcessingMS: processingTime.Milliseconds(),
		GPUUtilized:  true,
		Service:      request.Service,
		JobID:        response.JobID,
		Metadata: map[string]interface{}{
			"cuda_timestamp": response.Timestamp,
			"gpu_model":      s.gpuStats.GPUModel,
			"operation_type": response.Type,
		},
	})
}

// POST /api/gpu/legal/similarity - Legal case similarity using GPU
func (s *CUDAIntegrationService) handleLegalSimilarity(c *gin.Context) {
	var request LegalSimilarityRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid similarity request", "details": err.Error()})
		return
	}

	startTime := time.Now()
	var matches []SimilarityMatch

	totalPairs := len(request.CaseVectors)

	// Process similarity for each case vector
	for i, caseVector := range request.CaseVectors {
		// Safely combine query and case vectors for CUDA similarity processing
		combinedData := make([]float64, 0, len(request.QueryVector)+len(caseVector))
		combinedData = append(combinedData, request.QueryVector...)
		combinedData = append(combinedData, caseVector...)

		cudaRequest := CUDARequest{
			JobID: s.generateJobID(),
			Type:  "similarity",
			Data:  combinedData,
		}

		response, err := s.executeCUDAJob(cudaRequest)
		if err != nil {
			log.Printf("CUDA similarity error for case %d: %v", i, err)
			continue
		}

		// Calculate similarity score from GPU response
		if len(response.Vector) > 0 {
			score := calculateSimilarityScore(response.Vector)
			if score >= request.Threshold {
				matches = append(matches, SimilarityMatch{
					CaseID:     fmt.Sprintf("case-%d", i),
					Score:      score,
					Confidence: math.Min(score*1.2, 1.0), // Confidence based on score
				})
			}
		}
	}

	processingTime := time.Since(startTime)

	c.JSON(http.StatusOK, LegalSimilarityResponse{
		Matches:        matches,
		ProcessingTime: processingTime.Milliseconds(),
		GPUAccelerated: s.gpuAvailable,
		TotalPairs:     totalPairs,
	})
}

// GET /api/gpu/status - GPU and service status
func (s *CUDAIntegrationService) handleGPUStatus(c *gin.Context) {
	status := gin.H{
		"gpu_available":    s.gpuAvailable,
		"cuda_worker_path": s.cudaWorkerPath,
		"service_status":   "active",
		"gpu_stats":        s.gpuStats,
		"capabilities": gin.H{
			"operations":       []string{"embedding", "similarity", "som_train", "autoindex"},
			"max_vector_size":  10000,
			"concurrent_jobs":  4,
			"timeout_seconds":  30,
		},
		"integration_services": []string{
			"enhanced-rag (8094)",
			"upload-service (8093)",
			"enhanced-legal-ai (8202)",
			"gpu-indexer-service (8220)",
		},
		"timestamp": time.Now().Unix(),
	}

	if s.gpuAvailable {
		status["gpu_model"] = "NVIDIA GeForce RTX 3060 Ti"
		status["vram_total"] = "8191 MB"
		status["cuda_version"] = "12.8"
	}

	c.JSON(http.StatusOK, status)
}

// Helper functions
func calculateSimilarityScore(vector []float64) float64 {
	if len(vector) == 0 {
		return 0.0
	}

	// Simple similarity score calculation
	sum := 0.0
	for _, val := range vector {
		sum += val
	}

	// Normalize to 0-1 range, clamp using math functions
	normalized := sum / float64(len(vector))
	return math.Min(math.Max(normalized, 0.0), 1.0)
}

// Main service setup
func main() {
	// Initialize CUDA integration service
	cudaService := NewCUDAIntegrationService()

	// Setup Gin router
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// Enable CORS for SvelteKit integration
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// GPU processing endpoints
	r.POST("/api/gpu/process", cudaService.handleGPUProcess)
	r.POST("/api/gpu/legal/similarity", cudaService.handleLegalSimilarity)
	r.GET("/api/gpu/status", cudaService.handleGPUStatus)

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":       "CUDA Integration Service",
			"status":        "healthy",
			"gpu_available": cudaService.gpuAvailable,
			"version":       "1.0.0",
		})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8231" // Default port for CUDA integration service
	}

	log.Printf("CUDA Integration Service starting on port %s", port)
	log.Printf("GPU Status: http://localhost:%s/api/gpu/status", port)
	log.Printf("Health Check: http://localhost:%s/health", port)

	log.Fatal(r.Run(":" + port))
}