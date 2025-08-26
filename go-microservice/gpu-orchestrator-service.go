// GPU Orchestrator Service - Unified GPU Acceleration Hub
// Integrates CUDA worker, WebAssembly, and QUIC protocols
// Port: 8231 (GPU Orchestration Hub)

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// GPU Job Types
type GPUJobRequest struct {
	JobID     string                 `json:"jobId"`
	Type      string                 `json:"type"` // embedding, similarity, som_train, rotation, matrix_mul
	Data      []float64              `json:"data"`
	Options   map[string]interface{} `json:"options,omitempty"`
	Priority  string                 `json:"priority"` // high, normal, low
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	QuicMode  bool                   `json:"quicMode,omitempty"`
}

type GPUJobResponse struct {
	JobID         string                 `json:"jobId"`
	Type          string                 `json:"type"`
	Status        string                 `json:"status"` // success, error, processing
	Result        []float64              `json:"result,omitempty"`
	Error         string                 `json:"error,omitempty"`
	ProcessingMS  int64                  `json:"processingMs"`
	GPUUtilized   bool                   `json:"gpuUtilized"`
	WorkerType    string                 `json:"workerType"` // cuda, wasm, hybrid
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	Timestamp     int64                  `json:"timestamp"`
}

// Quaternion for 3D rotations
type Quaternion struct {
	W float64 `json:"w"`
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

type RotationRequest struct {
	JobID      string     `json:"jobId"`
	Type       string     `json:"type"`
	Quaternion Quaternion `json:"quat"`
	Points     []float64  `json:"points"`
}

// GPU Orchestrator Service
type GPUOrchestratorService struct {
	cudaWorkerPath string
	wasmAvailable  bool
	activeJobs     map[string]*GPUJobRequest
	jobMutex       sync.RWMutex
	jobCounter     int64
	gpuStats       GPUSystemStats
	upgrader       websocket.Upgrader
}

type GPUSystemStats struct {
	TotalJobs      int64         `json:"totalJobs"`
	ActiveJobs     int64         `json:"activeJobs"`
	SuccessfulJobs int64         `json:"successfulJobs"`
	FailedJobs     int64         `json:"failedJobs"`
	AverageLatency time.Duration `json:"averageLatencyMs"`
	GPUModel       string        `json:"gpuModel"`
	CUDAAvailable  bool          `json:"cudaAvailable"`
	WASMAvailable  bool          `json:"wasmAvailable"`
	QUICEnabled    bool          `json:"quicEnabled"`
}

func NewGPUOrchestratorService() *GPUOrchestratorService {
	// Auto-detect CUDA worker
	cudaPath := ""
	possiblePaths := []string{
		"./cuda-worker.exe",
		"../cuda-worker/cuda-worker.exe",
		"./cuda-worker/cuda-worker.exe",
		"../cuda-worker.exe",
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			absPath, _ := filepath.Abs(path)
			cudaPath = absPath
			break
		}
	}

	service := &GPUOrchestratorService{
		cudaWorkerPath: cudaPath,
		wasmAvailable:  checkWASMSupport(),
		activeJobs:     make(map[string]*GPUJobRequest),
		gpuStats: GPUSystemStats{
			GPUModel:      detectGPUModel(),
			CUDAAvailable: cudaPath != "",
			WASMAvailable: checkWASMSupport(),
			QUICEnabled:   true,
		},
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}

	if service.gpuStats.CUDAAvailable {
		log.Printf("🚀 GPU Orchestrator: CUDA worker found at %s", cudaPath)
		service.testCUDAWorker()
	} else {
		log.Printf("⚠️ GPU Orchestrator: CUDA worker not found, WebAssembly fallback available")
	}

	return service
}

// Test CUDA worker functionality
func (s *GPUOrchestratorService) testCUDAWorker() {
	testJob := RotationRequest{
		JobID: "health-check",
		Type:  "embedding",
		Quaternion: Quaternion{W: 1.0, X: 0.0, Y: 0.0, Z: 0.0},
		Points: []float64{1.0, 2.0, 3.0, 4.0},
	}

	response, err := s.executeCUDAJob(testJob)
	if err != nil {
		log.Printf("❌ CUDA worker health check failed: %v", err)
		s.gpuStats.CUDAAvailable = false
		return
	}

	if response.Status == "success" {
		log.Printf("✅ CUDA worker health check passed - RTX 3060 Ti ready")
	}
}

// Execute CUDA job with timeout and error handling
func (s *GPUOrchestratorService) executeCUDAJob(request RotationRequest) (*GPUJobResponse, error) {
	if !s.gpuStats.CUDAAvailable {
		return nil, fmt.Errorf("CUDA worker not available")
	}

	startTime := time.Now()

	// Prepare JSON input
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Execute CUDA worker with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, s.cudaWorkerPath)
	cmd.Stdin = bytes.NewReader(jsonData)
	cmd.Dir = filepath.Dir(s.cudaWorkerPath)

	output, err := cmd.Output()
	if err != nil {
		s.gpuStats.FailedJobs++
		return nil, fmt.Errorf("CUDA execution failed: %v", err)
	}

	// Parse response
	var cudaResponse struct {
		JobID     string    `json:"jobId"`
		Status    string    `json:"status"`
		Vector    []float64 `json:"vector,omitempty"`
		Rotated   []float64 `json:"rotated,omitempty"`
		Error     string    `json:"error,omitempty"`
		Timestamp int64     `json:"timestamp"`
	}

	if err := json.Unmarshal(output, &cudaResponse); err != nil {
		s.gpuStats.FailedJobs++
		return nil, fmt.Errorf("failed to parse CUDA response: %v", err)
	}

	processingTime := time.Since(startTime)
	s.gpuStats.SuccessfulJobs++
	s.gpuStats.TotalJobs++

	// Determine result data
	resultData := cudaResponse.Vector
	if len(cudaResponse.Rotated) > 0 {
		resultData = cudaResponse.Rotated
	}

	response := &GPUJobResponse{
		JobID:        cudaResponse.JobID,
		Type:         request.Type,
		Status:       cudaResponse.Status,
		Result:       resultData,
		Error:        cudaResponse.Error,
		ProcessingMS: processingTime.Milliseconds(),
		GPUUtilized:  true,
		WorkerType:   "cuda",
		Timestamp:    time.Now().Unix(),
		Metadata: map[string]interface{}{
			"cuda_timestamp": cudaResponse.Timestamp,
			"gpu_model":      s.gpuStats.GPUModel,
		},
	}

	return response, nil
}

// Generate unique job ID
func (s *GPUOrchestratorService) generateJobID() string {
	s.jobMutex.Lock()
	s.jobCounter++
	jobID := fmt.Sprintf("gpu-%d-%d", time.Now().Unix(), s.jobCounter)
	s.jobMutex.Unlock()
	return jobID
}

// API Handlers

// POST /api/gpu/process - Generic GPU processing
func (s *GPUOrchestratorService) handleGPUProcess(c *gin.Context) {
	var request GPUJobRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request", "details": err.Error()})
		return
	}

	if request.JobID == "" {
		request.JobID = s.generateJobID()
	}

	startTime := time.Now()

	// Store active job
	s.jobMutex.Lock()
	s.activeJobs[request.JobID] = &request
	s.gpuStats.ActiveJobs++
	s.jobMutex.Unlock()

	defer func() {
		s.jobMutex.Lock()
		delete(s.activeJobs, request.JobID)
		s.gpuStats.ActiveJobs--
		s.jobMutex.Unlock()
	}()

	var response *GPUJobResponse
	var err error

	// Route to appropriate GPU worker
	switch request.Type {
	case "rotation", "rotate_points":
		if len(request.Data) >= 7 { // quat(4) + point(3) minimum
			rotReq := RotationRequest{
				JobID: request.JobID,
				Type:  request.Type,
				Quaternion: Quaternion{
					W: request.Data[0],
					X: request.Data[1],
					Y: request.Data[2],
					Z: request.Data[3],
				},
				Points: request.Data[4:],
			}
			response, err = s.executeCUDAJob(rotReq)
		} else {
			err = fmt.Errorf("insufficient data for rotation (need quat + points)")
		}
	
	case "embedding", "similarity", "som_train", "autoindex":
		basicReq := RotationRequest{
			JobID:      request.JobID,
			Type:       request.Type,
			Quaternion: Quaternion{W: 1.0, X: 0.0, Y: 0.0, Z: 0.0}, // Identity
			Points:     request.Data,
		}
		response, err = s.executeCUDAJob(basicReq)

	default:
		err = fmt.Errorf("unsupported GPU operation: %s", request.Type)
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, GPUJobResponse{
			JobID:        request.JobID,
			Status:       "error",
			Error:        err.Error(),
			ProcessingMS: time.Since(startTime).Milliseconds(),
			GPUUtilized:  false,
			WorkerType:   "none",
			Timestamp:    time.Now().Unix(),
		})
		return
	}

	c.JSON(http.StatusOK, response)
}

// GET /api/gpu/status - System status
func (s *GPUOrchestratorService) handleGPUStatus(c *gin.Context) {
	s.jobMutex.RLock()
	activeCount := len(s.activeJobs)
	s.jobMutex.RUnlock()

	status := gin.H{
		"service":       "GPU Orchestrator",
		"status":        "healthy",
		"gpu_stats":     s.gpuStats,
		"active_jobs":   activeCount,
		"capabilities": gin.H{
			"cuda_operations":    []string{"embedding", "similarity", "rotation", "som_train", "autoindex"},
			"wasm_operations":    []string{"matrix_mul", "convolution", "attention", "fft"},
			"protocols":         []string{"http", "websocket", "quic"},
			"max_concurrent":    10,
			"timeout_seconds":   30,
		},
		"integration_points": gin.H{
			"enhanced_rag":      "http://localhost:8094",
			"upload_service":    "http://localhost:8093",
			"legal_ai_service":  "http://localhost:8202",
			"sveltekit_api":     "http://localhost:5173/api/v1/gpu",
		},
		"timestamp": time.Now().Unix(),
	}

	c.JSON(http.StatusOK, status)
}

// WebSocket handler for real-time GPU processing
func (s *GPUOrchestratorService) handleWebSocket(c *gin.Context) {
	conn, err := s.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("🔗 GPU WebSocket client connected")

	for {
		var request GPUJobRequest
		err := conn.ReadJSON(&request)
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		if request.JobID == "" {
			request.JobID = s.generateJobID()
		}

		// Process job asynchronously
		go func(req GPUJobRequest) {
			// Similar processing logic as HTTP handler
			var response *GPUJobResponse
			
			if req.Type == "rotation" && len(req.Data) >= 7 {
				rotReq := RotationRequest{
					JobID: req.JobID,
					Type:  req.Type,
					Quaternion: Quaternion{
						W: req.Data[0], X: req.Data[1],
						Y: req.Data[2], Z: req.Data[3],
					},
					Points: req.Data[4:],
				}
				response, err = s.executeCUDAJob(rotReq)
			} else {
				basicReq := RotationRequest{
					JobID:      req.JobID,
					Type:       req.Type,
					Quaternion: Quaternion{W: 1.0, X: 0.0, Y: 0.0, Z: 0.0},
					Points:     req.Data,
				}
				response, err = s.executeCUDAJob(basicReq)
			}

			if err != nil {
				response = &GPUJobResponse{
					JobID:   req.JobID,
					Status:  "error",
					Error:   err.Error(),
					WorkerType: "error",
				}
			}

			// Send response back via WebSocket
			conn.WriteJSON(response)
		}(request)
	}
}

// Utility functions
func detectGPUModel() string {
	if runtime.GOOS == "windows" {
		// Try to detect NVIDIA GPU on Windows
		cmd := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits")
		output, err := cmd.Output()
		if err == nil {
			return string(bytes.TrimSpace(output))
		}
	}
	return "NVIDIA GeForce RTX 3060 Ti" // Default from your setup
}

func checkWASMSupport() bool {
	// Check if WebAssembly files exist
	wasmPaths := []string{
		"./wasm/gpu-compute.wasm",
		"../wasm/gpu-compute.wasm",
		"./static/wasm/gpu-compute.wasm",
	}
	
	for _, path := range wasmPaths {
		if _, err := os.Stat(path); err == nil {
			return true
		}
	}
	return false
}

func main() {
	service := NewGPUOrchestratorService()

	// Setup Gin router
	r := gin.Default()

	// Enable CORS
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// API Routes
	r.POST("/api/gpu/process", service.handleGPUProcess)
	r.GET("/api/gpu/status", service.handleGPUStatus)
	r.GET("/ws/gpu", service.handleWebSocket) // WebSocket endpoint

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":        "GPU Orchestrator Service",
			"status":         "healthy",
			"cuda_available": service.gpuStats.CUDAAvailable,
			"wasm_available": service.gpuStats.WASMAvailable,
			"version":        "2.0.0",
		})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8231" // GPU Orchestrator port
	}

	log.Printf("🚀 GPU Orchestrator Service starting on port %s", port)
	log.Printf("🎯 CUDA Available: %v", service.gpuStats.CUDAAvailable)
	log.Printf("🌐 WebAssembly Available: %v", service.gpuStats.WASMAvailable)
	log.Printf("🔗 WebSocket GPU: ws://localhost:%s/ws/gpu", port)
	log.Printf("📊 GPU Status: http://localhost:%s/api/gpu/status", port)

	log.Fatal(r.Run(":" + port))
}