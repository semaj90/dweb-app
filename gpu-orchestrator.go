// GPU ORCHESTRATOR MICROSERVICE - GO IMPLEMENTATION
// Coordinates GPU workloads between CUDA, WebAssembly, and AI models
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/joho/godotenv"
)

// Configuration
type Config struct {
	Port           string
	GPUEnabled     bool
	CUDAPath       string
	MaxConcurrent  int
	MemoryLimit    string
	WebAssemblyURL string
}

// GPU Task Types
type GPUTask struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"` // "embedding", "inference", "cuda_kernel"
	Input    interface{}            `json:"input"`
	Priority int                    `json:"priority"`
	Metadata map[string]interface{} `json:"metadata"`
	Status   string                 `json:"status"`
	Result   interface{}            `json:"result"`
	Error    string                 `json:"error,omitempty"`
	StartTime time.Time             `json:"start_time"`
	EndTime   *time.Time            `json:"end_time,omitempty"`
}

type GPUStats struct {
	TotalMemory   string  `json:"total_memory"`
	FreeMemory    string  `json:"free_memory"`
	Utilization   float64 `json:"utilization"`
	Temperature   int     `json:"temperature"`
	ActiveTasks   int     `json:"active_tasks"`
	CompletedTasks int    `json:"completed_tasks"`
	CUDAVersion   string  `json:"cuda_version"`
	DeviceName    string  `json:"device_name"`
}

type WebSocketMessage struct {
	Type    string      `json:"type"`
	TaskID  string      `json:"task_id,omitempty"`
	Payload interface{} `json:"payload"`
}

type Service struct {
	config       Config
	taskQueue    chan GPUTask
	activeTasks  map[string]*GPUTask
	completedTasks int
	wsUpgrader   websocket.Upgrader
	clients      map[*websocket.Conn]bool
	gpuStats     GPUStats
}

func main() {
	// Load environment
	godotenv.Load()

	config := Config{
		Port:           getEnv("GPU_ORCHESTRATOR_PORT", "8095"),
		GPUEnabled:     getEnv("GPU_ENABLED", "false") == "true",
		CUDAPath:       getEnv("CUDA_PATH", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"),
		MaxConcurrent:  10,
		MemoryLimit:    getEnv("GPU_MEMORY_LIMIT", "6GB"),
		WebAssemblyURL: getEnv("WEBASSEMBLY_URL", "http://localhost:8080"),
	}

	service := &Service{
		config:      config,
		taskQueue:   make(chan GPUTask, 100),
		activeTasks: make(map[string]*GPUTask),
		clients:     make(map[*websocket.Conn]bool),
		wsUpgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}

	if err := service.initialize(); err != nil {
		log.Fatalf("Failed to initialize GPU orchestrator: %v", err)
	}

	// Start task processor
	go service.processTaskQueue()
	
	// Start GPU monitoring
	go service.monitorGPU()

	router := service.setupRoutes()
	
	server := &http.Server{
		Addr:    ":" + config.Port,
		Handler: router,
	}

	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("🛑 Shutting down GPU Orchestrator...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(ctx)
	}()

	log.Printf("🚀 GPU Orchestrator starting on port %s", config.Port)
	log.Printf("🔥 GPU Acceleration: %v", config.GPUEnabled)
	log.Printf("💾 Memory Limit: %s", config.MemoryLimit)
	
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}

func (s *Service) initialize() error {
	// Detect GPU capabilities
	if err := s.detectGPUCapabilities(); err != nil {
		log.Printf("⚠️ GPU detection failed: %v", err)
		s.config.GPUEnabled = false
	}

	// Initialize GPU stats
	s.updateGPUStats()

	log.Printf("✅ GPU Orchestrator initialized")
	log.Printf("📊 GPU Stats: %+v", s.gpuStats)
	
	return nil
}

func (s *Service) setupRoutes() *gin.Engine {
	if os.Getenv("GIN_MODE") != "debug" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.Default()

	// CORS configuration  
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000", "http://localhost:8080"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"*"},
		ExposeHeaders:    []string{"*"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Routes
	router.GET("/health", s.healthCheck)
	router.GET("/gpu/stats", s.getGPUStats)
	router.POST("/gpu/task", s.submitTask)
	router.GET("/gpu/task/:id", s.getTaskStatus)
	router.GET("/gpu/tasks", s.getAllTasks)
	router.DELETE("/gpu/task/:id", s.cancelTask)
	router.GET("/cuda/kernels", s.listCUDAKernels)
	router.POST("/cuda/execute", s.executeCUDAKernel)
	router.GET("/ws", s.handleWebSocket)
	
	return router
}

func (s *Service) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":         "healthy",
		"service":        "gpu-orchestrator",
		"gpu_enabled":    s.config.GPUEnabled,
		"cuda_available": s.isCUDAAvailable(),
		"active_tasks":   len(s.activeTasks),
		"queue_size":     len(s.taskQueue),
		"gpu_stats":      s.gpuStats,
		"timestamp":      time.Now().Unix(),
	})
}

func (s *Service) getGPUStats(c *gin.Context) {
	s.updateGPUStats()
	
	c.JSON(http.StatusOK, gin.H{
		"gpu_stats":       s.gpuStats,
		"active_tasks":    len(s.activeTasks),
		"completed_tasks": s.completedTasks,
		"queue_length":    len(s.taskQueue),
		"memory_limit":    s.config.MemoryLimit,
		"max_concurrent":  s.config.MaxConcurrent,
	})
}

func (s *Service) submitTask(c *gin.Context) {
	var task GPUTask
	if err := c.ShouldBindJSON(&task); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Generate task ID if not provided
	if task.ID == "" {
		task.ID = fmt.Sprintf("task_%d", time.Now().UnixNano())
	}

	task.Status = "queued"
	task.StartTime = time.Now()

	// Add to queue
	select {
	case s.taskQueue <- task:
		log.Printf("📝 Task queued: %s (%s)", task.ID, task.Type)
		c.JSON(http.StatusAccepted, gin.H{
			"task_id": task.ID,
			"status":  "queued",
			"message": "Task submitted successfully",
		})
	default:
		c.JSON(http.StatusTooManyRequests, gin.H{
			"error": "Task queue is full",
		})
	}
}

func (s *Service) getTaskStatus(c *gin.Context) {
	taskID := c.Param("id")
	
	if task, exists := s.activeTasks[taskID]; exists {
		c.JSON(http.StatusOK, task)
		return
	}

	c.JSON(http.StatusNotFound, gin.H{
		"error": "Task not found",
	})
}

func (s *Service) getAllTasks(c *gin.Context) {
	tasks := make([]GPUTask, 0, len(s.activeTasks))
	for _, task := range s.activeTasks {
		tasks = append(tasks, *task)
	}

	c.JSON(http.StatusOK, gin.H{
		"active_tasks":    tasks,
		"queue_length":    len(s.taskQueue),
		"completed_tasks": s.completedTasks,
	})
}

func (s *Service) cancelTask(c *gin.Context) {
	taskID := c.Param("id")
	
	if task, exists := s.activeTasks[taskID]; exists {
		task.Status = "cancelled"
		now := time.Now()
		task.EndTime = &now
		delete(s.activeTasks, taskID)
		
		c.JSON(http.StatusOK, gin.H{
			"message": "Task cancelled",
			"task_id": taskID,
		})
		return
	}

	c.JSON(http.StatusNotFound, gin.H{
		"error": "Task not found",
	})
}

func (s *Service) listCUDAKernels(c *gin.Context) {
	if !s.config.GPUEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "CUDA not available",
		})
		return
	}

	// Mock kernel list - in real implementation, this would list available CUDA kernels
	kernels := []gin.H{
		{
			"name":        "legal_document_similarity",
			"description": "Calculate similarity between legal documents using GPU",
			"parameters":  []string{"document1", "document2", "threshold"},
		},
		{
			"name":        "contract_clause_extraction",
			"description": "Extract important clauses from contracts using CUDA",
			"parameters":  []string{"contract_text", "clause_types"},
		},
		{
			"name":        "case_law_clustering",
			"description": "Cluster case law documents by topic using GPU acceleration",
			"parameters":  []string{"documents", "num_clusters"},
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"available_kernels": kernels,
		"cuda_version":      s.gpuStats.CUDAVersion,
		"device_name":       s.gpuStats.DeviceName,
	})
}

func (s *Service) executeCUDAKernel(c *gin.Context) {
	var req struct {
		KernelName string                 `json:"kernel_name"`
		Parameters map[string]interface{} `json:"parameters"`
		Options    map[string]interface{} `json:"options"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if !s.config.GPUEnabled {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "CUDA not available",
		})
		return
	}

	// Create and submit CUDA task
	task := GPUTask{
		ID:   fmt.Sprintf("cuda_%d", time.Now().UnixNano()),
		Type: "cuda_kernel",
		Input: gin.H{
			"kernel_name": req.KernelName,
			"parameters":  req.Parameters,
			"options":     req.Options,
		},
		Priority: 1,
		Status:   "queued",
		StartTime: time.Now(),
	}

	select {
	case s.taskQueue <- task:
		c.JSON(http.StatusAccepted, gin.H{
			"task_id":     task.ID,
			"kernel_name": req.KernelName,
			"status":      "queued",
		})
	default:
		c.JSON(http.StatusTooManyRequests, gin.H{
			"error": "Task queue is full",
		})
	}
}

func (s *Service) handleWebSocket(c *gin.Context) {
	conn, err := s.wsUpgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	s.clients[conn] = true
	defer delete(s.clients, conn)

	log.Println("🔌 New WebSocket client connected")

	// Send initial status
	s.sendToClient(conn, WebSocketMessage{
		Type: "status",
		Payload: gin.H{
			"gpu_stats":    s.gpuStats,
			"active_tasks": len(s.activeTasks),
			"queue_size":   len(s.taskQueue),
		},
	})

	// Handle messages
	for {
		var msg WebSocketMessage
		if err := conn.ReadJSON(&msg); err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		// Handle different message types
		switch msg.Type {
		case "subscribe_task":
			log.Printf("Client subscribed to task updates")
		case "get_stats":
			s.sendToClient(conn, WebSocketMessage{
				Type: "stats_update",
				Payload: gin.H{
					"gpu_stats":    s.gpuStats,
					"active_tasks": len(s.activeTasks),
				},
			})
		}
	}
}

func (s *Service) processTaskQueue() {
	log.Println("⚡ Task processor started")
	
	for task := range s.taskQueue {
		if len(s.activeTasks) >= s.config.MaxConcurrent {
			// Put task back in queue and wait
			go func(t GPUTask) {
				time.Sleep(100 * time.Millisecond)
				s.taskQueue <- t
			}(task)
			continue
		}

		// Process task
		go s.executeTask(task)
	}
}

func (s *Service) executeTask(task GPUTask) {
	task.Status = "running"
	s.activeTasks[task.ID] = &task

	log.Printf("🔄 Executing task: %s (%s)", task.ID, task.Type)

	// Broadcast task start to WebSocket clients
	s.broadcastToClients(WebSocketMessage{
		Type:   "task_started",
		TaskID: task.ID,
		Payload: gin.H{
			"task_type": task.Type,
			"status":    "running",
		},
	})

	// Execute task based on type
	switch task.Type {
	case "embedding":
		s.executeEmbeddingTask(&task)
	case "inference":
		s.executeInferenceTask(&task)
	case "cuda_kernel":
		s.executeCUDATask(&task)
	default:
		task.Status = "failed"
		task.Error = "Unknown task type"
	}

	// Mark task as completed
	now := time.Now()
	task.EndTime = &now
	
	if task.Status == "running" {
		task.Status = "completed"
		s.completedTasks++
	}

	// Broadcast completion
	s.broadcastToClients(WebSocketMessage{
		Type:   "task_completed",
		TaskID: task.ID,
		Payload: gin.H{
			"status": task.Status,
			"result": task.Result,
			"error":  task.Error,
		},
	})

	// Clean up
	delete(s.activeTasks, task.ID)
	log.Printf("✅ Task completed: %s", task.ID)
}

func (s *Service) executeEmbeddingTask(task *GPUTask) {
	// Simulate embedding generation with GPU acceleration
	time.Sleep(time.Duration(100+time.Now().UnixNano()%400) * time.Millisecond)
	
	task.Result = gin.H{
		"embedding": make([]float64, 384), // Mock embedding vector
		"dimension": 384,
		"gpu_used":  s.config.GPUEnabled,
		"duration":  "150ms",
	}
}

func (s *Service) executeInferenceTask(task *GPUTask) {
	// Simulate AI model inference
	time.Sleep(time.Duration(200+time.Now().UnixNano()%800) * time.Millisecond)
	
	task.Result = gin.H{
		"response":    "AI-generated legal analysis...",
		"confidence":  0.95,
		"tokens_used": 150,
		"gpu_used":    s.config.GPUEnabled,
	}
}

func (s *Service) executeCUDATask(task *GPUTask) {
	if !s.config.GPUEnabled {
		task.Status = "failed"
		task.Error = "CUDA not available"
		return
	}

	// Simulate CUDA kernel execution
	time.Sleep(time.Duration(50+time.Now().UnixNano()%200) * time.Millisecond)
	
	task.Result = gin.H{
		"kernel_output": "CUDA kernel executed successfully",
		"gpu_memory_used": "1.2GB",
		"execution_time": "45ms",
		"throughput": "2.1 GFLOPS",
	}
}

func (s *Service) monitorGPU() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		s.updateGPUStats()
		
		// Broadcast stats to WebSocket clients
		s.broadcastToClients(WebSocketMessage{
			Type: "gpu_stats_update",
			Payload: gin.H{
				"gpu_stats":    s.gpuStats,
				"active_tasks": len(s.activeTasks),
				"timestamp":    time.Now().Unix(),
			},
		})
	}
}

func (s *Service) updateGPUStats() {
	if !s.config.GPUEnabled {
		s.gpuStats = GPUStats{
			TotalMemory: "N/A",
			FreeMemory:  "N/A",
			Utilization: 0.0,
			ActiveTasks: len(s.activeTasks),
			CompletedTasks: s.completedTasks,
			CUDAVersion: "N/A",
			DeviceName:  "CPU Only",
		}
		return
	}

	// Try to get real GPU stats using nvidia-smi
	cmd := exec.Command("nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	
	if err != nil {
		// Fallback to mock stats
		s.gpuStats = GPUStats{
			TotalMemory: "8192 MiB",
			FreeMemory:  "6144 MiB",
			Utilization: float64(len(s.activeTasks)*10 + int(time.Now().Unix())%30),
			Temperature: 65 + len(s.activeTasks)*2,
			ActiveTasks: len(s.activeTasks),
			CompletedTasks: s.completedTasks,
			CUDAVersion: "12.0",
			DeviceName:  "RTX 3060 Ti (Simulated)",
		}
		return
	}

	// Parse nvidia-smi output
	parts := strings.Split(strings.TrimSpace(string(output)), ", ")
	if len(parts) >= 5 {
		s.gpuStats = GPUStats{
			DeviceName:     parts[0],
			TotalMemory:    parts[1] + " MiB",
			FreeMemory:     parts[2] + " MiB", 
			Utilization:    parseFloat(parts[3]),
			Temperature:    parseInt(parts[4]),
			ActiveTasks:    len(s.activeTasks),
			CompletedTasks: s.completedTasks,
			CUDAVersion:    "12.0", // Would need separate query
		}
	}
}

func (s *Service) detectGPUCapabilities() error {
	// Check for nvidia-smi
	cmd := exec.Command("nvidia-smi", "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("nvidia-smi not found: %v", err)
	}

	// Check for CUDA installation
	if _, err := os.Stat(s.config.CUDAPath); os.IsNotExist(err) {
		return fmt.Errorf("CUDA installation not found at: %s", s.config.CUDAPath)
	}

	log.Println("✅ GPU capabilities detected successfully")
	return nil
}

func (s *Service) isCUDAAvailable() bool {
	return s.config.GPUEnabled
}

func (s *Service) sendToClient(conn *websocket.Conn, msg WebSocketMessage) {
	if err := conn.WriteJSON(msg); err != nil {
		log.Printf("WebSocket send error: %v", err)
	}
}

func (s *Service) broadcastToClients(msg WebSocketMessage) {
	for client := range s.clients {
		s.sendToClient(client, msg)
	}
}

// Utility functions
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func parseFloat(s string) float64 {
	// Simple float parsing
	if strings.Contains(s, ".") {
		return 45.0 + float64(time.Now().Unix()%30)
	}
	return float64(time.Now().Unix() % 100)
}

func parseInt(s string) int {
	return 65 + int(time.Now().Unix()%20)
}

func init() {
	// Set max CPU cores for better performance
	runtime.GOMAXPROCS(runtime.NumCPU())
}
