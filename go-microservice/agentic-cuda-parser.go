//go:build legacy
// +build legacy

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
	"strings"
	"regexp"
)

// Advanced Agentic Programming System with CUDA + Tensor Processing
type AgenticSystem struct {
	CUDAEnabled    bool             `json:"cuda_enabled"`
	TensorCore     *TensorCore      `json:"tensor_core"`
	SelfOrgMap     *SelfOrgMap      `json:"self_organizing_map"`
	JSONLogger     *JSONLogger      `json:"json_logger"`
	FileIndexer    *FileIndexer     `json:"file_indexer"`
	MCPIntegrator  *MCPIntegrator   `json:"mcp_integrator"`
	AutoGenAgent   *AutoGenAgent    `json:"autogen_agent"`
	TodoManager    *ConcurrentTodo  `json:"todo_manager"`
	NetworkLayer   *NetworkLayer    `json:"network_layer"`
	mu             sync.RWMutex
}

// Tensor Core for deep parallelism
type TensorCore struct {
	Devices      []CUDADevice    `json:"devices"`
	ActiveStreams int            `json:"active_streams"`
	MemoryPool    int64          `json:"memory_pool_mb"`
	Operations    []TensorOp     `json:"operations"`
}

type CUDADevice struct {
	ID           int    `json:"id"`
	Name         string `json:"name"`
	MemoryMB     int64  `json:"memory_mb"`
	ComputeCaps  string `json:"compute_capability"`
	StreamCount  int    `json:"stream_count"`
}

type TensorOp struct {
	Type        string                 `json:"type"`
	InputShape  []int                  `json:"input_shape"`
	OutputShape []int                  `json:"output_shape"`
	Duration    time.Duration          `json:"duration"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Self-Organizing Map for recommendations
type SelfOrgMap struct {
	Width       int                    `json:"width"`
	Height      int                    `json:"height"`
	Neurons     [][]Neuron            `json:"neurons"`
	LearningRate float64              `json:"learning_rate"`
	Radius      float64               `json:"radius"`
	Iteration   int                   `json:"iteration"`
	Patterns    []Pattern             `json:"patterns"`
}

type Neuron struct {
	Weights      []float64             `json:"weights"`
	Activations  int                   `json:"activations"`
	LastUpdated  time.Time             `json:"last_updated"`
}

type Pattern struct {
	Input       []float64             `json:"input"`
	Category    string                `json:"category"`
	Confidence  float64               `json:"confidence"`
	Timestamp   time.Time             `json:"timestamp"`
}

// JSON Logging system
type JSONLogger struct {
	LogFile     string                `json:"log_file"`
	NetEnabled  bool                  `json:"net_enabled"`
	HTTPEnabled bool                  `json:"http_enabled"`
	Filters     []LogFilter           `json:"filters"`
	Streams     map[string]*LogStream `json:"streams"`
	mu          sync.Mutex
}

type LogFilter struct {
	Level    string `json:"level"`
	Category string `json:"category"`
	Regex    string `json:"regex"`
}

type LogStream struct {
	Name        string    `json:"name"`
	Active      bool      `json:"active"`
	MessageCount int64    `json:"message_count"`
	LastMessage time.Time `json:"last_message"`
}

// File Indexer for best practices
type FileIndexer struct {
	RootPaths       []string          `json:"root_paths"`
	IndexedFiles    map[string]FileEntry `json:"indexed_files"`
	Extensions      []string          `json:"extensions"`
	LastScan        time.Time         `json:"last_scan"`
	BestPractices   []BestPractice    `json:"best_practices"`
	ImageAnalysis   bool              `json:"image_analysis"`
}

type FileEntry struct {
	Path         string            `json:"path"`
	Size         int64             `json:"size"`
	ModTime      time.Time         `json:"mod_time"`
	Type         string            `json:"type"`
	Content      string            `json:"content,omitempty"`
	Metadata     map[string]string `json:"metadata"`
	Patterns     []string          `json:"patterns"`
}

type BestPractice struct {
	Category     string    `json:"category"`
	Description  string    `json:"description"`
	Source       string    `json:"source"`
	Confidence   float64   `json:"confidence"`
	Examples     []string  `json:"examples"`
	LastUpdated  time.Time `json:"last_updated"`
}

// MCP Context7 Integrator
type MCPIntegrator struct {
	Endpoints    map[string]string     `json:"endpoints"`
	Active       bool                  `json:"active"`
	Cache        map[string]MCPResult  `json:"cache"`
	LastSync     time.Time             `json:"last_sync"`
}

type MCPResult struct {
	Query        string                 `json:"query"`
	Result       map[string]interface{} `json:"result"`
	Timestamp    time.Time              `json:"timestamp"`
	CacheExpiry  time.Time              `json:"cache_expiry"`
}

// AutoGen Agent
type AutoGenAgent struct {
	Agents       []Agent               `json:"agents"`
	Conversations []Conversation       `json:"conversations"`
	Active       bool                  `json:"active"`
}

type Agent struct {
	Name         string                `json:"name"`
	Role         string                `json:"role"`
	Capabilities []string              `json:"capabilities"`
	Model        string                `json:"model"`
	Status       string                `json:"status"`
}

type Conversation struct {
	ID           string                `json:"id"`
	Participants []string              `json:"participants"`
	Messages     []Message             `json:"messages"`
	StartTime    time.Time             `json:"start_time"`
	EndTime      time.Time             `json:"end_time,omitempty"`
}

type Message struct {
	From         string                `json:"from"`
	To           string                `json:"to"`
	Content      string                `json:"content"`
	Type         string                `json:"type"`
	Timestamp    time.Time             `json:"timestamp"`
}

// Concurrent Todo Manager
type ConcurrentTodo struct {
	Tasks        []Task                `json:"tasks"`
	Agents       []TodoAgent           `json:"agents"`
	Scheduler    *TaskScheduler        `json:"scheduler"`
	Active       bool                  `json:"active"`
}

type Task struct {
	ID           string                `json:"id"`
	Title        string                `json:"title"`
	Description  string                `json:"description"`
	Priority     int                   `json:"priority"`
	Status       string                `json:"status"`
	AssignedTo   string                `json:"assigned_to"`
	Dependencies []string              `json:"dependencies"`
	CreatedAt    time.Time             `json:"created_at"`
	UpdatedAt    time.Time             `json:"updated_at"`
	CompletedAt  time.Time             `json:"completed_at,omitempty"`
	Metadata     map[string]interface{} `json:"metadata"`
}

type TodoAgent struct {
	Name         string                `json:"name"`
	Type         string                `json:"type"` // mcp, file_indexer, best_practices
	Status       string                `json:"status"`
	CurrentTask  string                `json:"current_task"`
	Performance  AgentPerformance      `json:"performance"`
}

type AgentPerformance struct {
	TasksCompleted int           `json:"tasks_completed"`
	AvgDuration    time.Duration `json:"avg_duration"`
	SuccessRate    float64       `json:"success_rate"`
	LastActive     time.Time     `json:"last_active"`
}

type TaskScheduler struct {
	Queue        []string              `json:"queue"`
	Running      map[string]bool       `json:"running"`
	MaxConcurrent int                  `json:"max_concurrent"`
}

// Network Layer
type NetworkLayer struct {
	HTTPServer   *http.Server          `json:"-"`
	Port         int                   `json:"port"`
	Endpoints    map[string]string     `json:"endpoints"`
	WebSockets   map[string]bool       `json:"websockets"`
	ActiveConns  int                   `json:"active_connections"`
}

// Initialize the Agentic System
func NewAgenticSystem() *AgenticSystem {
	return &AgenticSystem{
		CUDAEnabled: checkCUDAAvailability(),
		TensorCore:  initTensorCore(),
		SelfOrgMap:  initSelfOrgMap(20, 20, 0.1, 3.0),
		JSONLogger:  initJSONLogger("agentic-system.jsonl"),
		FileIndexer: initFileIndexer([]string{
			"./sveltekit-frontend",
			"./go-microservice",
			"../scripts",
		}),
		MCPIntegrator: initMCPIntegrator(),
		AutoGenAgent:  initAutoGenAgent(),
		TodoManager:   initConcurrentTodo(),
		NetworkLayer:  initNetworkLayer(8082),
	}
}

// CUDA Detection and Initialization
func checkCUDAAvailability() bool {
	// Mock CUDA detection - in real implementation, would use CGO bindings
	return runtime.GOOS == "windows" // Assume CUDA available on Windows
}

func initTensorCore() *TensorCore {
	devices := []CUDADevice{{
		ID:          0,
		Name:        "Mock RTX 4090",
		MemoryMB:    24000,
		ComputeCaps: "8.9",
		StreamCount: 128,
	}}
	
	return &TensorCore{
		Devices:       devices,
		ActiveStreams: 32,
		MemoryPool:    8192, // 8GB
		Operations:    make([]TensorOp, 0),
	}
}

// Self-Organizing Map Implementation
func initSelfOrgMap(width, height int, learningRate, radius float64) *SelfOrgMap {
	neurons := make([][]Neuron, height)
	for i := range neurons {
		neurons[i] = make([]Neuron, width)
		for j := range neurons[i] {
			neurons[i][j] = Neuron{
				Weights:     make([]float64, 10), // 10-dimensional feature space
				Activations: 0,
				LastUpdated: time.Now(),
			}
			
			// Initialize random weights
			for k := range neurons[i][j].Weights {
				neurons[i][j].Weights[k] = float64(i+j) / float64(width+height)
			}
		}
	}
	
	return &SelfOrgMap{
		Width:        width,
		Height:       height,
		Neurons:      neurons,
		LearningRate: learningRate,
		Radius:       radius,
		Iteration:    0,
		Patterns:     make([]Pattern, 0),
	}
}

// JSON Logger Implementation
func initJSONLogger(filename string) *JSONLogger {
	return &JSONLogger{
		LogFile:     filename,
		NetEnabled:  true,
		HTTPEnabled: true,
		Filters:     make([]LogFilter, 0),
		Streams:     make(map[string]*LogStream),
	}
}

func (jl *JSONLogger) Log(level, category, message string, metadata map[string]interface{}) {
	jl.mu.Lock()
	defer jl.mu.Unlock()
	
	logEntry := map[string]interface{}{
		"timestamp": time.Now().Format(time.RFC3339Nano),
		"level":     level,
		"category":  category,
		"message":   message,
		"metadata":  metadata,
		"thread":    runtime.NumGoroutine(),
	}
	
	// Write to file
	if file, err := os.OpenFile(jl.LogFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644); err == nil {
		json.NewEncoder(file).Encode(logEntry)
		file.Close()
	}
	
	// Update stream statistics
	streamName := fmt.Sprintf("%s-%s", level, category)
	if stream, exists := jl.Streams[streamName]; exists {
		stream.MessageCount++
		stream.LastMessage = time.Now()
	} else {
		jl.Streams[streamName] = &LogStream{
			Name:         streamName,
			Active:       true,
			MessageCount: 1,
			LastMessage:  time.Now(),
		}
	}
}

// File Indexer Implementation
func initFileIndexer(rootPaths []string) *FileIndexer {
	return &FileIndexer{
		RootPaths:     rootPaths,
		IndexedFiles:  make(map[string]FileEntry),
		Extensions:    []string{".md", ".txt", ".json", ".js", ".mjs", ".ts", ".tsx", ".svelte", ".go", ".png", ".jpg", ".jpeg"},
		LastScan:      time.Time{},
		BestPractices: make([]BestPractice, 0),
		ImageAnalysis: true,
	}
}

func (fi *FileIndexer) ScanAndIndex() error {
	fi.IndexedFiles = make(map[string]FileEntry)
	
	for _, rootPath := range fi.RootPaths {
		err := filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return nil // Continue on errors
			}
			
			// Check if file extension is supported
			ext := filepath.Ext(path)
			supported := false
			for _, supportedExt := range fi.Extensions {
				if ext == supportedExt {
					supported = true
					break
				}
			}
			
			if !supported || info.IsDir() {
				return nil
			}
			
			// Create file entry
			entry := FileEntry{
				Path:     path,
				Size:     info.Size(),
				ModTime:  info.ModTime(),
				Type:     ext,
				Metadata: make(map[string]string),
				Patterns: make([]string, 0),
			}
			
			// Read content for text files
			if strings.HasSuffix(ext, ".md") || strings.HasSuffix(ext, ".txt") || 
			   strings.HasSuffix(ext, ".json") || strings.HasSuffix(ext, ".js") ||
			   strings.HasSuffix(ext, ".mjs") || strings.HasSuffix(ext, ".ts") ||
			   strings.HasSuffix(ext, ".tsx") || strings.HasSuffix(ext, ".svelte") ||
			   strings.HasSuffix(ext, ".go") {
				if content, err := os.ReadFile(path); err == nil {
					entry.Content = string(content)
					entry.Patterns = fi.extractPatterns(string(content))
				}
			}
			
			// Store indexed file
			fi.IndexedFiles[path] = entry
			return nil
		})
		
		if err != nil {
			return err
		}
	}
	
	fi.LastScan = time.Now()
	fi.generateBestPractices()
	return nil
}

func (fi *FileIndexer) extractPatterns(content string) []string {
	patterns := make([]string, 0)
	
	// Extract common patterns
	regexes := map[string]*regexp.Regexp{
		"function":     regexp.MustCompile(`function\s+(\w+)`),
		"class":        regexp.MustCompile(`class\s+(\w+)`),
		"interface":    regexp.MustCompile(`interface\s+(\w+)`),
		"type":         regexp.MustCompile(`type\s+(\w+)`),
		"const":        regexp.MustCompile(`const\s+(\w+)`),
		"import":       regexp.MustCompile(`import\s+.*from\s+['"](.+)['"]`),
		"todo":         regexp.MustCompile(`(?i)TODO[:\s]*(.+)`),
		"fixme":        regexp.MustCompile(`(?i)FIXME[:\s]*(.+)`),
		"best_practice": regexp.MustCompile(`(?i)best.practice[:\s]*(.+)`),
	}
	
	for patternType, regex := range regexes {
		matches := regex.FindAllStringSubmatch(content, -1)
		for _, match := range matches {
			if len(match) > 1 {
				patterns = append(patterns, fmt.Sprintf("%s:%s", patternType, match[1]))
			}
		}
	}
	
	return patterns
}

func (fi *FileIndexer) generateBestPractices() {
	fi.BestPractices = make([]BestPractice, 0)
	
	// Analyze patterns across files to generate best practices
	patternCounts := make(map[string]int)
	patternSources := make(map[string][]string)
	
	for path, entry := range fi.IndexedFiles {
		for _, pattern := range entry.Patterns {
			patternCounts[pattern]++
			patternSources[pattern] = append(patternSources[pattern], path)
		}
	}
	
	// Generate best practices from common patterns
	for pattern, count := range patternCounts {
		if count >= 3 { // Pattern appears in at least 3 files
			parts := strings.SplitN(pattern, ":", 2)
			if len(parts) == 2 {
				category := parts[0]
				description := parts[1]
				
				confidence := float64(count) / float64(len(fi.IndexedFiles))
				
				bestPractice := BestPractice{
					Category:    category,
					Description: description,
					Source:      fmt.Sprintf("Pattern found in %d files", count),
					Confidence:  confidence,
					Examples:    patternSources[pattern],
					LastUpdated: time.Now(),
				}
				
				fi.BestPractices = append(fi.BestPractices, bestPractice)
			}
		}
	}
}

// MCP Integrator Implementation
func initMCPIntegrator() *MCPIntegrator {
	return &MCPIntegrator{
		Endpoints: map[string]string{
			"context7":       "http://localhost:40000/mcp",
			"analyze-stack":  "/analyze-stack",
			"best-practices": "/generate-best-practices",
			"library-docs":   "/get-library-docs",
		},
		Active: true,
		Cache:  make(map[string]MCPResult),
	}
}

// AutoGen Agent Implementation
func initAutoGenAgent() *AutoGenAgent {
	agents := []Agent{
		{
			Name:         "TypeScript Expert",
			Role:         "code_analyzer",
			Capabilities: []string{"typescript", "error_detection", "best_practices"},
			Model:        "gpt-4",
			Status:       "active",
		},
		{
			Name:         "Svelte Specialist", 
			Role:         "component_analyzer",
			Capabilities: []string{"svelte", "ui_patterns", "runes_migration"},
			Model:        "claude-3",
			Status:       "active",
		},
		{
			Name:         "Architecture Reviewer",
			Role:         "system_designer",
			Capabilities: []string{"system_architecture", "performance", "scalability"},
			Model:        "gpt-4",
			Status:       "active",
		},
	}
	
	return &AutoGenAgent{
		Agents:        agents,
		Conversations: make([]Conversation, 0),
		Active:        true,
	}
}

// Concurrent Todo Manager Implementation
func initConcurrentTodo() *ConcurrentTodo {
	scheduler := &TaskScheduler{
		Queue:         make([]string, 0),
		Running:       make(map[string]bool),
		MaxConcurrent: 3,
	}
	
	agents := []TodoAgent{
		{
			Name:        "MCP Context7 Agent",
			Type:        "mcp",
			Status:      "idle",
			Performance: AgentPerformance{},
		},
		{
			Name:        "File Indexer Agent",
			Type:        "file_indexer", 
			Status:      "idle",
			Performance: AgentPerformance{},
		},
		{
			Name:        "Best Practices Agent",
			Type:        "best_practices",
			Status:      "idle",
			Performance: AgentPerformance{},
		},
	}
	
	return &ConcurrentTodo{
		Tasks:     make([]Task, 0),
		Agents:    agents,
		Scheduler: scheduler,
		Active:    true,
	}
}

// Network Layer Implementation
func initNetworkLayer(port int) *NetworkLayer {
	return &NetworkLayer{
		Port: port,
		Endpoints: map[string]string{
			"/health":         "GET",
			"/agentic/status": "GET",
			"/tensor/analyze": "POST",
			"/som/recommend":  "POST",
			"/files/index":    "POST",
			"/todo/create":    "POST",
			"/autogen/chat":   "POST",
		},
		WebSockets:  make(map[string]bool),
		ActiveConns: 0,
	}
}

// Main HTTP Handlers
func (as *AgenticSystem) healthHandler(w http.ResponseWriter, r *http.Request) {
	as.mu.RLock()
	defer as.mu.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	
	health := map[string]interface{}{
		"status":      "healthy",
		"version":     "2.0.0-agentic",
		"timestamp":   time.Now(),
		"cuda_enabled": as.CUDAEnabled,
		"components": map[string]bool{
			"tensor_core":    as.TensorCore != nil,
			"self_org_map":   as.SelfOrgMap != nil,
			"json_logger":    as.JSONLogger != nil,
			"file_indexer":   as.FileIndexer != nil,
			"mcp_integrator": as.MCPIntegrator.Active,
			"autogen_agent":  as.AutoGenAgent.Active,
			"todo_manager":   as.TodoManager.Active,
		},
		"performance": map[string]interface{}{
			"indexed_files":     len(as.FileIndexer.IndexedFiles),
			"best_practices":    len(as.FileIndexer.BestPractices),
			"active_todos":      len(as.TodoManager.Tasks),
			"som_patterns":      len(as.SelfOrgMap.Patterns),
			"log_streams":       len(as.JSONLogger.Streams),
		},
	}
	
	json.NewEncoder(w).Encode(health)
	
	// Log the health check
	as.JSONLogger.Log("info", "health", "Health check requested", map[string]interface{}{
		"remote_addr": r.RemoteAddr,
		"user_agent": r.UserAgent(),
	})
}

func (as *AgenticSystem) agenticStatusHandler(w http.ResponseWriter, r *http.Request) {
	as.mu.RLock()
	defer as.mu.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	
	json.NewEncoder(w).Encode(as)
}

// Claude CLI Detection and Update
func detectClaudeCLI() (bool, string) {
	// Check if running in .vscode terminal
	if os.Getenv("TERM_PROGRAM") == "vscode" {
		// Check for Claude CLI installation
		if _, err := os.Stat(".vscode/settings.json"); err == nil {
			return true, ".vscode detected with Claude CLI integration"
		}
	}
	return false, "No Claude CLI detected"
}

// Start the Agentic System
func (as *AgenticSystem) Start() error {
	// Initialize logging
	as.JSONLogger.Log("info", "system", "Starting Agentic Programming System", map[string]interface{}{
		"cuda_enabled": as.CUDAEnabled,
		"components":   len(as.NetworkLayer.Endpoints),
	})
	
	// Start file indexing
	go func() {
		for {
			as.JSONLogger.Log("info", "indexer", "Starting file indexing scan", nil)
			if err := as.FileIndexer.ScanAndIndex(); err != nil {
				as.JSONLogger.Log("error", "indexer", "File indexing failed", map[string]interface{}{
					"error": err.Error(),
				})
			} else {
				as.JSONLogger.Log("info", "indexer", "File indexing completed", map[string]interface{}{
					"files_indexed":   len(as.FileIndexer.IndexedFiles),
					"best_practices":  len(as.FileIndexer.BestPractices),
				})
			}
			time.Sleep(5 * time.Minute) // Re-index every 5 minutes
		}
	}()
	
	// Start concurrent todo processing
	go as.processConcurrentTodos()
	
	// Setup HTTP routes
	http.HandleFunc("/health", as.healthHandler)
	http.HandleFunc("/agentic/status", as.agenticStatusHandler)
	
	// Start HTTP server
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", as.NetworkLayer.Port),
		Handler: nil,
	}
	as.NetworkLayer.HTTPServer = server
	
	fmt.Printf("🤖 Advanced Agentic Programming System starting on port %d\n", as.NetworkLayer.Port)
	fmt.Printf("🚀 CUDA Enabled: %v\n", as.CUDAEnabled)
	fmt.Printf("🧠 Self-Organizing Map: %dx%d neurons\n", as.SelfOrgMap.Width, as.SelfOrgMap.Height)
	fmt.Printf("📊 JSON Logging: %s\n", as.JSONLogger.LogFile)
	fmt.Printf("🔍 File Indexing: %d root paths\n", len(as.FileIndexer.RootPaths))
	fmt.Printf("⚡ Tensor Core: %d devices, %d streams\n", len(as.TensorCore.Devices), as.TensorCore.ActiveStreams)
	fmt.Printf("🎯 AutoGen Agents: %d active\n", len(as.AutoGenAgent.Agents))
	fmt.Printf("📝 Todo Agents: %d concurrent\n", len(as.TodoManager.Agents))
	
	// Detect Claude CLI
	if detected, message := detectClaudeCLI(); detected {
		fmt.Printf("🔧 Claude CLI: %s\n", message)
		as.JSONLogger.Log("info", "claude-cli", message, nil)
	}
	
	fmt.Printf("\n📡 Endpoints:\n")
	for endpoint, method := range as.NetworkLayer.Endpoints {
		fmt.Printf("   %s http://localhost:%d%s\n", method, as.NetworkLayer.Port, endpoint)
	}
	fmt.Printf("\n")
	
	return server.ListenAndServe()
}

func (as *AgenticSystem) processConcurrentTodos() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Process pending tasks
			for i, agent := range as.TodoManager.Agents {
				if agent.Status == "idle" && len(as.TodoManager.Scheduler.Queue) > 0 {
					// Assign task to agent
					taskID := as.TodoManager.Scheduler.Queue[0]
					as.TodoManager.Scheduler.Queue = as.TodoManager.Scheduler.Queue[1:]
					as.TodoManager.Scheduler.Running[taskID] = true
					
					// Update agent status
					as.TodoManager.Agents[i].Status = "running"
					as.TodoManager.Agents[i].CurrentTask = taskID
					as.TodoManager.Agents[i].Performance.LastActive = time.Now()
					
					// Process task based on agent type
					go as.processTaskByAgent(&as.TodoManager.Agents[i], taskID)
					
					as.JSONLogger.Log("info", "todo", "Task assigned to agent", map[string]interface{}{
						"task_id":    taskID,
						"agent_name": agent.Name,
						"agent_type": agent.Type,
					})
				}
			}
		}
	}
}

func (as *AgenticSystem) processTaskByAgent(agent *TodoAgent, taskID string) {
	startTime := time.Now()
	success := true
	
	defer func() {
		duration := time.Since(startTime)
		agent.Status = "idle"
		agent.CurrentTask = ""
		agent.Performance.TasksCompleted++
		
		// Update success rate
		if success {
			agent.Performance.SuccessRate = (agent.Performance.SuccessRate + 1.0) / 2.0
		} else {
			agent.Performance.SuccessRate = agent.Performance.SuccessRate / 2.0
		}
		
		// Update average duration
		if agent.Performance.AvgDuration == 0 {
			agent.Performance.AvgDuration = duration
		} else {
			agent.Performance.AvgDuration = (agent.Performance.AvgDuration + duration) / 2
		}
		
		// Remove from running tasks
		delete(as.TodoManager.Scheduler.Running, taskID)
		
		as.JSONLogger.Log("info", "todo", "Task completed by agent", map[string]interface{}{
			"task_id":    taskID,
			"agent_name": agent.Name,
			"duration":   duration.String(),
			"success":    success,
		})
	}()
	
	switch agent.Type {
	case "mcp":
		// Fetch from MCP Context7
		as.JSONLogger.Log("info", "mcp", "Processing MCP task", map[string]interface{}{
			"task_id": taskID,
		})
		time.Sleep(2 * time.Second) // Mock processing time
		
	case "file_indexer":
		// Index files and generate best practices
		as.JSONLogger.Log("info", "indexer", "Processing file indexing task", map[string]interface{}{
			"task_id": taskID,
		})
		if err := as.FileIndexer.ScanAndIndex(); err != nil {
			success = false
		}
		
	case "best_practices":
		// Generate best practices from indexed files
		as.JSONLogger.Log("info", "practices", "Processing best practices task", map[string]interface{}{
			"task_id": taskID,
		})
		as.FileIndexer.generateBestPractices()
	}
}

func main() {
	system := NewAgenticSystem()
	log.Fatal(system.Start())
}