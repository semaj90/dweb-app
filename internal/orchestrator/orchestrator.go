package orchestrator

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/joho/godotenv"
)

// --------------------------- Config ---------------------------
type OrchestratorConfig struct {
	Port           string
	GPUEnabled     bool
	CUDAPath       string
	MaxConcurrent  int
	MemoryLimit    string
	WebAssemblyURL string
	EnableUCB      bool
	UCBC           float64
}

func ogpuEnv(key, def string) string { if v := os.Getenv(key); v != "" { return v }; return def }

// --------------------------- Task Model ---------------------------
type TaskType string

const (
	TaskEmbedding  TaskType = "embedding"
	TaskInference  TaskType = "inference"
	TaskCUDAKernel TaskType = "cuda_kernel"
	TaskTensorOp   TaskType = "tensor_op"
)

type OrchestratorTask struct {
	ID        string                 `json:"id"`
	Type      TaskType               `json:"type"`
	Input     interface{}            `json:"input"`
	Priority  int                    `json:"priority"`
	Metadata  map[string]interface{} `json:"metadata"`
	Status    string                 `json:"status"`
	StartTime time.Time              `json:"start_time"`
	EndTime   *time.Time             `json:"end_time,omitempty"`
	Result    interface{}            `json:"result,omitempty"`
	Error     string                 `json:"error,omitempty"`
	visits    int64                  `json:"-"`
	value     float64                `json:"-"`
	added     time.Time
}

// --------------------------- GPU Stats ---------------------------
type GPUStats struct {
	TotalMemory     string  `json:"total_memory"`
	FreeMemory      string  `json:"free_memory"`
	Utilization     float64 `json:"utilization"`
	Temperature     int     `json:"temperature"`
	ActiveTasks     int     `json:"active_tasks"`
	CompletedTasks  int     `json:"completed_tasks"`
	CUDAVersion     string  `json:"cuda_version"`
	DeviceName      string  `json:"device_name"`
	LastUpdatedUnix int64   `json:"last_updated_unix"`
}

// --------------------------- Tensor ---------------------------
type Tensor4D struct {
	Shape   [4]int       `json:"shape"`
	Data    []float32    `json:"-"`
	Key     string       `json:"key"`
	Created time.Time    `json:"created"`
	Meta    TensorMeta   `json:"meta"`
	Access  atomic.Int64 `json:"-"`
}

type TensorMeta struct {
	CaseID         string   `json:"case_id"`
	DocumentID     string   `json:"document_id"`
	EmbeddingModel string   `json:"embedding_model"`
	ParagraphCats  []string `json:"paragraph_categories"`
}

type TensorCache struct {
	mu       sync.RWMutex
	items    map[string]*Tensor4D
	order    []string
	maxItems int
}

func NewTensorCache(max int) *TensorCache { return &TensorCache{items: map[string]*Tensor4D{}, maxItems: max} }
func (tc *TensorCache) Put(t *Tensor4D) { tc.mu.Lock(); defer tc.mu.Unlock(); if _, ok := tc.items[t.Key]; !ok { tc.order = append(tc.order, t.Key) }; tc.items[t.Key] = t; if len(tc.items) > tc.maxItems { tc.evictLRU() } }
func (tc *TensorCache) Get(key string) (*Tensor4D, bool) { tc.mu.RLock(); t, ok := tc.items[key]; tc.mu.RUnlock(); if ok { t.Access.Add(1) }; return t, ok }
func (tc *TensorCache) evictLRU() { if len(tc.order) == 0 { return }; victim := tc.order[0]; tc.order = tc.order[1:]; delete(tc.items, victim) }

// --------------------------- WebSocket ---------------------------
type WebSocketMessage struct {
	Type    string      `json:"type"`
	TaskID  string      `json:"task_id,omitempty"`
	Payload interface{} `json:"payload"`
}

// --------------------------- Service ---------------------------
type OrchestratorService struct {
	config          OrchestratorConfig
	taskQueue       chan *OrchestratorTask
	activeMu        sync.RWMutex
	activeTasks     map[string]*OrchestratorTask
	completedTasks  atomic.Int64
	gpuStats        atomic.Value
	upgrader        websocket.Upgrader
	clientsMu       sync.RWMutex
	clients         map[*websocket.Conn]bool
	quit            chan struct{}
	schedulerWake   chan struct{}
	tensorCache     *TensorCache
	totalSelections atomic.Int64
	totalLatencyMs  atomic.Int64
}

func newOrchestratorService(cfg OrchestratorConfig) *OrchestratorService {
	s := &OrchestratorService{
		config:        cfg,
		taskQueue:     make(chan *OrchestratorTask, 256),
		activeTasks:   map[string]*OrchestratorTask{},
		upgrader:      websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }},
		clients:       map[*websocket.Conn]bool{},
		quit:          make(chan struct{}),
		schedulerWake: make(chan struct{}, 1),
		tensorCache:   NewTensorCache(64),
	}
	s.gpuStats.Store(GPUStats{DeviceName: "Unknown", TotalMemory: "N/A", FreeMemory: "N/A"})
	return s
}

// Run starts the orchestrator HTTP server (replaces main())
func Run() {
	_ = godotenv.Load()
	cfg := OrchestratorConfig{
		Port:           ogpuEnv("GPU_ORCHESTRATOR_PORT", "8095"),
		GPUEnabled:     ogpuEnv("GPU_ENABLED", "false") == "true",
		CUDAPath:       ogpuEnv("CUDA_PATH", ogpuDefaultCUDAPath()),
		MaxConcurrent:  ogpuAtoiDefault(ogpuEnv("GPU_MAX_CONCURRENT", "8"), 8),
		MemoryLimit:    ogpuEnv("GPU_MEMORY_LIMIT", "6GB"),
		WebAssemblyURL: ogpuEnv("WEBASSEMBLY_URL", "http://localhost:8080"),
		EnableUCB:      true,
		UCBC:           1.25,
	}
	svc := newOrchestratorService(cfg)
	initOrchestratorMetrics()
	if err := svc.detectGPUCapabilities(); err != nil { log.Printf("⚠️ GPU capability detection: %v (CPU fallback)", err); cfg.GPUEnabled = false }
	go svc.monitorGPU(); go svc.schedulerLoop(); go svc.workerSupervisor()
	mux := http.NewServeMux(); svc.registerRoutes(mux); handler := withSimpleCORS(mux); srv := &http.Server{Addr: ":" + cfg.Port, Handler: handler}
	go func() { log.Printf("🚀 GPU Orchestrator listening on :%s (GPU=%v UCB=%v)", cfg.Port, cfg.GPUEnabled, cfg.EnableUCB); if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) { log.Fatalf("server: %v", err) } }()
	stop := make(chan os.Signal, 1); signal.Notify(stop, os.Interrupt); <-stop; close(svc.quit); _ = srv.Close(); log.Println("🛑 GPU Orchestrator stopped")
}

// --------------------------- Routing & Handlers (same as original) ---------------------------
func (s *OrchestratorService) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/gpu/stats", s.handleGPUStats)
	mux.HandleFunc("/gpu/task", s.handleSubmitTask)
	mux.HandleFunc("/gpu/task/status", s.handleGetTask)
	mux.HandleFunc("/gpu/tasks", s.handleListTasks)
	mux.HandleFunc("/gpu/task/cancel", s.handleCancelTask)
	mux.HandleFunc("/cuda/kernels", s.handleListKernels)
	mux.HandleFunc("/cuda/execute", s.handleExecuteKernel)
	mux.HandleFunc("/ws", s.handleWebSocket)
	mux.HandleFunc("/tensor/cache/put", s.handleTensorPut)
	mux.HandleFunc("/tensor/cache/get", s.handleTensorGet)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/metrics/prom", s.exposePromMetrics)
}

func (s *OrchestratorService) writeJSON(w http.ResponseWriter, status int, v interface{}) { w.Header().Set("Content-Type", "application/json"); w.WriteHeader(status); _ = json.NewEncoder(w).Encode(v) }
func (s *OrchestratorService) handleHealth(w http.ResponseWriter, r *http.Request) { st := s.gpuStats.Load().(GPUStats); s.writeJSON(w, 200, map[string]interface{}{"status": "healthy", "cuda_available": s.config.GPUEnabled, "active_tasks": s.activeCount(), "queue_depth": len(s.taskQueue), "gpu_stats": st, "timestamp": time.Now().Unix()}) }
func (s *OrchestratorService) handleGPUStats(w http.ResponseWriter, _ *http.Request) { s.writeJSON(w, 200, s.gpuStats.Load()) }
func (s *OrchestratorService) handleSubmitTask(w http.ResponseWriter, r *http.Request) { if r.Method != http.MethodPost { s.writeJSON(w, 405, map[string]string{"error": "POST required"}); return }; var t OrchestratorTask; if err := json.NewDecoder(r.Body).Decode(&t); err != nil { s.writeJSON(w, 400, map[string]string{"error": err.Error()}); return }; if t.Type == "" { t.Type = TaskInference }; if t.Priority == 0 { t.Priority = 1 }; if t.ID == "" { t.ID = fmt.Sprintf("task_%d", time.Now().UnixNano()) }; t.Status = "queued"; t.added = time.Now(); s.enqueue(&t); s.writeJSON(w, 202, map[string]string{"task_id": t.ID, "status": "queued"}) }
func (s *OrchestratorService) handleGetTask(w http.ResponseWriter, r *http.Request) { id := r.URL.Query().Get("id"); if id == "" { s.writeJSON(w, 400, map[string]string{"error": "id required"}); return }; s.activeMu.RLock(); t, ok := s.activeTasks[id]; s.activeMu.RUnlock(); if ok { s.writeJSON(w, 200, t); return }; s.writeJSON(w, 404, map[string]string{"error": "not found"}) }
func (s *OrchestratorService) handleListTasks(w http.ResponseWriter, _ *http.Request) { s.activeMu.RLock(); list := make([]*OrchestratorTask, 0, len(s.activeTasks)); for _, t := range s.activeTasks { list = append(list, t) }; s.activeMu.RUnlock(); s.writeJSON(w, 200, map[string]interface{}{"active": list, "queue_depth": len(s.taskQueue), "completed": s.completedTasks.Load()}) }
func (s *OrchestratorService) handleCancelTask(w http.ResponseWriter, r *http.Request) { id := r.URL.Query().Get("id"); if id == "" { s.writeJSON(w, 400, map[string]string{"error": "id required"}); return }; s.activeMu.Lock(); t, ok := s.activeTasks[id]; if ok { delete(s.activeTasks, id) }; s.activeMu.Unlock(); if ok { now := time.Now(); t.EndTime = &now; t.Status = "cancelled"; s.writeJSON(w, 200, t); return }; s.writeJSON(w, 404, map[string]string{"error": "not found"}) }
func (s *OrchestratorService) handleListKernels(w http.ResponseWriter, _ *http.Request) { kernels := []map[string]interface{}{{"name": "legal_document_similarity", "description": "GPU cosine similarity for legal docs", "parameters": []string{"doc_a", "doc_b"}}, {"name": "contract_clause_extraction", "description": "Extract clauses via pattern GPU", "parameters": []string{"text", "clauses"}}, {"name": "case_law_clustering", "description": "Cluster case law w/ embeddings", "parameters": []string{"docs", "k"}}}; s.writeJSON(w, 200, map[string]interface{}{"available_kernels": kernels, "cuda_version": s.gpuStats.Load().(GPUStats).CUDAVersion}) }
func (s *OrchestratorService) handleExecuteKernel(w http.ResponseWriter, r *http.Request) { if r.Method != http.MethodPost { s.writeJSON(w, 405, map[string]string{"error": "POST required"}); return }; var req struct{ KernelName string `json:"kernel_name"`; Parameters map[string]interface{} `json:"parameters"`; Options map[string]interface{} `json:"options"` }; if err := json.NewDecoder(r.Body).Decode(&req); err != nil { s.writeJSON(w, 400, map[string]string{"error": err.Error()}); return }; task := &OrchestratorTask{ID: fmt.Sprintf("cuda_%d", time.Now().UnixNano()), Type: TaskCUDAKernel, Input: req, Priority: 1, Status: "queued", added: time.Now()}; s.enqueue(task); s.writeJSON(w, 202, map[string]string{"task_id": task.ID, "status": "queued"}) }
func (s *OrchestratorService) handleMetrics(w http.ResponseWriter, _ *http.Request) { st := s.gpuStats.Load().(GPUStats); completed := s.completedTasks.Load(); avgLatency := float64(0); if completed > 0 { avgLatency = float64(s.totalLatencyMs.Load()) / float64(completed) }; w.Header().Set("Content-Type", "text/plain; version=0.0.4"); fmt.Fprintf(w, "gpu_active_tasks %d\n", s.activeCount()); fmt.Fprintf(w, "gpu_queue_depth %d\n", len(s.taskQueue)); fmt.Fprintf(w, "gpu_completed_tasks %d\n", completed); fmt.Fprintf(w, "gpu_avg_task_latency_ms %.2f\n", avgLatency); fmt.Fprintf(w, "gpu_utilization_percent %.2f\n", st.Utilization); fmt.Fprintf(w, "gpu_temperature_celsius %d\n", st.Temperature) }
func (s *OrchestratorService) handleWebSocket(w http.ResponseWriter, r *http.Request) { conn, err := s.upgrader.Upgrade(w, r, nil); if err != nil { log.Printf("ws upgrade: %v", err); return }; s.clientsMu.Lock(); s.clients[conn] = true; s.clientsMu.Unlock(); st := s.gpuStats.Load().(GPUStats); _ = conn.WriteJSON(WebSocketMessage{Type: "status", Payload: map[string]interface{}{"gpu": st, "queue": len(s.taskQueue)}}); go func(c *websocket.Conn) { defer func() { s.clientsMu.Lock(); delete(s.clients, c); s.clientsMu.Unlock(); _ = c.Close() }(); for { if _, _, err := c.NextReader(); err != nil { return } } }(conn) }
func (s *OrchestratorService) handleTensorPut(w http.ResponseWriter, r *http.Request) { if r.Method != http.MethodPost { s.writeJSON(w, 405, map[string]string{"error": "POST required"}); return }; var req struct{ Key string `json:"key"`; Shape [4]int `json:"shape"`; Meta TensorMeta `json:"meta"` }; if err := json.NewDecoder(r.Body).Decode(&req); err != nil { s.writeJSON(w, 400, map[string]string{"error": err.Error()}); return }; total := req.Shape[0] * req.Shape[1] * req.Shape[2] * req.Shape[3]; data := make([]float32, total); t := &Tensor4D{Key: req.Key, Shape: req.Shape, Data: data, Created: time.Now(), Meta: req.Meta}; s.tensorCache.Put(t); s.writeJSON(w, 200, map[string]string{"status": "stored", "key": req.Key, "elements": fmt.Sprintf("%d", total)}) }
func (s *OrchestratorService) handleTensorGet(w http.ResponseWriter, r *http.Request) { key := r.URL.Query().Get("key"); if key == "" { s.writeJSON(w, 400, map[string]string{"error": "key required"}); return }; if t, ok := s.tensorCache.Get(key); ok { s.writeJSON(w, 200, map[string]interface{}{"key": t.Key, "shape": t.Shape, "meta": t.Meta, "access_count": t.Access.Load()}); return }; s.writeJSON(w, 404, map[string]string{"error": "not found"}) }

// --------------------------- Scheduling ---------------------------
func (s *OrchestratorService) enqueue(t *OrchestratorTask) { select { case s.taskQueue <- t: default: log.Printf("⚠️ queue full dropping task %s", t.ID) }; s.wakeScheduler() }
func (s *OrchestratorService) wakeScheduler() { select { case s.schedulerWake <- struct{}{}: default: } }
func (s *OrchestratorService) schedulerLoop() { ticker := time.NewTicker(500 * time.Millisecond); defer ticker.Stop(); for { select { case <-s.quit: return; case <-ticker.C: s.dispatch(); case <-s.schedulerWake: s.dispatch() } } }
func (s *OrchestratorService) dispatch() { for s.activeCount() < s.config.MaxConcurrent { var picked *OrchestratorTask; drained := make([]*OrchestratorTask, 0); Drain: for { select { case t := <-s.taskQueue: drained = append(drained, t); default: break Drain } }; if len(drained) == 0 { return }; if s.config.EnableUCB { picked = s.selectUCB(drained) } else { picked = drained[0]; drained = drained[1:] }; for _, t := range drained { if t != picked { s.enqueue(t) } }; if picked != nil { go s.startExecution(picked) } } }
func (s *OrchestratorService) selectUCB(tasks []*OrchestratorTask) *OrchestratorTask { totalSel := float64(s.totalSelections.Add(1)); bestScore := -1e18; var best *OrchestratorTask; for _, t := range tasks { visits := float64(t.visits); if visits == 0 { visits = 1 }; exploitation := t.value / visits; exploration := s.config.UCBC * math.Sqrt(math.Log(totalSel)/visits); priorityBoost := float64(t.Priority) * 0.05; age := time.Since(t.added).Seconds(); ageBoost := math.Min(age/30.0, 0.5); score := exploitation + exploration + priorityBoost + ageBoost; if score > bestScore { bestScore = score; best = t } }; return best }
func (s *OrchestratorService) startExecution(task *OrchestratorTask) { s.activeMu.Lock(); task.Status = "running"; task.StartTime = time.Now(); s.activeTasks[task.ID] = task; s.activeMu.Unlock(); s.broadcast(WebSocketMessage{Type: "task_started", TaskID: task.ID, Payload: map[string]interface{}{"type": task.Type}}); execStart := time.Now(); switch task.Type { case TaskEmbedding: s.executeEmbedding(task); case TaskInference: s.executeInference(task); case TaskCUDAKernel: s.executeCUDA(task); case TaskTensorOp: s.executeTensorOp(task); default: task.Status = "failed"; task.Error = "unknown task type" }; now := time.Now(); task.EndTime = &now; if task.Status == "running" { task.Status = "completed" }; s.completedTasks.Add(1); latencyMs := now.Sub(execStart).Milliseconds(); s.totalLatencyMs.Add(latencyMs); task.visits++; if task.Status == "completed" && latencyMs > 0 { task.value += 1000.0 / float64(latencyMs+10) } else if task.Status != "completed" { task.value -= 1.0 }; if taskCounter != nil { taskCounter.WithLabelValues(string(task.Type), task.Status).Inc() }; if taskLatency != nil { taskLatency.WithLabelValues(string(task.Type), task.Status).Observe(float64(latencyMs)) }; s.broadcast(WebSocketMessage{Type: "task_completed", TaskID: task.ID, Payload: task}); s.activeMu.Lock(); delete(s.activeTasks, task.ID); s.activeMu.Unlock(); s.wakeScheduler() }

// --------------------------- Execution Simulations ---------------------------
func (s *OrchestratorService) executeEmbedding(task *OrchestratorTask) { base := 120 * time.Millisecond; if s.config.GPUEnabled { base = 60 * time.Millisecond }; time.Sleep(base + time.Duration(ogpuRandJitter(50))*time.Millisecond); dim := 384; vec := make([]float64, dim); for i := range vec { vec[i] = 0.001 * float64(i%17) }; task.Result = map[string]interface{}{"embedding": vec[:8], "dimension": dim, "truncated_preview": true, "gpu": s.config.GPUEnabled} }
func (s *OrchestratorService) executeInference(task *OrchestratorTask) { base := 220 * time.Millisecond; if s.config.GPUEnabled { base = 110 * time.Millisecond }; time.Sleep(base + time.Duration(ogpuRandJitter(120))*time.Millisecond); task.Result = map[string]interface{}{"tokens_used": 128 + ogpuRandJitter(32), "confidence": 0.9, "gpu": s.config.GPUEnabled} }
func (s *OrchestratorService) executeCUDA(task *OrchestratorTask) { if !s.config.GPUEnabled { task.Status = "failed"; task.Error = "CUDA disabled"; return }; time.Sleep(40*time.Millisecond + time.Duration(ogpuRandJitter(40))*time.Millisecond); task.Result = map[string]interface{}{"kernel_output": "ok", "throughput_gflops": 2.1 + float64(ogpuRandJitter(30))/100.0} }
func (s *OrchestratorService) executeTensorOp(task *OrchestratorTask) { time.Sleep(80*time.Millisecond + time.Duration(ogpuRandJitter(90))*time.Millisecond); op := "tricubic_search"; if m, ok := task.Input.(map[string]interface{}); ok { if v, ok2 := m["op"].(string); ok2 && v != "" { op = v } }; task.Result = map[string]interface{}{"operation": op, "matches": 5, "approx": true, "ucb_guided": s.config.EnableUCB} }

// --------------------------- Monitoring ---------------------------
func (s *OrchestratorService) monitorGPU() { ticker := time.NewTicker(5 * time.Second); defer ticker.Stop(); for { select { case <-s.quit: return; case <-ticker.C: st := s.collectGPUStats(); s.gpuStats.Store(st); s.broadcast(WebSocketMessage{Type: "gpu_stats_update", Payload: st}) } } }
func (s *OrchestratorService) collectGPUStats() GPUStats { if !s.config.GPUEnabled { return GPUStats{DeviceName: "CPU Fallback", TotalMemory: "N/A", FreeMemory: "N/A", Temperature: 42, ActiveTasks: s.activeCount(), CompletedTasks: int(s.completedTasks.Load()), CUDAVersion: "N/A", Utilization: 0, LastUpdatedUnix: time.Now().Unix()} }; cmd := exec.Command("nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"); out, err := cmd.Output(); if err != nil { return GPUStats{DeviceName: "RTX 3060 Ti (Simulated)", TotalMemory: "8192 MiB", FreeMemory: fmt.Sprintf("%d MiB", 6000+ogpuRandJitter(512)), Temperature: 60 + ogpuRandJitter(8), ActiveTasks: s.activeCount(), CompletedTasks: int(s.completedTasks.Load()), CUDAVersion: "12.0", Utilization: float64(20 + ogpuRandJitter(40)), LastUpdatedUnix: time.Now().Unix()} }; parts := strings.Split(strings.TrimSpace(string(out)), ", "); if len(parts) < 5 { return GPUStats{DeviceName: "Unknown", TotalMemory: "?", FreeMemory: "?", Utilization: 0, Temperature: 0, ActiveTasks: s.activeCount(), CompletedTasks: int(s.completedTasks.Load()), LastUpdatedUnix: time.Now().Unix()} }; util, _ := strconv.ParseFloat(parts[3], 64); temp, _ := strconv.Atoi(parts[4]); return GPUStats{DeviceName: parts[0], TotalMemory: parts[1] + " MiB", FreeMemory: parts[2] + " MiB", Utilization: util, Temperature: temp, ActiveTasks: s.activeCount(), CompletedTasks: int(s.completedTasks.Load()), CUDAVersion: "12.x", LastUpdatedUnix: time.Now().Unix()} }
func (s *OrchestratorService) workerSupervisor() { <-s.quit }
func (s *OrchestratorService) detectGPUCapabilities() error { if _, err := exec.LookPath("nvidia-smi"); err != nil { return fmt.Errorf("nvidia-smi not found in PATH") }; if _, err := os.Stat(s.config.CUDAPath); err != nil { return fmt.Errorf("CUDA path not found: %s", s.config.CUDAPath) }; return nil }

// --------------------------- Utilities ---------------------------
func (s *OrchestratorService) activeCount() int { s.activeMu.RLock(); n := len(s.activeTasks); s.activeMu.RUnlock(); return n }
func (s *OrchestratorService) broadcast(msg WebSocketMessage) { s.clientsMu.RLock(); defer s.clientsMu.RUnlock(); for c := range s.clients { _ = c.WriteJSON(msg) } }
func ogpuRandJitter(mod int) int { return int(time.Now().UnixNano() % int64(mod+1)) }
func ogpuAtoiDefault(s string, def int) int { n, err := strconv.Atoi(s); if err != nil { return def }; return n }
func ogpuDefaultCUDAPath() string { if runtime.GOOS == "windows" { return `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA` }; return "/usr/local/cuda" }
func withSimpleCORS(next http.Handler) http.Handler { return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Header().Set("Access-Control-Allow-Origin", "*"); w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization"); w.Header().Set("Access-Control-Allow-Methods", "GET,POST,OPTIONS"); if r.Method == http.MethodOptions { w.WriteHeader(204); return }; next.ServeHTTP(w, r) }) }

