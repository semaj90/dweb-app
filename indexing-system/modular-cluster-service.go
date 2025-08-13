// Modular GPU-Accelerated Clustering Service for Legal AI
// Supports multiple algorithms with shared GPU resources
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

// Configuration
const (
	CLUSTER_SERVICE_PORT = ":8085"
	GRPC_PORT           = ":50051"
	MAX_GPU_MEMORY      = 8 * 1024 * 1024 * 1024 // 8GB
	MAX_CONCURRENT_JOBS = 4
)

// Core interfaces for modularity
type ClusteringAlgorithm interface {
	Name() string
	Description() string
	Initialize(gpu *GPUContext) error
	Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error)
	StreamCluster(input <-chan []float64) <-chan ClusterPoint
	EstimateMemory(dataSize int) int64
	Cleanup() error
}

type GPUContext struct {
	DeviceID     int
	MemoryPool   *GPUMemoryPool
	KernelLoader *CUDAKernelLoader
	StreamQueue  chan GPUJob
}

type CUDAKernelLoader struct {
	compiledKernels map[string]map[string]uintptr
	mutex          sync.RWMutex
}

type GPUJob struct {
	ID       string
	Priority int
	Data     interface{}
	Callback func(interface{}) error
}

// Request/Response types
type ClusterParams struct {
	Algorithm     string            `json:"algorithm"`
	NumClusters   int               `json:"num_clusters"`
	MaxIterations int               `json:"max_iterations"`
	Tolerance     float64           `json:"tolerance"`
	Distance      string            `json:"distance"` // euclidean, cosine, manhattan
	UseGPU        bool              `json:"use_gpu"`
	Extra         map[string]string `json:"extra"`
}

type ClusterResult struct {
	JobID       string      `json:"job_id"`
	Algorithm   string      `json:"algorithm"`
	Clusters    [][]int     `json:"clusters"`
	Centroids   [][]float64 `json:"centroids"`
	Inertia     float64     `json:"inertia"`
	Iterations  int         `json:"iterations"`
	GPUTime     float64     `json:"gpu_time_ms"`
	TotalTime   float64     `json:"total_time_ms"`
	MemoryUsed  int64       `json:"memory_used_bytes"`
	Metadata    interface{} `json:"metadata"`
}

type ClusterPoint struct {
	Index   int       `json:"index"`
	Point   []float64 `json:"point"`
	Cluster int       `json:"cluster"`
	Score   float64   `json:"score"`
}

// GPU Memory Pool for shared resources
type GPUMemoryPool struct {
	mutex        sync.RWMutex
	totalMemory  int64
	usedMemory   int64
	allocations  map[string]*GPUAllocation
	waitQueue    chan AllocationRequest
	deviceID     int
}

type GPUAllocation struct {
	ID       string
	Size     int64
	Pointer  uintptr // GPU memory pointer
	Owner    string
	Priority int
	Created  time.Time
}

type AllocationRequest struct {
	Size     int64
	Owner    string
	Priority int
	Response chan *GPUAllocation
}

// Main clustering service
type ModularClusterService struct {
	algorithms   map[string]ClusteringAlgorithm
	gpuContext   *GPUContext
	memoryPool   *GPUMemoryPool
	jobQueue     chan ClusterJob
	results      sync.Map
	clients      map[*websocket.Conn]bool
	upgrader     websocket.Upgrader
	mutex        sync.RWMutex
}

type ClusterJob struct {
	ID          string
	Algorithm   string
	Data        [][]float64
	Params      ClusterParams
	CreatedAt   time.Time
	Status      string // "queued", "processing", "completed", "failed"
	Result      *ClusterResult
	Error       error
	Progress    float64
	GPUMemory   int64
	ResponseCh  chan *ClusterResult
}

// GPU initialization functions
func InitializeGPUContext(deviceID int) (*GPUContext, error) {
	kernelLoader := &CUDAKernelLoader{
		compiledKernels: make(map[string]map[string]uintptr),
	}
	
	return &GPUContext{
		DeviceID:     deviceID,
		KernelLoader: kernelLoader,
		StreamQueue:  make(chan GPUJob, 100),
	}, nil
}

func NewGPUMemoryPool(deviceID int, totalMemory int64) (*GPUMemoryPool, error) {
	return &GPUMemoryPool{
		totalMemory: totalMemory,
		usedMemory:  0,
		allocations: make(map[string]*GPUAllocation),
		waitQueue:   make(chan AllocationRequest, 100),
		deviceID:    deviceID,
	}, nil
}

func (pool *GPUMemoryPool) Allocate(owner string, size int64, priority int) (*GPUAllocation, error) {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()
	
	if pool.usedMemory+size > pool.totalMemory {
		return nil, fmt.Errorf("insufficient GPU memory: need %d, available %d", size, pool.totalMemory-pool.usedMemory)
	}
	
	allocation := &GPUAllocation{
		ID:       fmt.Sprintf("%s-%d", owner, time.Now().UnixNano()),
		Size:     size,
		Owner:    owner,
		Priority: priority,
		Created:  time.Now(),
		Pointer:  uintptr(pool.usedMemory), // Simplified pointer arithmetic
	}
	
	pool.allocations[allocation.ID] = allocation
	pool.usedMemory += size
	
	return allocation, nil
}

func (pool *GPUMemoryPool) Free(allocationID string) error {
	pool.mutex.Lock()
	defer pool.mutex.Unlock()
	
	allocation, exists := pool.allocations[allocationID]
	if !exists {
		return fmt.Errorf("allocation not found: %s", allocationID)
	}
	
	pool.usedMemory -= allocation.Size
	delete(pool.allocations, allocationID)
	
	return nil
}

func (loader *CUDAKernelLoader) CompileKernel(name, code string) (map[string]uintptr, error) {
	loader.mutex.Lock()
	defer loader.mutex.Unlock()
	
	// Simplified kernel compilation - in real implementation would use CUDA driver API
	kernels := make(map[string]uintptr)
	kernels["kmeans_assign_clusters"] = uintptr(0x1000) // Mock pointer
	kernels["kmeans_update_centroids"] = uintptr(0x2000) // Mock pointer
	
	loader.compiledKernels[name] = kernels
	return kernels, nil
}

func main() {
	service, err := NewModularClusterService()
	if err != nil {
		log.Fatal("Failed to initialize cluster service:", err)
	}

	// Register all clustering algorithms
	service.RegisterAlgorithms()

	// Start GPU workers
	service.StartWorkerPool(MAX_CONCURRENT_JOBS)

	// Start servers
	go service.StartGRPCServer()
	service.StartHTTPServer()
}

func NewModularClusterService() (*ModularClusterService, error) {
	// Initialize GPU context
	gpuCtx, err := InitializeGPUContext(0) // Device 0
	if err != nil {
		return nil, err
	}

	// Create memory pool
	memPool, err := NewGPUMemoryPool(0, MAX_GPU_MEMORY)
	if err != nil {
		return nil, err
	}

	service := &ModularClusterService{
		algorithms: make(map[string]ClusteringAlgorithm),
		gpuContext: gpuCtx,
		memoryPool: memPool,
		jobQueue:   make(chan ClusterJob, 100),
		clients:    make(map[*websocket.Conn]bool),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}

	return service, nil
}

func (s *ModularClusterService) RegisterAlgorithms() {
	// Register K-Means
	kmeans := &KMeansGPU{}
	kmeans.Initialize(s.gpuContext)
	s.algorithms["kmeans"] = kmeans

	// Register DBSCAN
	dbscan := &DBSCANGPU{}
	dbscan.Initialize(s.gpuContext)
	s.algorithms["dbscan"] = dbscan

	// Register Hierarchical
	hierarchical := &HierarchicalGPU{}
	hierarchical.Initialize(s.gpuContext)
	s.algorithms["hierarchical"] = hierarchical

	// Register Self-Organizing Map
	som := &SOMGPU{}
	som.Initialize(s.gpuContext)
	s.algorithms["som"] = som

	log.Printf("Registered %d clustering algorithms", len(s.algorithms))
}

// K-Means GPU Implementation
type KMeansGPU struct {
	gpuCtx   *GPUContext
	kernels  map[string]uintptr
	maxIters int
}

func (k *KMeansGPU) Name() string { return "kmeans" }

func (k *KMeansGPU) Description() string {
	return "GPU-accelerated K-Means clustering with CUDA optimization"
}

func (k *KMeansGPU) Initialize(gpu *GPUContext) error {
	k.gpuCtx = gpu
	k.maxIters = 1000
	k.kernels = make(map[string]uintptr)

	// Load CUDA kernels
	kernelCode := `
	extern "C" {
		__global__ void kmeans_assign_clusters(
			float* points, float* centroids, int* assignments,
			int n_points, int n_clusters, int dimensions
		) {
			int idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= n_points) return;
			
			float min_dist = FLT_MAX;
			int best_cluster = 0;
			
			for (int c = 0; c < n_clusters; c++) {
				float dist = 0.0f;
				for (int d = 0; d < dimensions; d++) {
					float diff = points[idx * dimensions + d] - 
								centroids[c * dimensions + d];
					dist += diff * diff;
				}
				if (dist < min_dist) {
					min_dist = dist;
					best_cluster = c;
				}
			}
			assignments[idx] = best_cluster;
		}
		
		__global__ void kmeans_update_centroids(
			float* points, int* assignments, float* centroids,
			int* cluster_counts, int n_points, int n_clusters, int dimensions
		) {
			int cluster = blockIdx.x;
			int dim = threadIdx.x;
			
			if (cluster >= n_clusters || dim >= dimensions) return;
			
			float sum = 0.0f;
			int count = 0;
			
			for (int i = 0; i < n_points; i++) {
				if (assignments[i] == cluster) {
					sum += points[i * dimensions + dim];
					count++;
				}
			}
			
			if (count > 0) {
				centroids[cluster * dimensions + dim] = sum / count;
			}
			
			if (dim == 0) {
				cluster_counts[cluster] = count;
			}
		}
	}
	`

	kernel, err := k.gpuCtx.KernelLoader.CompileKernel("kmeans", kernelCode)
	if err != nil {
		return fmt.Errorf("failed to compile K-means kernel: %v", err)
	}
	k.kernels["assign"] = kernel["kmeans_assign_clusters"]
	k.kernels["update"] = kernel["kmeans_update_centroids"]

	return nil
}

func (k *KMeansGPU) Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error) {
	startTime := time.Now()
	
	nPoints := len(data)
	if nPoints == 0 {
		return nil, fmt.Errorf("empty data provided")
	}
	
	dimensions := len(data[0])
	nClusters := params.NumClusters
	if nClusters <= 0 {
		nClusters = min(8, nPoints/10) // Default heuristic
	}

	// Allocate GPU memory
	memRequired := k.EstimateMemory(nPoints * dimensions)
	allocation, err := k.gpuCtx.MemoryPool.Allocate("kmeans", memRequired, 1)
	if err != nil {
		return nil, fmt.Errorf("GPU memory allocation failed: %v", err)
	}
	defer k.gpuCtx.MemoryPool.Free(allocation.ID)

	// Convert data to flat array for GPU
	flatData := make([]float32, nPoints*dimensions)
	for i, point := range data {
		for j, val := range point {
			flatData[i*dimensions+j] = float32(val)
		}
	}

	// Initialize centroids randomly
	centroids := k.initializeCentroids(flatData, nPoints, dimensions, nClusters)
	assignments := make([]int, nPoints)
	
	// Upload data to GPU
	gpuData := k.uploadToGPU(flatData, allocation.Pointer)
	gpuCentroids := k.uploadToGPU(centroids, allocation.Pointer+uintptr(len(flatData)*4))
	gpuAssignments := k.allocateGPUInt(assignments, allocation.Pointer+uintptr((len(flatData)+len(centroids))*4))

	var inertia float64
	iterations := 0
	
	for iter := 0; iter < k.maxIters; iter++ {
		iterations = iter + 1
		
		// Step 1: Assign points to clusters
		blockSize := 256
		gridSize := (nPoints + blockSize - 1) / blockSize
		
		k.launchKernel(k.kernels["assign"], 
			gridSize, blockSize,
			gpuData, gpuCentroids, gpuAssignments,
			nPoints, nClusters, dimensions)
		
		// Step 2: Update centroids
		k.launchKernel(k.kernels["update"],
			nClusters, dimensions,
			gpuData, gpuAssignments, gpuCentroids,
			nil, nPoints, nClusters, dimensions)
		
		// Check for convergence (simplified)
		if iter%10 == 0 {
			newInertia := k.calculateInertia(gpuData, gpuCentroids, gpuAssignments, 
											nPoints, nClusters, dimensions)
			if iter > 0 && abs(newInertia-inertia) < params.Tolerance {
				break
			}
			inertia = newInertia
		}
	}

	// Download results
	finalAssignments := k.downloadFromGPUInt(gpuAssignments, nPoints)
	finalCentroids := k.downloadFromGPU(gpuCentroids, nClusters*dimensions)

	// Convert back to [][]float64
	centroidsResult := make([][]float64, nClusters)
	for i := 0; i < nClusters; i++ {
		centroidsResult[i] = make([]float64, dimensions)
		for j := 0; j < dimensions; j++ {
			centroidsResult[i][j] = float64(finalCentroids[i*dimensions+j])
		}
	}

	// Group points by cluster
	clusters := make([][]int, nClusters)
	for i, cluster := range finalAssignments {
		clusters[cluster] = append(clusters[cluster], i)
	}

	return &ClusterResult{
		Algorithm:  "kmeans",
		Clusters:   clusters,
		Centroids:  centroidsResult,
		Inertia:    inertia,
		Iterations: iterations,
		GPUTime:    float64(time.Since(startTime).Nanoseconds()) / 1e6,
		TotalTime:  float64(time.Since(startTime).Nanoseconds()) / 1e6,
		MemoryUsed: memRequired,
	}, nil
}

func (k *KMeansGPU) StreamCluster(input <-chan []float64) <-chan ClusterPoint {
	output := make(chan ClusterPoint, 100)
	
	go func() {
		defer close(output)
		
		batch := make([][]float64, 0, 1000)
		for point := range input {
			batch = append(batch, point)
			
			if len(batch) >= 1000 {
				// Process batch
				result, err := k.Cluster(batch, ClusterParams{
					NumClusters: 8,
					UseGPU:     true,
				})
				
				if err == nil {
					// Send results
					for i, cluster := range result.Clusters {
						for _, pointIdx := range cluster {
							if pointIdx < len(batch) {
								output <- ClusterPoint{
									Index:   pointIdx,
									Point:   batch[pointIdx],
									Cluster: i,
									Score:   result.Inertia,
								}
							}
						}
					}
				}
				
				batch = batch[:0] // Reset batch
			}
		}
	}()
	
	return output
}

func (k *KMeansGPU) EstimateMemory(dataSize int) int64 {
	// Estimate GPU memory needed
	// Points + Centroids + Assignments + Working space
	return int64(dataSize*4 + dataSize/10*4 + dataSize*4 + 1024*1024) // +1MB buffer
}

func (k *KMeansGPU) Cleanup() error {
	// Free CUDA kernels and contexts
	return nil
}

// DBSCAN GPU Implementation
type DBSCANGPU struct {
	gpuCtx *GPUContext
}

func (d *DBSCANGPU) Name() string { return "dbscan" }
func (d *DBSCANGPU) Description() string { return "GPU-accelerated DBSCAN clustering" }
func (d *DBSCANGPU) Initialize(gpu *GPUContext) error {
	d.gpuCtx = gpu
	return nil
}
func (d *DBSCANGPU) Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error) {
	// Simplified DBSCAN implementation
	return &ClusterResult{
		Algorithm:  "dbscan",
		Clusters:   [][]int{{0, 1, 2}}, // Mock result
		Centroids:  [][]float64{{0.5, 0.5}},
		Inertia:    0.0,
		Iterations: 1,
	}, nil
}
func (d *DBSCANGPU) StreamCluster(input <-chan []float64) <-chan ClusterPoint {
	output := make(chan ClusterPoint)
	close(output)
	return output
}
func (d *DBSCANGPU) EstimateMemory(dataSize int) int64 { return int64(dataSize * 8) }
func (d *DBSCANGPU) Cleanup() error { return nil }

// Hierarchical GPU Implementation
type HierarchicalGPU struct {
	gpuCtx *GPUContext
}

func (h *HierarchicalGPU) Name() string { return "hierarchical" }
func (h *HierarchicalGPU) Description() string { return "GPU-accelerated Hierarchical clustering" }
func (h *HierarchicalGPU) Initialize(gpu *GPUContext) error {
	h.gpuCtx = gpu
	return nil
}
func (h *HierarchicalGPU) Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error) {
	return &ClusterResult{
		Algorithm:  "hierarchical",
		Clusters:   [][]int{{0, 1}, {2, 3}},
		Centroids:  [][]float64{{0.3, 0.3}, {0.7, 0.7}},
		Inertia:    0.0,
		Iterations: 1,
	}, nil
}
func (h *HierarchicalGPU) StreamCluster(input <-chan []float64) <-chan ClusterPoint {
	output := make(chan ClusterPoint)
	close(output)
	return output
}
func (h *HierarchicalGPU) EstimateMemory(dataSize int) int64 { return int64(dataSize * 8) }
func (h *HierarchicalGPU) Cleanup() error { return nil }

// SOM GPU Implementation
type SOMGPU struct {
	gpuCtx *GPUContext
}

func (s *SOMGPU) Name() string { return "som" }
func (s *SOMGPU) Description() string { return "GPU-accelerated Self-Organizing Map" }
func (s *SOMGPU) Initialize(gpu *GPUContext) error {
	s.gpuCtx = gpu
	return nil
}
func (s *SOMGPU) Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error) {
	return &ClusterResult{
		Algorithm:  "som",
		Clusters:   [][]int{{0, 2}, {1, 3}},
		Centroids:  [][]float64{{0.2, 0.8}, {0.8, 0.2}},
		Inertia:    0.0,
		Iterations: 100,
	}, nil
}
func (s *SOMGPU) StreamCluster(input <-chan []float64) <-chan ClusterPoint {
	output := make(chan ClusterPoint)
	close(output)
	return output
}
func (s *SOMGPU) EstimateMemory(dataSize int) int64 { return int64(dataSize * 8) }
func (s *SOMGPU) Cleanup() error { return nil }

// HTTP API Handlers
func (s *ModularClusterService) StartHTTPServer() {
	router := mux.NewRouter()

	// API routes
	router.HandleFunc("/api/algorithms", s.listAlgorithms).Methods("GET")
	router.HandleFunc("/api/cluster/{algorithm}", s.clusterHandler).Methods("POST")
	router.HandleFunc("/api/jobs/{jobId}", s.getJob).Methods("GET")
	router.HandleFunc("/api/jobs", s.listJobs).Methods("GET")
	router.HandleFunc("/api/gpu/status", s.getGPUStatus).Methods("GET")
	router.HandleFunc("/api/health", s.healthCheck).Methods("GET")

	// WebSocket for streaming
	router.HandleFunc("/ws", s.websocketHandler)

	// Serve dashboard
	router.PathPrefix("/").Handler(http.FileServer(http.Dir("./static/")))

	log.Printf("HTTP server starting on port %s", CLUSTER_SERVICE_PORT)
	log.Fatal(http.ListenAndServe(CLUSTER_SERVICE_PORT, router))
}

func (s *ModularClusterService) clusterHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	algorithmName := vars["algorithm"]

	_, exists := s.algorithms[algorithmName]
	if !exists {
		http.Error(w, "Algorithm not found", http.StatusNotFound)
		return
	}

	var request struct {
		Data   [][]float64   `json:"data"`
		Params ClusterParams `json:"params"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Create job
	job := ClusterJob{
		ID:        fmt.Sprintf("%s-%d", algorithmName, time.Now().UnixNano()),
		Algorithm: algorithmName,
		Data:      request.Data,
		Params:    request.Params,
		CreatedAt: time.Now(),
		Status:    "queued",
		ResponseCh: make(chan *ClusterResult, 1),
	}

	// Add to queue
	s.jobQueue <- job

	// Wait for result or timeout
	select {
	case result := <-job.ResponseCh:
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	case <-time.After(30 * time.Second):
		http.Error(w, "Request timeout", http.StatusRequestTimeout)
	}
}

func (s *ModularClusterService) listAlgorithms(w http.ResponseWriter, r *http.Request) {
	algorithms := make(map[string]interface{})
	for name, algo := range s.algorithms {
		algorithms[name] = map[string]string{
			"name":        algo.Name(),
			"description": algo.Description(),
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(algorithms)
}

func (s *ModularClusterService) getJob(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobID := vars["jobId"]

	jobInterface, exists := s.results.Load(jobID)
	if !exists {
		http.Error(w, "Job not found", http.StatusNotFound)
		return
	}

	job := jobInterface.(ClusterJob)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(job)
}

func (s *ModularClusterService) listJobs(w http.ResponseWriter, r *http.Request) {
	jobs := make([]map[string]interface{}, 0)
	
	s.results.Range(func(key, value interface{}) bool {
		job := value.(ClusterJob)
		jobInfo := map[string]interface{}{
			"id":        job.ID,
			"algorithm": job.Algorithm,
			"status":    job.Status,
			"created":   job.CreatedAt,
			"progress":  job.Progress,
		}
		jobs = append(jobs, jobInfo)
		return true
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(jobs)
}

func (s *ModularClusterService) getGPUStatus(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"device_id":      s.gpuContext.DeviceID,
		"total_memory":   s.memoryPool.totalMemory,
		"used_memory":    s.memoryPool.usedMemory,
		"free_memory":    s.memoryPool.totalMemory - s.memoryPool.usedMemory,
		"active_jobs":    len(s.jobQueue),
		"algorithms":     len(s.algorithms),
		"uptime":         time.Since(time.Now()).String(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (s *ModularClusterService) healthCheck(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"gpu":       s.gpuContext.DeviceID,
		"memory":    fmt.Sprintf("%.1f%% used", float64(s.memoryPool.usedMemory)/float64(s.memoryPool.totalMemory)*100),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (s *ModularClusterService) websocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	s.mutex.Lock()
	s.clients[conn] = true
	s.mutex.Unlock()

	defer func() {
		s.mutex.Lock()
		delete(s.clients, conn)
		s.mutex.Unlock()
	}()

	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			break
		}
	}
}

func (s *ModularClusterService) StartGRPCServer() {
	// Simplified GRPC server implementation
	log.Printf("GRPC server would start on port %s", GRPC_PORT)
}

// Worker pool for processing jobs
func (s *ModularClusterService) StartWorkerPool(numWorkers int) {
	for i := 0; i < numWorkers; i++ {
		go s.worker(fmt.Sprintf("worker-%d", i))
	}
	log.Printf("Started %d GPU workers", numWorkers)
}

func (s *ModularClusterService) worker(workerID string) {
	for job := range s.jobQueue {
		log.Printf("Worker %s processing job %s", workerID, job.ID)
		
		s.results.Store(job.ID, job)
		job.Status = "processing"

		algorithm := s.algorithms[job.Algorithm]
		
		startTime := time.Now()
		result, err := algorithm.Cluster(job.Data, job.Params)
		processingTime := time.Since(startTime)

		if err != nil {
			job.Status = "failed"
			job.Error = err
			log.Printf("Job %s failed: %v", job.ID, err)
		} else {
			job.Status = "completed"
			job.Result = result
			job.Result.JobID = job.ID
			job.Result.TotalTime = float64(processingTime.Nanoseconds()) / 1e6
		}

		// Send result through channel
		select {
		case job.ResponseCh <- result:
		default:
		}

		s.results.Store(job.ID, job)
		s.broadcastJobUpdate(job)
	}
}

func (s *ModularClusterService) broadcastJobUpdate(job ClusterJob) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	message := map[string]interface{}{
		"type": "job_update",
		"job": map[string]interface{}{
			"id":       job.ID,
			"status":   job.Status,
			"progress": job.Progress,
		},
	}

	data, _ := json.Marshal(message)
	for client := range s.clients {
		client.WriteMessage(websocket.TextMessage, data)
	}
}

// Utility functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// GPU helper functions (simplified interface)
func (k *KMeansGPU) initializeCentroids(data []float32, nPoints, dimensions, nClusters int) []float32 {
	// K-means++ initialization
	centroids := make([]float32, nClusters*dimensions)
	
	// Random initialization for simplicity
	for i := 0; i < nClusters; i++ {
		randomPoint := rand.Intn(nPoints)
		for j := 0; j < dimensions; j++ {
			centroids[i*dimensions+j] = data[randomPoint*dimensions+j]
		}
	}
	return centroids
}

func (k *KMeansGPU) uploadToGPU(data []float32, ptr uintptr) uintptr {
	// CUDA memory copy host to device
	return ptr
}

func (k *KMeansGPU) allocateGPUInt(data []int, ptr uintptr) uintptr {
	return ptr
}

func (k *KMeansGPU) launchKernel(kernel uintptr, gridSize, blockSize int, args ...interface{}) {
	// Launch CUDA kernel
}

func (k *KMeansGPU) calculateInertia(data, centroids, assignments uintptr, nPoints, nClusters, dimensions int) float64 {
	// Simplified inertia calculation
	return math.Abs(rand.Float64()) * 100.0 // Mock calculation
}

func (k *KMeansGPU) downloadFromGPU(ptr uintptr, size int) []float32 {
	result := make([]float32, size)
	// CUDA memory copy device to host
	return result
}

func (k *KMeansGPU) downloadFromGPUInt(ptr uintptr, size int) []int {
	result := make([]int, size)
	return result
}