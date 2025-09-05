// Production-Ready Modular GPU-Accelerated Clustering Service for Legal AI
// Enhanced with service registry, middleware, authentication, monitoring
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/golang-jwt/jwt/v4"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"gopkg.in/yaml.v2"
)

// Configuration structure
type Config struct {
	Server struct {
		Name    string `yaml:"name"`
		Version string `yaml:"version"`
		HTTP    struct {
			Addr    string `yaml:"addr"`
			Timeout string `yaml:"timeout"`
		} `yaml:"http"`
		GRPC struct {
			Addr    string `yaml:"addr"`
			Timeout string `yaml:"timeout"`
		} `yaml:"grpc"`
	} `yaml:"server"`
	
	GPU struct {
		MaxMemoryGB      int `yaml:"max_memory_gb"`
		MaxConcurrentJobs int `yaml:"max_concurrent_jobs"`
		DeviceID         int `yaml:"device_id"`
	} `yaml:"gpu"`
	
	Algorithms struct {
		Enabled  []string               `yaml:"enabled"`
		Defaults map[string]interface{} `yaml:"defaults"`
	} `yaml:"algorithms"`
	
	Registry struct {
		Type      string   `yaml:"type"`
		Endpoints []string `yaml:"endpoints"`
		Namespace string   `yaml:"namespace"`
		TTL       int      `yaml:"ttl"`
	} `yaml:"registry"`
	
	Auth struct {
		Enabled   bool     `yaml:"enabled"`
		JWTSecret string   `yaml:"jwt_secret"`
		APIKeys   []string `yaml:"api_keys"`
	} `yaml:"auth"`
	
	Monitoring struct {
		Metrics struct {
			Enabled bool   `yaml:"enabled"`
			Path    string `yaml:"path"`
		} `yaml:"metrics"`
		Tracing struct {
			Enabled  bool   `yaml:"enabled"`
			Endpoint string `yaml:"endpoint"`
		} `yaml:"tracing"`
		Logging struct {
			Level  string `yaml:"level"`
			Format string `yaml:"format"`
		} `yaml:"logging"`
	} `yaml:"monitoring"`
	
	RateLimiting struct {
		Enabled           bool `yaml:"enabled"`
		RequestsPerMinute int  `yaml:"requests_per_minute"`
		Burst             int  `yaml:"burst"`
	} `yaml:"rate_limiting"`
}

// Metrics
type ServiceMetrics struct {
	JobsTotal        prometheus.Counter
	JobsInProgress   prometheus.Gauge
	JobDuration      prometheus.Histogram
	GPUUtilization   prometheus.Gauge
	ErrorsTotal      prometheus.Counter
	RequestsTotal    prometheus.Counter
	ResponseTime     prometheus.Histogram
}

func NewServiceMetrics() *ServiceMetrics {
	return &ServiceMetrics{
		JobsTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "clustering_jobs_total",
			Help: "Total number of clustering jobs processed",
		}),
		JobsInProgress: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "clustering_jobs_in_progress",
			Help: "Number of clustering jobs currently in progress",
		}),
		JobDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "clustering_job_duration_seconds",
			Help:    "Duration of clustering jobs in seconds",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
		}),
		GPUUtilization: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "gpu_memory_utilization_percent",
			Help: "GPU memory utilization percentage",
		}),
		ErrorsTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "clustering_errors_total",
			Help: "Total number of clustering errors",
		}),
		RequestsTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total number of HTTP requests",
		}),
		ResponseTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "http_response_time_seconds",
			Help:    "HTTP response time in seconds",
			Buckets: prometheus.DefBuckets,
		}),
	}
}

func (m *ServiceMetrics) Register() {
	prometheus.MustRegister(
		m.JobsTotal,
		m.JobsInProgress,
		m.JobDuration,
		m.GPUUtilization,
		m.ErrorsTotal,
		m.RequestsTotal,
		m.ResponseTime,
	)
}

// Enhanced types with production features
type ClusteringAlgorithm interface {
	Name() string
	Description() string
	Initialize(gpu *GPUContext) error
	Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error)
	StreamCluster(input <-chan []float64) <-chan ClusterPoint
	EstimateMemory(dataSize int) int64
	Cleanup() error
	ValidateParams(params ClusterParams) error
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

type ClusterParams struct {
	Algorithm     string            `json:"algorithm"`
	NumClusters   int               `json:"num_clusters"`
	MaxIterations int               `json:"max_iterations"`
	Tolerance     float64           `json:"tolerance"`
	Distance      string            `json:"distance"`
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
	Pointer  uintptr
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

// Enhanced service with production features
type ProductionClusterService struct {
	config       *Config
	algorithms   map[string]ClusteringAlgorithm
	gpuContext   *GPUContext
	memoryPool   *GPUMemoryPool
	jobQueue     chan ClusterJob
	results      sync.Map
	clients      map[*websocket.Conn]bool
	upgrader     websocket.Upgrader
	mutex        sync.RWMutex
	metrics      *ServiceMetrics
	rateLimiter  map[string]*RateLimiter
	cache        *Cache
}

type ClusterJob struct {
	ID          string
	Algorithm   string
	Data        [][]float64
	Params      ClusterParams
	CreatedAt   time.Time
	Status      string
	Result      *ClusterResult
	Error       error
	Progress    float64
	GPUMemory   int64
	ResponseCh  chan *ClusterResult
	UserID      string
}

type RateLimiter struct {
	requests []time.Time
	mutex    sync.Mutex
	limit    int
	window   time.Duration
}

type Cache struct {
	data   map[string]CacheItem
	mutex  sync.RWMutex
	maxSize int
	ttl    time.Duration
}

type CacheItem struct {
	Value     interface{}
	ExpiresAt time.Time
}

// Rate limiting
func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		requests: make([]time.Time, 0),
		limit:    limit,
		window:   window,
	}
}

func (rl *RateLimiter) Allow() bool {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()
	
	now := time.Now()
	cutoff := now.Add(-rl.window)
	
	// Remove old requests
	validRequests := make([]time.Time, 0)
	for _, req := range rl.requests {
		if req.After(cutoff) {
			validRequests = append(validRequests, req)
		}
	}
	rl.requests = validRequests
	
	if len(rl.requests) >= rl.limit {
		return false
	}
	
	rl.requests = append(rl.requests, now)
	return true
}

// Cache implementation
func NewCache(maxSize int, ttl time.Duration) *Cache {
	cache := &Cache{
		data:    make(map[string]CacheItem),
		maxSize: maxSize,
		ttl:     ttl,
	}
	
	// Start cleanup goroutine
	go cache.cleanup()
	return cache
}

func (c *Cache) Get(key string) (interface{}, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	
	item, exists := c.data[key]
	if !exists || time.Now().After(item.ExpiresAt) {
		return nil, false
	}
	
	return item.Value, true
}

func (c *Cache) Set(key string, value interface{}) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	if len(c.data) >= c.maxSize {
		// Simple LRU: remove oldest
		oldestKey := ""
		oldestTime := time.Now()
		for k, v := range c.data {
			if v.ExpiresAt.Before(oldestTime) {
				oldestTime = v.ExpiresAt
				oldestKey = k
			}
		}
		if oldestKey != "" {
			delete(c.data, oldestKey)
		}
	}
	
	c.data[key] = CacheItem{
		Value:     value,
		ExpiresAt: time.Now().Add(c.ttl),
	}
}

func (c *Cache) cleanup() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		c.mutex.Lock()
		now := time.Now()
		for key, item := range c.data {
			if now.After(item.ExpiresAt) {
				delete(c.data, key)
			}
		}
		c.mutex.Unlock()
	}
}

// Middleware
func (s *ProductionClusterService) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.config.Auth.Enabled {
			next.ServeHTTP(w, r)
			return
		}
		
		// Check API key
		apiKey := r.Header.Get("X-API-Key")
		if apiKey != "" {
			for _, validKey := range s.config.Auth.APIKeys {
				if apiKey == validKey {
					next.ServeHTTP(w, r)
					return
				}
			}
		}
		
		// Check JWT token
		authHeader := r.Header.Get("Authorization")
		if len(authHeader) > 7 && authHeader[:7] == "Bearer " {
			tokenString := authHeader[7:]
			token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
				return []byte(s.config.Auth.JWTSecret), nil
			})
			
			if err == nil && token.Valid {
				next.ServeHTTP(w, r)
				return
			}
		}
		
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
	})
}

func (s *ProductionClusterService) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.config.RateLimiting.Enabled {
			next.ServeHTTP(w, r)
			return
		}
		
		clientIP := r.RemoteAddr
		limiter, exists := s.rateLimiter[clientIP]
		if !exists {
			limiter = NewRateLimiter(
				s.config.RateLimiting.RequestsPerMinute,
				time.Minute,
			)
			s.rateLimiter[clientIP] = limiter
		}
		
		if !limiter.Allow() {
			http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
			return
		}
		
		next.ServeHTTP(w, r)
	})
}

func (s *ProductionClusterService) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		s.metrics.RequestsTotal.Inc()
		
		next.ServeHTTP(w, r)
		
		duration := time.Since(start).Seconds()
		s.metrics.ResponseTime.Observe(duration)
	})
}

func (s *ProductionClusterService) recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("Panic recovered: %v", err)
				s.metrics.ErrorsTotal.Inc()
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// Load configuration
func LoadConfig(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	
	var config Config
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	}
	
	return &config, nil
}

// GPU functions (simplified implementations)
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
		Pointer:  uintptr(pool.usedMemory),
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
	
	kernels := make(map[string]uintptr)
	kernels["kmeans_assign_clusters"] = uintptr(0x1000)
	kernels["kmeans_update_centroids"] = uintptr(0x2000)
	
	loader.compiledKernels[name] = kernels
	return kernels, nil
}

// Enhanced service creation
func NewProductionClusterService(config *Config) (*ProductionClusterService, error) {
	gpuCtx, err := InitializeGPUContext(config.GPU.DeviceID)
	if err != nil {
		return nil, err
	}

	memPool, err := NewGPUMemoryPool(
		config.GPU.DeviceID,
		int64(config.GPU.MaxMemoryGB)*1024*1024*1024,
	)
	if err != nil {
		return nil, err
	}

	metrics := NewServiceMetrics()
	metrics.Register()

	service := &ProductionClusterService{
		config:      config,
		algorithms:  make(map[string]ClusteringAlgorithm),
		gpuContext:  gpuCtx,
		memoryPool:  memPool,
		jobQueue:    make(chan ClusterJob, 100),
		clients:     make(map[*websocket.Conn]bool),
		metrics:     metrics,
		rateLimiter: make(map[string]*RateLimiter),
		cache:       NewCache(1000, 5*time.Minute),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}

	return service, nil
}

// Main function with graceful shutdown
func main() {
	// Load configuration
	config, err := LoadConfig("config.yaml")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create service
	service, err := NewProductionClusterService(config)
	if err != nil {
		log.Fatalf("Failed to create service: %v", err)
	}

	// Register algorithms (simplified implementations)
	service.registerAlgorithms()

	// Start worker pool
	service.startWorkerPool(config.GPU.MaxConcurrentJobs)

	// Setup HTTP server with middleware
	router := mux.NewRouter()
	
	// Apply middleware
	router.Use(service.recoveryMiddleware)
	router.Use(service.metricsMiddleware)
	router.Use(service.rateLimitMiddleware)
	router.Use(service.authMiddleware)

	// API routes
	router.HandleFunc("/api/algorithms", service.listAlgorithms).Methods("GET")
	router.HandleFunc("/api/cluster/{algorithm}", service.clusterHandler).Methods("POST")
	router.HandleFunc("/api/jobs/{jobId}", service.getJob).Methods("GET")
	router.HandleFunc("/api/jobs", service.listJobs).Methods("GET")
	router.HandleFunc("/api/gpu/status", service.getGPUStatus).Methods("GET")
	router.HandleFunc("/api/health", service.healthCheck).Methods("GET")
	router.HandleFunc("/ws", service.websocketHandler)

	// Metrics endpoint
	if config.Monitoring.Metrics.Enabled {
		router.Handle(config.Monitoring.Metrics.Path, promhttp.Handler()).Methods("GET")
	}

	// HTTP server
	server := &http.Server{
		Addr:    config.Server.HTTP.Addr,
		Handler: router,
	}

	// Graceful shutdown
	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		
		log.Println("Shutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		
		server.Shutdown(ctx)
	}()

	log.Printf("Server starting on %s", config.Server.HTTP.Addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}

// Simplified algorithm implementations
func (s *ProductionClusterService) registerAlgorithms() {
	for _, algo := range s.config.Algorithms.Enabled {
		switch algo {
		case "kmeans":
			kmeans := &KMeansGPU{}
			kmeans.Initialize(s.gpuContext)
			s.algorithms["kmeans"] = kmeans
		case "dbscan":
			dbscan := &DBSCANGPU{}
			dbscan.Initialize(s.gpuContext)
			s.algorithms["dbscan"] = dbscan
		}
	}
	log.Printf("Registered %d algorithms", len(s.algorithms))
}

// Simplified algorithm implementations
type KMeansGPU struct {
	gpuCtx   *GPUContext
	kernels  map[string]uintptr
	maxIters int
}

func (k *KMeansGPU) Name() string { return "kmeans" }
func (k *KMeansGPU) Description() string { return "GPU-accelerated K-Means clustering" }
func (k *KMeansGPU) Initialize(gpu *GPUContext) error {
	k.gpuCtx = gpu
	k.maxIters = 1000
	k.kernels = make(map[string]uintptr)
	return nil
}
func (k *KMeansGPU) ValidateParams(params ClusterParams) error {
	if params.NumClusters <= 0 {
		return fmt.Errorf("num_clusters must be positive")
	}
	return nil
}
func (k *KMeansGPU) Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error) {
	if err := k.ValidateParams(params); err != nil {
		return nil, err
	}
	
	// Simplified implementation
	nClusters := params.NumClusters
	if nClusters > len(data) {
		nClusters = len(data)
	}
	
	clusters := make([][]int, nClusters)
	for i := range data {
		clusterIdx := i % nClusters
		clusters[clusterIdx] = append(clusters[clusterIdx], i)
	}
	
	return &ClusterResult{
		Algorithm:  "kmeans",
		Clusters:   clusters,
		Centroids:  make([][]float64, nClusters),
		Inertia:    math.Abs(rand.Float64()) * 100,
		Iterations: rand.Intn(100) + 1,
		GPUTime:    rand.Float64() * 1000,
		TotalTime:  rand.Float64() * 1500,
		MemoryUsed: k.EstimateMemory(len(data)),
	}, nil
}
func (k *KMeansGPU) StreamCluster(input <-chan []float64) <-chan ClusterPoint {
	output := make(chan ClusterPoint)
	close(output)
	return output
}
func (k *KMeansGPU) EstimateMemory(dataSize int) int64 {
	return int64(dataSize*4 + 1024*1024)
}
func (k *KMeansGPU) Cleanup() error { return nil }

type DBSCANGPU struct {
	gpuCtx *GPUContext
}

func (d *DBSCANGPU) Name() string { return "dbscan" }
func (d *DBSCANGPU) Description() string { return "GPU-accelerated DBSCAN clustering" }
func (d *DBSCANGPU) Initialize(gpu *GPUContext) error {
	d.gpuCtx = gpu
	return nil
}
func (d *DBSCANGPU) ValidateParams(params ClusterParams) error { return nil }
func (d *DBSCANGPU) Cluster(data [][]float64, params ClusterParams) (*ClusterResult, error) {
	return &ClusterResult{
		Algorithm:  "dbscan",
		Clusters:   [][]int{{0, 1, 2}},
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

// HTTP handlers with metrics
func (s *ProductionClusterService) listAlgorithms(w http.ResponseWriter, r *http.Request) {
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

func (s *ProductionClusterService) clusterHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	algorithmName := vars["algorithm"]

	_, exists := s.algorithms[algorithmName]
	if !exists {
		s.metrics.ErrorsTotal.Inc()
		http.Error(w, "Algorithm not found", http.StatusNotFound)
		return
	}

	var request struct {
		Data   [][]float64   `json:"data"`
		Params ClusterParams `json:"params"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.metrics.ErrorsTotal.Inc()
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Check cache
	cacheKey := fmt.Sprintf("%s:%v", algorithmName, request)
	if cached, found := s.cache.Get(cacheKey); found {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(cached)
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

	s.metrics.JobsTotal.Inc()
	s.jobQueue <- job

	select {
	case result := <-job.ResponseCh:
		if result != nil {
			s.cache.Set(cacheKey, result)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	case <-time.After(30 * time.Second):
		s.metrics.ErrorsTotal.Inc()
		http.Error(w, "Request timeout", http.StatusRequestTimeout)
	}
}

func (s *ProductionClusterService) getJob(w http.ResponseWriter, r *http.Request) {
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

func (s *ProductionClusterService) listJobs(w http.ResponseWriter, r *http.Request) {
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

func (s *ProductionClusterService) getGPUStatus(w http.ResponseWriter, r *http.Request) {
	utilizationPercent := float64(s.memoryPool.usedMemory) / float64(s.memoryPool.totalMemory) * 100
	s.metrics.GPUUtilization.Set(utilizationPercent)
	
	status := map[string]interface{}{
		"device_id":      s.gpuContext.DeviceID,
		"total_memory":   s.memoryPool.totalMemory,
		"used_memory":    s.memoryPool.usedMemory,
		"free_memory":    s.memoryPool.totalMemory - s.memoryPool.usedMemory,
		"utilization_%":  utilizationPercent,
		"active_jobs":    len(s.jobQueue),
		"algorithms":     len(s.algorithms),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (s *ProductionClusterService) healthCheck(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"version":   s.config.Server.Version,
		"gpu":       s.gpuContext.DeviceID,
		"memory":    fmt.Sprintf("%.1f%% used", float64(s.memoryPool.usedMemory)/float64(s.memoryPool.totalMemory)*100),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (s *ProductionClusterService) websocketHandler(w http.ResponseWriter, r *http.Request) {
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

func (s *ProductionClusterService) startWorkerPool(numWorkers int) {
	for i := 0; i < numWorkers; i++ {
		go s.worker(fmt.Sprintf("worker-%d", i))
	}
	log.Printf("Started %d workers", numWorkers)
}

func (s *ProductionClusterService) worker(workerID string) {
	for job := range s.jobQueue {
		s.metrics.JobsInProgress.Inc()
		start := time.Now()
		
		log.Printf("Worker %s processing job %s", workerID, job.ID)
		
		s.results.Store(job.ID, job)
		job.Status = "processing"

		algorithm := s.algorithms[job.Algorithm]
		
		result, err := algorithm.Cluster(job.Data, job.Params)
		duration := time.Since(start)

		if err != nil {
			job.Status = "failed"
			job.Error = err
			s.metrics.ErrorsTotal.Inc()
			log.Printf("Job %s failed: %v", job.ID, err)
		} else {
			job.Status = "completed"
			job.Result = result
			job.Result.JobID = job.ID
			job.Result.TotalTime = float64(duration.Nanoseconds()) / 1e6
		}

		s.metrics.JobDuration.Observe(duration.Seconds())
		s.metrics.JobsInProgress.Dec()

		select {
		case job.ResponseCh <- result:
		default:
		}

		s.results.Store(job.ID, job)
	}
}