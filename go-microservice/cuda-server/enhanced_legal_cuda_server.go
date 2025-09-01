//go:build enhanced
// +build enhanced

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	// gRPC and Protocol Buffers
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	// HTTP/2 and REST API
	"github.com/gin-gonic/gin"
	"golang.org/x/net/http2"

	// Monitoring and Metrics
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	// Database and Cache
	"github.com/jackc/pgx/v5/pgxpool"

	// Import our cache system (alias internal redis to avoid name clash)
	"legal-ai-production/internal/cache"
	iredis "legal-ai-production/internal/redis"

	// Protocol Buffer generated files
	pb "legal-ai-production/proto/legal_cuda_streaming"
)
	// Protocol Buffer generated files
	pb "legal-ai-production/proto/legal_cuda_streaming"
)

// =====================================
// Configuration & Environment
// =====================================

type CudaServerConfig struct {
	// Server Configuration
	HTTPPort      string `json:"http_port" env:"HTTP_PORT" default:"8080"`
	GRPCPort      string `json:"grpc_port" env:"GRPC_PORT" default:"50052"`
	MetricsPort   string `json:"metrics_port" env:"METRICS_PORT" default:"9090"`
	Environment   string `json:"environment" env:"ENVIRONMENT" default:"development"`

	// Security
	TLSEnabled bool   `json:"tls_enabled" env:"TLS_ENABLED" default:"false"`
	CertFile   string `json:"cert_file" env:"TLS_CERT_FILE"`
	KeyFile    string `json:"key_file" env:"TLS_KEY_FILE"`

	// Database Configuration
	PostgresURL string `json:"postgres_url" env:"POSTGRES_URL" default:"postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db"`
	RedisURL    string `json:"redis_url" env:"REDIS_URL" default:"localhost:6379"`

	// CUDA Configuration
	CudaDeviceID     int    `json:"cuda_device_id" env:"CUDA_DEVICE_ID" default:"0"`
	CudaMemoryPool   int64  `json:"cuda_memory_pool" env:"CUDA_MEMORY_POOL" default:"2147483648"` // 2GB
	CudaStreams      int    `json:"cuda_streams" env:"CUDA_STREAMS" default:"8"`
	TensorCores      bool   `json:"tensor_cores" env:"TENSOR_CORES" default:"true"`
	GpuUtilTarget    float64 `json:"gpu_util_target" env:"GPU_UTIL_TARGET" default:"0.85"`

	// Cache Configuration
	CacheConfig cache.MultiLayerCacheConfig `json:"cache_config"`

	// Performance & Limits
	MaxWorkers        int               `json:"max_workers" env:"MAX_WORKERS"`
	MaxConcurrentReqs int               `json:"max_concurrent_reqs" env:"MAX_CONCURRENT_REQS" default:"100"`
	RequestTimeout    time.Duration     `json:"request_timeout" env:"REQUEST_TIMEOUT" default:"5m"`
	RateLimits        map[string]int    `json:"rate_limits"`
	CORSOrigins       []string          `json:"cors_origins"`

	// Monitoring
	Monitoring MonitoringConfig `json:"monitoring"`
}

type MonitoringConfig struct {
	PrometheusEnabled bool   `json:"prometheus_enabled" default:"true"`
	TracingEnabled    bool   `json:"tracing_enabled" default:"false"`
	LogLevel          string `json:"log_level" default:"info"`
	HealthChecks      bool   `json:"health_checks" default:"true"`
}

// =====================================
// CUDA Performance Metrics
// =====================================

type CudaMetricsCollector struct {
	// Processing metrics
	documentProcessingDuration *prometheus.HistogramVec
	embeddingGenerationTime    *prometheus.HistogramVec
	searchQueryDuration        *prometheus.HistogramVec
	cudaKernelExecutionTime    *prometheus.HistogramVec

	// GPU metrics
	gpuUtilization      *prometheus.GaugeVec
	gpuMemoryUsage      *prometheus.GaugeVec
	gpuTemperature      *prometheus.GaugeVec
	tensorCoreUtilization *prometheus.GaugeVec

	// Request metrics
	grpcRequestsTotal     *prometheus.CounterVec
	httpRequestsTotal     *prometheus.CounterVec
	activeConnections     *prometheus.GaugeVec
	streamingConnections  *prometheus.GaugeVec

	// Cache metrics
	cacheOperations       *prometheus.CounterVec
	cacheHitRatio         *prometheus.GaugeVec
	cacheResponseTime     *prometheus.HistogramVec

	// Error tracking
	errorRate             *prometheus.CounterVec
	cudaErrors            *prometheus.CounterVec

	// Business metrics
	documentsProcessed    *prometheus.CounterVec
	embeddingsGenerated   *prometheus.CounterVec
	searchesPerformed     *prometheus.CounterVec
}

func NewCudaMetricsCollector() *CudaMetricsCollector {
	return &CudaMetricsCollector{
		documentProcessingDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "legal_cuda_document_processing_duration_seconds",
				Help:    "Time taken to process legal documents with CUDA",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"document_type", "processing_stage", "gpu_accelerated"},
		),

		embeddingGenerationTime: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "legal_cuda_embedding_generation_duration_seconds",
				Help:    "Time taken to generate embeddings with CUDA",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
			},
			[]string{"model_type", "embedding_dim", "batch_size"},
		),

		searchQueryDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "legal_cuda_search_query_duration_seconds",
				Help:    "Time taken for CUDA-accelerated similarity search",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
			},
			[]string{"collection", "search_type", "top_k"},
		),

		cudaKernelExecutionTime: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "cuda_kernel_execution_duration_seconds",
				Help:    "Time taken for CUDA kernel execution",
				Buckets: prometheus.ExponentialBuckets(0.0001, 2, 15),
			},
			[]string{"kernel_name", "block_size", "grid_size"},
		),

		gpuUtilization: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gpu_utilization_percent",
				Help: "GPU utilization percentage",
			},
			[]string{"gpu_id", "gpu_model"},
		),

		gpuMemoryUsage: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gpu_memory_usage_bytes",
				Help: "GPU memory usage in bytes",
			},
			[]string{"gpu_id", "memory_type"},
		),

		gpuTemperature: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gpu_temperature_celsius",
				Help: "GPU temperature in Celsius",
			},
			[]string{"gpu_id"},
		),

		tensorCoreUtilization: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "tensor_core_utilization_percent",
				Help: "Tensor Core utilization percentage",
			},
			[]string{"gpu_id"},
		),

		grpcRequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_cuda_grpc_requests_total",
				Help: "Total number of gRPC requests",
			},
			[]string{"method", "status", "streaming"},
		),

		httpRequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_cuda_http_requests_total",
				Help: "Total number of HTTP requests",
			},
			[]string{"method", "endpoint", "status"},
		),

		activeConnections: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "legal_cuda_active_connections",
				Help: "Current number of active connections",
			},
			[]string{"connection_type", "protocol"},
		),

		streamingConnections: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "legal_cuda_streaming_connections",
				Help: "Current number of streaming connections",
			},
			[]string{"stream_type", "session_state"},
		),

		cacheOperations: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_cuda_cache_operations_total",
				Help: "Total cache operations",
			},
			[]string{"operation", "cache_layer", "result"},
		),

		cacheHitRatio: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "legal_cuda_cache_hit_ratio",
				Help: "Cache hit ratio by layer",
			},
			[]string{"cache_layer"},
		),

		cacheResponseTime: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "legal_cuda_cache_response_time_seconds",
				Help:    "Cache response time",
				Buckets: prometheus.ExponentialBuckets(0.0001, 2, 12),
			},
			[]string{"cache_layer", "operation"},
		),

		errorRate: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_cuda_errors_total",
				Help: "Total number of errors by type",
			},
			[]string{"error_type", "service", "severity"},
		),

		cudaErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cuda_errors_total",
				Help: "Total number of CUDA errors",
			},
			[]string{"error_type", "kernel_name"},
		),

		documentsProcessed: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "legal_documents_processed_total",
				Help: "Total legal documents processed",
			},
			[]string{"document_type", "processing_result"},
		),

		embeddingsGenerated: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "embeddings_generated_total",
				Help: "Total embeddings generated",
			},
			[]string{"model_type", "embedding_dimension"},
		),

		searchesPerformed: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "searches_performed_total",
				Help: "Total searches performed",
			},
			[]string{"search_type", "collection"},
		),
	}
}

func (c *CudaMetricsCollector) RegisterMetrics() {
	prometheus.MustRegister(
		c.documentProcessingDuration,
		c.embeddingGenerationTime,
		c.searchQueryDuration,
		c.cudaKernelExecutionTime,
		c.gpuUtilization,
		c.gpuMemoryUsage,
		c.gpuTemperature,
		c.tensorCoreUtilization,
		c.grpcRequestsTotal,
		c.httpRequestsTotal,
		c.activeConnections,
		c.streamingConnections,
		c.cacheOperations,
		c.cacheHitRatio,
		c.cacheResponseTime,
		c.errorRate,
		c.cudaErrors,
		c.documentsProcessed,
		c.embeddingsGenerated,
		c.searchesPerformed,
	)
}

// =====================================
// Enhanced Legal CUDA Service Implementation
// =====================================

type EnhancedLegalCudaService struct {
	pb.UnimplementedLegalCudaServiceServer

	// Configuration
	config *CudaServerConfig

	// Database & Cache
	pgPool      *pgxpool.Pool
	redisCache  *iredis.DistributedCache
	multiCache  *cache.MultiLayerCache

	// CUDA Resources
	cudaDevice      int
	cudaContext     uintptr // CUDA context handle
	cudaStreams     []uintptr // CUDA stream handles
	cudaMemoryPool  *CudaMemoryPool

	// Performance Monitoring
	metrics *CudaMetricsCollector

	// Session Management
	activeSessions    map[string]*CudaStreamingSession
	sessionsMutex     sync.RWMutex

	// Resource Management
	workerPool        chan struct{}
	shutdownChan      chan struct{}
	wg                sync.WaitGroup

	// CUDA Device Properties
	deviceProps       CudaDeviceProperties
}

type CudaStreamingSession struct {
	ID              string
	StartTime       time.Time
	LastActivity    time.Time
	CudaStream      uintptr
	ProcessingQueue chan *pb.CudaRequest
	ResponseChan    chan *pb.CudaResponse
	Context         context.Context
	Cancel          context.CancelFunc
	Metrics         SessionMetrics
	Active          bool
	mutex           sync.RWMutex
}

type SessionMetrics struct {
	RequestsProcessed   int64
	TotalProcessingTime time.Duration
	AverageResponseTime time.Duration
	ErrorCount          int64
	CacheHits           int64
	CacheMisses         int64
}

type CudaDeviceProperties struct {
	Name                string
	Major               int
	Minor               int
	MultiProcessorCount int
	MaxThreadsPerBlock  int
	MaxBlockDim         [3]int
	MaxGridDim          [3]int
	SharedMemPerBlock   int
	TotalGlobalMem      int64
	ClockRate           int
	MemoryClockRate     int
	MemoryBusWidth      int
	L2CacheSize         int
	MaxTexture1D        int
	MaxTexture2D        [2]int
	MaxTexture3D        [3]int
	WarpSize            int
	MaxPitch            int64
}

// CUDA Memory Pool Implementation
type CudaMemoryPool struct {
	totalSize     int64
	allocatedSize int64
	freeBlocks    []CudaMemoryBlock
	usedBlocks    map[uintptr]CudaMemoryBlock
	mutex         sync.Mutex
	devicePtr     uintptr
}

type CudaMemoryBlock struct {
	Ptr    uintptr
	Size   int64
	Offset int64
}

// =====================================
// Service Initialization
// =====================================

func NewEnhancedLegalCudaService(config *CudaServerConfig) (*EnhancedLegalCudaService, error) {
	service := &EnhancedLegalCudaService{
		config:         config,
		activeSessions: make(map[string]*CudaStreamingSession),
		workerPool:     make(chan struct{}, config.MaxConcurrentReqs),
		shutdownChan:   make(chan struct{}),
		metrics:        NewCudaMetricsCollector(),
	}

	// Initialize worker pool
	for i := 0; i < config.MaxConcurrentReqs; i++ {
		service.workerPool <- struct{}{}
	}

	log.Printf("🚀 Initializing Enhanced Legal CUDA Service")

	// Initialize CUDA
	if err := service.initializeCUDA(); err != nil {
		return nil, fmt.Errorf("CUDA initialization failed: %w", err)
	}

	// Initialize database connection
	if err := service.initializeDatabase(); err != nil {
		return nil, fmt.Errorf("database initialization failed: %w", err)
	}

	// Initialize cache system
	if err := service.initializeCache(); err != nil {
		return nil, fmt.Errorf("cache initialization failed: %w", err)
	}

	// Register metrics
	service.metrics.RegisterMetrics()

	// Start background routines
	service.startBackgroundTasks()

	log.Printf("✅ Enhanced Legal CUDA Service initialized successfully")
	service.logSystemInfo()

	return service, nil
}

func (s *EnhancedLegalCudaService) initializeCUDA() error {
	// Initialize CUDA runtime
	log.Printf("🔧 Initializing CUDA Runtime...")

	// Set CUDA device
	s.cudaDevice = s.config.CudaDeviceID

	// Get device properties (placeholder - in real implementation would use CUDA API)
	s.deviceProps = CudaDeviceProperties{
		Name:                "RTX 3060 Ti",
		Major:               8,
		Minor:               6,
		MultiProcessorCount: 38,
		MaxThreadsPerBlock:  1024,
		MaxBlockDim:         [3]int{1024, 1024, 64},
		MaxGridDim:          [3]int{2147483647, 65535, 65535},
		SharedMemPerBlock:   49152,
		TotalGlobalMem:      8589934592, // 8GB
		ClockRate:           1665000,    // 1.665 GHz
		MemoryClockRate:     7001000,    // 14 Gbps effective
		MemoryBusWidth:      256,
		L2CacheSize:         4194304,    // 4MB
		WarpSize:            32,
	}

	// Initialize CUDA streams
	s.cudaStreams = make([]uintptr, s.config.CudaStreams)
	for i := 0; i < s.config.CudaStreams; i++ {
		// In real implementation: cudaStreamCreate(&s.cudaStreams[i])
		s.cudaStreams[i] = uintptr(i + 1) // Placeholder
	}

	// Initialize memory pool
	s.cudaMemoryPool = &CudaMemoryPool{
		totalSize:  s.config.CudaMemoryPool,
		freeBlocks: make([]CudaMemoryBlock, 0),
		usedBlocks: make(map[uintptr]CudaMemoryBlock),
	}

	log.Printf("✅ CUDA initialized: %s (Compute %d.%d)",
		s.deviceProps.Name, s.deviceProps.Major, s.deviceProps.Minor)
	log.Printf("📊 GPU Memory: %.2f GB, Multiprocessors: %d",
		float64(s.deviceProps.TotalGlobalMem)/1024/1024/1024, s.deviceProps.MultiProcessorCount)

	return nil
}

func (s *EnhancedLegalCudaService) initializeDatabase() error {
	log.Printf("🔧 Initializing PostgreSQL connection...")

	config, err := pgxpool.ParseConfig(s.config.PostgresURL)
	if err != nil {
		return fmt.Errorf("failed to parse PostgreSQL URL: %w", err)
	}

	// Configure connection pool
	config.MaxConns = 25
	config.MinConns = 5
	config.MaxConnLifetime = time.Hour
	config.MaxConnIdleTime = time.Minute * 30

	s.pgPool, err = pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		return fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := s.pgPool.Ping(ctx); err != nil {
		return fmt.Errorf("database ping failed: %w", err)
	}

	// Create cache tables if they don't exist
	if err := s.createCacheTables(ctx); err != nil {
		log.Printf("Warning: Failed to create cache tables: %v", err)
	}

	log.Printf("✅ PostgreSQL connected and ready")
	return nil
}

func (s *EnhancedLegalCudaService) createCacheTables(ctx context.Context) error {
	query := `
		CREATE TABLE IF NOT EXISTS cache_entries (
			id BIGSERIAL PRIMARY KEY,
			cache_key TEXT UNIQUE NOT NULL,
			cache_namespace TEXT NOT NULL DEFAULT 'default',
			cache_value JSONB NOT NULL,
			ttl_seconds INTEGER,
			size_bytes INTEGER,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			expires_at TIMESTAMPTZ,
			last_accessed TIMESTAMPTZ DEFAULT NOW(),
			access_count BIGINT DEFAULT 1
		);

		CREATE INDEX IF NOT EXISTS idx_cache_entries_key ON cache_entries(cache_key);
		CREATE INDEX IF NOT EXISTS idx_cache_entries_namespace ON cache_entries(cache_namespace);
		CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at ON cache_entries(expires_at);
		CREATE INDEX IF NOT EXISTS idx_cache_entries_created_at ON cache_entries(created_at);

		-- CUDA-specific cache tables
		CREATE TABLE IF NOT EXISTS cuda_embeddings_cache (
			id BIGSERIAL PRIMARY KEY,
			text_hash TEXT UNIQUE NOT NULL,
			model_type TEXT NOT NULL,
			embedding_dim INTEGER NOT NULL,
			embedding_data REAL[] NOT NULL,
			processing_time_ms INTEGER,
			gpu_utilization REAL,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			access_count BIGINT DEFAULT 1
		);

		CREATE INDEX IF NOT EXISTS idx_cuda_embeddings_hash ON cuda_embeddings_cache(text_hash);
		CREATE INDEX IF NOT EXISTS idx_cuda_embeddings_model ON cuda_embeddings_cache(model_type);

		CREATE TABLE IF NOT EXISTS cuda_search_cache (
			id BIGSERIAL PRIMARY KEY,
			query_hash TEXT UNIQUE NOT NULL,
			collection_name TEXT NOT NULL,
			top_k INTEGER NOT NULL,
			search_results JSONB NOT NULL,
			processing_time_ms INTEGER,
			created_at TIMESTAMPTZ DEFAULT NOW(),
			expires_at TIMESTAMPTZ,
			access_count BIGINT DEFAULT 1
		);

		CREATE INDEX IF NOT EXISTS idx_cuda_search_hash ON cuda_search_cache(query_hash);
		CREATE INDEX IF NOT EXISTS idx_cuda_search_collection ON cuda_search_cache(collection_name);
	`

	_, err := s.pgPool.Exec(ctx, query)
	return err
}

func (s *EnhancedLegalCudaService) initializeCache() error {
	log.Printf("🔧 Initializing Multi-Layer Cache System...")

	// Initialize Redis distributed cache (internal wrapper)
	var err error
	s.redisCache, err = iredis.InitializeDistributedCache(s.config.RedisURL)
	if err != nil {
		log.Printf("Warning: Redis cache initialization failed: %v", err)
	}

	// Configure multi-layer cache
	cacheConfig := cache.MultiLayerCacheConfig{
		// L1 Configuration (In-memory)
		L1MaxSize:    10000,
		L1DefaultTTL: 5 * time.Minute,
		EnableL1:     true,

		// L2 Configuration (Redis)
		L2Address:    s.config.RedisURL,
		L2DefaultTTL: 30 * time.Minute,
		L2MaxRetries: 3,
		EnableL2:     s.redisCache.IsEnabled(),

		// L3 Configuration (PostgreSQL)
		L3DefaultTTL:      2 * time.Hour,
		L3CleanupInterval: 1 * time.Hour,
		EnableL3:          true,

		// General settings
		EnableMetrics: true,
		WriteThrough:  true,
		ReadThrough:   true,
	}

	s.multiCache, err = cache.NewMultiLayerCache(cacheConfig, s.pgPool)
	if err != nil {
		return fmt.Errorf("multi-layer cache initialization failed: %w", err)
	}

	log.Printf("✅ Multi-Layer Cache initialized: L1=true L2=%v L3=true",
		s.redisCache.IsEnabled())

	return nil
}

func (s *EnhancedLegalCudaService) startBackgroundTasks() {
	// Start GPU monitoring
	go s.monitorGPUMetrics()

	// Start cache metrics collection
	go s.monitorCacheMetrics()

	// Start session cleanup
	go s.cleanupInactiveSessions()

	// Start memory pool maintenance
	go s.maintainMemoryPool()

	log.Printf("✅ Background monitoring tasks started")
}

func (s *EnhancedLegalCudaService) logSystemInfo() {
	log.Printf("🖥️  System Information:")
	log.Printf("   GPU: %s (Compute %d.%d)", s.deviceProps.Name, s.deviceProps.Major, s.deviceProps.Minor)
	log.Printf("   GPU Memory: %.2f GB", float64(s.deviceProps.TotalGlobalMem)/1024/1024/1024)
	log.Printf("   CUDA Streams: %d", len(s.cudaStreams))
	log.Printf("   Tensor Cores: %v", s.config.TensorCores)
	log.Printf("   Worker Pool: %d", s.config.MaxConcurrentReqs)
	log.Printf("   Cache Layers: L1 + L2(Redis) + L3(PostgreSQL)")
	log.Printf("   Environment: %s", s.config.Environment)
}

// =====================================
// gRPC Service Implementation
// =====================================

// BidirectionalLegalStream implements the main streaming interface
func (s *EnhancedLegalCudaService) BidirectionalLegalStream(stream pb.LegalCudaService_BidirectionalLegalStreamServer) error {
	startTime := time.Now()
	s.metrics.streamingConnections.WithLabelValues("bidirectional", "active").Inc()
	defer s.metrics.streamingConnections.WithLabelValues("bidirectional", "active").Dec()

	// Create session
	session := s.createStreamingSession(stream.Context())
	defer s.closeStreamingSession(session.ID)

	log.Printf("📡 Started bidirectional stream: %s", session.ID)

	// Handle streaming in separate goroutines
	errorChan := make(chan error, 2)

	// Request processor
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		errorChan <- s.processStreamRequests(stream, session)
	}()

	// Response sender
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		errorChan <- s.sendStreamResponses(stream, session)
	}()

	// Wait for completion or error
	select {
	case err := <-errorChan:
		if err != nil {
			s.metrics.errorRate.WithLabelValues("stream_processing", "cuda_service", "error").Inc()
			log.Printf("❌ Stream error for session %s: %v", session.ID, err)
			return err
		}
	case <-stream.Context().Done():
		log.Printf("🔌 Stream context cancelled for session %s", session.ID)
	case <-s.shutdownChan:
		log.Printf("🛑 Shutting down stream for session %s", session.ID)
	}

	duration := time.Since(startTime)
	s.metrics.grpcRequestsTotal.WithLabelValues("BidirectionalLegalStream", "completed", "true").Inc()

	log.Printf("✅ Completed bidirectional stream: %s (duration: %v)", session.ID, duration)
	return nil
}

func (s *EnhancedLegalCudaService) processStreamRequests(stream pb.LegalCudaService_BidirectionalLegalStreamServer, session *CudaStreamingSession) error {
	for {
		request, err := stream.Recv()
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("failed to receive request: %w", err)
		}

		select {
		case session.ProcessingQueue <- request:
			session.updateActivity()
		case <-stream.Context().Done():
			return stream.Context().Err()
		case <-s.shutdownChan:
			return fmt.Errorf("service shutting down")
		}

		if request.IsFinalChunk {
			break
		}
	}

	close(session.ProcessingQueue)
	return nil
}

func (s *EnhancedLegalCudaService) sendStreamResponses(stream pb.LegalCudaService_BidirectionalLegalStreamServer, session *CudaStreamingSession) error {
	for response := range session.ResponseChan {
		if err := stream.Send(response); err != nil {
			return fmt.Errorf("failed to send response: %w", err)
		}
		session.updateActivity()
	}
	return nil
}

func (s *EnhancedLegalCudaService) createStreamingSession(ctx context.Context) *CudaStreamingSession {
	sessionID := fmt.Sprintf("session_%d_%d", time.Now().UnixNano(), len(s.activeSessions))

	sessionCtx, cancel := context.WithCancel(ctx)

	session := &CudaStreamingSession{
		ID:              sessionID,
		StartTime:       time.Now(),
		LastActivity:    time.Now(),
		CudaStream:      s.getAvailableCudaStream(),
		ProcessingQueue: make(chan *pb.CudaRequest, 100),
		ResponseChan:    make(chan *pb.CudaResponse, 100),
		Context:         sessionCtx,
		Cancel:          cancel,
		Active:          true,
	}

	s.sessionsMutex.Lock()
	s.activeSessions[sessionID] = session
	s.sessionsMutex.Unlock()

	// Start session processor
	go s.processSessionRequests(session)

	return session
}

func (s *EnhancedLegalCudaService) processSessionRequests(session *CudaStreamingSession) {
	defer close(session.ResponseChan)

	for request := range session.ProcessingQueue {
		startTime := time.Now()

		var response *pb.CudaResponse
		var err error

		// Acquire a worker safely and run the processing function
		err = s.withWorker(session.Context, func() error {
			var e error
			response, e = s.processCudaRequest(session, request)
			return e
		})

		if err != nil {
			// ensure a response exists to report failure
			if response == nil {
				response = &pb.CudaResponse{
					SessionId:     request.SessionId,
					OperationType: request.OperationType,
					Status:        pb.ProcessingStatus_FAILED,
					ErrorMessage:  err.Error(),
				}
			} else {
				response.Status = pb.ProcessingStatus_FAILED
				response.ErrorMessage = err.Error()
			}
			s.metrics.errorRate.WithLabelValues("cuda_processing", "cuda_service", "error").Inc()
		}

		// Add performance metrics
		if response.CudaMetrics == nil {
			response.CudaMetrics = &pb.CudaPerformanceMetrics{}
		}
		response.CudaMetrics.TotalProcessingTimeUs = time.Since(startTime).Microseconds()
		response.CudaMetrics.GpuModel = s.deviceProps.Name

		session.ResponseChan <- response
		session.Metrics.RequestsProcessed++
	}
}

func (s *EnhancedLegalCudaService) processCudaRequest(session *CudaStreamingSession, request *pb.CudaRequest) (*pb.CudaResponse, error) {
	response := &pb.CudaResponse{
		SessionId:     request.SessionId,
		OperationType: request.OperationType,
		Status:        pb.ProcessingStatus_PROCESSING,
	}

	switch request.OperationType {
	case "embed":
		return s.processEmbeddingRequest(session, request, response)
	case "search":
		return s.processSearchRequest(session, request, response)
	case "analyze":
		return s.processAnalysisRequest(session, request, response)
	case "cluster":
		return s.processClusteringRequest(session, request, response)
	default:
		return nil, fmt.Errorf("unsupported operation type: %s", request.OperationType)
	}
}

// Implement specific CUDA processing methods
func (s *EnhancedLegalCudaService) processEmbeddingRequest(session *CudaStreamingSession, request *pb.CudaRequest, response *pb.CudaResponse) (*pb.CudaResponse, error) {
	startTime := time.Now()

	// Extract text data
	var textData string
	switch data := request.Data.(type) {
	case *pb.CudaRequest_RawText:
	if cachedEmbedding, found := s.multiCache.Get(context.Background(), request.SessionId, cacheKey); found {
		if embedding, ok := cachedEmbedding.([]float32); ok {
			response.Result = &pb.CudaResponse_ComputedEmbedding{
				ComputedEmbedding: embedding,
			}
			response.Status = pb.ProcessingStatus_COMPLETED

			s.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "hit").Inc()
			session.Metrics.CacheHits++

			return response, nil
		}
	}
			}
			response.Status = pb.ProcessingStatus_COMPLETED

			s.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "hit").Inc()
			session.Metrics.CacheHits++

			return response, nil
		}
	}

	s.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "miss").Inc()
	session.Metrics.CacheMisses++

	// Perform CUDA embedding generation
	embedding, err := s.generateEmbeddingWithCuda(textData, session.CudaStream, request.CudaOptions)
	if err != nil {
		return nil, fmt.Errorf("CUDA embedding generation failed: %w", err)
	}

	// Cache the result
	ttl := 30 * 60 // 30 minutes
	s.multiCache.Set(context.Background(), request.SessionId, cacheKey, embedding, ttl)
	s.metrics.cacheOperations.WithLabelValues("set", "multi_layer", "success").Inc()

	response.Result = &pb.CudaResponse_ComputedEmbedding{
		ComputedEmbedding: embedding,
	}
	response.Status = pb.ProcessingStatus_COMPLETED

	// Update metrics
	duration := time.Since(startTime)
	s.metrics.embeddingGenerationTime.WithLabelValues("legal_bert", "768", "1").Observe(duration.Seconds())
	s.metrics.embeddingsGenerated.WithLabelValues("legal_bert", "768").Inc()

	return response, nil
}

func (s *EnhancedLegalCudaService) processSearchRequest(session *CudaStreamingSession, request *pb.CudaRequest, response *pb.CudaResponse) (*pb.CudaResponse, error) {
	startTime := time.Now()

	// Extract embedding vector
	var queryEmbedding []float32
	if cachedResults, found := s.multiCache.Get(context.Background(), request.SessionId, cacheKey); found {
		if matches, ok := cachedResults.([]*pb.SearchMatch); ok {
			response.Result = &pb.CudaResponse_SearchMatches{
				SearchMatches: matches,
			}
			response.Status = pb.ProcessingStatus_COMPLETED

			s.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "hit").Inc()
			session.Metrics.CacheHits++

			return response, nil
		}
	}
			}
			response.Status = pb.ProcessingStatus_COMPLETED

			s.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "hit").Inc()
			session.Metrics.CacheHits++

			return response, nil
		}
	}

	s.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "miss").Inc()
	session.Metrics.CacheMisses++

	// Perform CUDA similarity search
	matches, err := s.performCudaSimilaritySearch(queryEmbedding, "legal_documents", 10, session.CudaStream)
	if err != nil {
		return nil, fmt.Errorf("CUDA similarity search failed: %w", err)
	}

	// Cache results
	ttl := 15 * 60 // 15 minutes
	s.multiCache.Set(context.Background(), request.SessionId, cacheKey, matches, ttl)
	s.metrics.cacheOperations.WithLabelValues("set", "multi_layer", "success").Inc()

	response.Result = &pb.CudaResponse_SearchMatches{
		SearchMatches: matches,
	}
	response.Status = pb.ProcessingStatus_COMPLETED

	// Update metrics
	duration := time.Since(startTime)
	s.metrics.searchQueryDuration.WithLabelValues("legal_documents", "similarity", "10").Observe(duration.Seconds())
	s.metrics.searchesPerformed.WithLabelValues("similarity", "legal_documents").Inc()

	return response, nil
}

func (s *EnhancedLegalCudaService) processAnalysisRequest(session *CudaStreamingSession, request *pb.CudaRequest, response *pb.CudaResponse) (*pb.CudaResponse, error) {
	// Implement legal document analysis with CUDA
	analysisResult := &pb.AnalysisResult{
		AnalysisType: "legal_document_analysis",
		Results: map[string]string{
			"contract_type": "service_agreement",
			"jurisdiction": "california",
			"key_terms": "termination_clause,payment_terms,liability",
		},
		Confidence: 0.95,
	}

	response.Result = &pb.CudaResponse_Analysis{
		Analysis: analysisResult,
	}
	response.Status = pb.ProcessingStatus_COMPLETED

	return response, nil
}

func (s *EnhancedLegalCudaService) processClusteringRequest(session *CudaStreamingSession, request *pb.CudaRequest, response *pb.CudaResponse) (*pb.CudaResponse, error) {
	// Implement CUDA-based clustering
	clusterResult := &pb.ClusterResult{
		Clusters: []*pb.DocumentCluster{
			{
				ClusterId: 1,
				DocumentIds: []string{"doc1", "doc2", "doc3"},
				Centroid: []float32{0.1, 0.2, 0.3},
				IntraClusterDistance: 0.15,
			},
		},
		ClusteringMethod: "cuda_kmeans",
	}

	response.Result = &pb.CudaResponse_Clusters{
		Clusters: clusterResult,
	}
	response.Status = pb.ProcessingStatus_COMPLETED

	return response, nil
}

// =====================================
// CUDA Processing Implementation (Placeholder)
// =====================================

func (s *EnhancedLegalCudaService) generateEmbeddingWithCuda(text string, cudaStream uintptr, options *pb.CudaOptions) ([]float32, error) {
	// Placeholder for CUDA embedding generation
	// In real implementation, this would:
	// 1. Tokenize text
	// 2. Run transformer model on GPU
	// 3. Extract embeddings
	// 4. Return float32 array

	embedding := make([]float32, 768)
	for i := range embedding {
		embedding[i] = float32(i) * 0.001 // Placeholder values
	}

	// Simulate GPU processing time
	time.Sleep(10 * time.Millisecond)

	return embedding, nil
}

func (s *EnhancedLegalCudaService) performCudaSimilaritySearch(queryEmbedding []float32, collection string, topK int, cudaStream uintptr) ([]*pb.SearchMatch, error) {
	// Placeholder for CUDA similarity search
	matches := make([]*pb.SearchMatch, topK)

	for i := 0; i < topK; i++ {
		matches[i] = &pb.SearchMatch{
			DocumentId:      fmt.Sprintf("doc_%d", i+1),
			SimilarityScore: float32(0.9 - float64(i)*0.05),
			DocumentTitle:   fmt.Sprintf("Legal Document %d", i+1),
			Snippet:         fmt.Sprintf("This is a snippet from document %d...", i+1),
			Metadata: map[string]string{
				"type": "contract",
				"jurisdiction": "california",
			},
		}
	}

	// Simulate GPU processing time
func (s *CudaStreamingSession) updateActivity() {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	s.LastActivity = time.Now()
}
// =====================================
var streamCounter int64

func (s *EnhancedLegalCudaService) withWorker(ctx context.Context, fn func() error) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-s.shutdownChan:
		return fmt.Errorf("service shutting down")
	case <-s.workerPool:
		// acquired a worker; ensure it's returned
		defer func() { s.workerPool <- struct{}{} }()
		return fn()
	}
}

func (s *EnhancedLegalCudaService) getAvailableCudaStream() uintptr {
	// Guard against zero streams
	if len(s.cudaStreams) == 0 {
		return 0
	}
	idx := atomic.AddInt64(&streamCounter, 1)
	return s.cudaStreams[int(idx)%len(s.cudaStreams)]
}
	s.LastActivity = time.Now()
	s.mutex.Unlock()
}

func (s *EnhancedLegalCudaService) getAvailableCudaStream() uintptr {
	// Simple round-robin allocation
	streamIndex := len(s.activeSessions) % len(s.cudaStreams)
	return s.cudaStreams[streamIndex]
}

func (s *EnhancedLegalCudaService) closeStreamingSession(sessionID string) {
	s.sessionsMutex.Lock()
	defer s.sessionsMutex.Unlock()

	if session, exists := s.activeSessions[sessionID]; exists {
		session.Active = false
		session.Cancel()
		delete(s.activeSessions, sessionID)
		log.Printf("🔌 Closed streaming session: %s", sessionID)
	}
}

func (s *EnhancedLegalCudaService) cleanupInactiveSessions() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.sessionsMutex.Lock()
			now := time.Now()
			for sessionID, session := range s.activeSessions {
				session.mutex.RLock()
				inactive := now.Sub(session.LastActivity) > 10*time.Minute
				session.mutex.RUnlock()

				if inactive {
					session.Cancel()
					delete(s.activeSessions, sessionID)
					log.Printf("🧹 Cleaned up inactive session: %s", sessionID)
				}
			}
			s.sessionsMutex.Unlock()
		case <-s.shutdownChan:
			return
		}
	}
}

// =====================================
// Background Monitoring
// =====================================

func (s *EnhancedLegalCudaService) monitorGPUMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Update GPU utilization metrics (placeholder)
			s.metrics.gpuUtilization.WithLabelValues(fmt.Sprintf("%d", s.cudaDevice), s.deviceProps.Name).Set(0.75)
			s.metrics.gpuMemoryUsage.WithLabelValues(fmt.Sprintf("%d", s.cudaDevice), "used").Set(float64(s.deviceProps.TotalGlobalMem) * 0.6)
			s.metrics.gpuTemperature.WithLabelValues(fmt.Sprintf("%d", s.cudaDevice)).Set(65.0)
			s.metrics.tensorCoreUtilization.WithLabelValues(fmt.Sprintf("%d", s.cudaDevice)).Set(0.8)

		case <-s.shutdownChan:
			return
		}
	}
}

func (s *EnhancedLegalCudaService) monitorCacheMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Get cache statistics
			stats := s.multiCache.GetStats()

			s.metrics.cacheHitRatio.WithLabelValues("L1").Set(stats.L1Stats.HitRatio)
			s.metrics.cacheHitRatio.WithLabelValues("L2").Set(stats.L2Stats.HitRatio)
			s.metrics.cacheHitRatio.WithLabelValues("L3").Set(stats.L3Stats.HitRatio)
			s.metrics.cacheHitRatio.WithLabelValues("total").Set(stats.Total.HitRatio)

		case <-s.shutdownChan:
			return
		}
	}
}

func (s *EnhancedLegalCudaService) maintainMemoryPool() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Maintain CUDA memory pool (placeholder)
			s.cudaMemoryPool.mutex.Lock()
			// Memory pool maintenance logic would go here
	_, cancel := context.WithCancel(context.Background())
	defer cancel()
		case <-s.shutdownChan:
			return
		}
	}
}

// =====================================
// Main Server Entry Point
// =====================================

func main() {
	log.Printf("🚀 Starting Enhanced Legal CUDA gRPC Server")
	log.Printf("📊 Runtime: Go %s on %s/%s", runtime.Version(), runtime.GOOS, runtime.GOARCH)

	// Load configuration
	config := loadServerConfig()
	log.Printf("⚙️  Configuration loaded - Environment: %s", config.Environment)

	// Initialize service
	service, err := NewEnhancedLegalCudaService(config)
	if err != nil {
		log.Fatalf("❌ Service initialization failed: %v", err)
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start gRPC server
	go func() {
		if err := startGRPCServer(service, config); err != nil {
			log.Fatalf("❌ gRPC server failed: %v", err)
		}
	}()

	// Start HTTP server for REST API and metrics
	go func() {
		if err := startHTTPServer(service, config); err != nil {
			log.Fatalf("❌ HTTP server failed: %v", err)
		}
	}()

	log.Printf("✅ All servers started successfully")
	log.Printf("🔧 gRPC Server: localhost:%s", config.GRPCPort)
	log.Printf("🌐 HTTP Server: http://localhost:%s", config.HTTPPort)
	log.Printf("📊 Metrics: http://localhost:%s/metrics", config.MetricsPort)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Printf("🛑 Shutdown signal received, gracefully shutting down...")

	// Signal shutdown to background tasks
	close(service.shutdownChan)

	// Wait for active sessions to complete
	service.wg.Wait()

	// Close resources
	if service.multiCache != nil {
		service.multiCache.Close()
	}
	if service.redisCache != nil {
		service.redisCache.Cleanup()
	}
	if service.pgPool != nil {
		service.pgPool.Close()
	}

	cancel()
	log.Printf("✅ Enhanced Legal CUDA gRPC Server shutdown complete")
}

func loadServerConfig() *CudaServerConfig {
	config := &CudaServerConfig{
		HTTPPort:    getEnv("HTTP_PORT", "8080"),
		GRPCPort:    getEnv("GRPC_PORT", "50052"),
		MetricsPort: getEnv("METRICS_PORT", "9090"),
		Environment: getEnv("ENVIRONMENT", "development"),

		PostgresURL: getEnv("POSTGRES_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		RedisURL:    getEnv("REDIS_URL", "localhost:6379"),

		CudaDeviceID:   0,
		CudaMemoryPool: 2147483648, // 2GB
		CudaStreams:    8,
		TensorCores:    true,

		MaxWorkers:        runtime.NumCPU(),
		MaxConcurrentReqs: 100,
		RequestTimeout:    5 * time.Minute,

		RateLimits: map[string]int{
			"embedding": 100,
			"search":    200,
			"analysis":  50,
		},

		CORSOrigins: []string{
			"http://localhost:3000",
			"http://localhost:5173",
			"http://localhost:8080",
		},

		Monitoring: MonitoringConfig{
			PrometheusEnabled: true,
			TracingEnabled:    false,
			LogLevel:          "info",
			HealthChecks:      true,
		},
	}

	return config
}

func startGRPCServer(service *EnhancedLegalCudaService, config *CudaServerConfig) error {
	lis, err := net.Listen("tcp", ":"+config.GRPCPort)
	if err != nil {
		return fmt.Errorf("failed to listen on gRPC port: %w", err)
	}

	var opts []grpc.ServerOption

	// TLS configuration
	if config.TLSEnabled {
		creds, err := credentials.NewServerTLSFromFile(config.CertFile, config.KeyFile)
		if err != nil {
			return fmt.Errorf("failed to create TLS credentials: %w", err)
		}
		opts = append(opts, grpc.Creds(creds))
	}

	// Configure for streaming
	opts = append(opts,
		grpc.MaxRecvMsgSize(64*1024*1024), // 64MB
		grpc.MaxSendMsgSize(64*1024*1024), // 64MB
		grpc.MaxConcurrentStreams(uint32(config.MaxConcurrentReqs)),
	)

	grpcServer := grpc.NewServer(opts...)

	// Register services
	pb.RegisterLegalCudaServiceServer(grpcServer, service)

	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for development
	if config.Environment != "production" {
		reflection.Register(grpcServer)
	}

	log.Printf("🔧 Enhanced Legal CUDA gRPC server listening on port %s", config.GRPCPort)
	return grpcServer.Serve(lis)
}

func startHTTPServer(service *EnhancedLegalCudaService, config *CudaServerConfig) error {
	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// Health endpoints
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"service":   "Enhanced Legal CUDA gRPC Server",
			"version":   "2.0.0",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"gpu":       service.deviceProps.Name,
			"cuda_streams": len(service.cudaStreams),
			"active_sessions": len(service.activeSessions),
		})
	})

	// Metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Cache statistics
	router.GET("/cache/stats", func(c *gin.Context) {
		stats := service.multiCache.GetStats()
		c.JSON(http.StatusOK, stats)
	})

	// GPU status
	router.GET("/gpu/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"device_id": service.cudaDevice,
			"properties": service.deviceProps,
			"memory_pool": gin.H{
				"total_size": service.cudaMemoryPool.totalSize,
				"allocated": service.cudaMemoryPool.allocatedSize,
			},
			"streams": len(service.cudaStreams),
		})
	})

	server := &http.Server{
		Addr:    ":" + config.HTTPPort,
		Handler: router,
	}

	// Enable HTTP/2
	if err := http2.ConfigureServer(server, nil); err != nil {
		return fmt.Errorf("failed to configure HTTP/2: %w", err)
	}

	log.Printf("🌐 Enhanced Legal CUDA HTTP server listening on port %s", config.HTTPPort)
	return server.ListenAndServe()
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}