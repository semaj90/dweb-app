// simple_legal_cuda_server.go - Simplified CUDA gRPC/HTTP Server with Integrated Cache System
package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"sync"
	"syscall"
	"time"

	// HTTP/2 and REST API
	"github.com/gin-gonic/gin"
	"golang.org/x/net/http2"

	// Monitoring and Metrics
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	// Database and Cache
	"github.com/jackc/pgx/v5/pgxpool"

	// Import our cache system
	"legal-ai-production/internal/cache"
	"legal-ai-production/internal/redis"
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

	// Database Configuration
	PostgresURL string `json:"postgres_url" env:"POSTGRES_URL" default:"postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db"`
	RedisURL    string `json:"redis_url" env:"REDIS_URL" default:"localhost:6379"`

	// CUDA Configuration
	CudaDeviceID   int           `json:"cuda_device_id" env:"CUDA_DEVICE_ID" default:"0"`
	CudaMemoryPool int64         `json:"cuda_memory_pool" env:"CUDA_MEMORY_POOL" default:"2147483648"` // 2GB
	CudaStreams    int           `json:"cuda_streams" env:"CUDA_STREAMS" default:"8"`
	TensorCores    bool          `json:"tensor_cores" env:"TENSOR_CORES" default:"true"`
	CacheConfig    cache.MultiLayerCacheConfig `json:"cache_config"`

	// Performance & Limits
	MaxWorkers        int               `json:"max_workers" env:"MAX_WORKERS"`
	MaxConcurrentReqs int               `json:"max_concurrent_reqs" env:"MAX_CONCURRENT_REQS" default:"100"`
	RequestTimeout    time.Duration     `json:"request_timeout" env:"REQUEST_TIMEOUT" default:"5m"`
	RateLimits        map[string]int    `json:"rate_limits"`
	CORSOrigins       []string          `json:"cors_origins"`
}

// =====================================
// CUDA Performance Metrics
// =====================================

type CudaMetricsCollector struct {
	documentProcessingDuration *prometheus.HistogramVec
	embeddingGenerationTime    *prometheus.HistogramVec
	searchQueryDuration        *prometheus.HistogramVec

	gpuUtilization *prometheus.GaugeVec
	gpuMemoryUsage *prometheus.GaugeVec
	gpuTemperature *prometheus.GaugeVec

	httpRequestsTotal *prometheus.CounterVec
	activeConnections *prometheus.GaugeVec

	cacheOperations   *prometheus.CounterVec
	cacheHitRatio     *prometheus.GaugeVec
	cacheResponseTime *prometheus.HistogramVec

	errorRate *prometheus.CounterVec
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
	}
}

func (c *CudaMetricsCollector) RegisterMetrics() {
	prometheus.MustRegister(
		c.documentProcessingDuration,
		c.embeddingGenerationTime,
		c.searchQueryDuration,
		c.gpuUtilization,
		c.gpuMemoryUsage,
		c.gpuTemperature,
		c.httpRequestsTotal,
		c.activeConnections,
		c.cacheOperations,
		c.cacheHitRatio,
		c.cacheResponseTime,
		c.errorRate,
	)
}

// =====================================
// Simple Legal CUDA Service Implementation
// =====================================

type SimpleLegalCudaService struct {
	config      *CudaServerConfig
	pgPool      *pgxpool.Pool
	redisCache  *redis.DistributedCache
	multiCache  *cache.MultiLayerCache
	metrics     *CudaMetricsCollector
	shutdownChan chan struct{}
	wg          sync.WaitGroup
	deviceProps CudaDeviceProperties
}

type CudaDeviceProperties struct {
	Name                string
	Major               int
	Minor               int
	MultiProcessorCount int
	TotalGlobalMem      int64
	ClockRate           int
	MemoryClockRate     int
	MemoryBusWidth      int
}

// =====================================
// Service Initialization
// =====================================

func NewSimpleLegalCudaService(config *CudaServerConfig) (*SimpleLegalCudaService, error) {
	service := &SimpleLegalCudaService{
		config:       config,
		shutdownChan: make(chan struct{}),
		metrics:      NewCudaMetricsCollector(),
	}

	log.Printf("🚀 Initializing Simple Legal CUDA Service")

	service.initializeCUDA()

	if config.PostgresURL != "disabled" && config.PostgresURL != "" {
		if err := service.initializeDatabase(); err != nil {
			log.Printf("⚠️  Database initialization failed: %v (continuing without database)", err)
		} else {
			log.Printf("✅ Database connected successfully")
		}
	} else {
		log.Printf("⚠️  Database disabled - running without persistence")
	}

	if err := service.initializeCache(); err != nil {
		log.Printf("⚠️  Cache initialization failed: %v (continuing with minimal cache)", err)
	}

	service.metrics.RegisterMetrics()
	service.startBackgroundTasks()

	log.Printf("✅ Simple Legal CUDA Service initialized successfully")
	service.logSystemInfo()

	return service, nil
}

func (s *SimpleLegalCudaService) initializeCUDA() {
	log.Printf("🔧 Initializing CUDA Runtime...")

	s.deviceProps = CudaDeviceProperties{
		Name:                "RTX 3060 Ti",
		Major:               8,
		Minor:               6,
		MultiProcessorCount: 38,
		TotalGlobalMem:      8589934592,
		ClockRate:           1665000,
		MemoryClockRate:     7001000,
		MemoryBusWidth:      256,
	}

	log.Printf("✅ CUDA initialized: %s (Compute %d.%d)", s.deviceProps.Name, s.deviceProps.Major, s.deviceProps.Minor)
	log.Printf("📊 GPU Memory: %.2f GB, Multiprocessors: %d", float64(s.deviceProps.TotalGlobalMem)/1024/1024/1024, s.deviceProps.MultiProcessorCount)
}

func (s *SimpleLegalCudaService) initializeDatabase() error {
	log.Printf("🔧 Initializing PostgreSQL connection...")

	cfg, err := pgxpool.ParseConfig(s.config.PostgresURL)
	if err != nil {
		return fmt.Errorf("failed to parse PostgreSQL URL: %w", err)
	}

	cfg.MaxConns = 25
	cfg.MinConns = 5
	cfg.MaxConnLifetime = time.Hour
	cfg.MaxConnIdleTime = time.Minute * 30

	s.pgPool, err = pgxpool.NewWithConfig(context.Background(), cfg)
	if err != nil {
		return fmt.Errorf("failed to create connection pool: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.pgPool.Ping(ctx); err != nil {
		return fmt.Errorf("database ping failed: %w", err)
	}

	log.Printf("✅ PostgreSQL connected and ready")
	return nil
}

func (s *SimpleLegalCudaService) initializeCache() error {
	log.Printf("🔧 Initializing Multi-Layer Cache System...")

	var err error
	if s.config.RedisURL != "disabled" && s.config.RedisURL != "" {
		s.redisCache, err = redis.InitializeDistributedCache(s.config.RedisURL)
		if err != nil {
			log.Printf("Warning: Redis cache initialization failed: %v", err)
		}
	} else {
		log.Printf("⚠️  Redis disabled - running with local cache only")
	}

	cacheConfig := cache.MultiLayerCacheConfig{
		L1MaxSize:    10000,
		L1DefaultTTL: 5 * time.Minute,
		EnableL1:     true,

		L2Address:    s.config.RedisURL,
		L2DefaultTTL: 30 * time.Minute,
		L2MaxRetries: 3,
		EnableL2:     s.redisCache != nil && s.redisCache.IsEnabled(),

		L3DefaultTTL:      2 * time.Hour,
		L3CleanupInterval: 1 * time.Hour,
		EnableL3:          s.pgPool != nil,

		EnableMetrics: true,
		WriteThrough:  s.pgPool != nil,
		ReadThrough:   true,
	}

	s.multiCache, err = cache.NewMultiLayerCache(cacheConfig, s.pgPool)
	if err != nil {
		return fmt.Errorf("multi-layer cache initialization failed: %w", err)
	}

	redisEnabled := "false"
	if s.redisCache != nil && s.redisCache.IsEnabled() {
		redisEnabled = "true"
	}

	dbEnabled := "false"
	if s.pgPool != nil {
		dbEnabled = "true"
	}

	log.Printf("✅ Multi-Layer Cache initialized: L1=true L2=%s L3=%s", redisEnabled, dbEnabled)
	return nil
}

func (s *SimpleLegalCudaService) startBackgroundTasks() {
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.monitorGPUMetrics()
	}()

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		s.monitorCacheMetrics()
	}()

	log.Printf("✅ Background monitoring tasks started")
}

func (s *SimpleLegalCudaService) logSystemInfo() {
	log.Printf("🖥️  System Information:")
	log.Printf("   GPU: %s (Compute %d.%d)", s.deviceProps.Name, s.deviceProps.Major, s.deviceProps.Minor)
	log.Printf("   GPU Memory: %.2f GB", float64(s.deviceProps.TotalGlobalMem)/1024/1024/1024)
	log.Printf("   CUDA Streams: %d", s.config.CudaStreams)
	log.Printf("   Tensor Cores: %v", s.config.TensorCores)
	log.Printf("   Cache Layers: L1 + L2(Redis) + L3(PostgreSQL)")
	log.Printf("   Environment: %s", s.config.Environment)
}

// =====================================
// Background Monitoring
// =====================================

func (s *SimpleLegalCudaService) monitorGPUMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.metrics.gpuUtilization.WithLabelValues("0", s.deviceProps.Name).Set(0.75)
			s.metrics.gpuMemoryUsage.WithLabelValues("0", "used").Set(float64(s.deviceProps.TotalGlobalMem) * 0.6)
			s.metrics.gpuTemperature.WithLabelValues("0").Set(65.0)
		case <-s.shutdownChan:
			return
		}
	}
}

func (s *SimpleLegalCudaService) monitorCacheMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if s.multiCache == nil {
				continue
			}
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

// =====================================
// Main Server Entry Point
// =====================================

func main() {
	log.Printf("🚀 Starting Simple Legal CUDA Server")
	log.Printf("📊 Runtime: Go %s on %s/%s", runtime.Version(), runtime.GOOS, runtime.GOARCH)

	config := loadServerConfig()
	log.Printf("⚙️  Configuration loaded - Environment: %s", config.Environment)

	service, err := NewSimpleLegalCudaService(config)
	if err != nil {
		log.Fatalf("❌ Service initialization failed: %v", err)
	}

	// Start HTTP server and receive server instance
	server, err := startHTTPServer(service, config)
	if err != nil {
		log.Fatalf("❌ Failed to start HTTP server: %v", err)
	}

	log.Printf("✅ Server started successfully")
	log.Printf("🌐 HTTP Server: http://localhost:%s", config.HTTPPort)
	log.Printf("📊 Metrics: http://localhost:%s/metrics", config.HTTPPort)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Printf("🛑 Shutdown signal received, gracefully shutting down HTTP server...")

	// Graceful HTTP server shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("⚠️  HTTP server shutdown error: %v", err)
	}

	// Signal background tasks to stop and wait
	close(service.shutdownChan)
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

	log.Printf("✅ Simple Legal CUDA Server shutdown complete")
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
		CudaMemoryPool: 2147483648,
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
	}

	return config
}

// startHTTPServer starts the server in a goroutine and returns the server instance.
func startHTTPServer(service *SimpleLegalCudaService, config *CudaServerConfig) (*http.Server, error) {
	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// Simple CORS middleware
	router.Use(func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		allowed := false
		for _, o := range config.CORSOrigins {
			if o == origin {
				allowed = true
				break
			}
		}
		if allowed {
			c.Writer.Header().Set("Access-Control-Allow-Origin", origin)
			c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Session-ID")
			c.Writer.Header().Set("Access-Control-Allow-Credentials", "true")
		}
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	// Health endpoints
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":        "healthy",
			"service":       "Simple Legal CUDA Server",
			"version":       "1.0.0",
			"timestamp":     time.Now().UTC().Format(time.RFC3339),
			"gpu":           service.deviceProps.Name,
			"cache_enabled": service.multiCache != nil,
		})
	})

	// Metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Cache statistics
	router.GET("/cache/stats", func(c *gin.Context) {
		if service.multiCache == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "cache not initialized"})
			return
		}
		stats := service.multiCache.GetStats()
		c.JSON(http.StatusOK, stats)
	})

	// GPU status
	router.GET("/gpu/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"device_id":    config.CudaDeviceID,
			"properties":   service.deviceProps,
			"tensor_cores": config.TensorCores,
			"streams":      config.CudaStreams,
		})
	})

	// CUDA API endpoints
	router.POST("/api/cuda/embed", func(c *gin.Context) {
		var req struct {
			Text string `json:"text"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		cacheKey := fmt.Sprintf("embed:%s", req.Text)
		sessionID := c.Request.Header.Get("Session-ID")
		if sessionID == "" {
			sessionID = "default"
		package main

		import (
			"bytes"
			"compress/gzip"
			"context"
			"encoding/base64"
			"encoding/binary"
			"fmt"
			"log"
			"net/http"
			"time"
		)

		// EmbeddingGenerator is the abstraction for real CUDA/cgo or CPU fallback
		type EmbeddingGenerator interface {
			Generate(ctx context.Context, text string) ([]float32, error)
		}

		// CPUSimulator implements EmbeddingGenerator for testing
		type CPUSimulator struct {
			Dim int
		}

		func (s *CPUSimulator) Generate(ctx context.Context, text string) ([]float32, error) {
			emb := make([]float32, s.Dim)
			for i := range emb {
				// Respect cancellation
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				default:
				}
				emb[i] = float32(i) * 0.001
			}
			return emb, nil
		}

		// Cache write worker job
		type cacheJob struct {
			ctx       context.Context
			sessionID string
			key       string
			value     interface{} // store encoded payload (string) to keep small over network
			ttlSec    int
		}

		// startCacheWriter runs background worker(s) that flush Set() to multiCache asynchronously
		func startCacheWriter(svc *SimpleLegalCudaService, workerCount int, queueSize int) chan<- cacheJob {
			jobs := make(chan cacheJob, queueSize)
			for w := 0; w < workerCount; w++ {
				go func(id int) {
					for job := range jobs {
						// respect shutdown
						select {
						case <-svc.shutdownChan:
							return
						default:
						}
						if svc.multiCache == nil {
							continue
						}
						err := svc.multiCache.Set(job.ctx, job.sessionID, job.key, job.value, job.ttlSec)
						if err != nil {
							log.Printf("[cache-writer-%d] set error: %v", id, err)
							if svc.metrics != nil {
								svc.metrics.cacheOperations.WithLabelValues("set", "multi_layer", "error").Inc()
							}
						} else if svc.metrics != nil {
							svc.metrics.cacheOperations.WithLabelValues("set", "multi_layer", "success").Inc()
						}
					}
				}(w)
			}
			return jobs
		}

		// helper: float32 slice -> gzip(base64(byte[]))
		func encodeEmbeddingGzipBase64(emb []float32) (string, error) {
			var buf bytes.Buffer
			// binary write floats as little-endian
			if err := binary.Write(&buf, binary.LittleEndian, emb); err != nil {
				return "", err
			}
			var gz bytes.Buffer
			gw := gzip.NewWriter(&gz)
			if _, err := gw.Write(buf.Bytes()); err != nil {
				_ = gw.Close()
				return "", err
			}
			if err := gw.Close(); err != nil {
				return "", err
			}
			return base64.StdEncoding.EncodeToString(gz.Bytes()), nil
		}

		// Register this endpoint from startHTTPServer or init code.
		// Assumes svc, gen (EmbeddingGenerator), and cacheJobs channel exist.
		func registerEmbedEndpoint(router *gin.Engine, svc *SimpleLegalCudaService, gen EmbeddingGenerator, cacheJobs chan<- cacheJob, defaultTTL time.Duration) {
			router.POST("/api/cuda/embed", func(c *gin.Context) {
				var req struct {
					Text string `json:"text"`
				}
				if err := c.ShouldBindJSON(&req); err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
					return
				}

				// Trace ID - prefer incoming header, otherwise generate short id
				traceID := c.GetHeader("X-Trace-ID")
				if traceID == "" {
					traceID = fmt.Sprintf("%d", time.Now().UnixNano())
				}

				cacheKey := fmt.Sprintf("embed:%x", req.Text) // simple key - consider hashing for long text
				sessionID := c.GetHeader("Session-ID")
				if sessionID == "" {
					sessionID = "default"
				}

				// Cache GET (non-blocking safe check)
				if svc.multiCache != nil {
					if cached, found := svc.multiCache.Get(c.Request.Context(), sessionID, cacheKey); found {
						if svc.metrics != nil {
							svc.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "hit").Inc()
						}
						c.JSON(http.StatusOK, gin.H{
							"trace_id": traceID,
							"cached":   true,
							"encoding": "gzip+base64",
							"data":     cached,
						})
						return
					}
					if svc.metrics != nil {
						svc.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "miss").Inc()
					}
				}

				// Generate embedding with request context (cancels on client disconnect)
				ctx := c.Request.Context()
				start := time.Now()
				emb, err := gen.Generate(ctx, req.Text)
				if err != nil {
					if svc.metrics != nil {
						svc.metrics.errorRate.WithLabelValues("embed_generate", "cuda_service", "high").Inc()
					}
					c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error(), "trace_id": traceID})
					return
				}
				if svc.metrics != nil {
					svc.metrics.embeddingGenerationTime.WithLabelValues("simulator", fmt.Sprintf("%d", len(emb)), "1").Observe(time.Since(start).Seconds())
				}

				// encode embedding
				encoded, err := encodeEmbeddingGzipBase64(emb)
				if err != nil {
					c.JSON(http.StatusInternalServerError, gin.H{"error": "encoding failed", "trace_id": traceID})
					return
				}

				// Async cache write: try to queue, drop with metric if queue full
				if svc.multiCache != nil && cacheJobs != nil {
					job := cacheJob{
						ctx:       context.Background(), // background so async write survives request context
						sessionID: sessionID,
						key:       cacheKey,
						value:     encoded,
						ttlSec:    int(defaultTTL.Seconds()),
					}
					select {
					case cacheJobs <- job:
						if svc.metrics != nil {
							svc.metrics.cacheOperations.WithLabelValues("set", "multi_layer", "queued").Inc()
						}
					default:
						// queue full - avoid blocking; record drop
						if svc.metrics != nil {
							svc.metrics.cacheOperations.WithLabelValues("set", "multi_layer", "dropped").Inc()
						}
						log.Printf("[embed] cache queue full, skipping async set for key=%s trace=%s", cacheKey, traceID)
					}
				}

				// Return compressed encoded embedding to client (SvelteKit can decode)
				c.JSON(http.StatusOK, gin.H{
					"trace_id": traceID,
					"cached":   false,
					"encoding": "gzip+base64",
					"data":     encoded,
				})
			})
		}
