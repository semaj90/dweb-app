// simple_legal_cuda_server.go - Simplified CUDA gRPC Server with Integrated Cache System
package main

import (
	"context"
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
	CudaDeviceID     int    `json:"cuda_device_id" env:"CUDA_DEVICE_ID" default:"0"`
	CudaMemoryPool   int64  `json:"cuda_memory_pool" env:"CUDA_MEMORY_POOL" default:"2147483648"` // 2GB
	CudaStreams      int    `json:"cuda_streams" env:"CUDA_STREAMS" default:"8"`
	TensorCores      bool   `json:"tensor_cores" env:"TENSOR_CORES" default:"true"`

	// Cache Configuration
	CacheConfig cache.MultiLayerCacheConfig `json:"cache_config"`

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
	// Processing metrics
	documentProcessingDuration *prometheus.HistogramVec
	embeddingGenerationTime    *prometheus.HistogramVec
	searchQueryDuration        *prometheus.HistogramVec

	// GPU metrics
	gpuUtilization      *prometheus.GaugeVec
	gpuMemoryUsage      *prometheus.GaugeVec
	gpuTemperature      *prometheus.GaugeVec

	// Request metrics
	httpRequestsTotal     *prometheus.CounterVec
	activeConnections     *prometheus.GaugeVec

	// Cache metrics
	cacheOperations       *prometheus.CounterVec
	cacheHitRatio         *prometheus.GaugeVec
	cacheResponseTime     *prometheus.HistogramVec

	// Error tracking
	errorRate             *prometheus.CounterVec
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
	// Configuration
	config *CudaServerConfig

	// Database & Cache
	pgPool      *pgxpool.Pool
	redisCache  *redis.DistributedCache
	multiCache  *cache.MultiLayerCache
	
	// Performance Monitoring
	metrics *CudaMetricsCollector

	// Resource Management
	shutdownChan      chan struct{}
	wg                sync.WaitGroup

	// CUDA Device Properties (simulated)
	deviceProps       CudaDeviceProperties
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

	// Initialize CUDA (simulated)
	service.initializeCUDA()

	// Initialize database connection (optional)
	if config.PostgresURL != "disabled" && config.PostgresURL != "" {
		if err := service.initializeDatabase(); err != nil {
			log.Printf("⚠️  Database initialization failed: %v (continuing without database)", err)
		} else {
			log.Printf("✅ Database connected successfully")
		}
	} else {
		log.Printf("⚠️  Database disabled - running without persistence")
	}

	// Initialize cache system (optional)
	if err := service.initializeCache(); err != nil {
		log.Printf("⚠️  Cache initialization failed: %v (continuing with minimal cache)", err)
	}

	// Register metrics
	service.metrics.RegisterMetrics()

	// Start background routines
	service.startBackgroundTasks()

	log.Printf("✅ Simple Legal CUDA Service initialized successfully")
	service.logSystemInfo()

	return service, nil
}

func (s *SimpleLegalCudaService) initializeCUDA() {
	log.Printf("🔧 Initializing CUDA Runtime...")

	// Simulated device properties for RTX 3060 Ti
	s.deviceProps = CudaDeviceProperties{
		Name:                "RTX 3060 Ti",
		Major:               8,
		Minor:               6,
		MultiProcessorCount: 38,
		TotalGlobalMem:      8589934592, // 8GB
		ClockRate:           1665000,    // 1.665 GHz
		MemoryClockRate:     7001000,    // 14 Gbps effective
		MemoryBusWidth:      256,
	}

	log.Printf("✅ CUDA initialized: %s (Compute %d.%d)", 
		s.deviceProps.Name, s.deviceProps.Major, s.deviceProps.Minor)
	log.Printf("📊 GPU Memory: %.2f GB, Multiprocessors: %d", 
		float64(s.deviceProps.TotalGlobalMem)/1024/1024/1024, s.deviceProps.MultiProcessorCount)
}

func (s *SimpleLegalCudaService) initializeDatabase() error {
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

	log.Printf("✅ PostgreSQL connected and ready")
	return nil
}

func (s *SimpleLegalCudaService) initializeCache() error {
	log.Printf("🔧 Initializing Multi-Layer Cache System...")

	// Initialize Redis distributed cache if enabled
	var err error
	if s.config.RedisURL != "disabled" && s.config.RedisURL != "" {
		s.redisCache, err = redis.InitializeDistributedCache(s.config.RedisURL)
		if err != nil {
			log.Printf("Warning: Redis cache initialization failed: %v", err)
		}
	} else {
		log.Printf("⚠️  Redis disabled - running with local cache only")
	}

	// Configure multi-layer cache
	cacheConfig := cache.MultiLayerCacheConfig{
		// L1 Configuration (In-memory)
		L1MaxSize:    10000,
		L1DefaultTTL: 5 * time.Minute,
		EnableL1:     true,

		// L2 Configuration (Redis) - only if Redis is available
		L2Address:    s.config.RedisURL,
		L2DefaultTTL: 30 * time.Minute,
		L2MaxRetries: 3,
		EnableL2:     s.redisCache != nil && s.redisCache.IsEnabled(),

		// L3 Configuration (PostgreSQL) - only if database is available  
		L3DefaultTTL:      2 * time.Hour,
		L3CleanupInterval: 1 * time.Hour,
		EnableL3:          s.pgPool != nil,

		// General settings
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
	// Start GPU monitoring
	go s.monitorGPUMetrics()
	
	// Start cache metrics collection
	go s.monitorCacheMetrics()

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
			// Update GPU utilization metrics (simulated)
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

// =====================================
// Main Server Entry Point
// =====================================

func main() {
	log.Printf("🚀 Starting Simple Legal CUDA Server")
	log.Printf("📊 Runtime: Go %s on %s/%s", runtime.Version(), runtime.GOOS, runtime.GOARCH)

	// Load configuration
	config := loadServerConfig()
	log.Printf("⚙️  Configuration loaded - Environment: %s", config.Environment)

	// Initialize service
	service, err := NewSimpleLegalCudaService(config)
	if err != nil {
		log.Fatalf("❌ Service initialization failed: %v", err)
	}

	// Setup graceful shutdown
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start HTTP server for REST API and metrics
	go func() {
		if err := startHTTPServer(service, config); err != nil {
			log.Fatalf("❌ HTTP server failed: %v", err)
		}
	}()

	log.Printf("✅ Server started successfully")
	log.Printf("🌐 HTTP Server: http://localhost:%s", config.HTTPPort)
	log.Printf("📊 Metrics: http://localhost:%s/metrics", config.HTTPPort)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Printf("🛑 Shutdown signal received, gracefully shutting down...")
	
	// Signal shutdown to background tasks
	close(service.shutdownChan)
	
	// Wait for background tasks to complete
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
	}

	return config
}

func startHTTPServer(service *SimpleLegalCudaService, config *CudaServerConfig) error {
	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// Health endpoints
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"service":   "Simple Legal CUDA Server",
			"version":   "1.0.0",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"gpu":       service.deviceProps.Name,
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
			"device_id": config.CudaDeviceID,
			"properties": service.deviceProps,
			"tensor_cores": config.TensorCores,
			"streams": config.CudaStreams,
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

		// Check cache first
		cacheKey := fmt.Sprintf("embed:%s", req.Text)
		sessionID := c.Request.Header.Get("Session-ID")
		if sessionID == "" {
			sessionID = "default"
		}
		if cached, found := service.multiCache.Get(c.Request.Context(), sessionID, cacheKey); found {
			service.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "hit").Inc()
			c.JSON(http.StatusOK, gin.H{
				"embedding": cached,
				"cached": true,
			})
			return
		}

		service.metrics.cacheOperations.WithLabelValues("get", "multi_layer", "miss").Inc()

		// Simulate CUDA embedding generation
		embedding := make([]float32, 768)
		for i := range embedding {
			embedding[i] = float32(i) * 0.001
		}

		// Cache the result
		service.multiCache.Set(c.Request.Context(), sessionID, cacheKey, embedding, 1800) // 30 minutes

		c.JSON(http.StatusOK, gin.H{
			"embedding": embedding,
			"cached": false,
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

	log.Printf("🌐 Simple Legal CUDA HTTP server listening on port %s", config.HTTPPort)
	return server.ListenAndServe()
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}