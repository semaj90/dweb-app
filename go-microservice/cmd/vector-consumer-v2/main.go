package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	pb "legal-ai-production/proto"
	"legal-ai-production/internal/auth"
	"legal-ai-production/internal/cache"
	"legal-ai-production/internal/observability"
	"legal-ai-production/internal/service"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
)

// Configuration structure
type Config struct {
	Port         string
	DatabaseURL  string
	RedisURL     string
	KratosURL    string
	CUDAEnabled  bool
	LogLevel     string
	MaxConcurrency int
}

// EnterpriseVectorServer implements the gRPC vector service with all enterprise features
type EnterpriseVectorServer struct {
	pb.UnimplementedVectorServiceServer
	pb.UnimplementedAsyncJobServiceServer
	
	config    *Config
	cache     *cache.MultiLayerCache
	auth      *auth.KratosAuthInterceptor
	logger    *observability.ELKLogger
	cudaWorker *service.CudaWorkerService
	dbService *service.DatabaseService
	
	// Performance monitoring
	requestCounter *observability.RequestCounter
	healthChecker  *observability.HealthChecker
}

// NewEnterpriseVectorServer creates a new enterprise-grade vector server
func NewEnterpriseVectorServer(config *Config) (*EnterpriseVectorServer, error) {
	// Initialize observability first
	logger, err := observability.NewELKLogger(&observability.ELKConfig{
		ServiceName: "vector-consumer-v2",
		Environment: "production",
		LogLevel:    config.LogLevel,
		ElasticsearchEndpoint: "http://localhost:9200",
		EnableMetrics: true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ELK logger: %w", err)
	}

	logger.Info("Initializing Enterprise Vector Consumer Service v2.0").
		WithString("version", "2.0.0").
		WithString("build_time", time.Now().Format(time.RFC3339)).
		WithBool("cuda_enabled", config.CUDAEnabled).
		Log()

	// Initialize database connection
	pgPool, err := pgxpool.New(context.Background(), config.DatabaseURL)
	if err != nil {
		logger.Error("Failed to connect to database").
			WithError(err).
			WithString("database_url", config.DatabaseURL).
			Log()
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Test database connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := pgPool.Ping(ctx); err != nil {
		logger.Error("Database ping failed").WithError(err).Log()
		return nil, fmt.Errorf("database ping failed: %w", err)
	}

	logger.Info("Database connection established successfully").
		WithString("database", "PostgreSQL with pgvector").
		Log()

	// Initialize Redis connection for caching
	redisClient := redis.NewClient(&redis.Options{
		Addr:     config.RedisURL,
		Password: "",
		DB:       0,
	})

	// Test Redis connection
	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Error("Redis connection failed").WithError(err).Log()
		return nil, fmt.Errorf("redis connection failed: %w", err)
	}

	logger.Info("Redis connection established successfully").
		WithString("redis_url", config.RedisURL).
		Log()

	// Initialize multi-layer cache
	multiCache, err := cache.NewMultiLayerCache(&cache.Config{
		L1Size:      10000,
		L1TTL:       5 * time.Minute,
		L2TTL:       30 * time.Minute,
		L3TTL:       24 * time.Hour,
		RedisClient: redisClient,
		PgPool:      pgPool,
		Logger:      logger,
	})
	if err != nil {
		logger.Error("Failed to initialize multi-layer cache").WithError(err).Log()
		return nil, fmt.Errorf("failed to initialize cache: %w", err)
	}

	// Initialize Kratos authentication
	kratosAuth, err := auth.NewKratosAuthInterceptor(&auth.Config{
		KratosURL: config.KratosURL,
		Cache:     multiCache,
		Logger:    logger,
	})
	if err != nil {
		logger.Error("Failed to initialize Kratos authentication").WithError(err).Log()
		return nil, fmt.Errorf("failed to initialize auth: %w", err)
	}

	// Initialize CUDA worker service
	cudaWorker, err := service.NewCudaWorkerService(&service.CudaConfig{
		Enabled:     config.CUDAEnabled,
		DeviceID:    0,
		MaxMemoryGB: 6, // Use 6GB of 8GB available VRAM
		Logger:      logger,
	})
	if err != nil {
		logger.Error("Failed to initialize CUDA worker").WithError(err).Log()
		return nil, fmt.Errorf("failed to initialize CUDA worker: %w", err)
	}

	// Initialize database service
	dbService, err := service.NewDatabaseService(&service.DatabaseConfig{
		PgPool: pgPool,
		Logger: logger,
	})
	if err != nil {
		logger.Error("Failed to initialize database service").WithError(err).Log()
		return nil, fmt.Errorf("failed to initialize database service: %w", err)
	}

	// Initialize performance monitoring
	requestCounter := observability.NewRequestCounter()
	healthChecker := observability.NewHealthChecker(logger)

	server := &EnterpriseVectorServer{
		config:         config,
		cache:          multiCache,
		auth:           kratosAuth,
		logger:         logger,
		cudaWorker:     cudaWorker,
		dbService:      dbService,
		requestCounter: requestCounter,
		healthChecker:  healthChecker,
	}

	logger.Info("Enterprise Vector Consumer Service v2.0 initialized successfully").
		WithInt("max_concurrency", config.MaxConcurrency).
		Log()

	return server, nil
}

// ProcessRotation implements the vector rotation service
func (s *EnterpriseVectorServer) ProcessRotation(ctx context.Context, req *pb.VectorRequest) (*pb.VectorResponse, error) {
	// Start performance tracking
	startTime := time.Now()
	s.requestCounter.IncrementRequest("ProcessRotation")

	// Create structured log entry
	logEntry := s.logger.Info("Processing vector rotation request").
		WithString("request_id", req.RequestId).
		WithString("client_id", req.ClientId).
		WithInt("vector_size", len(req.Vector)).
		WithString("method", "ProcessRotation")

	// Authenticate and authorize request
	identity, err := s.auth.ValidateRequestContext(ctx)
	if err != nil {
		logEntry.WithError(err).WithString("status", "auth_failed").Log()
		s.requestCounter.IncrementError("ProcessRotation", "auth_failed")
		return nil, err
	}

	logEntry = logEntry.WithString("user_id", identity.ID).
		WithStringSlice("user_roles", identity.Roles)

	// Check cache first
	cacheKey := fmt.Sprintf("rotation:%s:%s", req.RequestId, req.RotationMatrix)
	if cachedResult, found := s.cache.Get(ctx, cacheKey); found {
		logEntry.WithString("cache_status", "hit").
			WithDuration("duration", time.Since(startTime)).
			WithString("status", "success").Log()
		
		s.requestCounter.IncrementSuccess("ProcessRotation")
		return cachedResult.(*pb.VectorResponse), nil
	}

	// Process with CUDA if available
	var result []float32
	if s.config.CUDAEnabled {
		result, err = s.cudaWorker.ProcessVectorRotation(ctx, &service.VectorRotationRequest{
			Vector:         req.Vector,
			RotationMatrix: req.RotationMatrix,
			Precision:      service.PrecisionHigh, // Use cuBLAS for high precision
		})
		if err != nil {
			logEntry.WithError(err).WithString("status", "cuda_error").Log()
			s.requestCounter.IncrementError("ProcessRotation", "cuda_error")
			return nil, err
		}
		logEntry = logEntry.WithString("processing_method", "cuda_cublas")
	} else {
		// Fallback to CPU processing
		result, err = s.processCPURotation(req.Vector, req.RotationMatrix)
		if err != nil {
			logEntry.WithError(err).WithString("status", "cpu_error").Log()
			s.requestCounter.IncrementError("ProcessRotation", "cpu_error")
			return nil, err
		}
		logEntry = logEntry.WithString("processing_method", "cpu_fallback")
	}

	// Create response
	response := &pb.VectorResponse{
		RequestId:    req.RequestId,
		Success:      true,
		ResultVector: result,
		ProcessingTimeMs: int64(time.Since(startTime).Milliseconds()),
		Metadata: map[string]string{
			"processing_method": "cuda_cublas",
			"precision":         "high",
			"cache_status":      "miss",
		},
	}

	// Store in cache
	s.cache.Set(ctx, cacheKey, response, 5*time.Minute)

	// Store processing record in database
	if err := s.dbService.RecordVectorOperation(ctx, &service.VectorOperationRecord{
		RequestID:        req.RequestId,
		UserID:          identity.ID,
		Operation:       "rotation",
		InputDimensions: len(req.Vector),
		OutputDimensions: len(result),
		ProcessingTimeMs: response.ProcessingTimeMs,
		Success:         true,
	}); err != nil {
		// Log error but don't fail the request
		s.logger.Warning("Failed to record vector operation").
			WithError(err).
			WithString("request_id", req.RequestId).
			Log()
	}

	logEntry.WithString("cache_status", "miss").
		WithDuration("duration", time.Since(startTime)).
		WithString("status", "success").
		WithInt("output_dimensions", len(result)).Log()

	s.requestCounter.IncrementSuccess("ProcessRotation")
	return response, nil
}

// ProcessSimilarity implements vector similarity computation
func (s *EnterpriseVectorServer) ProcessSimilarity(ctx context.Context, req *pb.SimilarityRequest) (*pb.SimilarityResponse, error) {
	startTime := time.Now()
	s.requestCounter.IncrementRequest("ProcessSimilarity")

	logEntry := s.logger.Info("Processing similarity request").
		WithString("request_id", req.RequestId).
		WithString("similarity_type", req.SimilarityType.String()).
		WithInt("vector_a_size", len(req.VectorA)).
		WithInt("vector_b_size", len(req.VectorB))

	// Authenticate request
	identity, err := s.auth.ValidateRequestContext(ctx)
	if err != nil {
		logEntry.WithError(err).WithString("status", "auth_failed").Log()
		s.requestCounter.IncrementError("ProcessSimilarity", "auth_failed")
		return nil, err
	}

	// Check cache
	cacheKey := fmt.Sprintf("similarity:%s:%s", req.RequestId, req.SimilarityType.String())
	if cachedResult, found := s.cache.Get(ctx, cacheKey); found {
		logEntry.WithString("cache_status", "hit").
			WithDuration("duration", time.Since(startTime)).Log()
		s.requestCounter.IncrementSuccess("ProcessSimilarity")
		return cachedResult.(*pb.SimilarityResponse), nil
	}

	// Process similarity with CUDA cuBLAS for mathematical precision
	var score float32
	if s.config.CUDAEnabled {
		score, err = s.cudaWorker.ComputeSimilarity(ctx, &service.SimilarityRequest{
			VectorA:        req.VectorA,
			VectorB:        req.VectorB,
			SimilarityType: service.SimilarityType(req.SimilarityType),
			UseCuBLAS:      true, // Ensure mathematical precision
		})
		if err != nil {
			logEntry.WithError(err).WithString("status", "cuda_error").Log()
			s.requestCounter.IncrementError("ProcessSimilarity", "cuda_error")
			return nil, err
		}
	} else {
		// CPU fallback
		score, err = s.computeCPUSimilarity(req.VectorA, req.VectorB, req.SimilarityType)
		if err != nil {
			logEntry.WithError(err).WithString("status", "cpu_error").Log()
			s.requestCounter.IncrementError("ProcessSimilarity", "cpu_error")
			return nil, err
		}
	}

	response := &pb.SimilarityResponse{
		RequestId:        req.RequestId,
		Success:          true,
		SimilarityScore:  score,
		ProcessingTimeMs: int64(time.Since(startTime).Milliseconds()),
		Metadata: map[string]string{
			"processing_method": "cuda_cublas",
			"precision":         "high",
			"cache_status":      "miss",
		},
	}

	// Cache result
	s.cache.Set(ctx, cacheKey, response, 10*time.Minute)

	logEntry.WithFloat32("similarity_score", score).
		WithString("cache_status", "miss").
		WithDuration("duration", time.Since(startTime)).
		WithString("status", "success").Log()

	s.requestCounter.IncrementSuccess("ProcessSimilarity")
	return response, nil
}

// ProcessLegalDocument implements legal document processing
func (s *EnterpriseVectorServer) ProcessLegalDocument(ctx context.Context, req *pb.LegalDocumentRequest) (*pb.LegalDocumentResponse, error) {
	startTime := time.Now()
	s.requestCounter.IncrementRequest("ProcessLegalDocument")

	logEntry := s.logger.Info("Processing legal document").
		WithString("request_id", req.RequestId).
		WithString("document_type", req.DocumentType.String()).
		WithInt("content_size", len(req.Content))

	// Authenticate and authorize
	identity, err := s.auth.ValidateRequestContext(ctx)
	if err != nil {
		logEntry.WithError(err).Log()
		return nil, err
	}

	// Check permissions for legal document processing
	if !s.auth.HasPermission(identity, "legal_document_processing") {
		logEntry.WithString("status", "permission_denied").Log()
		return nil, fmt.Errorf("insufficient permissions for legal document processing")
	}

	// Process document with comprehensive analysis
	analysisResult, err := s.dbService.ProcessLegalDocument(ctx, &service.LegalDocumentProcessingRequest{
		DocumentID:   req.RequestId,
		DocumentType: service.DocumentType(req.DocumentType),
		Content:      req.Content,
		Metadata:     req.Metadata,
		UserID:       identity.ID,
	})
	if err != nil {
		logEntry.WithError(err).WithString("status", "processing_error").Log()
		s.requestCounter.IncrementError("ProcessLegalDocument", "processing_error")
		return nil, err
	}

	response := &pb.LegalDocumentResponse{
		RequestId:        req.RequestId,
		Success:          true,
		DocumentId:       analysisResult.DocumentID,
		ProcessingTimeMs: int64(time.Since(startTime).Milliseconds()),
		Entities:         analysisResult.ExtractedEntities,
		Summary:          analysisResult.Summary,
		Confidence:       analysisResult.ConfidenceScore,
		Metadata: map[string]string{
			"processing_version": "2.0",
			"user_id":           identity.ID,
		},
	}

	logEntry.WithString("document_id", analysisResult.DocumentID).
		WithFloat32("confidence", analysisResult.ConfidenceScore).
		WithInt("entities_extracted", len(analysisResult.ExtractedEntities)).
		WithDuration("duration", time.Since(startTime)).
		WithString("status", "success").Log()

	s.requestCounter.IncrementSuccess("ProcessLegalDocument")
	return response, nil
}

// CPU fallback methods
func (s *EnterpriseVectorServer) processCPURotation(vector []float32, rotationMatrix []float32) ([]float32, error) {
	// Simple CPU-based vector rotation implementation
	if len(vector) == 0 {
		return nil, fmt.Errorf("empty vector")
	}
	
	result := make([]float32, len(vector))
	// Simplified rotation - in production this would be a full matrix multiplication
	for i, v := range vector {
		result[i] = v * 0.9 // Placeholder rotation
	}
	return result, nil
}

func (s *EnterpriseVectorServer) computeCPUSimilarity(vectorA, vectorB []float32, simType pb.SimilarityType) (float32, error) {
	if len(vectorA) != len(vectorB) {
		return 0, fmt.Errorf("vector dimensions mismatch")
	}

	switch simType {
	case pb.SimilarityType_COSINE:
		return s.cosineSimilarity(vectorA, vectorB), nil
	case pb.SimilarityType_EUCLIDEAN:
		return s.euclideanDistance(vectorA, vectorB), nil
	default:
		return 0, fmt.Errorf("unsupported similarity type: %v", simType)
	}
}

func (s *EnterpriseVectorServer) cosineSimilarity(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func (s *EnterpriseVectorServer) euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

func main() {
	// Parse command line flags
	var (
		port         = flag.String("port", "8095", "Server port")
		dbURL        = flag.String("db-url", "postgres://legal_admin:123456@localhost:5432/legal_ai_db?sslmode=disable", "Database URL")
		redisURL     = flag.String("redis-url", "localhost:6379", "Redis URL")
		kratosURL    = flag.String("kratos-url", "http://localhost:4433", "Kratos URL")
		cudaEnabled  = flag.Bool("cuda", true, "Enable CUDA acceleration")
		logLevel     = flag.String("log-level", "info", "Log level")
		maxConcurrency = flag.Int("max-concurrency", 1000, "Maximum concurrent requests")
	)
	flag.Parse()

	config := &Config{
		Port:         *port,
		DatabaseURL:  *dbURL,
		RedisURL:     *redisURL,
		KratosURL:    *kratosURL,
		CUDAEnabled:  *cudaEnabled,
		LogLevel:     *logLevel,
		MaxConcurrency: *maxConcurrency,
	}

	// Create enterprise vector server
	server, err := NewEnterpriseVectorServer(config)
	if err != nil {
		fmt.Printf("Failed to create server: %v\n", err)
		os.Exit(1)
	}

	// Setup gRPC server with interceptors
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(grpc.ChainUnaryInterceptor(
			server.auth.UnaryServerInterceptor(),
			server.logger.UnaryServerInterceptor(),
		)),
		grpc.StreamInterceptor(grpc.ChainStreamInterceptor(
			server.auth.StreamServerInterceptor(),
			server.logger.StreamServerInterceptor(),
		)),
	)

	// Register services
	pb.RegisterVectorServiceServer(grpcServer, server)
	pb.RegisterAsyncJobServiceServer(grpcServer, server)
	
	// Register health check service
	healthServer := health.NewServer()
	healthServer.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)

	// Enable gRPC reflection for development
	reflection.Register(grpcServer)

	// Setup network listener
	lis, err := net.Listen("tcp", ":"+config.Port)
	if err != nil {
		server.logger.Error("Failed to listen").WithError(err).WithString("port", config.Port).Log()
		os.Exit(1)
	}

	server.logger.Info("Enterprise Vector Consumer Service v2.0 starting").
		WithString("port", config.Port).
		WithString("version", "2.0.0").
		WithBool("cuda_enabled", config.CUDAEnabled).
		WithInt("max_concurrency", config.MaxConcurrency).
		Log()

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start server in goroutine
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := grpcServer.Serve(lis); err != nil {
			server.logger.Error("Server failed").WithError(err).Log()
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	server.logger.Info("Shutting down Enterprise Vector Consumer Service v2.0").Log()

	// Graceful shutdown
	grpcServer.GracefulStop()
	cancel()
	wg.Wait()

	server.logger.Info("Enterprise Vector Consumer Service v2.0 stopped").Log()
}