package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	pb "github.com/legal-ai/inference-gateway/proto"
)

const (
	// Server configuration
	GRPC_PORT     = ":50051"
	HTTP_PORT     = ":8080"
	METRICS_PORT  = ":9090"
	WORKER_URL    = "http://127.0.0.1:5001"
	REDIS_URL     = "redis://localhost:6379"
	
	// Performance tuning
	MAX_CONCURRENT_REQUESTS = 100
	REQUEST_TIMEOUT        = 30 * time.Second
	CACHE_TTL             = 5 * time.Minute
)

// Metrics
var (
	requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "inference_requests_total",
			Help: "Total number of inference requests",
		},
		[]string{"method", "status"},
	)
	
	requestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "inference_request_duration_seconds",
			Help: "Duration of inference requests",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method"},
	)
	
	activeRequests = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "inference_active_requests",
			Help: "Number of active inference requests",
		},
	)
	
	cacheHits = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "inference_cache_hits_total",
			Help: "Total number of cache hits",
		},
	)
	
	cacheMisses = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "inference_cache_misses_total",
			Help: "Total number of cache misses",
		},
	)
)

func init() {
	prometheus.MustRegister(requestCounter)
	prometheus.MustRegister(requestDuration)
	prometheus.MustRegister(activeRequests)
	prometheus.MustRegister(cacheHits)
	prometheus.MustRegister(cacheMisses)
}

// InferenceServer implements the gRPC service
type InferenceServer struct {
	pb.UnimplementedInferenceServiceServer
	httpClient *http.Client
	redisClient *redis.Client
	requestLimiter chan struct{}
	stats *ServerStats
	mu sync.RWMutex
}

type ServerStats struct {
	TotalRequests   int64     `json:"total_requests"`
	ActiveRequests  int64     `json:"active_requests"`
	StartTime       time.Time `json:"start_time"`
	LastRequestTime time.Time `json:"last_request_time"`
}

// NewInferenceServer creates a new server instance
func NewInferenceServer() (*InferenceServer, error) {
	// Redis client
	opt, err := redis.ParseURL(REDIS_URL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %v", err)
	}
	redisClient := redis.NewClient(opt)
	
	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis connection failed: %v", err)
	}
	
	return &InferenceServer{
		httpClient: &http.Client{
			Timeout: REQUEST_TIMEOUT,
		},
		redisClient: redisClient,
		requestLimiter: make(chan struct{}, MAX_CONCURRENT_REQUESTS),
		stats: &ServerStats{
			StartTime: time.Now(),
		},
	}, nil
}

// Generate handles unary text generation
func (s *InferenceServer) Generate(ctx context.Context, req *pb.GenerateRequest) (*pb.GenerateResponse, error) {
	timer := prometheus.NewTimer(requestDuration.WithLabelValues("generate"))
	defer timer.ObserveDuration()
	
	activeRequests.Inc()
	defer activeRequests.Dec()
	
	requestCounter.WithLabelValues("generate", "attempt").Inc()
	
	s.mu.Lock()
	s.stats.TotalRequests++
	s.stats.ActiveRequests++
	s.stats.LastRequestTime = time.Now()
	s.mu.Unlock()
	
	defer func() {
		s.mu.Lock()
		s.stats.ActiveRequests--
		s.mu.Unlock()
	}()
	
	// Rate limiting
	select {
	case s.requestLimiter <- struct{}{}:
		defer func() { <-s.requestLimiter }()
	case <-ctx.Done():
		requestCounter.WithLabelValues("generate", "timeout").Inc()
		return nil, status.Error(codes.DeadlineExceeded, "Request timeout due to rate limiting")
	}
	
	// Check cache first
	cacheKey := s.generateCacheKey(req)
	if cached := s.getCachedResponse(ctx, cacheKey); cached != nil {
		cacheHits.Inc()
		requestCounter.WithLabelValues("generate", "cache_hit").Inc()
		return cached, nil
	}
	
	cacheMisses.Inc()
	
	// Forward to Python worker
	result, err := s.forwardToWorker(ctx, req)
	if err != nil {
		requestCounter.WithLabelValues("generate", "error").Inc()
		return nil, err
	}
	
	// Cache successful result
	s.cacheResponse(ctx, cacheKey, result)
	
	requestCounter.WithLabelValues("generate", "success").Inc()
	return result, nil
}

// StreamGenerate handles streaming text generation
func (s *InferenceServer) StreamGenerate(req *pb.GenerateRequest, stream pb.InferenceService_StreamGenerateServer) error {
	timer := prometheus.NewTimer(requestDuration.WithLabelValues("stream_generate"))
	defer timer.ObserveDuration()
	
	activeRequests.Inc()
	defer activeRequests.Dec()
	
	requestCounter.WithLabelValues("stream_generate", "attempt").Inc()
	
	s.mu.Lock()
	s.stats.TotalRequests++
	s.stats.ActiveRequests++
	s.stats.LastRequestTime = time.Now()
	s.mu.Unlock()
	
	defer func() {
		s.mu.Lock()
		s.stats.ActiveRequests--
		s.mu.Unlock()
	}()
	
	// Rate limiting
	select {
	case s.requestLimiter <- struct{}{}:
		defer func() { <-s.requestLimiter }()
	case <-stream.Context().Done():
		requestCounter.WithLabelValues("stream_generate", "timeout").Inc()
		return status.Error(codes.DeadlineExceeded, "Request timeout due to rate limiting")
	}
	
	// Create HTTP request to worker
	reqBody := map[string]interface{}{
		"prompt":      req.Prompt,
		"model":       req.Model,
		"temperature": req.Temperature,
		"max_tokens":  req.MaxTokens,
		"stream":      true,
		"metadata":    req.Metadata,
	}
	
	jsonBody, _ := json.Marshal(reqBody)
	httpReq, err := http.NewRequestWithContext(stream.Context(), "POST", WORKER_URL+"/stream_generate", 
		bytes.NewBuffer(jsonBody))
	if err != nil {
		requestCounter.WithLabelValues("stream_generate", "error").Inc()
		return status.Error(codes.Internal, fmt.Sprintf("Failed to create request: %v", err))
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	
	// Execute request
	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		requestCounter.WithLabelValues("stream_generate", "error").Inc()
		return status.Error(codes.Unavailable, fmt.Sprintf("Worker unavailable: %v", err))
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		requestCounter.WithLabelValues("stream_generate", "error").Inc()
		return status.Error(codes.Internal, fmt.Sprintf("Worker error: %d", resp.StatusCode))
	}
	
	// Stream response
	decoder := json.NewDecoder(resp.Body)
	requestId := fmt.Sprintf("req_%d", time.Now().UnixNano())
	
	for {
		select {
		case <-stream.Context().Done():
			requestCounter.WithLabelValues("stream_generate", "cancelled").Inc()
			return status.Error(codes.Cancelled, "Stream cancelled by client")
		default:
		}
		
		var chunk map[string]interface{}
		if err := decoder.Decode(&chunk); err != nil {
			if err == io.EOF {
				break
			}
			requestCounter.WithLabelValues("stream_generate", "error").Inc()
			return status.Error(codes.Internal, fmt.Sprintf("Failed to decode chunk: %v", err))
		}
		
		// Convert to proto chunk
		grpcChunk := &pb.GenerateChunk{
			Token:     getString(chunk, "token"),
			Done:      getBool(chunk, "done"),
			RequestId: requestId,
			Metadata:  getStringMap(chunk, "metadata"),
			CompletionProgress: getFloat32(chunk, "progress"),
		}
		
		if err := stream.Send(grpcChunk); err != nil {
			requestCounter.WithLabelValues("stream_generate", "error").Inc()
			return status.Error(codes.Internal, fmt.Sprintf("Failed to send chunk: %v", err))
		}
		
		if grpcChunk.Done {
			break
		}
	}
	
	requestCounter.WithLabelValues("stream_generate", "success").Inc()
	return nil
}

// BatchEmbed handles batch embedding generation
func (s *InferenceServer) BatchEmbed(ctx context.Context, req *pb.BatchEmbedRequest) (*pb.BatchEmbedResponse, error) {
	timer := prometheus.NewTimer(requestDuration.WithLabelValues("batch_embed"))
	defer timer.ObserveDuration()
	
	activeRequests.Inc()
	defer activeRequests.Dec()
	
	requestCounter.WithLabelValues("batch_embed", "attempt").Inc()
	
	// Rate limiting
	select {
	case s.requestLimiter <- struct{}{}:
		defer func() { <-s.requestLimiter }()
	case <-ctx.Done():
		requestCounter.WithLabelValues("batch_embed", "timeout").Inc()
		return nil, status.Error(codes.DeadlineExceeded, "Request timeout")
	}
	
	// Forward to worker
	reqBody := map[string]interface{}{
		"texts":     req.Texts,
		"model":     req.Model,
		"normalize": req.Normalize,
		"user_id":   req.UserId,
	}
	
	jsonBody, _ := json.Marshal(reqBody)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", WORKER_URL+"/batch_embed", 
		bytes.NewBuffer(jsonBody))
	if err != nil {
		requestCounter.WithLabelValues("batch_embed", "error").Inc()
		return nil, status.Error(codes.Internal, fmt.Sprintf("Failed to create request: %v", err))
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		requestCounter.WithLabelValues("batch_embed", "error").Inc()
		return nil, status.Error(codes.Unavailable, fmt.Sprintf("Worker unavailable: %v", err))
	}
	defer resp.Body.Close()
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		requestCounter.WithLabelValues("batch_embed", "error").Inc()
		return nil, status.Error(codes.Internal, fmt.Sprintf("Failed to decode response: %v", err))
	}
	
	// Convert to proto response
	embeddings := make([]*pb.Embedding, 0)
	if embeds, ok := result["embeddings"].([]interface{}); ok {
		for i, embed := range embeds {
			if embedMap, ok := embed.(map[string]interface{}); ok {
				vector := make([]float32, 0)
				if vecData, ok := embedMap["vector"].([]interface{}); ok {
					for _, v := range vecData {
						if f, ok := v.(float64); ok {
							vector = append(vector, float32(f))
						}
					}
				}
				
				embeddings = append(embeddings, &pb.Embedding{
					Vector: vector,
					Text:   req.Texts[i],
					Id:     fmt.Sprintf("embed_%d", i),
				})
			}
		}
	}
	
	requestCounter.WithLabelValues("batch_embed", "success").Inc()
	return &pb.BatchEmbedResponse{
		Embeddings:     embeddings,
		Dimensions:     int32(getInt(result, "dimensions")),
		Success:        getBool(result, "success"),
		ProcessingTime: getFloat32(result, "processing_time"),
	}, nil
}

// SimilaritySearch handles similarity search requests
func (s *InferenceServer) SimilaritySearch(ctx context.Context, req *pb.SimilarityRequest) (*pb.SimilarityResponse, error) {
	timer := prometheus.NewTimer(requestDuration.WithLabelValues("similarity_search"))
	defer timer.ObserveDuration()
	
	activeRequests.Inc()
	defer activeRequests.Dec()
	
	requestCounter.WithLabelValues("similarity_search", "attempt").Inc()
	
	// This would integrate with your existing SvelteKit similarity service
	// For now, return a placeholder response
	requestCounter.WithLabelValues("similarity_search", "success").Inc()
	return &pb.SimilarityResponse{
		Results:        make([]*pb.SimilarityResult, 0),
		Success:        true,
		ProcessingTime: 0.1,
		CacheHit:       false,
	}, nil
}

// HealthCheck returns server health status
func (s *InferenceServer) HealthCheck(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	// Check worker health
	workerHealthy := s.checkWorkerHealth(ctx)
	
	// Check Redis health
	redisHealthy := true
	if err := s.redisClient.Ping(ctx).Err(); err != nil {
		redisHealthy = false
	}
	
	services := map[string]string{
		"worker": "healthy",
		"redis":  "healthy",
	}
	
	if !workerHealthy {
		services["worker"] = "unhealthy"
	}
	if !redisHealthy {
		services["redis"] = "unhealthy"
	}
	
	s.mu.RLock()
	stats := &pb.SystemStats{
		ActiveRequests: s.stats.ActiveRequests,
		TotalRequests:  s.stats.TotalRequests,
	}
	s.mu.RUnlock()
	
	return &pb.HealthResponse{
		Healthy:  workerHealthy && redisHealthy,
		Version:  "1.0.0",
		Services: services,
		Stats:    stats,
	}, nil
}

// ClearCache handles cache clearing requests
func (s *InferenceServer) ClearCache(ctx context.Context, req *pb.ClearCacheRequest) (*pb.ClearCacheResponse, error) {
	pattern := req.Pattern
	if pattern == "" {
		pattern = "*"
	}
	
	var keys []string
	var err error
	
	switch req.CacheType {
	case "tokens":
		keys, err = s.redisClient.Keys(ctx, "tokens:"+pattern).Result()
	case "embeddings":
		keys, err = s.redisClient.Keys(ctx, "embed:"+pattern).Result()
	case "results":
		keys, err = s.redisClient.Keys(ctx, "result:"+pattern).Result()
	case "all":
		keys, err = s.redisClient.Keys(ctx, "*"+pattern).Result()
	default:
		return &pb.ClearCacheResponse{
			Success: false,
			Message: "Invalid cache type",
		}, nil
	}
	
	if err != nil {
		return &pb.ClearCacheResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to get keys: %v", err),
		}, nil
	}
	
	if len(keys) > 0 {
		deleted, err := s.redisClient.Del(ctx, keys...).Result()
		if err != nil {
			return &pb.ClearCacheResponse{
				Success: false,
				Message: fmt.Sprintf("Failed to delete keys: %v", err),
			}, nil
		}
		
		return &pb.ClearCacheResponse{
			Success:       true,
			ClearedEntries: int32(deleted),
			Message:       fmt.Sprintf("Cleared %d entries", deleted),
		}, nil
	}
	
	return &pb.ClearCacheResponse{
		Success:       true,
		ClearedEntries: 0,
		Message:       "No matching entries found",
	}, nil
}

// Helper methods
import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"io"
)

func (s *InferenceServer) generateCacheKey(req *pb.GenerateRequest) string {
	data := fmt.Sprintf("%s:%s:%.2f:%d", req.Prompt, req.Model, req.Temperature, req.MaxTokens)
	hash := sha256.Sum256([]byte(data))
	return "result:" + hex.EncodeToString(hash[:])[:16]
}

func (s *InferenceServer) getCachedResponse(ctx context.Context, key string) *pb.GenerateResponse {
	data, err := s.redisClient.Get(ctx, key).Result()
	if err != nil {
		return nil
	}
	
	var result pb.GenerateResponse
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		return nil
	}
	
	return &result
}

func (s *InferenceServer) cacheResponse(ctx context.Context, key string, resp *pb.GenerateResponse) {
	data, err := json.Marshal(resp)
	if err != nil {
		return
	}
	
	s.redisClient.Set(ctx, key, data, CACHE_TTL)
}

func (s *InferenceServer) forwardToWorker(ctx context.Context, req *pb.GenerateRequest) (*pb.GenerateResponse, error) {
	reqBody := map[string]interface{}{
		"prompt":      req.Prompt,
		"model":       req.Model,
		"temperature": req.Temperature,
		"max_tokens":  req.MaxTokens,
		"metadata":    req.Metadata,
	}
	
	jsonBody, _ := json.Marshal(reqBody)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", WORKER_URL+"/generate", 
		bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("Failed to create request: %v", err))
	}
	
	httpReq.Header.Set("Content-Type", "application/json")
	
	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, status.Error(codes.Unavailable, fmt.Sprintf("Worker unavailable: %v", err))
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, status.Error(codes.Internal, fmt.Sprintf("Worker error: %d", resp.StatusCode))
	}
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("Failed to decode response: %v", err))
	}
	
	return &pb.GenerateResponse{
		Id:             getString(result, "id"),
		Text:           getString(result, "text"),
		Success:        getBool(result, "success"),
		TokensUsed:     int32(getInt(result, "tokens_used")),
		ProcessingTime: getFloat32(result, "processing_time"),
		Metadata:       getStringMap(result, "metadata"),
	}, nil
}

func (s *InferenceServer) checkWorkerHealth(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, "GET", WORKER_URL+"/health", nil)
	if err != nil {
		return false
	}
	
	resp, err := s.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	
	return resp.StatusCode == http.StatusOK
}

// Utility functions for type conversion
func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getBool(m map[string]interface{}, key string) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return false
}

func getInt(m map[string]interface{}, key string) int {
	if v, ok := m[key].(float64); ok {
		return int(v)
	}
	return 0
}

func getFloat32(m map[string]interface{}, key string) float32 {
	if v, ok := m[key].(float64); ok {
		return float32(v)
	}
	return 0
}

func getStringMap(m map[string]interface{}, key string) map[string]string {
	result := make(map[string]string)
	if v, ok := m[key].(map[string]interface{}); ok {
		for k, val := range v {
			if s, ok := val.(string); ok {
				result[k] = s
			}
		}
	}
	return result
}

// HTTP REST Gateway
func setupHTTPGateway(server *InferenceServer) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	
	// CORS middleware
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})
	
	// Health endpoint
	r.GET("/health", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		health, err := server.HealthCheck(ctx, &pb.HealthRequest{Detailed: true})
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, health)
	})
	
	// Generation endpoint
	r.POST("/api/generate", func(c *gin.Context) {
		var req pb.GenerateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		
		ctx, cancel := context.WithTimeout(context.Background(), REQUEST_TIMEOUT)
		defer cancel()
		
		resp, err := server.Generate(ctx, &req)
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, resp)
	})
	
	return r
}

// Main function
func main() {
	log.Printf("Starting Legal AI Inference Gateway...")
	
	// Create server
	server, err := NewInferenceServer()
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}
	
	// Start gRPC server
	lis, err := net.Listen("tcp", GRPC_PORT)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	
	grpcServer := grpc.NewServer(
		grpc.MaxConcurrentStreams(MAX_CONCURRENT_REQUESTS),
	)
	pb.RegisterInferenceServiceServer(grpcServer, server)
	
	go func() {
		log.Printf("gRPC server listening on %s", GRPC_PORT)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()
	
	// Start HTTP gateway
	httpServer := &http.Server{
		Addr:    HTTP_PORT,
		Handler: setupHTTPGateway(server),
	}
	
	go func() {
		log.Printf("HTTP gateway listening on %s", HTTP_PORT)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to serve HTTP: %v", err)
		}
	}()
	
	// Start metrics server
	go func() {
		log.Printf("Metrics server listening on %s", METRICS_PORT)
		http.Handle("/metrics", promhttp.Handler())
		if err := http.ListenAndServe(METRICS_PORT, nil); err != nil {
			log.Printf("Metrics server error: %v", err)
		}
	}()
	
	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	
	log.Println("Shutting down servers...")
	
	// Shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	if err := httpServer.Shutdown(ctx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}
	
	// Shutdown gRPC server
	grpcServer.GracefulStop()
	
	// Close Redis connection
	server.redisClient.Close()
	
	log.Println("Gateway stopped")
}