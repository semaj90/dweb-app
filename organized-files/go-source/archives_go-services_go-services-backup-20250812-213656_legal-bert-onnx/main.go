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
	"runtime"
	"strconv"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

// Legal-BERT ONNX Service for high-performance legal text embeddings
type LegalBERTService struct {
	httpServer *http.Server
	grpcServer *grpc.Server
	onnxModel  *ONNXModel
	config     *ServiceConfig
}

type ServiceConfig struct {
	Port           int    `json:"port"`
	ModelPath      string `json:"model_path"`
	TensorThreads  int    `json:"tensor_threads"`
	CacheSize      string `json:"cache_size"`
	EnableMetrics  bool   `json:"enable_metrics"`
	EnableGRPC     bool   `json:"enable_grpc"`
}

type ONNXModel struct {
	ModelPath   string
	InputShape  []int64
	OutputShape []int64
	Tokenizer   *Tokenizer
}

type Tokenizer struct {
	VocabSize int
	MaxLength int
}

type EmbedRequest struct {
	Text  string   `json:"text,omitempty"`
	Texts []string `json:"texts,omitempty"`
}

type EmbedResponse struct {
	Embedding  []float32   `json:"embedding,omitempty"`
	Embeddings [][]float32 `json:"embeddings,omitempty"`
	Dimensions int         `json:"dimensions"`
	Model      string      `json:"model"`
	ProcessingTimeMs int64 `json:"processing_time_ms"`
}

func main() {
	// Load configuration
	config := loadConfig()
	
	// Initialize ONNX model
	model, err := initializeONNXModel(config.ModelPath)
	if err != nil {
		log.Fatalf("Failed to initialize ONNX model: %v", err)
	}

	// Create service
	service := &LegalBERTService{
		onnxModel: model,
		config:    config,
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start HTTP server
	go service.startHTTPServer(ctx)

	// Start gRPC server if enabled
	if config.EnableGRPC {
		go service.startGRPCServer(ctx)
	}

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down Legal-BERT ONNX service...")
	cancel()

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if service.httpServer != nil {
		service.httpServer.Shutdown(shutdownCtx)
	}
	if service.grpcServer != nil {
		service.grpcServer.GracefulStop()
	}

	log.Println("Legal-BERT ONNX service stopped")
}

func loadConfig() *ServiceConfig {
	config := &ServiceConfig{
		Port:          8081,
		ModelPath:     "./models/legal-bert.onnx",
		TensorThreads: runtime.NumCPU(),
		CacheSize:     "2GB",
		EnableMetrics: true,
		EnableGRPC:    true,
	}

	// Override from environment
	if port := os.Getenv("PORT"); port != "" {
		if p, err := strconv.Atoi(port); err == nil {
			config.Port = p
		}
	}

	if modelPath := os.Getenv("MODEL_PATH"); modelPath != "" {
		config.ModelPath = modelPath
	}

	if threads := os.Getenv("ONNX_THREADS"); threads != "" {
		if t, err := strconv.Atoi(threads); err == nil {
			config.TensorThreads = t
		}
	}

	if cacheSize := os.Getenv("TENSOR_CACHE_SIZE"); cacheSize != "" {
		config.CacheSize = cacheSize
	}

	return config
}

func initializeONNXModel(modelPath string) (*ONNXModel, error) {
	// Check if model file exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Printf("ONNX model not found at %s, using mock implementation", modelPath)
		return &ONNXModel{
			ModelPath:   modelPath,
			InputShape:  []int64{1, 512}, // BERT-like input shape
			OutputShape: []int64{1, 768}, // Legal-BERT embedding dimension
			Tokenizer: &Tokenizer{
				VocabSize: 30522, // BERT vocab size
				MaxLength: 512,
			},
		}, nil
	}

	// In a real implementation, this would load the actual ONNX model
	// using github.com/owulveryck/onnx-go or similar
	log.Printf("Loading ONNX model from %s", modelPath)
	
	return &ONNXModel{
		ModelPath:   modelPath,
		InputShape:  []int64{1, 512},
		OutputShape: []int64{1, 768},
		Tokenizer: &Tokenizer{
			VocabSize: 30522,
			MaxLength: 512,
		},
	}, nil
}

func (s *LegalBERTService) startHTTPServer(ctx context.Context) {
	// Setup Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// CORS middleware
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":      "healthy",
			"service":     "legal-bert-onnx",
			"version":     "1.0.0",
			"model_path":  s.config.ModelPath,
			"uptime":      time.Since(time.Now()).String(),
		})
	})

	// Embedding endpoint
	router.POST("/embed", s.handleEmbed)
	router.POST("/embeddings", s.handleEmbed) // Alternative endpoint name

	// Model info endpoint
	router.GET("/model/info", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"model_path":    s.onnxModel.ModelPath,
			"input_shape":   s.onnxModel.InputShape,
			"output_shape":  s.onnxModel.OutputShape,
			"vocab_size":    s.onnxModel.Tokenizer.VocabSize,
			"max_length":    s.onnxModel.Tokenizer.MaxLength,
			"tensor_threads": s.config.TensorThreads,
		})
	})

	// Start HTTP server
	s.httpServer = &http.Server{
		Addr:    fmt.Sprintf(":%d", s.config.Port),
		Handler: router,
	}

	log.Printf("Legal-BERT HTTP server starting on port %d", s.config.Port)
	if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("HTTP server failed: %v", err)
	}
}

func (s *LegalBERTService) startGRPCServer(ctx context.Context) {
	grpcPort := s.config.Port + 1000 // Use different port for gRPC
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", grpcPort))
	if err != nil {
		log.Fatalf("Failed to listen on gRPC port %d: %v", grpcPort, err)
	}

	s.grpcServer = grpc.NewServer()
	
	// Register health service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(s.grpcServer, healthServer)
	healthServer.SetServingStatus("legal-bert", grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for debugging
	reflection.Register(s.grpcServer)

	log.Printf("Legal-BERT gRPC server starting on port %d", grpcPort)
	if err := s.grpcServer.Serve(lis); err != nil {
		log.Fatalf("gRPC server failed: %v", err)
	}
}

func (s *LegalBERTService) handleEmbed(c *gin.Context) {
	startTime := time.Now()
	
	var req EmbedRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request format", "details": err.Error()})
		return
	}

	// Validate input
	if req.Text == "" && len(req.Texts) == 0 {
		c.JSON(400, gin.H{"error": "Either 'text' or 'texts' must be provided"})
		return
	}

	var embeddings [][]float32
	var err error

	if req.Text != "" {
		// Single text embedding
		embedding, err := s.generateEmbedding(req.Text)
		if err != nil {
			c.JSON(500, gin.H{"error": "Failed to generate embedding", "details": err.Error()})
			return
		}
		embeddings = [][]float32{embedding}
	} else {
		// Batch text embeddings
		embeddings, err = s.generateBatchEmbeddings(req.Texts)
		if err != nil {
			c.JSON(500, gin.H{"error": "Failed to generate batch embeddings", "details": err.Error()})
			return
		}
	}

	processingTime := time.Since(startTime).Milliseconds()

	// Format response
	response := EmbedResponse{
		Dimensions:       len(embeddings[0]),
		Model:           "legal-bert-onnx",
		ProcessingTimeMs: processingTime,
	}

	if req.Text != "" {
		response.Embedding = embeddings[0]
	} else {
		response.Embeddings = embeddings
	}

	c.JSON(200, response)
}

func (s *LegalBERTService) generateEmbedding(text string) ([]float32, error) {
	// Mock implementation - in production, this would use actual ONNX inference
	// This generates a deterministic embedding based on text hash for consistency
	
	// Tokenize text (simplified)
	tokens := s.tokenize(text)
	
	// Generate mock embedding (768 dimensions for Legal-BERT)
	embedding := make([]float32, 768)
	
	// Create deterministic but diverse embeddings based on input
	hash := 0
	for _, char := range text {
		hash = hash*31 + int(char)
	}
	
	for i := range embedding {
		// Generate values between -1.0 and 1.0
		seed := float32((hash + i*17) % 2000 - 1000) / 1000.0
		embedding[i] = seed * 0.1 // Scale down for realistic embeddings
	}
	
	// Normalize embedding
	embedding = normalizeVector(embedding)
	
	log.Printf("Generated embedding for text (length: %d, tokens: %d, dims: %d)", 
		len(text), len(tokens), len(embedding))
	
	return embedding, nil
}

func (s *LegalBERTService) generateBatchEmbeddings(texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	
	for i, text := range texts {
		embedding, err := s.generateEmbedding(text)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for text %d: %w", i, err)
		}
		embeddings[i] = embedding
	}
	
	return embeddings, nil
}

func (s *LegalBERTService) tokenize(text string) []int {
	// Simplified tokenization for mock implementation
	// In production, this would use proper BERT tokenization
	words := []rune(text)
	tokens := make([]int, 0, len(words))
	
	for _, word := range words {
		// Simple character-to-token mapping
		tokenId := int(word) % s.onnxModel.Tokenizer.VocabSize
		tokens = append(tokens, tokenId)
		
		if len(tokens) >= s.onnxModel.Tokenizer.MaxLength {
			break
		}
	}
	
	return tokens
}

func normalizeVector(vec []float32) []float32 {
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	
	if norm == 0 {
		return vec
	}
	
	norm = float32(1.0 / (norm + 1e-12)) // Add small epsilon to avoid division by zero
	
	normalized := make([]float32, len(vec))
	for i, v := range vec {
		normalized[i] = v * norm
	}
	
	return normalized
}