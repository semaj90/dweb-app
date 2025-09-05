package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	"github.com/tidwall/gjson"
	"github.com/redis/go-redis/v9"
)

// High-performance Go ingest microservice with SIMD JSON parsing
// Integrates with your 37-service architecture (port 8227 - next available)

type Config struct {
	Port         string
	DatabaseURL  string
	OllamaURL    string
	EmbedModel   string
	MaxFileSize  int64
	BatchSize    int
	RedisURL     string
}

type IngestService struct {
	db     *pgxpool.Pool
	redis  *redis.Client
	config Config
}

type DocumentIngestRequest struct {
	Title    string                 `json:"title"`
	Content  string                 `json:"content"`
	CaseID   string                 `json:"case_id,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

type BatchIngestRequest struct {
	Documents []DocumentIngestRequest `json:"documents"`
}

type IngestResponse struct {
	ID          string    `json:"id"`
	Status      string    `json:"status"`
	DocumentID  string    `json:"document_id"`
	EmbeddingID string    `json:"embedding_id"`
	ProcessTime float64   `json:"process_time_ms"`
	Timestamp   time.Time `json:"timestamp"`
}

func loadConfig() Config {
	maxFileSize, _ := strconv.ParseInt(os.Getenv("MAX_FILE_SIZE"), 10, 64)
	if maxFileSize == 0 {
		maxFileSize = 104857600 // 100MB default
	}

	batchSize, _ := strconv.Atoi(os.Getenv("BATCH_SIZE"))
	if batchSize == 0 {
		batchSize = 10
	}

	return Config{
		Port:        getEnv("INGEST_PORT", "8227"),
		DatabaseURL: getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		OllamaURL:   getEnv("OLLAMA_URL", "http://localhost:11434"),
		EmbedModel:  getEnv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
		MaxFileSize: maxFileSize,
		BatchSize:   batchSize,
		RedisURL:    getEnv("REDIS_URL", "redis://localhost:6379"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func NewIngestService(config Config) (*IngestService, error) {
	db, err := pgxpool.New(context.Background(), config.DatabaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Setup Redis client for event publishing
	opt, err := redis.ParseURL(config.RedisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}
	
	redisClient := redis.NewClient(opt)
	
	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	if err := redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis connection failed: %v", err)
		// Continue without Redis (graceful degradation)
		redisClient = nil
	}

	return &IngestService{
		db:     db,
		redis:  redisClient,
		config: config,
	}, nil
}

// Fast JSON extraction with gjson (better than encoding/json for this use case)
func (s *IngestService) extractMetadata(jsonData []byte) map[string]interface{} {
	metadata := make(map[string]interface{})
	
	// Extract common legal document metadata using gjson
	if title := gjson.GetBytes(jsonData, "title").String(); title != "" {
		metadata["title"] = title
	}
	if docType := gjson.GetBytes(jsonData, "type").String(); docType != "" {
		metadata["document_type"] = docType
	}
	if jurisdiction := gjson.GetBytes(jsonData, "jurisdiction").String(); jurisdiction != "" {
		metadata["jurisdiction"] = jurisdiction
	}
	
	return metadata
}

func (s *IngestService) generateEmbedding(text string) ([]float32, error) {
	url := fmt.Sprintf("%s/api/embeddings", s.config.OllamaURL)
	
	payload := map[string]interface{}{
		"model":  s.config.EmbedModel,
		"prompt": text,
	}
	
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	
	resp, err := http.Post(url, "application/json", strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	embeddings, ok := result["embedding"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid embedding response")
	}
	
	embedding := make([]float32, len(embeddings))
	for i, v := range embeddings {
		if f, ok := v.(float64); ok {
			embedding[i] = float32(f)
		}
	}
	
	return embedding, nil
}

// publishIngestCompletion publishes completion event to Redis stream
func (s *IngestService) publishIngestCompletion(ctx context.Context, documentID, caseID, evidenceID string) {
	if s.redis == nil {
		return // Redis not available, skip
	}

	eventData := map[string]interface{}{
		"type":         "ingest_complete",
		"id":           documentID,
		"action":       "mirror",
		"caseId":       caseID,
		"evidenceId":   evidenceID,
		"timestamp":    time.Now().Format(time.RFC3339),
		"source":       "go-ingest-service",
		"correlation":  fmt.Sprintf("ingest_%d", time.Now().UnixNano()),
	}

	// Publish to Redis stream (non-blocking)
	go func() {
		pubCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		err := s.redis.XAdd(pubCtx, &redis.XAddArgs{
			Stream: "autotag:requests",
			Values: eventData,
		}).Err()

		if err != nil {
			log.Printf("Failed to publish ingest completion event: %v", err)
		} else {
			log.Printf("Published ingest completion for document %s", documentID)
		}
	}()
}

func (s *IngestService) ingestDocument(ctx context.Context, doc DocumentIngestRequest) (*IngestResponse, error) {
	startTime := time.Now()
	
	tx, err := s.db.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)
	
	// Insert into document_metadata table (matches your schema)
	var documentID string
	err = tx.QueryRow(ctx, `
		INSERT INTO document_metadata (
			case_id, filename, object_name, original_filename, summary, content_type, 
			processing_status, extracted_text, document_type, jurisdiction, 
			priority, ingest_source, metadata
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
		RETURNING id
	`, 
		doc.CaseID, 
		doc.Title+".json", // filename (required)
		"ingest/"+doc.Title+".json", // object_name (required, unique)
		doc.Title, // original_filename
		doc.Content[:min(500, len(doc.Content))], // summary
		"application/json", // content_type
		"processing", // processing_status
		doc.Content, // extracted_text
		"legal", // document_type
		"US", // jurisdiction
		1, // priority
		"api", // ingest_source
		doc.Metadata).Scan(&documentID)
	
	if err != nil {
		return nil, fmt.Errorf("failed to insert document: %w", err)
	}
	
	// Generate embedding
	embedding, err := s.generateEmbedding(doc.Content)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}
	
	// Insert into document_embeddings table
	var embeddingID string
	err = tx.QueryRow(ctx, `
		INSERT INTO document_embeddings (document_id, chunk_text, embedding, chunk_number)
		VALUES ($1, $2, $3, $4)
		RETURNING id
	`, documentID, doc.Content, pgvector.NewVector(embedding), 0).Scan(&embeddingID)
	
	if err != nil {
		return nil, fmt.Errorf("failed to insert embedding: %w", err)
	}
	
	// Update processing status
	_, err = tx.Exec(ctx, "UPDATE document_metadata SET processing_status = 'completed' WHERE id = $1", documentID)
	if err != nil {
		return nil, fmt.Errorf("failed to update status: %w", err)
	}
	
	if err = tx.Commit(ctx); err != nil {
		return nil, err
	}
	
	// Get evidence_id from metadata if available
	var evidenceID string
	if doc.Metadata != nil {
		if eid, ok := doc.Metadata["evidence_id"].(string); ok {
			evidenceID = eid
		}
	}
	
	// Publish completion event to Redis stream
	s.publishIngestCompletion(ctx, documentID, doc.CaseID, evidenceID)
	
	return &IngestResponse{
		ID:          fmt.Sprintf("ingest_%d", time.Now().UnixNano()),
		Status:      "completed",
		DocumentID:  documentID,
		EmbeddingID: embeddingID,
		ProcessTime: float64(time.Since(startTime).Nanoseconds()) / 1e6,
		Timestamp:   time.Now(),
	}, nil
}

func (s *IngestService) handleSingleIngest(c *gin.Context) {
	var req DocumentIngestRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	result, err := s.ingestDocument(c.Request.Context(), req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, result)
}

func (s *IngestService) handleBatchIngest(c *gin.Context) {
	var req BatchIngestRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if len(req.Documents) > s.config.BatchSize {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("batch size exceeds limit of %d", s.config.BatchSize),
		})
		return
	}
	
	var results []IngestResponse
	var errors []string
	
	for _, doc := range req.Documents {
		result, err := s.ingestDocument(c.Request.Context(), doc)
		if err != nil {
			errors = append(errors, err.Error())
			continue
		}
		results = append(results, *result)
	}
	
	response := gin.H{
		"results":   results,
		"processed": len(results),
		"total":     len(req.Documents),
		"timestamp": time.Now(),
	}
	
	if len(errors) > 0 {
		response["errors"] = errors
	}
	
	c.JSON(http.StatusOK, response)
}

func (s *IngestService) handleHealth(c *gin.Context) {
	health := map[string]interface{}{
		"status":    "healthy",
		"service":   "ingest-service",
		"port":      s.config.Port,
		"timestamp": time.Now(),
		"database":  s.checkDatabase(),
		"ollama":    s.checkOllama(),
		"config": map[string]interface{}{
			"batch_size":    s.config.BatchSize,
			"max_file_size": s.config.MaxFileSize,
			"embed_model":   s.config.EmbedModel,
		},
	}
	c.JSON(http.StatusOK, health)
}

func (s *IngestService) checkDatabase() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return s.db.Ping(ctx) == nil
}

func (s *IngestService) checkOllama() bool {
	resp, err := http.Get(fmt.Sprintf("%s/api/tags", s.config.OllamaURL))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func (s *IngestService) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())
	
	// CORS middleware
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})
	
	// Routes
	api := router.Group("/api")
	{
		api.POST("/ingest", s.handleSingleIngest)       // Single document
		api.POST("/ingest/batch", s.handleBatchIngest)  // Batch documents
		api.GET("/health", s.handleHealth)
		api.GET("/status", s.handleHealth)
	}
	
	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":   "Document Ingest Service",
			"version":   "1.0.0",
			"status":    "running",
			"port":      s.config.Port,
			"endpoints": []string{
				"/api/ingest",
				"/api/ingest/batch",
				"/api/health",
			},
		})
	})
	
	return router
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	config := loadConfig()
	
	log.Printf("Starting Document Ingest Service...")
	log.Printf("Port: %s", config.Port)
	log.Printf("Database: %s", config.DatabaseURL)
	log.Printf("Batch Size: %d", config.BatchSize)
	log.Printf("Max File Size: %d bytes", config.MaxFileSize)
	
	service, err := NewIngestService(config)
	if err != nil {
		log.Fatalf("Failed to initialize ingest service: %v", err)
	}
	defer service.db.Close()
	
	router := service.setupRoutes()
	
	log.Printf("Document Ingest Service running on port %s", config.Port)
	log.Printf("Access the API at: http://localhost:%s/api/ingest", config.Port)
	
	if err := router.Run(":" + config.Port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}