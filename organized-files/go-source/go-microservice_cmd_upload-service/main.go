package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

type Config struct {
	Port         string
	DatabaseURL  string
	RedisURL     string
	OllamaURL    string
	MinIOURL     string
	EmbedModel   string
}

type UploadService struct {
	db     *pgxpool.Pool
	redis  *redis.Client
	config Config
}

type DocumentUpload struct {
	ID           string    `json:"id"`
	CaseID       string    `json:"case_id"`
	UserID       string    `json:"user_id"`
	Filename     string    `json:"filename"`
	FileSize     int64     `json:"file_size"`
	FileType     string    `json:"file_type"`
	Content      string    `json:"content,omitempty"`
	Summary      string    `json:"summary,omitempty"`
	Status       string    `json:"status"`
	ProcessedAt  time.Time `json:"processed_at"`
}

type ProcessingResult struct {
	DocumentID string    `json:"document_id"`
	ChunksCreated int    `json:"chunks_created"`
	Summary    string    `json:"summary"`
	ProcessingTime float64 `json:"processing_time_ms"`
}

func loadConfig() Config {
	return Config{
		Port:         getEnv("UPLOAD_PORT", "8093"),
		DatabaseURL:  getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		RedisURL:     getEnv("REDIS_URL", "redis://localhost:6379"),
		OllamaURL:    getEnv("OLLAMA_URL", "http://localhost:11434"),
		MinIOURL:     getEnv("MINIO_URL", "http://localhost:9000"),
		EmbedModel:   getEnv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func NewUploadService(config Config) (*UploadService, error) {
	// Initialize PostgreSQL connection
	db, err := pgxpool.New(context.Background(), config.DatabaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Initialize Redis connection
	opt, err := redis.ParseURL(config.RedisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse redis URL: %w", err)
	}
	
	redisClient := redis.NewClient(opt)
	ctx := context.Background()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis connection failed: %v", err)
	}

	return &UploadService{
		db:     db,
		redis:  redisClient,
		config: config,
	}, nil
}

func (u *UploadService) extractTextFromFile(filename string, content []byte) (string, error) {
	// Simple text extraction - in production, use proper PDF/DOCX parsers
	ext := strings.ToLower(filepath.Ext(filename))
	
	switch ext {
	case ".txt":
		return string(content), nil
	case ".pdf":
		// For now, return placeholder - integrate with pdf-parse or similar
		return fmt.Sprintf("PDF content extracted from %s (placeholder)", filename), nil
	case ".docx":
		// For now, return placeholder - integrate with docx parser
		return fmt.Sprintf("DOCX content extracted from %s (placeholder)", filename), nil
	default:
		return string(content), nil
	}
}

func (u *UploadService) generateEmbedding(text string) ([]float32, error) {
	url := fmt.Sprintf("%s/api/embeddings", u.config.OllamaURL)
	
	payload := map[string]interface{}{
		"model":  u.config.EmbedModel,
		"prompt": text,
	}
	
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	
	resp, err := http.Post(url, "application/json", bytes.NewReader(payloadBytes))
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

func (u *UploadService) chunkText(text string, chunkSize int) []string {
	if len(text) <= chunkSize {
		return []string{text}
	}
	
	var chunks []string
	for i := 0; i < len(text); i += chunkSize {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[i:end])
	}
	
	return chunks
}

func (u *UploadService) processDocument(docID, content string) error {
	start := time.Now()
	
	// Chunk the document
	chunks := u.chunkText(content, 500) // 500 character chunks
	
	// Generate embeddings for each chunk
	for i, chunk := range chunks {
		embedding, err := u.generateEmbedding(chunk)
		if err != nil {
			log.Printf("Failed to generate embedding for chunk %d: %v", i, err)
			continue
		}
		
		// Store chunk with embedding
		_, err = u.db.Exec(context.Background(),
			`INSERT INTO document_embeddings (document_id, chunk_number, chunk_text, embedding)
			 VALUES ($1, $2, $3, $4)`,
			docID, i+1, chunk, pgvector.NewVector(embedding))
		
		if err != nil {
			log.Printf("Failed to store chunk %d: %v", i, err)
		}
	}
	
	// Update processing status
	_, err := u.db.Exec(context.Background(),
		`UPDATE document_metadata 
		 SET processing_status = 'completed', updated_at = NOW()
		 WHERE id = $1`,
		docID)
	
	processingTime := time.Since(start)
	log.Printf("Document %s processed in %v with %d chunks", docID, processingTime, len(chunks))
	
	return err
}

func (u *UploadService) handleUpload(c *gin.Context) {
	// Parse multipart form
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to parse form"})
		return
	}
	
	files := form.File["files"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No files provided"})
		return
	}
	
	caseID := c.PostForm("case_id")
	userID := c.PostForm("user_id")
	
	var results []DocumentUpload
	
	for _, file := range files {
		// Read file content
		src, err := file.Open()
		if err != nil {
			continue
		}
		
		content, err := io.ReadAll(src)
		src.Close()
		if err != nil {
			continue
		}
		
		// Extract text
		extractedText, err := u.extractTextFromFile(file.Filename, content)
		if err != nil {
			log.Printf("Failed to extract text from %s: %v", file.Filename, err)
			extractedText = string(content) // Fallback
		}
		
		// Store document metadata
		var docID string
		err = u.db.QueryRow(context.Background(),
			`INSERT INTO document_metadata 
			 (case_id, user_id, original_filename, file_size, file_type, extracted_text, upload_status, processing_status)
			 VALUES ($1, $2, $3, $4, $5, $6, 'completed', 'pending')
			 RETURNING id`,
			caseID, userID, file.Filename, file.Size, file.Header.Get("Content-Type"), extractedText).Scan(&docID)
		
		if err != nil {
			log.Printf("Failed to store document metadata: %v", err)
			continue
		}
		
		// Process document asynchronously
		go u.processDocument(docID, extractedText)
		
		results = append(results, DocumentUpload{
			ID:       docID,
			CaseID:   caseID,
			UserID:   userID,
			Filename: file.Filename,
			FileSize: file.Size,
			FileType: file.Header.Get("Content-Type"),
			Status:   "processing",
		})
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Files uploaded successfully",
		"documents": results,
	})
}

func (u *UploadService) handleStatus(c *gin.Context) {
	docID := c.Param("id")
	
	var doc DocumentUpload
	err := u.db.QueryRow(context.Background(),
		`SELECT id, case_id, user_id, original_filename, file_size, file_type, 
		        upload_status, processing_status, created_at
		 FROM document_metadata WHERE id = $1`,
		docID).Scan(&doc.ID, &doc.CaseID, &doc.UserID, &doc.Filename, 
		           &doc.FileSize, &doc.FileType, &doc.Status, &doc.ProcessedAt)
	
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Document not found"})
		return
	}
	
	c.JSON(http.StatusOK, doc)
}

func (u *UploadService) handleHealth(c *gin.Context) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"services": map[string]bool{
			"database": u.checkDatabase(),
			"redis":    u.checkRedis(),
			"ollama":   u.checkOllama(),
		},
	}
	c.JSON(http.StatusOK, health)
}

func (u *UploadService) checkDatabase() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return u.db.Ping(ctx) == nil
}

func (u *UploadService) checkRedis() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return u.redis.Ping(ctx).Err() == nil
}

func (u *UploadService) checkOllama() bool {
	resp, err := http.Get(fmt.Sprintf("%s/api/tags", u.config.OllamaURL))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func (u *UploadService) setupRoutes() *gin.Engine {
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
	router.POST("/upload", u.handleUpload)
	router.GET("/status/:id", u.handleStatus)
	router.GET("/health", u.handleHealth)
	
	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Legal AI Upload Service",
			"version": "1.0.0",
			"status":  "running",
			"endpoints": []string{
				"/upload",
				"/status/:id",
				"/health",
			},
		})
	})
	
	return router
}

func main() {
	config := loadConfig()
	
	log.Printf("Starting Upload Service...")
	log.Printf("Port: %s", config.Port)
	log.Printf("Embed Model: %s", config.EmbedModel)
	
	service, err := NewUploadService(config)
	if err != nil {
		log.Fatalf("Failed to initialize upload service: %v", err)
	}
	defer service.db.Close()
	defer service.redis.Close()
	
	router := service.setupRoutes()
	
	log.Printf("Upload Service running on port %s", config.Port)
	log.Printf("Access the API at: http://localhost:%s/upload", config.Port)
	
	if err := router.Run(":" + config.Port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}