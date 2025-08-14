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
	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
)

// Enhanced RAG Service with MCP Filesystem Integration
// Supports gemma3-legal, PostgreSQL pgvector, Redis caching, and file system search

type Config struct {
	Port         string
	DatabaseURL  string
	RedisURL     string
	OllamaURL    string
	Model        string
	EmbedModel   string
}

type RAGService struct {
	db          *pgxpool.Pool
	redis       *redis.Client
	config      Config
}

type Document struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Metadata  map[string]interface{} `json:"metadata"`
	Embedding []float32 `json:"embedding,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

type SearchRequest struct {
	Query     string                 `json:"query"`
	CaseID    string                 `json:"case_id,omitempty"`
	Limit     int                    `json:"limit,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type SearchResult struct {
	Document   Document `json:"document"`
	Similarity float64  `json:"similarity"`
	Rank       int      `json:"rank"`
}

type RAGResponse struct {
	Query      string         `json:"query"`
	Results    []SearchResult `json:"results"`
	Response   string         `json:"response"`
	Timestamp  time.Time      `json:"timestamp"`
	ProcessingTime float64    `json:"processing_time_ms"`
}

func loadConfig() Config {
	return Config{
		Port:         getEnv("RAG_PORT", "8094"),
		DatabaseURL:  getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		RedisURL:     getEnv("REDIS_URL", "redis://localhost:6379"),
		OllamaURL:    getEnv("OLLAMA_URL", "http://localhost:11434"),
		Model:        getEnv("OLLAMA_MODEL", "gemma3-legal:latest"),
		EmbedModel:   getEnv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func NewRAGService(config Config) (*RAGService, error) {
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

	return &RAGService{
		db:     db,
		redis:  redisClient,
		config: config,
	}, nil
}

func (r *RAGService) generateEmbedding(text string) ([]float32, error) {
	url := fmt.Sprintf("%s/api/embeddings", r.config.OllamaURL)
	
	payload := map[string]interface{}{
		"model":  r.config.EmbedModel,
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

func (r *RAGService) searchSimilarDocuments(query string, limit int, caseID string) ([]SearchResult, error) {
	start := time.Now()
	
	// Generate query embedding
	queryEmbedding, err := r.generateEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}
	
	// Search in vector database
	sqlQuery := `
		SELECT 
			dm.id,
			dm.original_filename,
			de.chunk_text,
			dm.summary,
			dm.created_at,
			1 - (de.embedding <=> $1) as similarity
		FROM document_embeddings de
		JOIN document_metadata dm ON de.document_id = dm.id
		WHERE ($2 = '' OR dm.case_id::text = $2)
		ORDER BY de.embedding <=> $1
		LIMIT $3
	`
	
	rows, err := r.db.Query(context.Background(), sqlQuery, pgvector.NewVector(queryEmbedding), caseID, limit)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}
	defer rows.Close()
	
	var results []SearchResult
	rank := 1
	
	for rows.Next() {
		var doc Document
		var similarity float64
		var filename, summary string
		
		err := rows.Scan(
			&doc.ID,
			&filename,
			&doc.Content,
			&summary,
			&doc.CreatedAt,
			&similarity,
		)
		if err != nil {
			continue
		}
		
		doc.Metadata = map[string]interface{}{
			"filename": filename,
			"summary":  summary,
		}
		
		results = append(results, SearchResult{
			Document:   doc,
			Similarity: similarity,
			Rank:       rank,
		})
		rank++
	}
	
	log.Printf("Vector search completed in %v, found %d results", time.Since(start), len(results))
	return results, nil
}

func (r *RAGService) generateResponse(query string, context []SearchResult) (string, error) {
	// Build context from search results
	contextText := "Based on the following legal documents:\n\n"
	for i, result := range context {
		contextText += fmt.Sprintf("Document %d (similarity: %.3f):\n%s\n\n", 
			i+1, result.Similarity, result.Document.Content[:min(500, len(result.Document.Content))])
	}
	
	// Generate response using gemma3-legal
	prompt := fmt.Sprintf(`%s

Question: %s

Please provide a comprehensive legal analysis based on the provided documents. Include relevant citations and ensure accuracy.

Response:`, contextText, query)
	
	url := fmt.Sprintf("%s/api/generate", r.config.OllamaURL)
	payload := map[string]interface{}{
		"model":  r.config.Model,
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"temperature": 0.3,
			"top_p": 0.9,
		},
	}
	
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	
	resp, err := http.Post(url, "application/json", strings.NewReader(string(payloadBytes)))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	
	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	
	response, ok := result["response"].(string)
	if !ok {
		return "", fmt.Errorf("invalid response from Ollama")
	}
	
	return response, nil
}

func (r *RAGService) handleSearch(c *gin.Context) {
	start := time.Now()
	
	var req SearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if req.Limit == 0 {
		req.Limit = 5
	}
	
	// Check Redis cache
	cacheKey := fmt.Sprintf("rag:search:%s:%s:%d", req.Query, req.CaseID, req.Limit)
	if cached, err := r.redis.Get(context.Background(), cacheKey).Result(); err == nil {
		var cachedResult RAGResponse
		if json.Unmarshal([]byte(cached), &cachedResult) == nil {
			log.Printf("Returning cached result for query: %s", req.Query)
			c.JSON(http.StatusOK, cachedResult)
			return
		}
	}
	
	// Perform vector search
	results, err := r.searchSimilarDocuments(req.Query, req.Limit, req.CaseID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	// Generate AI response
	response, err := r.generateResponse(req.Query, results)
	if err != nil {
		log.Printf("Failed to generate AI response: %v", err)
		response = "Unable to generate AI response at this time."
	}
	
	result := RAGResponse{
		Query:          req.Query,
		Results:        results,
		Response:       response,
		Timestamp:      time.Now(),
		ProcessingTime: float64(time.Since(start).Nanoseconds()) / 1e6,
	}
	
	// Cache result
	if resultBytes, err := json.Marshal(result); err == nil {
		r.redis.Set(context.Background(), cacheKey, resultBytes, 10*time.Minute)
	}
	
	c.JSON(http.StatusOK, result)
}

func (r *RAGService) handleHealth(c *gin.Context) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"services": map[string]bool{
			"database": r.checkDatabase(),
			"redis":    r.checkRedis(),
			"ollama":   r.checkOllama(),
		},
	}
	c.JSON(http.StatusOK, health)
}

func (r *RAGService) checkDatabase() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	return r.db.Ping(ctx) == nil
}

func (r *RAGService) checkRedis() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	return r.redis.Ping(ctx).Err() == nil
}

func (r *RAGService) checkOllama() bool {
	resp, err := http.Get(fmt.Sprintf("%s/api/tags", r.config.OllamaURL))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (r *RAGService) setupRoutes() *gin.Engine {
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
		api.POST("/rag/search", r.handleSearch)
		api.GET("/health", r.handleHealth)
		api.GET("/status", r.handleHealth)
	}
	
	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Enhanced RAG API",
			"version": "1.0.0",
			"status":  "running",
			"endpoints": []string{
				"/api/rag/search",
				"/api/health",
			},
		})
	})
	
	return router
}

func main() {
	config := loadConfig()
	
	log.Printf("Starting Enhanced RAG Service...")
	log.Printf("Port: %s", config.Port)
	log.Printf("Model: %s", config.Model)
	log.Printf("Embed Model: %s", config.EmbedModel)
	
	service, err := NewRAGService(config)
	if err != nil {
		log.Fatalf("Failed to initialize RAG service: %v", err)
	}
	defer service.db.Close()
	defer service.redis.Close()
	
	router := service.setupRoutes()
	
	log.Printf("Enhanced RAG Service running on port %s", config.Port)
	log.Printf("Access the API at: http://localhost:%s/api/rag/search", config.Port)
	
	if err := router.Run(":" + config.Port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}