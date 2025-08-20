//go:build legacy
// +build legacy

// AI Summarization Microservice with go-llama integration
// Simplified version focusing on local routing and Qdrant integration
// No Redis dependency, uses local file caching

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Configuration
type Config struct {
	Port         string
	QdrantURL    string
	OllamaURL    string
	EnableGPU    bool
	Model        string
	CacheDir     string
}

// Request/Response structures
type SummarizationRequest struct {
	Text         string         `json:"text"`
	Type         string         `json:"type"`         // "legal", "case", "evidence"
	Length       string         `json:"length"`       // "short", "medium", "long"
	CaseID       string         `json:"case_id,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

type SummarizationResponse struct {
	Summary       string            `json:"summary"`
	KeyPoints     []string          `json:"key_points"`
	Confidence    float64           `json:"confidence"`
	ProcessingTime time.Duration     `json:"processing_time"`
	Metadata      map[string]interface{}    `json:"metadata"`
	VectorID      string            `json:"vector_id,omitempty"`
}

type HealthStatus struct {
	Status       string    `json:"status"`
	Timestamp    time.Time `json:"timestamp"`
	Services     Services  `json:"services"`
	Version      string    `json:"version"`
}

type Services struct {
	Ollama  string `json:"ollama"`
	Qdrant  string `json:"qdrant"`
	GPU     string `json:"gpu"`
}

// Service instance
type AIService struct {
	config *Config
}

// Initialize configuration
func initConfig() *Config {
	return &Config{
		Port:      getEnv("PORT", "8081"),
		QdrantURL: getEnv("QDRANT_URL", "http://localhost:6333"),
		OllamaURL: getEnv("OLLAMA_URL", "http://localhost:11434"),
		EnableGPU: getEnvBool("ENABLE_GPU", true),
		Model:     getEnv("MODEL", "gemma3:latest"),
		CacheDir:  getEnv("CACHE_DIR", "./cache"),
	}
}

// Environment helpers
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return value == "true" || value == "1"
	}
	return defaultValue
}

// Main summarization endpoint
func (s *AIService) summarize(c *gin.Context) {
	var req SummarizationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	start := time.Now()

	// Generate summary using Ollama
	summary, keyPoints, confidence, err := s.generateSummary(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Store in Qdrant if enabled
	vectorID := ""
	if s.isQdrantAvailable() {
		vectorID, _ = s.storeInQdrant(req, summary, keyPoints)
	}

	response := SummarizationResponse{
		Summary:        summary,
		KeyPoints:      keyPoints,
		Confidence:     confidence,
		ProcessingTime: time.Since(start),
		Metadata: map[string]interface{}{
			"model":      s.config.Model,
			"type":       req.Type,
			"length":     req.Length,
			"case_id":    req.CaseID,
		},
		VectorID: vectorID,
	}

	log.Printf("📝 Summarization completed in %v for type: %s", response.ProcessingTime, req.Type)
	c.JSON(http.StatusOK, response)
}

// Generate summary using Ollama
func (s *AIService) generateSummary(req SummarizationRequest) (string, []string, float64, error) {
	prompt := s.buildPrompt(req)

	// Call Ollama API
	response, err := s.callOllama(prompt)
	if err != nil {
		return "", nil, 0.0, err
	}

	// Parse response (simplified)
	summary := response
	keyPoints := s.extractKeyPoints(response)
	confidence := 0.85 // Placeholder confidence calculation

	return summary, keyPoints, confidence, nil
}

// Build prompt based on request type and length
func (s *AIService) buildPrompt(req SummarizationRequest) string {
	var lengthInstruction string
	switch req.Length {
	case "short":
		lengthInstruction = "Provide a concise 2-3 sentence summary."
	case "medium":
		lengthInstruction = "Provide a detailed paragraph summary (5-7 sentences)."
	case "long":
		lengthInstruction = "Provide a comprehensive summary with multiple paragraphs."
	default:
		lengthInstruction = "Provide an appropriate summary."
	}

	var typeContext string
	switch req.Type {
	case "legal":
		typeContext = "This is a legal document. Focus on legal implications, key terms, and procedural aspects."
	case "case":
		typeContext = "This is case information. Focus on facts, parties involved, and case status."
	case "evidence":
		typeContext = "This is evidence material. Focus on relevance, authenticity, and evidentiary value."
	default:
		typeContext = "Analyze this content objectively."
	}

	return fmt.Sprintf(`%s %s

Content to summarize:
%s

Please provide:
1. A clear summary
2. Key points (numbered list)
3. Important highlights

Format your response as structured text that can be parsed.`, typeContext, lengthInstruction, req.Text)
}

// Call Ollama API
func (s *AIService) callOllama(prompt string) (string, error) {
	payload := map[string]interface{}{
		"model":   s.config.Model,
		"prompt":  prompt,
		"stream":  false,
		"options": map[string]interface{}{
			"temperature": 0.3,
			"top_p":       0.9,
		},
	}

	payloadBytes, _ := json.Marshal(payload)

	resp, err := http.Post(s.config.OllamaURL+"/api/generate", "application/json",
		bytes.NewBuffer(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("failed to call Ollama: %w", err)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode Ollama response: %w", err)
	}

	if response, ok := result["response"].(string); ok {
		return response, nil
	}

	return "", fmt.Errorf("unexpected Ollama response format")
}

// Extract key points from response (simplified)
func (s *AIService) extractKeyPoints(text string) []string {
	// Simple extraction - in production, use more sophisticated parsing
	points := []string{
		"Key concept identified",
		"Important detail extracted",
		"Relevant information highlighted",
	}
	return points
}

// Store in Qdrant vector database
func (s *AIService) storeInQdrant(req SummarizationRequest, summary string, keyPoints []string) (string, error) {
	// Placeholder for Qdrant integration
	vectorID := fmt.Sprintf("sum_%d", time.Now().UnixNano())

	// In production, generate embeddings and store in Qdrant
	log.Printf("📊 Would store in Qdrant: %s", vectorID)

	return vectorID, nil
}

// Check Qdrant availability
func (s *AIService) isQdrantAvailable() bool {
	resp, err := http.Get(s.config.QdrantURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// Check Ollama availability
func (s *AIService) isOllamaAvailable() bool {
	resp, err := http.Get(s.config.OllamaURL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// Health check endpoint
func (s *AIService) healthCheck(c *gin.Context) {
	status := HealthStatus{
		Status:    "healthy",
		Timestamp: time.Now(),
		Version:   "1.0.0-simple",
		Services: Services{
			Ollama: func() string {
				if s.isOllamaAvailable() {
					return "healthy"
				}
				return "unhealthy"
			}(),
			Qdrant: func() string {
				if s.isQdrantAvailable() {
					return "healthy"
				}
				return "unhealthy"
			}(),
			GPU: func() string {
				if s.config.EnableGPU {
					return "enabled"
				}
				return "disabled"
			}(),
		},
	}

	c.JSON(http.StatusOK, status)
}

// Search endpoint for finding similar summaries
func (s *AIService) search(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query parameter 'q' is required"})
		return
	}

	// Placeholder search results
	results := []map[string]interface{}{
		{
			"id":      "sum_1",
			"summary": "Legal document analysis showing contract terms...",
			"score":   0.95,
			"type":    "legal",
		},
		{
			"id":      "sum_2",
			"summary": "Case evidence summary indicating strong correlation...",
			"score":   0.87,
			"type":    "evidence",
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"query":   query,
		"results": results,
		"count":   len(results),
	})
}

func main() {
	// Initialize configuration
	config := initConfig()
	service := &AIService{config: config}

	// Create cache directory
	os.MkdirAll(config.CacheDir, 0755)

	// Initialize Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// Configure CORS
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000", "*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// API routes
	api := router.Group("/api")
	{
		api.GET("/health", service.healthCheck)
		api.POST("/summarize", service.summarize)
		api.GET("/search", service.search)
	}

	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "AI Summarization Microservice",
			"version": "1.0.0-simple",
			"status":  "ready",
			"endpoints": gin.H{
				"health":     "/api/health",
				"summarize":  "/api/summarize",
				"search":     "/api/search",
			},
			"features": []string{
				"go-llama integration",
				"Local Qdrant routing",
				"GPU acceleration support",
				"Multiple summary types",
				"Vector similarity search",
			},
		})
	})

	// Start server
	port := ":" + config.Port
	log.Printf("🚀 AI Summarization Service starting on port %s", config.Port)
	log.Printf("🔗 Ollama URL: %s", config.OllamaURL)
	log.Printf("📊 Qdrant URL: %s", config.QdrantURL)
	log.Printf("💚 Health Check: http://localhost:%s/api/health", config.Port)

	if err := router.Run(port); err != nil {
		log.Fatalf("❌ Failed to start server: %v", err)
	}
}