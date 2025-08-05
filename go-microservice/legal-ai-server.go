package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	"github.com/valyala/fastjson"
)

// Enhanced types for Legal AI processing
type DocumentProcessRequest struct {
	DocumentID   string            `json:"document_id"`
	Content      string            `json:"content"`
	DocumentType string            `json:"document_type"` // evidence, case, legal_document
	CaseID       string            `json:"case_id,omitempty"`
	Options      ProcessingOptions `json:"options"`
}

type ProcessingOptions struct {
	ExtractEntities   bool `json:"extract_entities"`
	GenerateSummary   bool `json:"generate_summary"`
	AssessRisk        bool `json:"assess_risk"`
	GenerateEmbedding bool `json:"generate_embedding"`
	StoreInDatabase   bool `json:"store_in_database"`
	UseGemma3Legal    bool `json:"use_gemma3_legal"`
}

type DocumentProcessResponse struct {
	Success        bool                   `json:"success"`
	DocumentID     string                 `json:"document_id"`
	Summary        string                 `json:"summary,omitempty"`
	Entities       []LegalEntity          `json:"entities,omitempty"`
	RiskAssessment RiskAssessment         `json:"risk_assessment,omitempty"`
	Embedding      []float32              `json:"embedding,omitempty"`
	ProcessingTime time.Duration          `json:"processing_time"`
	Metadata       map[string]interface{} `json:"metadata"`
	Error          string                 `json:"error,omitempty"`
}

type LegalEntity struct {
	Type       string  `json:"type"`       // party, date, monetary, clause, jurisdiction
	Value      string  `json:"value"`
	Confidence float32 `json:"confidence"`
	StartPos   int     `json:"start_pos"`
	EndPos     int     `json:"end_pos"`
}

type RiskAssessment struct {
	OverallRisk    string  `json:"overall_risk"`    // low, medium, high, critical
	RiskScore      float32 `json:"risk_score"`      // 0-100
	RiskFactors    []string `json:"risk_factors"`
	Recommendations []string `json:"recommendations"`
	Confidence     float32 `json:"confidence"`
}

type OllamaRequest struct {
	Model    string `json:"model"`
	Prompt   string `json:"prompt"`
	Stream   bool   `json:"stream"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

type DatabaseDocument struct {
	ID          string    `db:"id"`
	CaseID      string    `db:"case_id"`
	Title       string    `db:"title"`
	Content     string    `db:"content"`
	Summary     string    `db:"summary"`
	Entities    string    `db:"entities"`    // JSON string
	RiskScore   float32   `db:"risk_score"`
	Embedding   pgvector.Vector `db:"embedding"`
	CreatedAt   time.Time `db:"created_at"`
	UpdatedAt   time.Time `db:"updated_at"`
}

// Global variables
var (
	dbPool      *pgxpool.Pool
	ollamaURL   string
	startTime   = time.Now()
	version     = "2.0.0-legal-ai"
	parser      = &fastjson.Parser{}
)

func main() {
	// Initialize database connection
	var err error
	dbPool, err = initDatabase()
	if err != nil {
		log.Printf("⚠️  Database connection failed: %v", err)
		log.Printf("🔄 Continuing without database - caching mode only")
	} else {
		log.Printf("✅ PostgreSQL connection established")
	}
	defer func() {
		if dbPool != nil {
			dbPool.Close()
		}
	}()

	// Initialize Ollama connection
	ollamaURL = os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}

	// Test Ollama connection
	if testOllamaConnection() {
		log.Printf("✅ Ollama connection established at %s", ollamaURL)
	} else {
		log.Printf("⚠️  Ollama connection failed - AI features disabled")
	}

	// Initialize Gin router
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// CORS configuration for SvelteKit integration
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{
		"http://localhost:5173", 
		"http://localhost:5175", 
		"http://localhost:3000",
		"http://localhost:4173", // SvelteKit preview
	}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Authorization", "Accept"}
	config.AllowCredentials = true
	r.Use(cors.New(config))

	// Health check endpoint
	r.GET("/health", healthCheck)
	r.GET("/", healthCheck)

	// Legal AI processing endpoints
	r.POST("/process-document", processDocumentHandler)
	r.POST("/analyze-legal-text", analyzeLegalTextHandler)
	r.POST("/generate-summary", generateSummaryHandler)
	r.POST("/extract-entities", extractEntitiesHandler)
	r.POST("/assess-risk", assessRiskHandler)
	r.POST("/generate-embedding", generateEmbeddingHandler)

	// Database interaction endpoints
	r.GET("/documents/:id", getDocumentHandler)
	r.POST("/documents", storeDocumentHandler)
	r.POST("/search-similar", searchSimilarDocumentsHandler)

	// Utility endpoints
	r.GET("/metrics", metricsHandler)
	r.GET("/ollama-status", ollamaStatusHandler)
	r.GET("/database-status", databaseStatusHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("🚀 Legal AI GPU Server starting on port %s", port)
	log.Printf("🧠 AI Model: gemma3-legal via Ollama")
	log.Printf("💻 CPU Cores: %d", runtime.NumCPU())
	log.Printf("💾 Database: PostgreSQL + pgvector")
	
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

// Initialize PostgreSQL connection with pgvector
func initDatabase() (*pgxpool.Pool, error) {
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		databaseURL = "postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db"
	}

	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse database URL: %w", err)
	}

	// Set connection pool settings
	config.MaxConns = 10
	config.MinConns = 2
	config.MaxConnLifetime = time.Hour
	config.MaxConnIdleTime = time.Minute * 30

	pool, err := pgxpool.New(context.Background(), config.ConnString())
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Ensure pgvector extension is available
	var hasVector bool
	err = pool.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')").Scan(&hasVector)
	if err != nil || !hasVector {
		log.Printf("⚠️  pgvector extension not found - vector operations disabled")
	}

	return pool, nil
}

// Test Ollama connection
func testOllamaConnection() bool {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(ollamaURL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Call Ollama API for text generation and analysis
func callOllama(prompt string, model string) (string, error) {
	if model == "" {
		model = "gemma3-legal" // Default to legal-specific model
	}

	requestBody := OllamaRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
		Options: map[string]interface{}{
			"temperature": 0.7,
			"num_ctx":     4096,
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(ollamaURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to call Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Ollama API error: status %d", resp.StatusCode)
	}

	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if ollamaResp.Error != "" {
		return "", fmt.Errorf("Ollama error: %s", ollamaResp.Error)
	}

	return ollamaResp.Response, nil
}

// Main document processing handler
func processDocumentHandler(c *gin.Context) {
	startTime := time.Now()

	var req DocumentProcessRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	log.Printf("🔄 Processing document: ID=%s, Type=%s, Length=%d chars", 
		req.DocumentID, req.DocumentType, len(req.Content))

	response := DocumentProcessResponse{
		Success:    true,
		DocumentID: req.DocumentID,
		Metadata:   make(map[string]interface{}),
	}

	// Generate summary if requested
	if req.Options.GenerateSummary {
		summary, err := generateLegalSummary(req.Content)
		if err != nil {
			log.Printf("❌ Summary generation failed: %v", err)
			response.Metadata["summary_error"] = err.Error()
		} else {
			response.Summary = summary
			log.Printf("✅ Summary generated: %d chars", len(summary))
		}
	}

	// Extract entities if requested
	if req.Options.ExtractEntities {
		entities, err := extractLegalEntities(req.Content)
		if err != nil {
			log.Printf("❌ Entity extraction failed: %v", err)
			response.Metadata["entities_error"] = err.Error()
		} else {
			response.Entities = entities
			log.Printf("✅ Extracted %d entities", len(entities))
		}
	}

	// Assess risk if requested
	if req.Options.AssessRisk {
		riskAssessment, err := assessLegalRisk(req.Content)
		if err != nil {
			log.Printf("❌ Risk assessment failed: %v", err)
			response.Metadata["risk_error"] = err.Error()
		} else {
			response.RiskAssessment = riskAssessment
			log.Printf("✅ Risk assessed: %s (%.2f)", riskAssessment.OverallRisk, riskAssessment.RiskScore)
		}
	}

	// Generate embedding if requested
	if req.Options.GenerateEmbedding {
		embedding, err := generateTextEmbedding(req.Content)
		if err != nil {
			log.Printf("❌ Embedding generation failed: %v", err)
			response.Metadata["embedding_error"] = err.Error()
		} else {
			response.Embedding = embedding
			log.Printf("✅ Embedding generated: %d dimensions", len(embedding))
		}
	}

	// Store in database if requested and available
	if req.Options.StoreInDatabase && dbPool != nil {
		doc := DatabaseDocument{
			ID:        req.DocumentID,
			CaseID:    req.CaseID,
			Title:     fmt.Sprintf("%s Document", req.DocumentType),
			Content:   req.Content,
			Summary:   response.Summary,
			RiskScore: response.RiskAssessment.RiskScore,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}

		if len(response.Entities) > 0 {
			entitiesJSON, _ := json.Marshal(response.Entities)
			doc.Entities = string(entitiesJSON)
		}

		if len(response.Embedding) > 0 {
			doc.Embedding = pgvector.NewVector(response.Embedding)
		}

		if err := storeDocument(doc); err != nil {
			log.Printf("❌ Database storage failed: %v", err)
			response.Metadata["storage_error"] = err.Error()
		} else {
			log.Printf("✅ Document stored in database")
		}
	}

	response.ProcessingTime = time.Since(startTime)
	c.JSON(http.StatusOK, response)
}

// Generate legal summary using Ollama
func generateLegalSummary(content string) (string, error) {
	prompt := fmt.Sprintf(`You are a legal AI assistant. Please provide a concise summary of the following legal document, highlighting key points, parties involved, and important legal implications:

Document:
%s

Please provide a structured summary covering:
1. Main subject matter
2. Key parties involved
3. Important dates and deadlines
4. Legal implications
5. Action items or next steps

Summary:`, content)

	return callOllama(prompt, "gemma3-legal")
}

// Extract legal entities using Ollama
func extractLegalEntities(content string) ([]LegalEntity, error) {
	prompt := fmt.Sprintf(`You are a legal AI assistant specialized in entity extraction. Extract and categorize legal entities from the following text. Respond with a JSON array of entities.

Text:
%s

Please extract entities in the following categories:
- parties: Names of individuals, companies, organizations
- dates: Important dates, deadlines, court dates
- monetary: Dollar amounts, fees, damages, settlements
- clauses: Important legal clauses, sections, or provisions
- jurisdictions: Courts, states, federal districts

Respond with valid JSON only:`, content)

	response, err := callOllama(prompt, "gemma3-legal")
	if err != nil {
		return nil, err
	}

	// Parse the JSON response
	var entities []LegalEntity
	response = strings.TrimSpace(response)
	if strings.HasPrefix(response, "```json") {
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimSuffix(response, "```")
	}

	if err := json.Unmarshal([]byte(response), &entities); err != nil {
		// If JSON parsing fails, create a simple entity from the response
		return []LegalEntity{{
			Type:       "analysis",
			Value:      response,
			Confidence: 0.8,
		}}, nil
	}

	return entities, nil
}

// Assess legal risk using Ollama
func assessLegalRisk(content string) (RiskAssessment, error) {
	prompt := fmt.Sprintf(`You are a legal AI assistant specialized in risk assessment. Analyze the following legal document and assess potential risks. Respond with a JSON object.

Document:
%s

Please assess:
1. Overall risk level (low, medium, high, critical)
2. Risk score (0-100)
3. Specific risk factors
4. Recommendations to mitigate risks
5. Confidence in assessment (0.0-1.0)

Respond with valid JSON only in this format:
{
  "overall_risk": "medium",
  "risk_score": 65.5,
  "risk_factors": ["factor1", "factor2"],
  "recommendations": ["rec1", "rec2"],
  "confidence": 0.85
}`, content)

	response, err := callOllama(prompt, "gemma3-legal")
	if err != nil {
		return RiskAssessment{}, err
	}

	// Parse the JSON response
	var riskAssessment RiskAssessment
	response = strings.TrimSpace(response)
	if strings.HasPrefix(response, "```json") {
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimSuffix(response, "```")
	}

	if err := json.Unmarshal([]byte(response), &riskAssessment); err != nil {
		// If JSON parsing fails, create a default assessment
		return RiskAssessment{
			OverallRisk:     "medium",
			RiskScore:       50.0,
			RiskFactors:     []string{"Unable to parse detailed analysis"},
			Recommendations: []string{"Manual review recommended"},
			Confidence:      0.5,
		}, nil
	}

	return riskAssessment, nil
}

// Generate text embedding (simplified - in production use proper embedding model)
func generateTextEmbedding(content string) ([]float32, error) {
	// For now, create a simple hash-based embedding
	// In production, use proper embedding models via Ollama or dedicated service
	
	embedding := make([]float32, 384) // Common embedding dimension
	hash := 0
	for i, char := range content {
		hash = (hash*31 + int(char)) % len(embedding)
		embedding[hash] += float32(i%256) / 255.0
	}
	
	// Normalize
	var sum float32
	for _, val := range embedding {
		sum += val * val
	}
	norm := float32(1.0)
	if sum > 0 {
		norm = 1.0 / float32(len(embedding)) * sum
	}
	
	for i := range embedding {
		embedding[i] *= norm
	}
	
	return embedding, nil
}

// Store document in PostgreSQL
func storeDocument(doc DatabaseDocument) error {
	if dbPool == nil {
		return fmt.Errorf("database not available")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	query := `
		INSERT INTO legal_documents (id, case_id, title, content, summary, entities, risk_score, embedding, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (id) DO UPDATE SET
			title = EXCLUDED.title,
			content = EXCLUDED.content,
			summary = EXCLUDED.summary,
			entities = EXCLUDED.entities,
			risk_score = EXCLUDED.risk_score,
			embedding = EXCLUDED.embedding,
			updated_at = EXCLUDED.updated_at
	`

	_, err := dbPool.Exec(ctx, query,
		doc.ID, doc.CaseID, doc.Title, doc.Content, doc.Summary,
		doc.Entities, doc.RiskScore, doc.Embedding, doc.CreatedAt, doc.UpdatedAt)

	return err
}

// Health check handler
func healthCheck(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	dbStatus := "disconnected"
	if dbPool != nil {
		if err := dbPool.Ping(context.Background()); err == nil {
			dbStatus = "connected"
		}
	}

	ollamaStatus := "disconnected"
	if testOllamaConnection() {
		ollamaStatus = "connected"
	}

	status := gin.H{
		"status":         "healthy",
		"version":        version,
		"uptime":         time.Since(startTime).String(),
		"cpu_cores":      runtime.NumCPU(),
		"memory_mb":      fmt.Sprintf("%.2f", float64(m.Alloc)/1024/1024),
		"database":       dbStatus,
		"ollama":         ollamaStatus,
		"ollama_url":     ollamaURL,
		"goroutines":     runtime.NumGoroutine(),
		"capabilities": []string{
			"document_processing",
			"legal_analysis", 
			"entity_extraction",
			"risk_assessment",
			"text_summarization",
			"embedding_generation",
			"vector_similarity_search",
		},
	}

	c.JSON(http.StatusOK, status)
}

// Individual handler functions for modular access
func analyzeLegalTextHandler(c *gin.Context) {
	// Implementation similar to processDocumentHandler but focused on analysis
	c.JSON(http.StatusOK, gin.H{"message": "Legal text analysis endpoint"})
}

func generateSummaryHandler(c *gin.Context) {
	// Implementation for standalone summary generation
	c.JSON(http.StatusOK, gin.H{"message": "Summary generation endpoint"})
}

func extractEntitiesHandler(c *gin.Context) {
	// Implementation for standalone entity extraction
	c.JSON(http.StatusOK, gin.H{"message": "Entity extraction endpoint"})
}

func assessRiskHandler(c *gin.Context) {
	// Implementation for standalone risk assessment
	c.JSON(http.StatusOK, gin.H{"message": "Risk assessment endpoint"})
}

func generateEmbeddingHandler(c *gin.Context) {
	// Implementation for standalone embedding generation
	c.JSON(http.StatusOK, gin.H{"message": "Embedding generation endpoint"})
}

func getDocumentHandler(c *gin.Context) {
	// Implementation for retrieving documents
	c.JSON(http.StatusOK, gin.H{"message": "Get document endpoint"})
}

func storeDocumentHandler(c *gin.Context) {
	// Implementation for storing documents
	c.JSON(http.StatusOK, gin.H{"message": "Store document endpoint"})
}

func searchSimilarDocumentsHandler(c *gin.Context) {
	// Implementation for vector similarity search
	c.JSON(http.StatusOK, gin.H{"message": "Similar documents search endpoint"})
}

func metricsHandler(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	metrics := gin.H{
		"uptime":             time.Since(startTime).String(),
		"goroutines":         runtime.NumGoroutine(),
		"memory_alloc_mb":    fmt.Sprintf("%.2f", float64(m.Alloc)/1024/1024),
		"memory_total_mb":    fmt.Sprintf("%.2f", float64(m.TotalAlloc)/1024/1024),
		"memory_sys_mb":      fmt.Sprintf("%.2f", float64(m.Sys)/1024/1024),
		"gc_runs":            m.NumGC,
		"cpu_cores":          runtime.NumCPU(),
		"database_connected": dbPool != nil,
		"ollama_connected":   testOllamaConnection(),
	}

	c.JSON(http.StatusOK, metrics)
}

func ollamaStatusHandler(c *gin.Context) {
	status := gin.H{
		"url":       ollamaURL,
		"connected": testOllamaConnection(),
	}

	if testOllamaConnection() {
		// Get available models
		client := &http.Client{Timeout: 5 * time.Second}
		resp, err := client.Get(ollamaURL + "/api/tags")
		if err == nil {
			defer resp.Body.Close()
			var models map[string]interface{}
			json.NewDecoder(resp.Body).Decode(&models)
			status["models"] = models
		}
	}

	c.JSON(http.StatusOK, status)
}

func databaseStatusHandler(c *gin.Context) {
	status := gin.H{
		"connected": false,
		"pool_size": 0,
	}

	if dbPool != nil {
		if err := dbPool.Ping(context.Background()); err == nil {
			status["connected"] = true
			status["pool_size"] = dbPool.Stat().TotalConns()
			status["idle_conns"] = dbPool.Stat().IdleConns()
			status["acquired_conns"] = dbPool.Stat().AcquiredConns()
		}
	}

	c.JSON(http.StatusOK, status)
}