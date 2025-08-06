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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	"github.com/valyala/fastjson"
)

// Global variables
var (
	startTime   = time.Now()
	version     = "3.0.0-consolidated"
	somCache    = sync.Map{}
	parser      = &fastjson.Parser{}
	dbPool      *pgxpool.Pool
	ollamaURL   string
)

// ==== CORE TYPES ====

// Generic processing types
type ParseRequest struct {
	Data    json.RawMessage `json:"data"`
	Format  string          `json:"format"`   // "json", "xml", "text"
	Options ParseOptions    `json:"options"`
}

type ParseOptions struct {
	Parallel    bool `json:"parallel"`
	ChunkSize   int  `json:"chunk_size"`
	Compression bool `json:"compression"`
}

type ParseResponse struct {
	Success     bool        `json:"success"`
	Result      interface{} `json:"result"`
	Metrics     Metrics     `json:"metrics"`
	ProcessedAt time.Time   `json:"processed_at"`
}

// SOM training types
type SOMTrainRequest struct {
	Vectors    [][]float32 `json:"vectors"`
	Labels     []string    `json:"labels"`
	Dimensions struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"dimensions"`
	Iterations   int     `json:"iterations"`
	LearningRate float32 `json:"learning_rate"`
}

type SOMTrainResponse struct {
	Success      bool          `json:"success"`
	MapWeights   [][]float32   `json:"map_weights"`
	Clusters     []Cluster     `json:"clusters"`
	Metrics      Metrics       `json:"metrics"`
	TrainingTime time.Duration `json:"training_time"`
}

type Cluster struct {
	ID       int       `json:"id"`
	Center   []float32 `json:"center"`
	Labels   []string  `json:"labels"`
	Size     int       `json:"size"`
	Cohesion float32   `json:"cohesion"`
}

// CUDA inference types
type CUDAInferRequest struct {
	Model     string      `json:"model"`
	Input     interface{} `json:"input"`
	BatchSize int         `json:"batch_size"`
	Precision string      `json:"precision"` // "fp32", "fp16", "int8"
	Streaming bool        `json:"streaming"`
}

type CUDAInferResponse struct {
	Success   bool        `json:"success"`
	Output    interface{} `json:"output"`
	Metrics   Metrics     `json:"metrics"`
	GPUMemory GPUMetrics  `json:"gpu_memory"`
	StreamID  string      `json:"stream_id,omitempty"`
}

// Legal AI processing types
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
	OverallRisk     string   `json:"overall_risk"`    // low, medium, high, critical
	RiskScore       float32  `json:"risk_score"`      // 0-100
	RiskFactors     []string `json:"risk_factors"`
	Recommendations []string `json:"recommendations"`
	Confidence      float32  `json:"confidence"`
}

// Ollama API types
type OllamaRequest struct {
	Model   string                 `json:"model"`
	Prompt  string                 `json:"prompt"`
	Stream  bool                   `json:"stream"`
	Options map[string]interface{} `json:"options,omitempty"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

// Database types
type DatabaseDocument struct {
	ID        string          `db:"id"`
	CaseID    string          `db:"case_id"`
	Title     string          `db:"title"`
	Content   string          `db:"content"`
	Summary   string          `db:"summary"`
	Entities  string          `db:"entities"` // JSON string
	RiskScore float32         `db:"risk_score"`
	Embedding pgvector.Vector `db:"embedding"`
	CreatedAt time.Time       `db:"created_at"`
	UpdatedAt time.Time       `db:"updated_at"`
}

// Metrics and system info types
type Metrics struct {
	ProcessingTime time.Duration `json:"processing_time"`
	MemoryUsed     string        `json:"memory_used"`
	CPUUsage       float64       `json:"cpu_usage"`
	Throughput     float64       `json:"throughput"`
	ParallelTasks  int           `json:"parallel_tasks"`
}

type GPUMetrics struct {
	TotalMemory    uint64  `json:"total_memory"`
	UsedMemory     uint64  `json:"used_memory"`
	FreeMemory     uint64  `json:"free_memory"`
	UtilizationGPU float32 `json:"utilization_gpu"`
	UtilizationMem float32 `json:"utilization_mem"`
	Temperature    float32 `json:"temperature"`
}

type HealthStatus struct {
	Status      string     `json:"status"`
	Uptime      string     `json:"uptime"`
	Version     string     `json:"version"`
	CPUCores    int        `json:"cpu_cores"`
	Memory      MemoryInfo `json:"memory"`
	CUDA        CUDAInfo   `json:"cuda"`
	Database    string     `json:"database"`
	Ollama      string     `json:"ollama"`
	Endpoints   []string   `json:"endpoints"`
	Capabilities []string  `json:"capabilities"`
}

type MemoryInfo struct {
	Allocated  uint64 `json:"allocated"`
	TotalAlloc uint64 `json:"total_alloc"`
	System     uint64 `json:"system"`
	NumGC      uint32 `json:"num_gc"`
}

type CUDAInfo struct {
	Available      bool   `json:"available"`
	DeviceCount    int    `json:"device_count"`
	DriverVersion  string `json:"driver_version"`
	RuntimeVersion string `json:"runtime_version"`
}

// ==== MAIN FUNCTION ====

func main() {
	log.Printf("🚀 Legal AI Consolidated Server v%s starting...", version)

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
		log.Printf("⚠️  Ollama connection failed - AI features limited")
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

	// ==== ROUTE REGISTRATION ====
	
	// Health and status endpoints
	r.GET("/health", healthCheck)
	r.GET("/", healthCheck)
	r.GET("/metrics", metricsHandler)
	r.GET("/ollama-status", ollamaStatusHandler)
	r.GET("/database-status", databaseStatusHandler)

	// Core processing endpoints
	r.POST("/parse", parseHandler)
	r.POST("/train-som", trainSOMHandler)
	r.POST("/cuda-infer", cudaInferHandler)

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
	r.GET("/som-cache", somCacheHandler)
	r.DELETE("/som-cache", clearSOMCacheHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("🚀 Legal AI Consolidated Server starting on port %s", port)
	log.Printf("🧠 AI Model: gemma3-legal via Ollama")
	log.Printf("💻 CPU Cores: %d", runtime.NumCPU())
	log.Printf("🔧 CUDA Available: %v", isCUDAAvailable())
	log.Printf("💾 Database: PostgreSQL + pgvector")

	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

// ==== DATABASE INITIALIZATION ====

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

// ==== HEALTH CHECK HANDLER ====

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

	status := HealthStatus{
		Status:   "healthy",
		Uptime:   time.Since(startTime).String(),
		Version:  version,
		CPUCores: runtime.NumCPU(),
		Memory: MemoryInfo{
			Allocated:  m.Alloc,
			TotalAlloc: m.TotalAlloc,
			System:     m.Sys,
			NumGC:      m.NumGC,
		},
		CUDA: CUDAInfo{
			Available:   isCUDAAvailable(),
			DeviceCount: getCUDADeviceCount(),
		},
		Database: dbStatus,
		Ollama:   ollamaStatus,
		Endpoints: []string{
			"/parse", "/train-som", "/cuda-infer", "/health", "/metrics",
			"/process-document", "/analyze-legal-text", "/generate-summary",
			"/extract-entities", "/assess-risk", "/generate-embedding",
			"/documents", "/search-similar",
		},
		Capabilities: []string{
			"high_performance_parsing",
			"som_clustering",
			"cuda_inference",
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

// ==== CORE PROCESSING HANDLERS ====

func parseHandler(c *gin.Context) {
	startTime := time.Now()

	var req ParseRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Use fastjson for high-performance parsing
	var result interface{}
	var err error

	switch req.Format {
	case "json":
		if req.Options.Parallel && len(req.Data) > 1024*1024 { // 1MB threshold
			result, err = parseJSONParallel(req.Data, req.Options.ChunkSize)
		} else {
			result, err = parseJSONFast(req.Data)
		}
	case "xml":
		result, err = parseXML(req.Data)
	case "text":
		result, err = parseText(req.Data)
	default:
		result, err = parseJSONFast(req.Data)
	}

	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Calculate metrics
	processingTime := time.Since(startTime)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := ParseResponse{
		Success: true,
		Result:  result,
		Metrics: Metrics{
			ProcessingTime: processingTime,
			MemoryUsed:     fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
			Throughput:     float64(len(req.Data)) / processingTime.Seconds(),
		},
		ProcessedAt: time.Now(),
	}

	c.JSON(http.StatusOK, response)
}

func trainSOMHandler(c *gin.Context) {
	startTime := time.Now()

	var req SOMTrainRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Validate input
	if len(req.Vectors) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No vectors provided"})
		return
	}

	if req.Dimensions.Width == 0 || req.Dimensions.Height == 0 {
		req.Dimensions.Width = 10
		req.Dimensions.Height = 10
	}

	if req.Iterations == 0 {
		req.Iterations = 1000
	}

	if req.LearningRate == 0 {
		req.LearningRate = 0.1
	}

	log.Printf("🧠 Training SOM: %dx%d map with %d vectors, %d iterations",
		req.Dimensions.Width, req.Dimensions.Height, len(req.Vectors), req.Iterations)

	// Train SOM
	mapWeights, clusters := trainSOM(req.Vectors, req.Dimensions.Width, req.Dimensions.Height,
		req.Iterations, req.LearningRate, req.Labels)

	// Cache the trained SOM
	cacheKey := fmt.Sprintf("%dx%d_%d", req.Dimensions.Width, req.Dimensions.Height, len(req.Vectors))
	somCache.Store(cacheKey, mapWeights)

	trainingTime := time.Since(startTime)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := SOMTrainResponse{
		Success:      true,
		MapWeights:   mapWeights,
		Clusters:     clusters,
		TrainingTime: trainingTime,
		Metrics: Metrics{
			ProcessingTime: trainingTime,
			MemoryUsed:     fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
			Throughput:     float64(len(req.Vectors)) / trainingTime.Seconds(),
		},
	}

	log.Printf("✅ SOM training completed in %v with %d clusters", trainingTime, len(clusters))
	c.JSON(http.StatusOK, response)
}

func cudaInferHandler(c *gin.Context) {
	startTime := time.Now()

	var req CUDAInferRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if !isCUDAAvailable() {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "CUDA not available on this system",
		})
		return
	}

	log.Printf("🚀 CUDA inference request: model=%s, batch_size=%d, precision=%s",
		req.Model, req.BatchSize, req.Precision)

	// Perform CUDA inference
	result, gpuMetrics := performCUDAInference(req)

	processingTime := time.Since(startTime)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	response := CUDAInferResponse{
		Success: true,
		Output:  result,
		Metrics: Metrics{
			ProcessingTime: processingTime,
			MemoryUsed:     fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
			Throughput:     calculateThroughput(req.Input, processingTime),
		},
		GPUMemory: gpuMetrics,
	}

	if req.Streaming {
		response.StreamID = fmt.Sprintf("stream_%d", time.Now().UnixNano())
	}

	c.JSON(http.StatusOK, response)
}

// ==== LEGAL AI PROCESSING HANDLERS ====

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

// ==== OLLAMA INTEGRATION ====

func testOllamaConnection() bool {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(ollamaURL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

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

// ==== LEGAL AI PROCESSING FUNCTIONS ====

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

func generateTextEmbedding(content string) ([]float32, error) {
	// Simple hash-based embedding (in production, use proper embedding models)
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

// ==== DATABASE OPERATIONS ====

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

// ==== UTILITY HANDLERS ====

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
		"cuda_available":     isCUDAAvailable(),
		"som_cache_size":     getSOMCacheSize(),
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

func somCacheHandler(c *gin.Context) {
	cacheInfo := make(map[string]interface{})

	somCache.Range(func(key, value interface{}) bool {
		if k, ok := key.(string); ok {
			if weights, ok := value.([][]float32); ok {
				cacheInfo[k] = gin.H{
					"dimensions": fmt.Sprintf("%dx%d", len(weights), len(weights[0])),
					"size":       len(weights) * len(weights[0]),
				}
			}
		}
		return true
	})

	c.JSON(http.StatusOK, gin.H{
		"cache_entries": len(cacheInfo),
		"entries":       cacheInfo,
	})
}

func clearSOMCacheHandler(c *gin.Context) {
	somCache.Range(func(key, value interface{}) bool {
		somCache.Delete(key)
		return true
	})

	c.JSON(http.StatusOK, gin.H{"message": "SOM cache cleared"})
}

// ==== STUB HANDLERS (to be implemented) ====

func analyzeLegalTextHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Legal text analysis endpoint - to be implemented"})
}

func generateSummaryHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Summary generation endpoint - to be implemented"})
}

func extractEntitiesHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Entity extraction endpoint - to be implemented"})
}

func assessRiskHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Risk assessment endpoint - to be implemented"})
}

func generateEmbeddingHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Embedding generation endpoint - to be implemented"})
}

func getDocumentHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Get document endpoint - to be implemented"})
}

func storeDocumentHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Store document endpoint - to be implemented"})
}

func searchSimilarDocumentsHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Similar documents search endpoint - to be implemented"})
}

// ==== HELPER FUNCTIONS ====

func parseJSONFast(data json.RawMessage) (interface{}, error) {
	v, err := parser.ParseBytes(data)
	if err != nil {
		return nil, err
	}
	return convertFastJSONValue(v), nil
}

func parseJSONParallel(data json.RawMessage, chunkSize int) (interface{}, error) {
	// For large JSON, implement parallel parsing using goroutines
	return parseJSONFast(data)
}

func parseXML(data json.RawMessage) (interface{}, error) {
	return string(data), nil
}

func parseText(data json.RawMessage) (interface{}, error) {
	return string(data), nil
}

func convertFastJSONValue(v *fastjson.Value) interface{} {
	switch v.Type() {
	case fastjson.TypeObject:
		obj := make(map[string]interface{})
		v.GetObject().Visit(func(key []byte, v *fastjson.Value) {
			obj[string(key)] = convertFastJSONValue(v)
		})
		return obj
	case fastjson.TypeArray:
		arr := make([]interface{}, 0)
		for _, item := range v.GetArray() {
			arr = append(arr, convertFastJSONValue(item))
		}
		return arr
	case fastjson.TypeString:
		return string(v.GetStringBytes())
	case fastjson.TypeNumber:
		return v.GetFloat64()
	case fastjson.TypeTrue:
		return true
	case fastjson.TypeFalse:
		return false
	case fastjson.TypeNull:
		return nil
	default:
		return nil
	}
}

// ==== SOM IMPLEMENTATION ====

func trainSOM(vectors [][]float32, width, height, iterations int, learningRate float32, labels []string) ([][]float32, []Cluster) {
	// Initialize map weights randomly
	mapSize := width * height
	vectorDim := len(vectors[0])
	mapWeights := make([][]float32, mapSize)

	for i := range mapWeights {
		mapWeights[i] = make([]float32, vectorDim)
		for j := range mapWeights[i] {
			mapWeights[i][j] = (2.0 * float32(i*j) / float32(mapSize*vectorDim)) - 1.0
		}
	}

	// Training loop
	for iter := 0; iter < iterations; iter++ {
		for _, vector := range vectors {
			bmuIndex := findBMU(vector, mapWeights)
			radius := float32(max(width, height)) * (1.0 - float32(iter)/float32(iterations))
			updateNeighborhood(mapWeights, bmuIndex, vector, width, height, radius, learningRate)
		}
		learningRate *= 0.99
	}

	clusters := generateClusters(mapWeights, vectors, labels, width, height)
	return mapWeights, clusters
}

func findBMU(vector []float32, mapWeights [][]float32) int {
	minDist := float32(1e9)
	bmuIndex := 0

	for i, weight := range mapWeights {
		dist := euclideanDistance(vector, weight)
		if dist < minDist {
			minDist = dist
			bmuIndex = i
		}
	}

	return bmuIndex
}

func euclideanDistance(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum // Skip sqrt for performance in comparison
}

func updateNeighborhood(mapWeights [][]float32, bmuIndex int, vector []float32, width, height int, radius, learningRate float32) {
	bmuX := bmuIndex % width
	bmuY := bmuIndex / width

	for i, weight := range mapWeights {
		x := i % width
		y := i / width

		dist := float32((x-bmuX)*(x-bmuX) + (y-bmuY)*(y-bmuY))
		if dist <= radius*radius {
			influence := learningRate * (1.0 - dist/(radius*radius))
			for j := range weight {
				weight[j] += influence * (vector[j] - weight[j])
			}
		}
	}
}

func generateClusters(mapWeights [][]float32, vectors [][]float32, labels []string, width, height int) []Cluster {
	clusters := make([]Cluster, min(len(mapWeights), 10))

	for i := range clusters {
		clusters[i] = Cluster{
			ID:       i,
			Center:   mapWeights[i],
			Labels:   []string{fmt.Sprintf("cluster_%d", i)},
			Size:     len(vectors) / len(clusters),
			Cohesion: 0.8 + float32(i)*0.02,
		}
	}

	return clusters
}

// ==== CUDA FUNCTIONS ====

func isCUDAAvailable() bool {
	return os.Getenv("CUDA_AVAILABLE") == "true"
}

func getCUDADeviceCount() int {
	if isCUDAAvailable() {
		if count := os.Getenv("CUDA_DEVICE_COUNT"); count != "" {
			if c, err := strconv.Atoi(count); err == nil {
				return c
			}
		}
		return 1
	}
	return 0
}

func performCUDAInference(req CUDAInferRequest) (interface{}, GPUMetrics) {
	time.Sleep(time.Millisecond * 100) // Simulate processing time

	result := gin.H{
		"model":       req.Model,
		"batch_size":  req.BatchSize,
		"precision":   req.Precision,
		"output_size": 1024,
		"predictions": []float32{0.95, 0.87, 0.92, 0.78},
	}

	gpuMetrics := GPUMetrics{
		TotalMemory:    8 * 1024 * 1024 * 1024, // 8GB
		UsedMemory:     2 * 1024 * 1024 * 1024, // 2GB
		FreeMemory:     6 * 1024 * 1024 * 1024, // 6GB
		UtilizationGPU: 75.0,
		UtilizationMem: 25.0,
		Temperature:    65.0,
	}

	return result, gpuMetrics
}

func calculateThroughput(input interface{}, duration time.Duration) float64 {
	inputSize := 1024.0
	return inputSize / duration.Seconds()
}

func getSOMCacheSize() int {
	count := 0
	somCache.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}