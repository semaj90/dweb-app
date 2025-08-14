package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/go-redis/redis/v8"
	"gonum.org/v1/gonum/mat"
)

// Enhanced RAG System with Self-Organizing Map (SOM) for User Intent Analysis
// Integrates with PostgreSQL pgvector, provides semantic clustering and intent prediction

type EnhancedRAGService struct {
	db          *pgxpool.Pool
	redis       *redis.Client
	config      RAGConfig
	somNetwork  *SOMNetwork
	intentCache map[string]CachedIntent
	
	// Performance metrics
	queryCount    int64
	avgQueryTime  float64
	cacheHitRate  float64
}

type RAGConfig struct {
	Port         string
	DatabaseURL  string
	RedisURL     string
	EmbeddingDim int
	SOMWidth     int
	SOMHeight    int
	LearningRate float64
	MaxEpochs    int
	VectorThreshold float64
}

type RAGRequest struct {
	Query       string                 `json:"query"`
	UserID      string                 `json:"user_id"`
	CaseID      string                 `json:"case_id,omitempty"`
	Context     map[string]interface{} `json:"context,omitempty"`
	MaxResults  int                    `json:"max_results,omitempty"`
	MinScore    float64                `json:"min_score,omitempty"`
	IntentHint  string                 `json:"intent_hint,omitempty"`
	EnableSOM   bool                   `json:"enable_som,omitempty"`
}

type RAGResponse struct {
	Query          string                 `json:"query"`
	Results        []DocumentResult       `json:"results"`
	UserIntent     string                 `json:"user_intent"`
	IntentScore    float64                `json:"intent_score"`
	SOMCluster     int                    `json:"som_cluster"`
	ProcessingTime float64                `json:"processing_time_ms"`
	TotalDocs      int                    `json:"total_docs"`
	Suggestions    []string               `json:"suggestions"`
	Context        map[string]interface{} `json:"context"`
	CacheHit       bool                   `json:"cache_hit"`
}

type DocumentResult struct {
	ID           int                    `json:"id"`
	CaseID       string                 `json:"case_id"`
	Title        string                 `json:"title"`
	Content      string                 `json:"content"`
	DocumentType string                 `json:"document_type"`
	Score        float64                `json:"score"`
	Relevance    string                 `json:"relevance"`
	Highlights   []string               `json:"highlights"`
	Metadata     map[string]interface{} `json:"metadata"`
	CreatedAt    time.Time              `json:"created_at"`
}

type CachedIntent struct {
	Intent    string    `json:"intent"`
	Score     float64   `json:"score"`
	Cluster   int       `json:"cluster"`
	CachedAt  time.Time `json:"cached_at"`
}

// Self-Organizing Map (SOM) implementation for intent clustering
type SOMNetwork struct {
	Width        int
	Height       int
	InputDim     int
	Weights      [][][]float64
	LearningRate float64
	Radius       float64
	Epoch        int
	MaxEpochs    int
	
	// Intent mapping
	IntentMap    map[string]int  // Intent name -> cluster ID
	ClusterMap   map[int]string  // Cluster ID -> intent name
}

type SOMNode struct {
	X, Y    int
	Weights []float64
}

// Legal intent categories for SOM training
var LegalIntents = map[string][]string{
	"search":        {"find", "search", "locate", "lookup", "discover"},
	"analyze":       {"analyze", "examine", "review", "assess", "evaluate"},
	"summarize":     {"summarize", "overview", "brief", "abstract", "synopsis"},
	"draft":         {"draft", "write", "compose", "create", "prepare"},
	"research":      {"research", "investigate", "study", "explore", "examine"},
	"compare":       {"compare", "contrast", "differentiate", "versus", "against"},
	"explain":       {"explain", "clarify", "describe", "define", "elaborate"},
	"calculate":     {"calculate", "compute", "estimate", "determine", "assess"},
	"validate":      {"validate", "verify", "confirm", "check", "ensure"},
	"recommend":     {"recommend", "suggest", "advise", "propose", "counsel"},
}

func NewEnhancedRAGService() *EnhancedRAGService {
	config := loadRAGConfig()
	
	return &EnhancedRAGService{
		config:      config,
		intentCache: make(map[string]CachedIntent),
		somNetwork:  NewSOMNetwork(config.SOMWidth, config.SOMHeight, config.EmbeddingDim, config.LearningRate, config.MaxEpochs),
	}
}

func loadRAGConfig() RAGConfig {
	return RAGConfig{
		Port:            getEnv("RAG_PORT", "8095"),
		DatabaseURL:     getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		RedisURL:        getEnv("REDIS_URL", "redis://localhost:6379"),
		EmbeddingDim:    getEnvInt("EMBEDDING_DIM", 384),
		SOMWidth:        getEnvInt("SOM_WIDTH", 8),
		SOMHeight:       getEnvInt("SOM_HEIGHT", 8),
		LearningRate:    getEnvFloat("SOM_LEARNING_RATE", 0.1),
		MaxEpochs:       getEnvInt("SOM_MAX_EPOCHS", 100),
		VectorThreshold: getEnvFloat("VECTOR_THRESHOLD", 0.7),
	}
}

func (r *EnhancedRAGService) Initialize() error {
	log.Println("🧠 Initializing Enhanced RAG System with SOM...")
	
	// Initialize database
	var err error
	r.db, err = pgxpool.New(context.Background(), r.config.DatabaseURL)
	if err != nil {
		return fmt.Errorf("database connection failed: %w", err)
	}
	
	// Initialize Redis
	opt, err := redis.ParseURL(r.config.RedisURL)
	if err != nil {
		return fmt.Errorf("redis URL parsing failed: %w", err)
	}
	r.redis = redis.NewClient(opt)
	
	// Initialize and train SOM network
	if err := r.trainSOMNetwork(); err != nil {
		log.Printf("⚠️ SOM training failed: %v", err)
	}
	
	log.Println("✅ Enhanced RAG System initialized")
	return nil
}

func NewSOMNetwork(width, height, inputDim int, learningRate float64, maxEpochs int) *SOMNetwork {
	som := &SOMNetwork{
		Width:        width,
		Height:       height,
		InputDim:     inputDim,
		LearningRate: learningRate,
		MaxEpochs:    maxEpochs,
		Radius:       float64(max(width, height)) / 2.0,
		IntentMap:    make(map[string]int),
		ClusterMap:   make(map[int]string),
	}
	
	// Initialize weights randomly
	som.Weights = make([][][]float64, width)
	for x := 0; x < width; x++ {
		som.Weights[x] = make([][]float64, height)
		for y := 0; y < height; y++ {
			som.Weights[x][y] = make([]float64, inputDim)
			for d := 0; d < inputDim; d++ {
				som.Weights[x][y][d] = (rand.Float64() - 0.5) * 0.1
			}
		}
	}
	
	return som
}

func (r *EnhancedRAGService) trainSOMNetwork() error {
	log.Println("🎯 Training SOM network for legal intent analysis...")
	
	// Generate training data from legal intents
	trainingData := r.generateTrainingData()
	
	for epoch := 0; epoch < r.somNetwork.MaxEpochs; epoch++ {
		r.somNetwork.Epoch = epoch
		
		for _, sample := range trainingData {
			r.somNetwork.train(sample.Vector, sample.Intent)
		}
		
		// Decay learning rate and radius
		decayFactor := 1.0 - float64(epoch)/float64(r.somNetwork.MaxEpochs)
		r.somNetwork.LearningRate *= decayFactor
		r.somNetwork.Radius *= decayFactor
	}
	
	// Map intents to clusters
	r.mapIntentsToClusters(trainingData)
	
	log.Printf("✅ SOM training completed (%d epochs)", r.somNetwork.MaxEpochs)
	return nil
}

type TrainingSample struct {
	Vector []float64
	Intent string
}

func (r *EnhancedRAGService) generateTrainingData() []TrainingSample {
	var samples []TrainingSample
	
	for intent, keywords := range LegalIntents {
		for _, keyword := range keywords {
			// Generate simple embedding (in production, use real embeddings)
			vector := r.generateSimpleEmbedding(keyword)
			samples = append(samples, TrainingSample{
				Vector: vector,
				Intent: intent,
			})
		}
	}
	
	return samples
}

func (r *EnhancedRAGService) generateSimpleEmbedding(text string) []float64 {
	// Simplified embedding generation (replace with real embeddings)
	embedding := make([]float64, r.config.EmbeddingDim)
	
	for i, char := range text {
		if i < len(embedding) {
			embedding[i] = float64(char) / 1000.0
		}
	}
	
	// Normalize
	norm := 0.0
	for _, v := range embedding {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	
	if norm > 0 {
		for i := range embedding {
			embedding[i] /= norm
		}
	}
	
	return embedding
}

func (som *SOMNetwork) train(input []float64, intent string) {
	// Find best matching unit (BMU)
	bmuX, bmuY := som.findBMU(input)
	
	// Update weights in neighborhood
	for x := 0; x < som.Width; x++ {
		for y := 0; y < som.Height; y++ {
			distance := math.Sqrt(float64((x-bmuX)*(x-bmuX) + (y-bmuY)*(y-bmuY)))
			
			if distance <= som.Radius {
				influence := math.Exp(-distance*distance / (2*som.Radius*som.Radius))
				
				for d := 0; d < som.InputDim; d++ {
					som.Weights[x][y][d] += som.LearningRate * influence * (input[d] - som.Weights[x][y][d])
				}
			}
		}
	}
}

func (som *SOMNetwork) findBMU(input []float64) (int, int) {
	minDistance := math.Inf(1)
	bmuX, bmuY := 0, 0
	
	for x := 0; x < som.Width; x++ {
		for y := 0; y < som.Height; y++ {
			distance := som.euclideanDistance(input, som.Weights[x][y])
			if distance < minDistance {
				minDistance = distance
				bmuX, bmuY = x, y
			}
		}
	}
	
	return bmuX, bmuY
}

func (som *SOMNetwork) euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func (r *EnhancedRAGService) mapIntentsToClusters(trainingData []TrainingSample) {
	clusterIntents := make(map[int]map[string]int)
	
	for _, sample := range trainingData {
		x, y := r.somNetwork.findBMU(sample.Vector)
		clusterID := y*r.somNetwork.Width + x
		
		if clusterIntents[clusterID] == nil {
			clusterIntents[clusterID] = make(map[string]int)
		}
		clusterIntents[clusterID][sample.Intent]++
	}
	
	// Assign dominant intent to each cluster
	for clusterID, intents := range clusterIntents {
		maxCount := 0
		dominantIntent := ""
		
		for intent, count := range intents {
			if count > maxCount {
				maxCount = count
				dominantIntent = intent
			}
		}
		
		r.somNetwork.ClusterMap[clusterID] = dominantIntent
		r.somNetwork.IntentMap[dominantIntent] = clusterID
	}
}

func (r *EnhancedRAGService) HandleRAGQuery(ctx *gin.Context) {
	var req RAGRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	start := time.Now()
	
	// Set defaults
	if req.MaxResults == 0 {
		req.MaxResults = 10
	}
	if req.MinScore == 0 {
		req.MinScore = 0.3
	}
	
	// Check cache first
	cacheKey := fmt.Sprintf("rag:%s:%s", req.UserID, req.Query)
	cached, err := r.getCachedResponse(cacheKey)
	if err == nil && cached != nil {
		cached.CacheHit = true
		r.cacheHitRate = (r.cacheHitRate*float64(r.queryCount) + 1.0) / float64(r.queryCount+1)
		r.queryCount++
		ctx.JSON(http.StatusOK, cached)
		return
	}
	
	// Generate query embedding
	queryEmbedding := r.generateSimpleEmbedding(req.Query)
	
	// Analyze user intent using SOM
	intent, intentScore, cluster := r.analyzeIntentWithSOM(queryEmbedding)
	
	// Perform semantic search
	results, err := r.performSemanticSearch(queryEmbedding, req, intent)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	// Generate suggestions based on intent
	suggestions := r.generateIntentBasedSuggestions(intent, results)
	
	// Build response
	response := RAGResponse{
		Query:          req.Query,
		Results:        results,
		UserIntent:     intent,
		IntentScore:    intentScore,
		SOMCluster:     cluster,
		ProcessingTime: float64(time.Since(start).Nanoseconds()) / 1e6,
		TotalDocs:      len(results),
		Suggestions:    suggestions,
		Context:        req.Context,
		CacheHit:       false,
	}
	
	// Cache response
	r.cacheResponse(cacheKey, &response)
	
	// Update metrics
	r.updateMetrics(time.Since(start))
	
	ctx.JSON(http.StatusOK, response)
}

func (r *EnhancedRAGService) analyzeIntentWithSOM(embedding []float64) (string, float64, int) {
	if r.somNetwork == nil {
		return "general", 0.5, 0
	}
	
	// Find best matching cluster
	x, y := r.somNetwork.findBMU(embedding)
	clusterID := y*r.somNetwork.Width + x
	
	// Get intent for cluster
	intent, exists := r.somNetwork.ClusterMap[clusterID]
	if !exists {
		intent = "general"
	}
	
	// Calculate intent confidence based on distance to cluster center
	distance := r.somNetwork.euclideanDistance(embedding, r.somNetwork.Weights[x][y])
	confidence := math.Max(0.0, 1.0-distance)
	
	return intent, confidence, clusterID
}

func (r *EnhancedRAGService) performSemanticSearch(queryEmbedding []float64, req RAGRequest, intent string) ([]DocumentResult, error) {
	// Convert embedding to PostgreSQL format
	embeddingStr := r.floatArrayToPGVector(queryEmbedding)
	
	// Build query based on intent
	query := r.buildSemanticQuery(intent, req)
	
	rows, err := r.db.Query(context.Background(), query, embeddingStr, req.MinScore, req.MaxResults)
	if err != nil {
		return nil, fmt.Errorf("semantic search query failed: %w", err)
	}
	defer rows.Close()
	
	var results []DocumentResult
	for rows.Next() {
		var result DocumentResult
		var metadataJSON []byte
		
		err := rows.Scan(
			&result.ID,
			&result.CaseID,
			&result.Title,
			&result.Content,
			&result.DocumentType,
			&result.Score,
			&metadataJSON,
			&result.CreatedAt,
		)
		if err != nil {
			continue
		}
		
		// Parse metadata
		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &result.Metadata)
		}
		
		// Calculate relevance
		result.Relevance = r.calculateRelevance(result.Score, intent)
		
		// Generate highlights
		result.Highlights = r.generateHighlights(result.Content, req.Query)
		
		results = append(results, result)
	}
	
	return results, nil
}

func (r *EnhancedRAGService) buildSemanticQuery(intent string, req RAGRequest) string {
	baseQuery := `
		SELECT 
			id, case_id, title, content, document_type,
			1 - (embedding <=> $1::vector) as score,
			metadata, created_at
		FROM legal_documents
		WHERE 1 - (embedding <=> $1::vector) > $2
	`
	
	// Add intent-specific filters
	switch intent {
	case "search", "research":
		baseQuery += ` AND document_type IN ('evidence', 'case_file', 'report')`
	case "analyze", "compare":
		baseQuery += ` AND document_type IN ('analysis', 'expert_opinion', 'case_law')`
	case "draft":
		baseQuery += ` AND document_type IN ('template', 'precedent', 'form')`
	}
	
	// Add case filter if provided
	if req.CaseID != "" {
		baseQuery += fmt.Sprintf(` AND case_id = '%s'`, req.CaseID)
	}
	
	baseQuery += `
		ORDER BY score DESC
		LIMIT $3
	`
	
	return baseQuery
}

func (r *EnhancedRAGService) floatArrayToPGVector(arr []float64) string {
	strArr := make([]string, len(arr))
	for i, v := range arr {
		strArr[i] = strconv.FormatFloat(v, 'f', -1, 64)
	}
	return "[" + strings.Join(strArr, ",") + "]"
}

func (r *EnhancedRAGService) calculateRelevance(score float64, intent string) string {
	// Adjust relevance based on intent
	adjustedScore := score
	
	switch intent {
	case "analyze", "research":
		adjustedScore *= 1.1 // Boost analytical content
	case "draft":
		adjustedScore *= 0.9 // Templates may be less specific
	}
	
	if adjustedScore > 0.8 {
		return "high"
	} else if adjustedScore > 0.6 {
		return "medium"
	} else {
		return "low"
	}
}

func (r *EnhancedRAGService) generateHighlights(content, query string) []string {
	// Simple highlighting (in production, use proper text processing)
	words := strings.Fields(strings.ToLower(query))
	contentLower := strings.ToLower(content)
	
	var highlights []string
	contentWords := strings.Fields(content)
	
	for i, word := range contentWords {
		wordLower := strings.ToLower(word)
		for _, queryWord := range words {
			if strings.Contains(wordLower, queryWord) && len(highlights) < 3 {
				start := max(0, i-3)
				end := min(len(contentWords), i+4)
				highlight := strings.Join(contentWords[start:end], " ")
				highlights = append(highlights, "..."+highlight+"...")
				break
			}
		}
	}
	
	return highlights
}

func (r *EnhancedRAGService) generateIntentBasedSuggestions(intent string, results []DocumentResult) []string {
	var suggestions []string
	
	switch intent {
	case "search":
		suggestions = []string{
			"Refine search with specific terms",
			"Filter by document type",
			"Search within case documents",
		}
	case "analyze":
		suggestions = []string{
			"Compare with similar cases",
			"Generate detailed analysis",
			"Export analysis report",
		}
	case "summarize":
		suggestions = []string{
			"Create executive summary",
			"Generate bullet points",
			"Extract key findings",
		}
	case "draft":
		suggestions = []string{
			"Use document template",
			"Add legal citations",
			"Review similar documents",
		}
	default:
		suggestions = []string{
			"Explore related topics",
			"Get more details",
			"Save for later",
		}
	}
	
	return suggestions
}

func (r *EnhancedRAGService) getCachedResponse(key string) (*RAGResponse, error) {
	data, err := r.redis.Get(context.Background(), key).Result()
	if err != nil {
		return nil, err
	}
	
	var response RAGResponse
	err = json.Unmarshal([]byte(data), &response)
	return &response, err
}

func (r *EnhancedRAGService) cacheResponse(key string, response *RAGResponse) {
	data, _ := json.Marshal(response)
	r.redis.Set(context.Background(), key, data, 10*time.Minute)
}

func (r *EnhancedRAGService) updateMetrics(duration time.Duration) {
	r.queryCount++
	processingTime := float64(duration.Nanoseconds()) / 1e6
	r.avgQueryTime = (r.avgQueryTime*float64(r.queryCount-1) + processingTime) / float64(r.queryCount)
}

func (r *EnhancedRAGService) HandleStatus(ctx *gin.Context) {
	status := map[string]interface{}{
		"service":        "Enhanced RAG System",
		"status":         "running",
		"som_enabled":    r.somNetwork != nil,
		"som_dimensions": fmt.Sprintf("%dx%d", r.config.SOMWidth, r.config.SOMHeight),
		"query_count":    r.queryCount,
		"avg_query_time": r.avgQueryTime,
		"cache_hit_rate": r.cacheHitRate,
		"embedding_dim":  r.config.EmbeddingDim,
		"intents":        len(LegalIntents),
		"timestamp":      time.Now(),
	}
	
	ctx.JSON(http.StatusOK, status)
}

func (r *EnhancedRAGService) HandleAnalyzeIntent(ctx *gin.Context) {
	var req struct {
		Query string `json:"query"`
	}
	
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	embedding := r.generateSimpleEmbedding(req.Query)
	intent, score, cluster := r.analyzeIntentWithSOM(embedding)
	
	ctx.JSON(http.StatusOK, map[string]interface{}{
		"query":       req.Query,
		"intent":      intent,
		"score":       score,
		"cluster":     cluster,
		"suggestions": r.generateIntentBasedSuggestions(intent, nil),
	})
}

func (r *EnhancedRAGService) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())
	
	// CORS middleware
	router.Use(func(ctx *gin.Context) {
		ctx.Header("Access-Control-Allow-Origin", "*")
		ctx.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		ctx.Header("Access-Control-Allow-Headers", "Content-Type")
		
		if ctx.Request.Method == "OPTIONS" {
			ctx.AbortWithStatus(204)
			return
		}
		
		ctx.Next()
	})
	
	// API routes
	api := router.Group("/api")
	{
		api.POST("/rag/query", r.HandleRAGQuery)
		api.POST("/rag/analyze-intent", r.HandleAnalyzeIntent)
		api.GET("/rag/status", r.HandleStatus)
	}
	
	// Root endpoint
	router.GET("/", func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"service":     "Enhanced RAG System with SOM",
			"version":     "2.0.0",
			"status":      "running",
			"som_enabled": r.somNetwork != nil,
			"endpoints": []string{
				"/api/rag/query", "/api/rag/analyze-intent", "/api/rag/status",
			},
		})
	})
	
	return router
}

func (r *EnhancedRAGService) Run() error {
	if err := r.Initialize(); err != nil {
		return err
	}
	
	router := r.setupRoutes()
	
	log.Printf("🧠 Enhanced RAG System starting on port %s", r.config.Port)
	log.Printf("🎯 SOM Network: %dx%d (%d intents)", r.config.SOMWidth, r.config.SOMHeight, len(LegalIntents))
	log.Printf("🔍 Embedding Dimension: %d", r.config.EmbeddingDim)
	log.Printf("📊 Vector Threshold: %.2f", r.config.VectorThreshold)
	
	return router.Run(":" + r.config.Port)
}

func (r *EnhancedRAGService) Cleanup() {
	if r.db != nil {
		r.db.Close()
	}
	
	if r.redis != nil {
		r.redis.Close()
	}
}

// Utility functions
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

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if f, err := strconv.ParseFloat(value, 64); err == nil {
			return f
		}
	}
	return defaultValue
}

func main() {
	service := NewEnhancedRAGService()
	defer service.Cleanup()
	
	if err := service.Run(); err != nil {
		log.Fatalf("💥 Enhanced RAG service failed: %v", err)
	}
}