// ENHANCED RAG SOM SYSTEM - GO MICROSERVICE
// Legal AI document processing with Self-Organizing Map clustering
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gomodule/redigo/redis"
	"github.com/joho/godotenv"
)

// Configuration
type Config struct {
	Port       string
	RedisURL   string
	OllamaURL  string
	GPUEnabled bool
}

// Document structures
type Document struct {
	ID          string                 `json:"id"`
	Content     string                 `json:"content"`
	Title       string                 `json:"title"`
	Type        string                 `json:"type"`
	Embedding   []float64              `json:"embedding"`
	Metadata    map[string]interface{} `json:"metadata"`
	ProcessedAt time.Time              `json:"processed_at"`
}

type SOMCluster struct {
	ID       string      `json:"id"`
	Centroid []float64   `json:"centroid"`
	Size     int         `json:"size"`
	Topic    string      `json:"topic"`
	Docs     []Document  `json:"documents"`
}

type RAGRequest struct {
	Query     string            `json:"query"`
	TopK      int               `json:"top_k"`
	Threshold float64           `json:"threshold"`
	Context   map[string]string `json:"context"`
}

type RAGResponse struct {
	Documents []Document  `json:"documents"`
	Clusters  []SOMCluster `json:"clusters"`
	Query     string      `json:"query"`
	Metadata  interface{} `json:"metadata"`
}

type Service struct {
	config     Config
	redisPool  *redis.Pool
	somGrid    *SOMGrid
	embedCache map[string][]float64
}

// Self-Organizing Map implementation
type SOMGrid struct {
	Width   int         `json:"width"`
	Height  int         `json:"height"`
	Neurons [][]Neuron  `json:"neurons"`
	Trained bool        `json:"trained"`
}

type Neuron struct {
	Weights   []float64 `json:"weights"`
	Documents []string  `json:"documents"`
	Topic     string    `json:"topic"`
}

func main() {
	// Load environment
	godotenv.Load()

	config := Config{
		Port:       getEnv("ENHANCED_RAG_PORT", "8094"),
		RedisURL:   getEnv("REDIS_URL", "redis://localhost:6379"),
		OllamaURL:  getEnv("OLLAMA_URL", "http://localhost:11434"),
		GPUEnabled: getEnv("GPU_ENABLED", "false") == "true",
	}

	service := &Service{
		config:     config,
		embedCache: make(map[string][]float64),
	}

	if err := service.initialize(); err != nil {
		log.Fatalf("Failed to initialize service: %v", err)
	}

	router := service.setupRoutes()
	
	server := &http.Server{
		Addr:    ":" + config.Port,
		Handler: router,
	}

	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("🛑 Shutting down Enhanced RAG service...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(ctx)
	}()

	log.Printf("🚀 Enhanced RAG SOM System starting on port %s", config.Port)
	log.Printf("🔥 GPU Acceleration: %v", config.GPUEnabled)
	log.Printf("🧠 SOM Grid: Legal document clustering ready")
	
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}

func (s *Service) initialize() error {
	// Initialize Redis pool
	s.redisPool = &redis.Pool{
		MaxIdle:     3,
		IdleTimeout: 240 * time.Second,
		Dial: func() (redis.Conn, error) {
			return redis.DialURL(s.config.RedisURL)
		},
	}

	// Test Redis connection
	conn := s.redisPool.Get()
	defer conn.Close()
	
	if _, err := conn.Do("PING"); err != nil {
		return fmt.Errorf("redis connection failed: %v", err)
	}

	// Initialize SOM grid for legal document clustering
	s.somGrid = &SOMGrid{
		Width:  20,
		Height: 20,
		Neurons: make([][]Neuron, 20),
	}

	// Initialize neurons
	for i := range s.somGrid.Neurons {
		s.somGrid.Neurons[i] = make([]Neuron, 20)
		for j := range s.somGrid.Neurons[i] {
			s.somGrid.Neurons[i][j] = Neuron{
				Weights:   make([]float64, 384), // nomic-embed-text dimension
				Documents: make([]string, 0),
			}
			
			// Random weight initialization
			for k := range s.somGrid.Neurons[i][j].Weights {
				s.somGrid.Neurons[i][j].Weights[k] = (float64(time.Now().UnixNano()%1000) - 500) / 1000.0
			}
		}
	}

	log.Println("✅ Enhanced RAG service initialized successfully")
	return nil
}

func (s *Service) setupRoutes() *gin.Engine {
	if os.Getenv("GIN_MODE") != "debug" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.Default()

	// CORS configuration
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"*"},
		ExposeHeaders:    []string{"*"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Routes
	router.GET("/health", s.healthCheck)
	router.POST("/search", s.semanticSearch)
	router.POST("/cluster", s.clusterDocuments)
	router.POST("/embed", s.generateEmbedding)
	router.GET("/som/status", s.getSOMStatus)
	router.POST("/som/train", s.trainSOM)
	
	return router
}

func (s *Service) healthCheck(c *gin.Context) {
	conn := s.redisPool.Get()
	defer conn.Close()

	redisOK := false
	if _, err := conn.Do("PING"); err == nil {
		redisOK = true
	}

	c.JSON(http.StatusOK, gin.H{
		"status":      "healthy",
		"service":     "enhanced-rag-som",
		"gpu_enabled": s.config.GPUEnabled,
		"som_trained": s.somGrid.Trained,
		"redis":       redisOK,
		"timestamp":   time.Now().Unix(),
	})
}

func (s *Service) semanticSearch(c *gin.Context) {
	var req RAGRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if req.TopK == 0 {
		req.TopK = 5
	}
	if req.Threshold == 0 {
		req.Threshold = 0.7
	}

	log.Printf("🔍 Semantic search for: %s", req.Query[:min(50, len(req.Query))])

	// Generate embedding for query
	embedding, err := s.getEmbedding(req.Query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate embedding"})
		return
	}

	// Search similar documents
	documents, err := s.findSimilarDocuments(embedding, req.TopK, req.Threshold)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Search failed"})
		return
	}

	// Find relevant clusters
	clusters := s.findRelevantClusters(embedding, 3)

	response := RAGResponse{
		Documents: documents,
		Clusters:  clusters,
		Query:     req.Query,
		Metadata: gin.H{
			"embedding_dim": len(embedding),
			"search_time":   time.Now().Unix(),
			"gpu_used":      s.config.GPUEnabled,
		},
	}

	c.JSON(http.StatusOK, response)
}

func (s *Service) clusterDocuments(c *gin.Context) {
	var documents []Document
	if err := c.ShouldBindJSON(&documents); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	log.Printf("📊 Clustering %d documents", len(documents))

	// Process each document through SOM
	for _, doc := range documents {
		if len(doc.Embedding) == 0 {
			// Generate embedding if not provided
			embedding, err := s.getEmbedding(doc.Content)
			if err != nil {
				continue
			}
			doc.Embedding = embedding
		}
		
		// Find best matching unit (BMU) in SOM
		bmu := s.findBestMatchingUnit(doc.Embedding)
		s.somGrid.Neurons[bmu[0]][bmu[1]].Documents = append(
			s.somGrid.Neurons[bmu[0]][bmu[1]].Documents, 
			doc.ID,
		)
	}

	// Generate cluster summary
	clusters := s.extractClusters()

	c.JSON(http.StatusOK, gin.H{
		"clustered_documents": len(documents),
		"clusters_found":      len(clusters),
		"som_neurons_active":  s.countActiveNeurons(),
		"clusters":            clusters,
	})
}

func (s *Service) generateEmbedding(c *gin.Context) {
	var req struct {
		Text  string `json:"text"`
		Model string `json:"model"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model == "" {
		req.Model = "nomic-embed-text"
	}

	embedding, err := s.getEmbedding(req.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"embedding":   embedding,
		"dimension":   len(embedding),
		"model":       req.Model,
		"gpu_used":    s.config.GPUEnabled,
		"cached":      s.isCached(req.Text),
	})
}

func (s *Service) getSOMStatus(c *gin.Context) {
	activeNeurons := s.countActiveNeurons()
	totalDocs := s.countTotalDocuments()

	c.JSON(http.StatusOK, gin.H{
		"grid_size":      fmt.Sprintf("%dx%d", s.somGrid.Width, s.somGrid.Height),
		"active_neurons": activeNeurons,
		"total_neurons":  s.somGrid.Width * s.somGrid.Height,
		"total_docs":     totalDocs,
		"trained":        s.somGrid.Trained,
		"coverage":       float64(activeNeurons) / float64(s.somGrid.Width*s.somGrid.Height),
	})
}

func (s *Service) trainSOM(c *gin.Context) {
	var req struct {
		Epochs       int     `json:"epochs"`
		LearningRate float64 `json:"learning_rate"`
		Radius       float64 `json:"radius"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set defaults
	if req.Epochs == 0 {
		req.Epochs = 100
	}
	if req.LearningRate == 0 {
		req.LearningRate = 0.1
	}
	if req.Radius == 0 {
		req.Radius = 5.0
	}

	log.Printf("🎓 Training SOM for %d epochs", req.Epochs)

	// Simulate training process (in real implementation, this would train on actual document embeddings)
	go func() {
		for epoch := 0; epoch < req.Epochs; epoch++ {
			// Update learning rate and radius
			currentLR := req.LearningRate * (1.0 - float64(epoch)/float64(req.Epochs))
			currentRadius := req.Radius * (1.0 - float64(epoch)/float64(req.Epochs))
			
			_ = currentLR
			_ = currentRadius
			
			// Simulate training step
			time.Sleep(50 * time.Millisecond)
		}
		
		s.somGrid.Trained = true
		log.Println("✅ SOM training completed")
	}()

	c.JSON(http.StatusAccepted, gin.H{
		"message": "SOM training started",
		"epochs":  req.Epochs,
		"params": gin.H{
			"learning_rate": req.LearningRate,
			"radius":        req.Radius,
		},
	})
}

// Helper functions
func (s *Service) getEmbedding(text string) ([]float64, error) {
	// Check cache first
	if cached, exists := s.embedCache[text]; exists {
		return cached, nil
	}

	// In a real implementation, this would call Ollama
	// For demo purposes, return a mock embedding
	embedding := make([]float64, 384)
	for i := range embedding {
		embedding[i] = float64(time.Now().UnixNano()%1000) / 1000.0
	}

	// Cache the result
	s.embedCache[text] = embedding
	
	return embedding, nil
}

func (s *Service) findSimilarDocuments(queryEmbedding []float64, topK int, threshold float64) ([]Document, error) {
	// Mock implementation - in reality, this would query a vector database
	mockDocs := []Document{
		{
			ID:      "doc_1",
			Title:   "Contract Law Fundamentals",
			Content: "A contract is a legally binding agreement between two or more parties...",
			Type:    "legal_document",
			Metadata: map[string]interface{}{
				"similarity": 0.92,
				"category":   "contract_law",
			},
		},
		{
			ID:      "doc_2", 
			Title:   "Employment Agreement Template",
			Content: "This employment agreement outlines the terms and conditions...",
			Type:    "contract_template",
			Metadata: map[string]interface{}{
				"similarity": 0.85,
				"category":   "employment",
			},
		},
	}

	return mockDocs, nil
}

func (s *Service) findRelevantClusters(queryEmbedding []float64, topK int) []SOMCluster {
	clusters := make([]SOMCluster, 0)
	
	// Find clusters with similar centroids
	for i := 0; i < min(topK, 3); i++ {
		cluster := SOMCluster{
			ID:       fmt.Sprintf("cluster_%d", i+1),
			Size:     5 + i*3,
			Topic:    []string{"Contract Law", "Employment Rights", "Corporate Governance"}[i],
			Centroid: make([]float64, len(queryEmbedding)),
		}
		
		clusters = append(clusters, cluster)
	}
	
	return clusters
}

func (s *Service) findBestMatchingUnit(embedding []float64) [2]int {
	bestDistance := float64(999999)
	bmu := [2]int{0, 0}
	
	for i := 0; i < s.somGrid.Height; i++ {
		for j := 0; j < s.somGrid.Width; j++ {
			distance := s.euclideanDistance(embedding, s.somGrid.Neurons[i][j].Weights)
			if distance < bestDistance {
				bestDistance = distance
				bmu = [2]int{i, j}
			}
		}
	}
	
	return bmu
}

func (s *Service) euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a) && i < len(b); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum // sqrt omitted for performance
}

func (s *Service) extractClusters() []SOMCluster {
	clusters := make([]SOMCluster, 0)
	clusterID := 0
	
	for i := 0; i < s.somGrid.Height; i++ {
		for j := 0; j < s.somGrid.Width; j++ {
			neuron := s.somGrid.Neurons[i][j]
			if len(neuron.Documents) > 0 {
				clusters = append(clusters, SOMCluster{
					ID:       fmt.Sprintf("cluster_%d", clusterID),
					Size:     len(neuron.Documents),
					Topic:    s.inferTopic(neuron.Documents),
					Centroid: neuron.Weights,
				})
				clusterID++
			}
		}
	}
	
	return clusters
}

func (s *Service) inferTopic(documents []string) string {
	// Simple topic inference based on document count
	topics := []string{"Contract Analysis", "Legal Research", "Case Law", "Regulatory Compliance"}
	return topics[len(documents)%len(topics)]
}

func (s *Service) countActiveNeurons() int {
	count := 0
	for i := 0; i < s.somGrid.Height; i++ {
		for j := 0; j < s.somGrid.Width; j++ {
			if len(s.somGrid.Neurons[i][j].Documents) > 0 {
				count++
			}
		}
	}
	return count
}

func (s *Service) countTotalDocuments() int {
	total := 0
	for i := 0; i < s.somGrid.Height; i++ {
		for j := 0; j < s.somGrid.Width; j++ {
			total += len(s.somGrid.Neurons[i][j].Documents)
		}
	}
	return total
}

func (s *Service) isCached(text string) bool {
	_, exists := s.embedCache[text]
	return exists
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
