package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// Configuration
const (
	viteDevServerURL = "http://localhost:5173"
	goServerPort     = ":3000"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for development
	},
}

// Simple types for batch embedding
type SimpleBatchEmbedRequest struct {
	DocID  string   `json:"docId"`
	Chunks []string `json:"chunks"`
	Model  string   `json:"model,omitempty"`
}

type SimpleBatchEmbedResponse struct {
	DocID      string      `json:"docId"`
	Embeddings [][]float32 `json:"embeddings"`
	Metadata   SimpleEmbedMeta `json:"metadata"`
}

type SimpleEmbedMeta struct {
	ProcessedAt   time.Time `json:"processedAt"`
	ChunkCount    int       `json:"chunkCount"`
	Model         string    `json:"model"`
	ProcessTimeMs int64     `json:"processTimeMs"`
}

type SimpleOllamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type SimpleOllamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

var embedCache sync.Map

func main() {
	// Parse Vite URL
	viteURL, err := url.Parse(viteDevServerURL)
	if err != nil {
		log.Fatalf("Invalid Vite server URL: %v", err)
	}

	// Create reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(viteURL)

	// Initialize Gin
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// CORS configuration
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:3000", "http://localhost:5173"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Authorization", "Accept"}
	config.AllowCredentials = true
	r.Use(cors.New(config))

	// Register Go API routes (these take priority)
	apiRoutes := r.Group("/api")
	{
		apiRoutes.POST("/batch-embed", simpleBatchEmbedHandler)
		apiRoutes.POST("/ai/chat", handleAIChat)
		apiRoutes.POST("/upload", handleFileUpload)
		apiRoutes.GET("/health", handleGoHealth)
	}

	// WebSocket routes
	r.GET("/ws", handleWebSocket)

	// Proxy everything else to Vite
	r.NoRoute(func(c *gin.Context) {
		path := c.Request.URL.Path
		
		// If it's an API route, let Gin handle it
		if strings.HasPrefix(path, "/api/") || path == "/ws" {
			c.JSON(http.StatusNotFound, gin.H{"error": "Endpoint not found"})
			return
		}

		// Log what we're proxying
		log.Printf("[PROXY] %s %s -> Vite", c.Request.Method, path)

		// Proxy to Vite
		director := func(req *http.Request) {
			req.URL.Scheme = viteURL.Scheme
			req.URL.Host = viteURL.Host
			req.Host = viteURL.Host
		}
		
		proxy.Director = director
		proxy.ServeHTTP(c.Writer, c.Request)
	})

	log.Printf("🚀 Go-Enhanced Vite Server starting on http://localhost%s", goServerPort)
	log.Printf("📦 Proxying frontend to %s", viteDevServerURL) 
	log.Printf("🤖 Go API endpoints: /api/*, /ws")

	if err := r.Run(goServerPort); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// Simplified batch embedding handler
func simpleBatchEmbedHandler(c *gin.Context) {
	start := time.Now()
	
	var req SimpleBatchEmbedRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
		return
	}

	log.Printf("📊 Processing %d chunks for docId: %s", len(req.Chunks), req.DocID)

	// Check cache first
	cacheKey := fmt.Sprintf("embed:%s", req.DocID)
	if cached, exists := embedCache.Load(cacheKey); exists {
		c.JSON(http.StatusOK, cached)
		return
	}
	
	// Process embeddings with parallel workers
	embeddings := make([][]float32, len(req.Chunks))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 4) // Limit concurrent Ollama calls
	
	for i, chunk := range req.Chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			
			embedding, err := getSimpleOllamaEmbedding(text, req.Model)
			if err != nil {
				fmt.Printf("Error getting embedding for chunk %d: %v\n", idx, err)
				embeddings[idx] = generateSimpleMockEmbedding(text)
			} else {
				embeddings[idx] = embedding
			}
		}(i, chunk)
	}
	
	wg.Wait()
	
	// Prepare response
	response := SimpleBatchEmbedResponse{
		DocID:      req.DocID,
		Embeddings: embeddings,
		Metadata: SimpleEmbedMeta{
			ProcessedAt:   time.Now(),
			ChunkCount:    len(req.Chunks),
			Model:         req.Model,
			ProcessTimeMs: time.Since(start).Milliseconds(),
		},
	}
	
	// Cache to memory
	embedCache.Store(cacheKey, response)
	
	c.JSON(http.StatusOK, response)
}

func getSimpleOllamaEmbedding(text string, model string) ([]float32, error) {
	// Check if Ollama is running
	ollamaURL := "http://localhost:11434/api/embeddings"
	
	if model == "" {
		model = "gemma-legal" // Default to custom model
	}
	
	reqBody := SimpleOllamaEmbedRequest{
		Model:  model,
		Prompt: text,
	}
	
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}
	
	resp, err := http.Post(ollamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		// Fallback to mock embedding for development
		return generateSimpleMockEmbedding(text), nil
	}
	defer resp.Body.Close()
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return generateSimpleMockEmbedding(text), nil
	}
	
	var embedResp SimpleOllamaEmbedResponse
	if err := json.Unmarshal(body, &embedResp); err != nil {
		return generateSimpleMockEmbedding(text), nil
	}
	
	return embedResp.Embedding, nil
}

func generateSimpleMockEmbedding(text string) []float32 {
	// Generate deterministic mock embedding based on text
	embedding := make([]float32, 384)
	hash := 0
	for _, ch := range text {
		hash = (hash * 31 + int(ch)) % 1000000
	}
	
	for i := range embedding {
		embedding[i] = float32((hash+i)%100) / 100.0
	}
	return embedding
}

// WebSocket handler with real-time updates
func handleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	docId := c.Query("docId")
	log.Printf("📡 WebSocket connected - docId: %s", docId)

	// Send welcome message
	conn.WriteJSON(map[string]interface{}{
		"type":    "status_update",
		"message": "🔗 Connected to Go-Enhanced Vite Server",
		"docId":   docId,
		"time":    time.Now(),
	})

	// Keep connection alive and handle messages
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		// Echo messages back with timestamp
		response := map[string]interface{}{
			"type":      "echo",
			"original":  msg,
			"timestamp": time.Now(),
			"server":    "go-enhanced-vite",
		}

		if err := conn.WriteJSON(response); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}

	log.Printf("📡 WebSocket disconnected - docId: %s", docId)
}

// AI Chat handler
func handleAIChat(c *gin.Context) {
	var request struct {
		Message string `json:"message"`
		Context string `json:"context"`
		Model   string `json:"model"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Use your custom models
	model := request.Model
	if model == "" {
		model = "gemma-legal" // Default to your custom model
	}

	log.Printf("🤖 AI Chat request - Model: %s, Message: %.50s...", model, request.Message)

	// Simulate AI response (replace with actual Ollama call)
	response := map[string]interface{}{
		"response":   "This is a response from " + model + " model for: " + request.Message,
		"model":      model,
		"timestamp":  time.Now(),
		"confidence": 0.85,
	}

	c.JSON(http.StatusOK, response)
}

// File upload handler
func handleFileUpload(c *gin.Context) {
	file, header, err := c.Request.FormFile("document")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}
	defer file.Close()

	log.Printf("📄 File uploaded: %s (%.2f KB)", header.Filename, float64(header.Size)/1024)

	// Process file (add your logic here)
	response := map[string]interface{}{
		"success":   true,
		"filename":  header.Filename,
		"size":      header.Size,
		"processed": time.Now(),
		"docId":     "doc-" + time.Now().Format("20060102-150405"),
	}

	c.JSON(http.StatusOK, response)
}

// Go service health
func handleGoHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":     "healthy",
		"service":    "go-enhanced-vite-server",
		"timestamp":  time.Now(),
		"proxy_url":  viteDevServerURL,
		"server_url": "http://localhost" + goServerPort,
	})
}