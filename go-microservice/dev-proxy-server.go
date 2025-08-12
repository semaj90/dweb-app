//go:build legacy
// +build legacy

package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// Configuration
const (
	viteDevServerURL = "http://localhost:5173" // SvelteKit dev server
	goServerPort     = ":3000"                 // Our unified server port
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for development
	},
}

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
		apiRoutes.POST("/batch-embed", BatchEmbedHandler)
		apiRoutes.POST("/ai/chat", handleAIChat)
		apiRoutes.POST("/upload", handleFileUpload)
		apiRoutes.GET("/health", handleGoHealth)
	}

	// WebSocket routes
	r.GET("/ws", handleWebSocket)

	// Go microservice routes from main.go
	r.GET("/health", healthCheck)
	r.POST("/parse", parseHandler)
	r.POST("/train-som", trainSOMHandler)
	r.GET("/metrics", metricsHandler)
	r.GET("/som-cache", somCacheHandler)
	r.DELETE("/som-cache", clearSOMCacheHandler)

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
	log.Printf("🤖 Go API endpoints: /api/*, /ws, /health")

	if err := r.Run(goServerPort); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
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