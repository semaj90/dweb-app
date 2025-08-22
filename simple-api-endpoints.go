package main

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Simple API service to provide missing /api/rag and /api/ai endpoints
// This is a minimal implementation for immediate testing

type RAGRequest struct {
	Query       string                 `json:"query"`
	UserID      string                 `json:"user_id,omitempty"`
	CaseID      string                 `json:"case_id,omitempty"`
	Context     map[string]interface{} `json:"context,omitempty"`
	MaxResults  int                    `json:"max_results,omitempty"`
}

type AIRequest struct {
	Prompt      string                 `json:"prompt"`
	Model       string                 `json:"model,omitempty"`
	Context     map[string]interface{} `json:"context,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
}

func main() {
	// Set Gin to release mode
	gin.SetMode(gin.ReleaseMode)
	
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	
	// CORS configuration
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	router.Use(cors.New(config))
	
	// API routes
	api := router.Group("/api")
	{
		// Health check
		api.GET("/health", handleHealth)
		
		// RAG endpoint for /api/v1/rag proxy
		api.POST("/rag", handleRAG)
		
		// AI endpoint for /api/v1/ai proxy
		api.POST("/ai", handleAI)
		
		// Original endpoints for compatibility
		api.POST("/rag/query", handleRAGQuery)
		api.GET("/rag/status", handleStatus)
	}
	
	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":    "Simple API Endpoints",
			"version":    "1.0.0",
			"status":     "running",
			"port":       8094,
			"endpoints": []string{
				"/api/health", "/api/rag", "/api/ai", 
				"/api/rag/query", "/api/rag/status",
			},
			"message":    "Providing missing REST API endpoints for Vite proxy",
			"timestamp":  time.Now(),
		})
	})
	
	port := "8094"
	log.Printf("🚀 Simple API Endpoints service starting on port %s", port)
	log.Printf("📡 Endpoints available:")
	log.Printf("   - POST /api/rag")
	log.Printf("   - POST /api/ai") 
	log.Printf("   - GET  /api/health")
	log.Printf("   - GET  /api/rag/status")
	log.Printf("✅ Ready to handle Vite proxy requests")
	
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("💥 Failed to start service: %v", err)
	}
}

func handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"service":   "Simple API Endpoints",
		"timestamp": time.Now(),
		"uptime":    "running",
	})
}

func handleRAG(c *gin.Context) {
	var req RAGRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if req.Query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query is required"})
		return
	}
	
	// Mock RAG response
	response := gin.H{
		"query":       req.Query,
		"status":      "success",
		"results": []gin.H{
			{
				"id":          1,
				"title":       "Sample Legal Document",
				"content":     fmt.Sprintf("Sample content related to: %s", req.Query),
				"score":       0.85,
				"relevance":   "high",
				"document_type": "contract",
			},
			{
				"id":          2,
				"title":       "Legal Precedent",
				"content":     fmt.Sprintf("Legal precedent for: %s", req.Query),
				"score":       0.72,
				"relevance":   "medium", 
				"document_type": "case_law",
			},
		},
		"total_results":   2,
		"processing_time": 45.2,
		"timestamp":       time.Now(),
		"service":         "simple-api-endpoints",
	}
	
	log.Printf("📡 RAG request processed: %s", req.Query)
	c.JSON(http.StatusOK, response)
}

func handleAI(c *gin.Context) {
	var req AIRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	if req.Prompt == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "prompt is required"})
		return
	}
	
	// Mock AI response
	response := gin.H{
		"prompt":   req.Prompt,
		"model":    "simple-api-mock",
		"response": fmt.Sprintf("AI analysis of: %s. This is a mock response from the simple API endpoints service. In production, this would connect to a real LLM.", req.Prompt),
		"status":   "success",
		"confidence": 0.90,
		"processing_time": 125.7,
		"timestamp": time.Now(),
		"service":   "simple-api-endpoints",
		"tokens": map[string]interface{}{
			"input":  len(req.Prompt),
			"output": 85,
			"total":  len(req.Prompt) + 85,
		},
	}
	
	log.Printf("🤖 AI request processed: %s", req.Prompt[:min(50, len(req.Prompt))])
	c.JSON(http.StatusOK, response)
}

func handleRAGQuery(c *gin.Context) {
	// Redirect to simplified RAG handler
	handleRAG(c)
}

func handleStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":    "Simple API Endpoints",
		"status":     "running",
		"version":    "1.0.0",
		"endpoints":  4,
		"timestamp":  time.Now(),
		"message":    "All endpoints operational",
	})
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}