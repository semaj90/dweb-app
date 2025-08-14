package main

import (
	"log"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

type HealthResponse struct {
	Status    string            `json:"status"`
	Timestamp time.Time         `json:"timestamp"`
	Services  map[string]string `json:"services"`
	Version   string            `json:"version"`
	Port      string            `json:"port"`
}

type ProcessRequest struct {
	Content      string `json:"content"`
	DocumentType string `json:"document_type"`
	Jurisdiction string `json:"jurisdiction"`
}

type ProcessResponse struct {
	Success          bool     `json:"success"`
	Message          string   `json:"message"`
	ProcessedContent string   `json:"processed_content,omitempty"`
	Summary          string   `json:"summary,omitempty"`
	Keywords         []string `json:"keywords,omitempty"`
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("Starting Legal AI Go microservice on port %s", port)

	// Initialize Gin
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// CORS configuration
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{
		"http://localhost:5173", 
		"http://localhost:5174", 
		"http://localhost:5175", 
		"http://localhost:5176",
		"http://localhost:3000",
	}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"*"}
	r.Use(cors.New(config))

	// Health endpoint
	r.GET("/health", func(c *gin.Context) {
		response := HealthResponse{
			Status:    "healthy",
			Timestamp: time.Now(),
			Services: map[string]string{
				"server": "running",
				"api":    "ready",
			},
			Version: "2.0.0",
			Port:    port,
		}
		c.JSON(200, response)
	})

	// API health endpoint
	r.GET("/api/health", func(c *gin.Context) {
		response := HealthResponse{
			Status:    "healthy",
			Timestamp: time.Now(),
			Services: map[string]string{
				"server": "running",
				"api":    "ready",
			},
			Version: "2.0.0",
			Port:    port,
		}
		c.JSON(200, response)
	})

	// Status endpoint
	r.GET("/api/status", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status": "running",
			"port":   port,
			"endpoints": []string{
				"/health",
				"/api/health",
				"/api/status",
				"/api/process",
			},
		})
	})

	// Basic document processing endpoint
	r.POST("/api/process", func(c *gin.Context) {
		var req ProcessRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, ProcessResponse{
				Success: false,
				Message: "Invalid request format: " + err.Error(),
			})
			return
		}

		// Mock processing
		response := ProcessResponse{
			Success:          true,
			Message:          "Document processed successfully",
			ProcessedContent: req.Content,
			Summary:          "This is a mock summary of the document for testing purposes.",
			Keywords:         []string{"legal", "document", "test"},
		}

		c.JSON(200, response)
	})

	// Simple ping endpoint
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "pong",
			"timestamp": time.Now(),
		})
	})

	log.Printf("Server starting on :%s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}