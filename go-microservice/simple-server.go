//go:build legacy
// +build legacy

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"context"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

type HealthResponse struct {
	Status    string            `json:"status"`
	Timestamp time.Time         `json:"timestamp"`
	Services  map[string]string `json:"services"`
	Version   string            `json:"version"`
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

	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}

	// Initialize Redis
	ctx := context.Background()
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	// Test Redis connection
	_, err := rdb.Ping(ctx).Result()
	redisStatus := "connected"
	if err != nil {
		log.Printf("Redis not available: %v", err)
		redisStatus = "disconnected"
	}

	// Initialize Gin
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// CORS configuration
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"*"}
	r.Use(cors.New(config))

	// Health endpoint
	r.GET("/api/health", func(c *gin.Context) {
		response := HealthResponse{
			Status:    "healthy",
			Timestamp: time.Now(),
			Services: map[string]string{
				"redis":  redisStatus,
				"server": "running",
			},
			Version: "1.0.0",
		}
		c.JSON(200, response)
	})

	// Status endpoint
	r.GET("/api/status", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status": "running",
			"port":   port,
			"redis":  redisStatus,
			"endpoints": []string{
				"/api/health",
				"/api/status",
				"/api/process",
			},
		})
	})

	// Document processing endpoint
	r.POST("/api/process", func(c *gin.Context) {
		var req ProcessRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, ProcessResponse{
				Success: false,
				Message: "Invalid request format",
			})
			return
		}

		// Simple processing simulation
		processed := fmt.Sprintf("Processed legal document: %s", req.Content[:min(len(req.Content), 100)])
		summary := fmt.Sprintf("Legal analysis for %s jurisdiction", req.Jurisdiction)
		keywords := []string{"legal", "document", "analysis", req.DocumentType}

		response := ProcessResponse{
			Success:          true,
			Message:          "Document processed successfully",
			ProcessedContent: processed,
			Summary:          summary,
			Keywords:         keywords,
		}

		// Cache result in Redis if available
		if redisStatus == "connected" {
			cacheKey := fmt.Sprintf("processed:%d", time.Now().Unix())
			cacheData, _ := json.Marshal(response)
			rdb.Set(ctx, cacheKey, cacheData, 1*time.Hour)
		}

		c.JSON(200, response)
	})

	// Vector search endpoint (placeholder)
	r.POST("/api/vector-search", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"results": []gin.H{
				{
					"id":       "doc1",
					"content":  "Sample legal document content",
					"score":    0.95,
					"metadata": gin.H{"type": "contract", "jurisdiction": "US"},
				},
				{
					"id":       "doc2",
					"content":  "Another legal document",
					"score":    0.87,
					"metadata": gin.H{"type": "statute", "jurisdiction": "CA"},
				},
			},
			"total": 2,
			"query": "legal search",
		})
	})

	// Root endpoint
	r.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"service": "Enhanced Legal AI Microservice",
			"version": "1.0.0",
			"status":  "running",
			"endpoints": gin.H{
				"health":        "/api/health",
				"status":        "/api/status",
				"process":       "/api/process (POST)",
				"vector_search": "/api/vector-search (POST)",
			},
		})
	})

	log.Printf("🚀 Simple Legal AI Server starting on port %s", port)
	log.Printf("🔗 Health: http://localhost:%s/api/health", port)
	log.Printf("📊 Status: http://localhost:%s/api/status", port)
	log.Printf("🔧 Redis: %s", redisStatus)

	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
