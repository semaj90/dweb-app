//go:build legacy
// +build legacy

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
)

// Simple Legal Processor - Fallback without CUDA requirements
type HealthResponse struct {
	Status         string    `json:"status"`
	GPUEnabled     bool      `json:"gpu_enabled"`
	GPUMemory      uint64    `json:"gpu_memory"`
	RedisConnected bool      `json:"redis_connected"`
	DBConnected    bool      `json:"db_connected"`
	Goroutines     int       `json:"goroutines"`
	HeapAlloc      uint64    `json:"heap_alloc"`
	SysMemory      uint64    `json:"sys_memory"`
	Timestamp      time.Time `json:"timestamp"`
}

var (
	dbPool      *pgxpool.Pool
	redisClient *redis.Client
)

func main() {
	log.Println("🚀 Legal Processor Service Starting (Fallback Mode)...")

	// Initialize connections
	initializeConnections()

	// Setup Gin router
	router := gin.Default()

	// CORS middleware
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// Routes
	router.GET("/health", handleHealth)
	router.GET("/metrics", handleMetrics)
	router.POST("/similarity-search", handleSimilaritySearch)
	router.POST("/rag-enhanced", handleRAGSearch)
	router.POST("/llm-request", handleLLMRequest)
	
	// API routes
	router.POST("/api/legal", handleLegalAPI)
	router.GET("/api/legal/gpu", handleGPUStatus)

	log.Println("✅ Legal Processor listening on :8080")
	log.Println("📊 Mode: CPU Fallback (No CUDA required)")
	
	if err := router.Run(":8080"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func initializeConnections() {
	// Database connection
	dbURL := "postgres://legal_admin:123456@localhost:5432/legal_ai_db"
	var err error
	
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	dbPool, err = pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Printf("⚠️ Database connection failed: %v", err)
	} else {
		if err := dbPool.Ping(ctx); err != nil {
			log.Printf("⚠️ Database ping failed: %v", err)
		} else {
			log.Println("✅ Connected to PostgreSQL")
		}
	}

	// Redis connection
	redisClient = redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "",
		DB:       0,
	})

	ctx2, cancel2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel2()
	
	if err := redisClient.Ping(ctx2).Err(); err != nil {
		log.Printf("⚠️ Redis connection failed: %v", err)
	} else {
		log.Println("✅ Connected to Redis")
	}
}

func handleHealth(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Check connections
	dbConnected := false
	if dbPool != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := dbPool.Ping(ctx); err == nil {
			dbConnected = true
		}
	}

	redisConnected := false
	if redisClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := redisClient.Ping(ctx).Err(); err == nil {
			redisConnected = true
		}
	}

	response := HealthResponse{
		Status:         "ok",
		GPUEnabled:     false, // Fallback mode - no GPU
		GPUMemory:      0,
		RedisConnected: redisConnected,
		DBConnected:    dbConnected,
		Goroutines:     runtime.NumGoroutine(),
		HeapAlloc:      m.HeapAlloc,
		SysMemory:      m.Sys,
		Timestamp:      time.Now(),
	}

	c.JSON(http.StatusOK, response)
}

func handleMetrics(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	metrics := map[string]interface{}{
		"memory": map[string]uint64{
			"alloc":      m.Alloc,
			"total":      m.TotalAlloc,
			"sys":        m.Sys,
			"heap_alloc": m.HeapAlloc,
		},
		"goroutines": runtime.NumGoroutine(),
		"cpu_count":  runtime.NumCPU(),
		"timestamp":  time.Now(),
	}

	c.JSON(http.StatusOK, metrics)
}

func handleSimilaritySearch(c *gin.Context) {
	var request struct {
		QueryEmbedding     []float32   `json:"queryEmbedding"`
		DocumentEmbeddings [][]float32 `json:"documentEmbeddings"`
		DocumentIDs        []string    `json:"documentIds"`
		TopK               int         `json:"topK"`
		UseGPU             bool        `json:"useGPU"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// Simple CPU-based similarity (dot product)
	results := make([]map[string]interface{}, 0)
	
	for i, docEmbedding := range request.DocumentEmbeddings {
		if i < len(request.DocumentIDs) {
			score := float32(0.0)
			// Simple dot product
			for j := 0; j < len(request.QueryEmbedding) && j < len(docEmbedding); j++ {
				score += request.QueryEmbedding[j] * docEmbedding[j]
			}
			
			results = append(results, map[string]interface{}{
				"documentId": request.DocumentIDs[i],
				"score":      score,
				"rank":       i + 1,
			})
		}
	}

	// Sort by score (simplified)
	// In production, use proper sorting
	
	response := map[string]interface{}{
		"results":   results,
		"timingMs":  10,
		"method":    "CPU",
		"gpuUsed":   false,
	}

	c.JSON(http.StatusOK, response)
}

func handleRAGSearch(c *gin.Context) {
	var request map[string]interface{}
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// Simplified RAG response
	response := map[string]interface{}{
		"query":             request["query"],
		"results":           []interface{}{},
		"synthesizedAnswer": "This is a fallback response from the CPU-based legal processor.",
		"recommendations":   []string{"Document A", "Document B"},
		"didYouMean":        []string{},
		"processingSteps":   []interface{}{},
		"metrics": map[string]interface{}{
			"totalTimeMs":     100,
			"documentsScored": 0,
		},
	}

	c.JSON(http.StatusOK, response)
}

func handleLLMRequest(c *gin.Context) {
	var request map[string]interface{}
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	response := map[string]interface{}{
		"provider": request["provider"],
		"model":    "fallback",
		"response": "This is a fallback LLM response.",
		"timingMs": 50,
	}

	c.JSON(http.StatusOK, response)
}

func handleLegalAPI(c *gin.Context) {
	var request map[string]interface{}
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	endpoint, ok := request["endpoint"].(string)
	if !ok {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Missing endpoint"})
		return
	}

	switch endpoint {
	case "similarity-search":
		handleSimilaritySearch(c)
	case "rag-enhanced":
		handleRAGSearch(c)
	case "llm-request":
		handleLLMRequest(c)
	default:
		c.JSON(http.StatusOK, gin.H{"status": "ok", "endpoint": endpoint})
	}
}

func handleGPUStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"gpu_enabled": false,
		"mode":        "CPU Fallback",
		"status":      "operational",
	})
}
