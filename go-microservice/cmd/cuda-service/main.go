package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

type CudaService struct {
	db          *gorm.DB
	redis       *redis.Client
	workerPath  string
	healthCache map[string]bool
}

type CudaRequest struct {
	JobID string    `json:"jobId"`
	Type  string    `json:"type"` // "embedding", "similarity", "som_train", "autoindex"
	Data  []float32 `json:"data"`
}

type CudaResponse struct {
	JobID     string    `json:"jobId"`
	Type      string    `json:"type"`
	Vector    []float32 `json:"vector"`
	Status    string    `json:"status"`
	Timestamp int64     `json:"timestamp"`
	Error     string    `json:"error,omitempty"`
}

type EmbeddingJob struct {
	ID        uint      `json:"id" gorm:"primaryKey"`
	JobID     string    `json:"job_id" gorm:"uniqueIndex"`
	Type      string    `json:"type"`
	Status    string    `json:"status"` // "pending", "processing", "completed", "failed"
	InputData string    `json:"input_data"`
	Result    string    `json:"result"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

func NewCudaService() *CudaService {
	// Database connection
	dsn := os.Getenv("POSTGRES_DSN")
	if dsn == "" {
		dsn = "host=localhost user=postgres password=123456 dbname=legal_ai_db port=5432 sslmode=disable"
	}
	
	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Printf("Failed to connect to database: %v", err)
	} else {
		// Auto-migrate the schema
		db.AutoMigrate(&EmbeddingJob{})
	}

	// Redis connection
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}
	
	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	// CUDA worker path
	workerPath := os.Getenv("CUDA_WORKER_PATH")
	if workerPath == "" {
		workerPath = "../cuda-worker/cuda-worker.exe"
	}

	return &CudaService{
		db:          db,
		redis:       rdb,
		workerPath:  workerPath,
		healthCache: make(map[string]bool),
	}
}

func (cs *CudaService) healthCheck() gin.HandlerFunc {
	return func(c *gin.Context) {
		health := map[string]interface{}{
			"service":   "cuda-service",
			"timestamp": time.Now().Unix(),
			"status":    "healthy",
			"checks":    map[string]bool{},
		}

		// Check database
		if cs.db != nil {
			sqlDB, err := cs.db.DB()
			if err == nil && sqlDB.Ping() == nil {
				health["checks"].(map[string]bool)["database"] = true
			} else {
				health["checks"].(map[string]bool)["database"] = false
				health["status"] = "degraded"
			}
		} else {
			health["checks"].(map[string]bool)["database"] = false
		}

		// Check Redis
		if cs.redis != nil {
			_, err := cs.redis.Ping(context.Background()).Result()
			health["checks"].(map[string]bool)["redis"] = err == nil
			if err != nil {
				health["status"] = "degraded"
			}
		}

		// Check CUDA worker
		cudaHealthy := cs.testCudaWorker()
		health["checks"].(map[string]bool)["cuda_worker"] = cudaHealthy
		if !cudaHealthy {
			health["status"] = "degraded"
		}

		if health["status"] == "healthy" {
			c.JSON(200, health)
		} else {
			c.JSON(503, health)
		}
	}
}

func (cs *CudaService) testCudaWorker() bool {
	// Test with simple embedding request
	testReq := CudaRequest{
		JobID: "health-check",
		Type:  "embedding",
		Data:  []float32{1.0, 2.0, 3.0, 4.0},
	}

	response, err := cs.executeCudaWorker(testReq)
	return err == nil && response.Status == "success"
}

func (cs *CudaService) executeCudaWorker(req CudaRequest) (*CudaResponse, error) {
	// Convert request to JSON
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Execute CUDA worker
	cmd := exec.Command(cs.workerPath)
	cmd.Stdin = strings.NewReader(string(reqJSON))
	
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("cuda worker execution failed: %v", err)
	}

	// Parse response
	var response CudaResponse
	if err := json.Unmarshal(output, &response); err != nil {
		return nil, fmt.Errorf("failed to parse cuda worker response: %v", err)
	}

	return &response, nil
}

func (cs *CudaService) processEmbedding() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req CudaRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid request format"})
			return
		}

		// Generate job ID if not provided
		if req.JobID == "" {
			req.JobID = fmt.Sprintf("emb_%d", time.Now().UnixNano())
		}

		// Save job to database (if available)
		if cs.db != nil {
			inputJSON, _ := json.Marshal(req.Data)
			job := EmbeddingJob{
				JobID:     req.JobID,
				Type:      req.Type,
				Status:    "processing",
				InputData: string(inputJSON),
			}
			cs.db.Create(&job)
		}

		// Execute CUDA worker
		response, err := cs.executeCudaWorker(req)
		if err != nil {
			// Update job status if DB available
			if cs.db != nil {
				cs.db.Model(&EmbeddingJob{}).Where("job_id = ?", req.JobID).Updates(map[string]interface{}{
					"status": "failed",
					"result": err.Error(),
				})
			}
			
			c.JSON(500, gin.H{
				"error":  "CUDA processing failed",
				"detail": err.Error(),
				"jobId":  req.JobID,
			})
			return
		}

		// Update job status and save result
		if cs.db != nil {
			resultJSON, _ := json.Marshal(response)
			cs.db.Model(&EmbeddingJob{}).Where("job_id = ?", req.JobID).Updates(map[string]interface{}{
				"status": "completed",
				"result": string(resultJSON),
			})
		}

		// Cache result in Redis (if available)
		if cs.redis != nil {
			resultJSON, _ := json.Marshal(response)
			cs.redis.Set(context.Background(), fmt.Sprintf("cuda:result:%s", req.JobID), resultJSON, time.Hour)
		}

		c.JSON(200, response)
	}
}

func (cs *CudaService) getJobStatus() gin.HandlerFunc {
	return func(c *gin.Context) {
		jobID := c.Param("jobId")
		
		// Try Redis cache first
		if cs.redis != nil {
			cachedResult, err := cs.redis.Get(context.Background(), fmt.Sprintf("cuda:result:%s", jobID)).Result()
			if err == nil {
				var response CudaResponse
				if json.Unmarshal([]byte(cachedResult), &response) == nil {
					c.JSON(200, response)
					return
				}
			}
		}

		// Fall back to database
		if cs.db != nil {
			var job EmbeddingJob
			if err := cs.db.Where("job_id = ?", jobID).First(&job).Error; err == nil {
				c.JSON(200, gin.H{
					"jobId":     job.JobID,
					"type":      job.Type,
					"status":    job.Status,
					"result":    job.Result,
					"createdAt": job.CreatedAt,
					"updatedAt": job.UpdatedAt,
				})
				return
			}
		}

		c.JSON(404, gin.H{"error": "Job not found"})
	}
}

func main() {
	// Initialize service
	service := NewCudaService()

	// Setup Gin router
	r := gin.Default()

	// CORS middleware
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})

	// Routes
	r.GET("/health", service.healthCheck())
	r.POST("/vectorize", service.processEmbedding())
	r.POST("/embedding", service.processEmbedding()) // Alias
	r.GET("/job/:jobId", service.getJobStatus())
	
	// Service info
	r.GET("/info", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"service":     "cuda-service",
			"version":     "1.0.0",
			"description": "CUDA GPU processing microservice for Legal AI",
			"endpoints": map[string]string{
				"POST /vectorize":     "Process embeddings with CUDA",
				"POST /embedding":     "Alias for /vectorize",
				"GET /job/:jobId":     "Get job status and results",
				"GET /health":         "Service health check",
				"GET /info":           "Service information",
			},
		})
	})

	// Start server
	port := os.Getenv("CUDA_SERVICE_PORT")
	if port == "" {
		port = "8096"
	}

	log.Printf("🚀 CUDA Service starting on port %s", port)
	log.Printf("📊 Health check: http://localhost:%s/health", port)
	log.Printf("🔧 CUDA Worker: %s", service.workerPath)
	
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}