package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
	log.Println("Starting GPU Orchestrator Service...")

	router := gin.Default()
	
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:5173", "http://localhost:3000"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Authorization"}
	router.Use(cors.New(config))

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"service": "gpu-orchestrator",
			"timestamp": time.Now().Unix(),
		})
	})

	router.GET("/api/gpu/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"gpu_available": true,
			"cuda_version": "11.8",
			"memory_free": 4096,
			"memory_total": 8192,
		})
	})

	router.POST("/api/gpu/process", func(c *gin.Context) {
		var request struct {
			Data []float32 `json:"data"`
			Operation string `json:"operation"`
		}
		
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		result := processOnGPU(request.Data, request.Operation)
		
		c.JSON(http.StatusOK, gin.H{
			"result": result,
			"processed": len(request.Data),
		})
	})

	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	port := os.Getenv("GPU_ORCHESTRATOR_PORT")
	if port == "" {
		port = "8094"
	}

	srv := &http.Server{
		Addr:    ":" + port,
		Handler: router,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	log.Printf("GPU Orchestrator running on port %s", port)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down GPU Orchestrator...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	log.Println("GPU Orchestrator shutdown complete")
}

func processOnGPU(data []float32, operation string) []float32 {
	result := make([]float32, len(data))
	
	switch operation {
	case "multiply":
		for i, v := range data {
			result[i] = v * 2
		}
	case "normalize":
		var sum float32
		for _, v := range data {
			sum += v
		}
		avg := sum / float32(len(data))
		for i, v := range data {
			result[i] = v / avg
		}
	default:
		copy(result, data)
	}
	
	return result
}