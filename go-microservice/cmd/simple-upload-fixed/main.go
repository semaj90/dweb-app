package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	router.Use(cors.New(config))
	
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":    "Simple Upload Service (Fixed)",
			"version":    "1.0.1",
			"status":     "running",
			"timestamp":  time.Now(),
			"message":    "Using dynamic port allocation",
		})
	})
	
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"service":   "Simple Upload Service (Fixed)",
			"timestamp": time.Now(),
		})
	})
	
	router.POST("/upload", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "Upload endpoint ready",
			"status":  "success",
			"timestamp": time.Now(),
		})
	})
	
	// Use HTTP_PORT environment variable with fallback
	port := getEnv("HTTP_PORT", "8103")
	
	log.Printf("🚀 Simple Upload Service (Fixed) starting on port %s", port)
	log.Printf("📁 Endpoints: /health, /upload")
	log.Printf("✅ Using HTTP_PORT environment variable: %s", port)
	
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("💥 Failed to start service: %v", err)
	}
}