// Main integration test for document processor with SvelteKit integration
package main

import (
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

func main() {
	// Initialize integration
	processor := InitializeIntegrationBridge()
	
	// Create Gin router
	router := gin.Default()
	
	// Health check endpoint with integration status
	router.GET("/api/health", func(c *gin.Context) {
		health := processor.HealthCheck()
		c.JSON(http.StatusOK, health)
	})
	
	// Document upload endpoint with SvelteKit integration
	router.POST("/api/upload", func(c *gin.Context) {
		var req DocumentUploadRequest
		if err := c.ShouldBind(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		
		response, err := processor.ProcessDocumentWithSvelteKit(&req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(http.StatusOK, response)
	})
	
	// Test interface
	router.GET("/test", func(c *gin.Context) {
		c.HTML(http.StatusOK, "test.html", gin.H{
			"title": "Document Processor Integration Test",
		})
	})
	
	// Determine port
	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}
	
	log.Printf("Starting integrated document processor on port %s", port)
	log.Printf("Health endpoint: http://localhost:%s/api/health", port)
	log.Printf("Upload endpoint: http://localhost:%s/api/upload", port)
	log.Printf("Test interface: http://localhost:%s/test", port)
	
	if err := router.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}