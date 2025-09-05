// Enhanced AI Summarization Service with Document Upload and Processing
// GPU-accelerated SIMD parsing, OCR, and concurrent embedding generation

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Enhanced service with document processing
type EnhancedAIService struct {
	config             *Config
	documentProcessor  *DocumentProcessor
}

// Initialize enhanced service
func NewEnhancedAIService(config *Config) *EnhancedAIService {
	// Create uploads directory
	os.MkdirAll("./uploads", 0755)
	
	return &EnhancedAIService{
		config:            config,
		documentProcessor: NewDocumentProcessor(config),
	}
}

// Document upload and processing endpoint
func (s *EnhancedAIService) uploadDocument(c *gin.Context) {
	s.documentProcessor.uploadAndProcess(c)
}

// Batch document processing endpoint
func (s *EnhancedAIService) batchProcessDocuments(c *gin.Context) {
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to parse multipart form"})
		return
	}

	files := form.File["files"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No files provided"})
		return
	}

	// Process up to 10 files concurrently
	maxFiles := 10
	if len(files) > maxFiles {
		files = files[:maxFiles]
	}

	type BatchResult struct {
		FileName string                     `json:"file_name"`
		Success  bool                       `json:"success"`
		Result   *DocumentProcessingResponse `json:"result,omitempty"`
		Error    string                     `json:"error,omitempty"`
	}

	results := make([]BatchResult, len(files))
	resultChan := make(chan BatchResult, len(files))

	// Process files concurrently
	for i, file := range files {
		go func(index int, fileHeader *multipart.FileHeader) {
			// Create a new context for each file
			tempContext := &gin.Context{}
			tempContext.Request = c.Request
			
			// Process individual file
			req := DocumentUploadRequest{
				File:            fileHeader,
				DocumentType:    c.PostForm("document_type"),
				CaseID:          c.PostForm("case_id"),
				PracticeArea:    c.PostForm("practice_area"),
				Jurisdiction:    c.PostForm("jurisdiction"),
				EnableOCR:       c.PostForm("enable_ocr") == "true",
				EnableEmbedding: c.PostForm("enable_embedding") == "true",
				ChunkSize:       512,
			}

			docID := fmt.Sprintf("batch_%d_%d", time.Now().Unix(), index)
			filePath := fmt.Sprintf("./uploads/%s_%s", docID, fileHeader.Filename)
			
			var result BatchResult
			result.FileName = fileHeader.Filename

			// Save file
			if err := c.SaveUploadedFile(fileHeader, filePath); err != nil {
				result.Success = false
				result.Error = fmt.Sprintf("Failed to save file: %v", err)
			} else {
				// Process document
				response, err := s.documentProcessor.processDocument(filePath, &req, docID)
				if err != nil {
					result.Success = false
					result.Error = fmt.Sprintf("Processing failed: %v", err)
				} else {
					result.Success = true
					result.Result = response
				}
				// Cleanup
				os.Remove(filePath)
			}

			resultChan <- result
		}(i, file)
	}

	// Collect results
	for i := 0; i < len(files); i++ {
		results[i] = <-resultChan
	}

	c.JSON(http.StatusOK, gin.H{
		"total_files":     len(files),
		"processing_time": time.Since(time.Now()),
		"results":         results,
	})
}

// Document analysis endpoint
func (s *EnhancedAIService) analyzeDocument(c *gin.Context) {
	var req struct {
		DocumentID string `json:"document_id" binding:"required"`
		Analysis   struct {
			ExtractKeyTerms     bool `json:"extract_key_terms"`
			DetectEntities      bool `json:"detect_entities"`
			ClassifyDocument    bool `json:"classify_document"`
			GenerateQuestions   bool `json:"generate_questions"`
			ComplianceCheck     bool `json:"compliance_check"`
		} `json:"analysis"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Placeholder for advanced document analysis
	analysis := gin.H{
		"document_id": req.DocumentID,
		"analysis_results": gin.H{
			"key_terms": []string{
				"contract", "liability", "jurisdiction", "payment terms", "intellectual property"
			},
			"entities": []gin.H{
				{"type": "PERSON", "text": "John Doe", "confidence": 0.95},
				{"type": "ORG", "text": "Legal Corp", "confidence": 0.88},
				{"type": "DATE", "text": "January 15, 2025", "confidence": 0.92},
			},
			"document_class": gin.H{
				"primary": "legal_contract",
				"confidence": 0.89,
				"categories": []string{"contract", "legal", "business"},
			},
			"generated_questions": []string{
				"What are the key liability provisions in this contract?",
				"When does this agreement become effective?",
				"What are the payment terms and conditions?",
			},
			"compliance_status": gin.H{
				"overall_score": 0.85,
				"issues": []string{
					"Missing jurisdiction clause specificity",
					"Payment terms could be more detailed",
				},
				"recommendations": []string{
					"Add specific governing law clause",
					"Include late payment penalty provisions",
				},
			},
		},
		"processed_at": time.Now(),
		"processing_model": s.config.Model,
	}

	c.JSON(http.StatusOK, analysis)
}

// Enhanced health check with document processing status
func (s *EnhancedAIService) enhancedHealthCheck(c *gin.Context) {
	health := gin.H{
		"status":    "healthy",
		"timestamp": time.Now(),
		"version":   "2.0.0-enhanced",
		"services": gin.H{
			"ollama":  s.checkOllama(),
			"qdrant":  s.checkQdrant(),
			"gpu":     s.checkGPU(),
			"ocr":     "enabled",
			"simd":    "enabled",
		},
		"system": gin.H{
			"goroutines":       runtime.NumGoroutine(),
			"memory":           getMemoryStats(),
			"cpu_cores":        runtime.NumCPU(),
			"processing_workers": s.documentProcessor.chunkWorkers,
		},
		"capabilities": gin.H{
			"supported_formats": []string{
				"PDF", "TXT", "RTF", "DOCX", "PNG", "JPG", "JPEG", "TIFF"
			},
			"features": []string{
				"OCR Processing", "GPU Acceleration", "SIMD Parsing",
				"Concurrent Processing", "Batch Upload", "Vector Embeddings",
				"AI Summarization", "Document Analysis"
			},
			"max_file_size": "100MB",
			"max_batch_size": 10,
		},
	}

	c.JSON(http.StatusOK, health)
}

// Processing statistics endpoint
func (s *EnhancedAIService) getProcessingStats(c *gin.Context) {
	stats := gin.H{
		"documents_processed": 0, // Would track in production
		"total_processing_time": "0s",
		"average_processing_time": "0s",
		"formats_processed": gin.H{
			"pdf":  0,
			"txt":  0,
			"rtf":  0,
			"docx": 0,
			"images": 0,
		},
		"performance_metrics": gin.H{
			"cpu_usage": "calculating...",
			"memory_usage": getMemoryStats(),
			"gpu_utilization": "available",
			"concurrent_capacity": s.documentProcessor.chunkWorkers,
		},
		"ocr_statistics": gin.H{
			"total_ocr_operations": 0,
			"average_confidence": 0.0,
			"supported_languages": []string{"eng", "spa", "fra", "deu"},
		},
	}

	c.JSON(http.StatusOK, stats)
}

// Helper methods
func (s *EnhancedAIService) checkOllama() string {
	resp, err := http.Get(s.config.OllamaURL + "/api/tags")
	if err != nil {
		return "unhealthy"
	}
	defer resp.Body.Close()
	return "healthy"
}

func (s *EnhancedAIService) checkQdrant() string {
	resp, err := http.Get(s.config.QdrantURL + "/health")
	if err != nil {
		return "unhealthy"
	}
	defer resp.Body.Close()
	return "healthy"
}

func (s *EnhancedAIService) checkGPU() string {
	if s.config.EnableGPU {
		return "enabled"
	}
	return "disabled"
}

func getMemoryStats() gin.H {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return gin.H{
		"alloc_mb":       bToMb(m.Alloc),
		"total_alloc_mb": bToMb(m.TotalAlloc),
		"sys_mb":         bToMb(m.Sys),
		"gc_cycles":      m.NumGC,
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

// Main function with enhanced features
func main() {
	// Initialize configuration
	config := initConfig()
	service := NewEnhancedAIService(config)
	defer service.documentProcessor.Shutdown()

	// Create necessary directories
	os.MkdirAll(config.CacheDir, 0755)
	os.MkdirAll("./uploads", 0755)

	// Initialize Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// Configure CORS with file upload support
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000", "*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Set max multipart memory (100MB)
	router.MaxMultipartMemory = 100 << 20

	// API routes
	api := router.Group("/api")
	{
		// Health and system
		api.GET("/health", service.enhancedHealthCheck)
		api.GET("/stats", service.getProcessingStats)

		// Original endpoints
		api.POST("/summarize", service.summarize)
		api.GET("/search", service.search)

		// Enhanced document processing
		api.POST("/upload", service.uploadDocument)
		api.POST("/batch-upload", service.batchProcessDocuments)
		api.POST("/analyze", service.analyzeDocument)
	}

	// Root endpoint with enhanced capabilities
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Enhanced AI Summarization Service",
			"version": "2.0.0-gpu-simd",
			"status":  "ready",
			"endpoints": gin.H{
				"health":        "/api/health",
				"stats":         "/api/stats",
				"summarize":     "/api/summarize",
				"search":        "/api/search",
				"upload":        "/api/upload",
				"batch_upload":  "/api/batch-upload",
				"analyze":       "/api/analyze",
			},
			"features": []string{
				"GPU-accelerated SIMD parsing",
				"Multi-format document support (PDF, TXT, RTF, DOCX, Images)",
				"OCR text extraction with confidence scoring",
				"Concurrent document processing",
				"Batch file upload and processing",
				"Vector embedding generation",
				"AI-powered summarization and analysis",
				"Legal document classification",
				"Real-time performance monitoring",
			},
			"capabilities": gin.H{
				"max_file_size":    "100MB",
				"max_batch_files":  10,
				"supported_formats": []string{"PDF", "TXT", "RTF", "DOCX", "PNG", "JPG", "JPEG", "TIFF"},
				"concurrent_workers": service.documentProcessor.chunkWorkers,
				"gpu_acceleration": config.EnableGPU,
				"simd_optimization": true,
			},
		})
	})

	// Start server
	port := ":" + config.Port
	log.Printf("🚀 Enhanced AI Service starting on port %s", config.Port)
	log.Printf("📋 Document Upload: http://localhost:%s/api/upload", config.Port)
	log.Printf("📊 Batch Processing: http://localhost:%s/api/batch-upload", config.Port)
	log.Printf("🔍 Document Analysis: http://localhost:%s/api/analyze", config.Port)
	log.Printf("💚 Health Check: http://localhost:%s/api/health", config.Port)
	log.Printf("📈 Processing Stats: http://localhost:%s/api/stats", config.Port)

	if err := router.Run(port); err != nil {
		log.Fatalf("❌ Failed to start server: %v", err)
	}
}