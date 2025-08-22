package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// SimpleAPIEndpoints provides a minimal working API service
type SimpleAPIEndpoints struct {
	aiProcessor *AIProcessor
	port        string
}

// NewSimpleAPIEndpoints creates a new simple API service
func NewSimpleAPIEndpoints() *SimpleAPIEndpoints {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8094"
	}

	return &SimpleAPIEndpoints{
		aiProcessor: NewAIProcessor(),
		port:        port,
	}
}

// StartSimpleServer starts the simple API server
func (s *SimpleAPIEndpoints) StartSimpleServer() error {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// CORS configuration
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"*"}
	r.Use(cors.New(config))

	// Health check endpoint
	r.GET("/", s.handleRoot)
	r.GET("/health", s.handleHealth)
	r.GET("/api/health", s.handleHealth)

	// AI processing endpoints
	r.POST("/api/rag", s.handleRAG)
	r.POST("/api/ai", s.handleAI)
	r.POST("/api/rag/query", s.handleRAGQuery)
	r.GET("/api/rag/status", s.handleRAGStatus)

	// Auto-solver specific endpoints
	r.POST("/api/auto-solve", s.handleAutoSolve)
	r.POST("/api/typescript-fix", s.handleTypeScriptFix)

	log.Printf("🚀 Simple API Endpoints starting on port %s", s.port)
	log.Printf("📍 Health check: http://localhost:%s/api/health", s.port)
	log.Printf("📍 RAG API: http://localhost:%s/api/rag", s.port)
	log.Printf("📍 Auto-solver: http://localhost:%s/api/auto-solve", s.port)

	return r.Run(":" + s.port)
}

// handleRoot provides service information
func (s *SimpleAPIEndpoints) handleRoot(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":   "Simple API Endpoints",
		"status":    "running",
		"port":      s.port,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   "1.0.0",
		"message":   "Providing missing REST API endpoints for Vite proxy",
		"endpoints": []string{
			"/api/health",
			"/api/rag",
			"/api/ai",
			"/api/rag/query",
			"/api/rag/status",
		},
	})
}

// handleHealth provides health status
func (s *SimpleAPIEndpoints) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":   "Simple API Endpoints",
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"uptime":    "running",
	})
}

// handleRAG handles RAG processing requests
func (s *SimpleAPIEndpoints) handleRAG(c *gin.Context) {
	var request LegalAnalysisRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Process with AI
	response, err := s.aiProcessor.ProcessLegalDocument(context.Background(), &request)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"result":  response,
		"service": "enhanced-rag",
	})
}

// handleAI handles general AI processing
func (s *SimpleAPIEndpoints) handleAI(c *gin.Context) {
	s.handleRAG(c) // Delegate to RAG handler
}

// handleRAGQuery handles RAG query requests
func (s *SimpleAPIEndpoints) handleRAGQuery(c *gin.Context) {
	s.handleRAG(c) // Delegate to RAG handler
}

// handleRAGStatus provides RAG service status
func (s *SimpleAPIEndpoints) handleRAGStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"rag_service": "operational",
		"ai_model":    "gemma3-legal:latest",
		"ollama_url":  "http://localhost:11434",
		"status":      "ready",
		"timestamp":   time.Now().UTC(),
	})
}

// TypeScriptError represents a TypeScript compilation error
type TypeScriptError struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Column  int    `json:"column"`
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
	Context string `json:"context,omitempty"`
}

// AutoSolveRequest represents an auto-solve request
type AutoSolveRequest struct {
	Errors       []TypeScriptError `json:"errors"`
	Strategy     string            `json:"strategy,omitempty"`
	MaxFixes     int               `json:"max_fixes,omitempty"`
	UseThinking  bool              `json:"use_thinking,omitempty"`
	Temperature  float32           `json:"temperature,omitempty"`
}

// AutoSolveResponse represents an auto-solve response
type AutoSolveResponse struct {
	Success        bool                   `json:"success"`
	FixesApplied   int                    `json:"fixes_applied"`
	RemainingErrors int                   `json:"remaining_errors"`
	Fixes          []TypeScriptFix        `json:"fixes"`
	ProcessingTime int64                  `json:"processing_time_ms"`
	Strategy       string                 `json:"strategy"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// TypeScriptFix represents a suggested fix
type TypeScriptFix struct {
	File        string `json:"file"`
	Line        int    `json:"line"`
	Column      int    `json:"column"`
	OriginalCode string `json:"original_code"`
	FixedCode   string `json:"fixed_code"`
	Explanation string `json:"explanation,omitempty"`
	Confidence  float64 `json:"confidence"`
}

// handleAutoSolve handles auto-solving of TypeScript errors
func (s *SimpleAPIEndpoints) handleAutoSolve(c *gin.Context) {
	var request AutoSolveRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	startTime := time.Now()

	// Process errors in batches
	maxFixes := request.MaxFixes
	if maxFixes == 0 || maxFixes > 50 {
		maxFixes = 50 // Limit to prevent overwhelming
	}

	fixes := make([]TypeScriptFix, 0)
	processed := 0

	for i, tsError := range request.Errors {
		if i >= maxFixes {
			break
		}

		fix, err := s.generateTypescriptFix(tsError, request.UseThinking)
		if err != nil {
			log.Printf("Failed to generate fix for error %d: %v", i, err)
			continue
		}

		fixes = append(fixes, *fix)
		processed++
	}

	processingTime := time.Since(startTime).Milliseconds()

	response := &AutoSolveResponse{
		Success:         true,
		FixesApplied:    processed,
		RemainingErrors: len(request.Errors) - processed,
		Fixes:           fixes,
		ProcessingTime:  processingTime,
		Strategy:        getStrategy(request.Strategy),
		Metadata: map[string]interface{}{
			"model":             "gemma3-legal:latest",
			"use_thinking":      request.UseThinking,
			"total_errors":      len(request.Errors),
			"batch_size":        maxFixes,
			"timestamp":         time.Now().UTC(),
		},
	}

	c.JSON(http.StatusOK, response)
}

// handleTypeScriptFix handles individual TypeScript error fixes
func (s *SimpleAPIEndpoints) handleTypeScriptFix(c *gin.Context) {
	var tsError TypeScriptError
	if err := c.ShouldBindJSON(&tsError); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	fix, err := s.generateTypescriptFix(tsError, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"fix":     fix,
	})
}

// generateTypescriptFix generates a fix for a TypeScript error using AI
func (s *SimpleAPIEndpoints) generateTypescriptFix(tsError TypeScriptError, useThinking bool) (*TypeScriptFix, error) {
	prompt := s.buildTypescriptFixPrompt(tsError, useThinking)

	analysisRequest := &LegalAnalysisRequest{
		Text:         prompt,
		DocumentType: "typescript_error",
		AnalysisType: "error_fix",
		UseThinking:  useThinking,
		Temperature:  0.2, // Lower temperature for more deterministic fixes
		MaxTokens:    1024,
		Metadata: map[string]interface{}{
			"file":    tsError.File,
			"line":    tsError.Line,
			"column":  tsError.Column,
			"message": tsError.Message,
		},
	}

	response, err := s.aiProcessor.ProcessLegalDocument(context.Background(), analysisRequest)
	if err != nil {
		return nil, fmt.Errorf("AI processing failed: %w", err)
	}

	// Parse AI response to extract fix
	fix := &TypeScriptFix{
		File:        tsError.File,
		Line:        tsError.Line,
		Column:      tsError.Column,
		OriginalCode: tsError.Code,
		FixedCode:   extractFixedCode(response.Analysis),
		Explanation: response.Summary,
		Confidence:  response.Confidence,
	}

	return fix, nil
}

// buildTypescriptFixPrompt builds a specialized prompt for TypeScript error fixing
func (s *SimpleAPIEndpoints) buildTypescriptFixPrompt(tsError TypeScriptError, useThinking bool) string {
	basePrompt := fmt.Sprintf(`Fix this TypeScript error in a Svelte 5 project:

File: %s
Line: %d, Column: %d
Error: %s

Code Context:
%s

Requirements:
- Provide ONLY the corrected code
- Ensure Svelte 5 compatibility (use runes, proper event handlers)
- Maintain type safety
- Follow TypeScript best practices
- Keep changes minimal and focused`,
		tsError.File,
		tsError.Line,
		tsError.Column,
		tsError.Message,
		tsError.Context,
	)

	if useThinking {
		return basePrompt + `

Use <thinking> tags to show your reasoning:
1. Analyze the error type and cause
2. Consider Svelte 5 specific requirements
3. Determine the minimal fix needed
4. Verify type safety

Then provide the corrected code.`
	}

	return basePrompt + `

Respond with the corrected code only.`
}

// extractFixedCode extracts the fixed code from AI response
func extractFixedCode(analysis string) string {
	// Simple extraction - in a real implementation, this would be more sophisticated
	if len(analysis) > 0 {
		return analysis
	}
	return "// Fix could not be generated"
}

// getStrategy returns the strategy name
func getStrategy(strategy string) string {
	if strategy == "" {
		return "parse_errors_first"
	}
	return strategy
}

// StartAPIService starts the simple API service (called from main.go)
func StartAPIService() error {
	service := NewSimpleAPIEndpoints()
	return service.StartSimpleServer()
}