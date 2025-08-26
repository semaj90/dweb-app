// Integration Bridge between document-processor.go and SvelteKit Ollama API endpoints
// Ensures compatibility and unified configuration between Go services and SvelteKit frontend

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
)

// Integration configuration for SvelteKit <-> Go service compatibility
type IntegrationConfig struct {
	// SvelteKit frontend endpoints
	SvelteKitBaseURL   string `json:"sveltekit_base_url"`
	SvelteKitPort      string `json:"sveltekit_port"`
	
	// Ollama configuration (shared between systems)
	OllamaURL          string `json:"ollama_url"`
	OllamaModel        string `json:"ollama_model"`
	EmbeddingModel     string `json:"embedding_model"`
	
	// Go service configuration
	DocumentProcessorPort string `json:"document_processor_port"`
	AIServicePort         string `json:"ai_service_port"`
	
	// GPU configuration
	EnableGPU          bool   `json:"enable_gpu"`
	GPUDevice          string `json:"gpu_device"`
}

// Default integration configuration
func NewIntegrationConfig() *IntegrationConfig {
	return &IntegrationConfig{
		SvelteKitBaseURL:      "http://localhost:5173",
		SvelteKitPort:         "5173",
		OllamaURL:            "http://localhost:11434",
		OllamaModel:          "gemma3-legal",
		EmbeddingModel:       "nomic-embed-text",
		DocumentProcessorPort: "8081",
		AIServicePort:        "8094",
		EnableGPU:            true,
		GPUDevice:            "cuda:0",
	}
}

// SvelteKit API client for Go services
type SvelteKitAPIClient struct {
	config     *IntegrationConfig
	httpClient *http.Client
}

func NewSvelteKitAPIClient(config *IntegrationConfig) *SvelteKitAPIClient {
	return &SvelteKitAPIClient{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Test SvelteKit Ollama endpoints from Go service
func (c *SvelteKitAPIClient) TestOllamaIntegration() error {
	endpoints := []string{
		"/api/ollama/models",
		"/api/ollama/gpu-status", 
		"/api/ollama/gpu-config",
	}

	for _, endpoint := range endpoints {
		url := c.config.SvelteKitBaseURL + endpoint
		resp, err := c.httpClient.Get(url)
		if err != nil {
			return fmt.Errorf("failed to connect to SvelteKit endpoint %s: %w", endpoint, err)
		}
		resp.Body.Close()
		
		if resp.StatusCode != 200 {
			return fmt.Errorf("SvelteKit endpoint %s returned status %d", endpoint, resp.StatusCode)
		}
	}
	
	return nil
}

// Generate embedding using SvelteKit Ollama API
func (c *SvelteKitAPIClient) GenerateEmbedding(text string) ([]float32, error) {
	payload := map[string]interface{}{
		"text":      text,
		"model":     c.config.EmbeddingModel + ":latest",
		"normalize": true,
		"truncate":  true,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	url := c.config.SvelteKitBaseURL + "/api/ollama/embed"
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to call SvelteKit embedding API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("SvelteKit embedding API returned status %d", resp.StatusCode)
	}

	var result struct {
		Success   bool      `json:"success"`
		Embedding []float32 `json:"embedding"`
		Error     string    `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode embedding response: %w", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("SvelteKit embedding failed: %s", result.Error)
	}

	return result.Embedding, nil
}

// Generate chat completion using SvelteKit Ollama API
func (c *SvelteKitAPIClient) GenerateCompletion(message, systemPrompt string, useVectorSearch bool) (string, error) {
	payload := map[string]interface{}{
		"message":         message,
		"model":           c.config.OllamaModel + ":latest",
		"temperature":     0.7,
		"maxTokens":       2048,
		"stream":          false,
		"systemPrompt":    systemPrompt,
		"useVectorSearch": useVectorSearch,
		"context":         []string{},
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat request: %w", err)
	}

	url := c.config.SvelteKitBaseURL + "/api/ollama/chat"
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("failed to call SvelteKit chat API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("SvelteKit chat API returned status %d", resp.StatusCode)
	}

	var result struct {
		Success  bool   `json:"success"`
		Response string `json:"response"`
		Error    string `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode chat response: %w", err)
	}

	if !result.Success {
		return "", fmt.Errorf("SvelteKit chat failed: %s", result.Error)
	}

	return result.Response, nil
}

// Check GPU status using SvelteKit API
func (c *SvelteKitAPIClient) CheckGPUStatus() (map[string]interface{}, error) {
	url := c.config.SvelteKitBaseURL + "/api/ollama/gpu-status"
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to check GPU status: %w", err)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode GPU status: %w", err)
	}

	return result, nil
}

// Enhanced DocumentProcessor with SvelteKit integration
type EnhancedDocumentProcessor struct {
	*DocumentProcessor
	svelteKitClient *SvelteKitAPIClient
	integrationConfig *IntegrationConfig
}

func NewEnhancedDocumentProcessor(config *IntegrationConfig) *EnhancedDocumentProcessor {
	// Create standard document processor with updated config
	docConfig := &Config{
		Port:           config.DocumentProcessorPort,
		OllamaURL:      config.OllamaURL,
		EnableGPU:      config.EnableGPU,
		MaxConcurrency: 16,
	}

	docProcessor := NewDocumentProcessor(docConfig)
	svelteKitClient := NewSvelteKitAPIClient(config)

	return &EnhancedDocumentProcessor{
		DocumentProcessor: docProcessor,
		svelteKitClient:   svelteKitClient,
		integrationConfig: config,
	}
}

// Process document with SvelteKit API integration
func (edp *EnhancedDocumentProcessor) ProcessDocumentWithSvelteKit(request *DocumentUploadRequest) (*DocumentProcessingResponse, error) {
	// First, test SvelteKit integration
	if err := edp.svelteKitClient.TestOllamaIntegration(); err != nil {
		return nil, fmt.Errorf("SvelteKit integration test failed: %w", err)
	}

	// Process document using standard processor (need to save file first)
	// This is a simplified version - in production, you'd handle file upload properly
	tempFilePath := fmt.Sprintf("./temp_%s_%s", request.CaseID, request.File.Filename)
	docID := fmt.Sprintf("doc_%d", time.Now().Unix())
	
	response, err := edp.DocumentProcessor.processDocument(tempFilePath, request, docID)
	if err != nil {
		return nil, fmt.Errorf("document processing failed: %w", err)
	}

	// Enhance embeddings using SvelteKit API
	if request.EnableEmbedding {
		for _, chunk := range response.Chunks {
			embedding, err := edp.svelteKitClient.GenerateEmbedding(chunk.Content)
			if err != nil {
				// Fallback to direct Ollama API
				embedding, err = edp.DocumentProcessor.generateEmbedding(chunk.Content)
				if err != nil {
					continue // Skip this embedding on error
				}
			}

			embeddingChunk := EmbeddingChunk{
				ChunkID:   chunk.ID,
				Embedding: embedding,
				Dimension: len(embedding),
				Model:     edp.integrationConfig.EmbeddingModel,
			}

			response.Embeddings = append(response.Embeddings, embeddingChunk)
		}
	}

	// Enhance summary using SvelteKit chat API
	if response.ExtractedText != "" {
		systemPrompt := fmt.Sprintf("You are a legal AI assistant specialized in %s documents. Provide a comprehensive analysis.", request.DocumentType)
		enhancedSummary, err := edp.svelteKitClient.GenerateCompletion(
			"Analyze this legal document: "+response.ExtractedText,
			systemPrompt,
			true, // use vector search
		)

		if err == nil {
			response.Summary = enhancedSummary
		}
	}

	// GPU status would be available in performance metrics
	_ = edp.svelteKitClient // Keep reference to avoid unused variable error

	return response, nil
}

// Integration health check endpoint
func (edp *EnhancedDocumentProcessor) HealthCheck() map[string]interface{} {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Format(time.RFC3339),
		"services": map[string]interface{}{},
	}

	// Test SvelteKit integration
	if err := edp.svelteKitClient.TestOllamaIntegration(); err != nil {
		health["services"].(map[string]interface{})["sveltekit"] = map[string]interface{}{
			"status": "unhealthy",
			"error":  err.Error(),
		}
	} else {
		health["services"].(map[string]interface{})["sveltekit"] = map[string]interface{}{
			"status": "healthy",
			"url":    edp.integrationConfig.SvelteKitBaseURL,
		}
	}

	// Test direct Ollama connection
	if edp.isOllamaHealthy() {
		health["services"].(map[string]interface{})["ollama"] = map[string]interface{}{
			"status": "healthy",
			"url":    edp.integrationConfig.OllamaURL,
		}
	} else {
		health["services"].(map[string]interface{})["ollama"] = map[string]interface{}{
			"status": "unhealthy",
			"url":    edp.integrationConfig.OllamaURL,
		}
	}

	// Check GPU status
	if gpuStatus, err := edp.svelteKitClient.CheckGPUStatus(); err == nil {
		health["services"].(map[string]interface{})["gpu"] = gpuStatus
	}

	return health
}

// Check if Ollama is healthy
func (edp *EnhancedDocumentProcessor) isOllamaHealthy() bool {
	resp, err := http.Get(edp.integrationConfig.OllamaURL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

// Initialize integration bridge
func InitializeIntegrationBridge() *EnhancedDocumentProcessor {
	config := NewIntegrationConfig()
	
	// Override with environment variables if available
	if ollamaURL := os.Getenv("OLLAMA_URL"); ollamaURL != "" {
		config.OllamaURL = ollamaURL
	}
	
	if svelteKitURL := os.Getenv("SVELTEKIT_URL"); svelteKitURL != "" {
		config.SvelteKitBaseURL = svelteKitURL
	}

	return NewEnhancedDocumentProcessor(config)
}