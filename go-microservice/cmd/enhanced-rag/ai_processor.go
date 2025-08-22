package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// AIProcessor handles AI processing with go-llama and Ollama integration
type AIProcessor struct {
	ollamaURL    string
	model        string
	httpClient   *http.Client
	temperature  float32
	maxTokens    int
}

// NewAIProcessor creates a new AI processor with go-llama integration
func NewAIProcessor() *AIProcessor {
	return &AIProcessor{
		ollamaURL:   "http://localhost:11434",
		model:       "gemma3-legal:latest",
		temperature: 0.3,
		maxTokens:   2048,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// LegalAnalysisRequest represents a request for legal AI analysis
type LegalAnalysisRequest struct {
	Text           string            `json:"text"`
	DocumentType   string            `json:"document_type,omitempty"`
	AnalysisType   string            `json:"analysis_type,omitempty"`
	Context        string            `json:"context,omitempty"`
	UseThinking    bool              `json:"use_thinking,omitempty"`
	Temperature    float32           `json:"temperature,omitempty"`
	MaxTokens      int               `json:"max_tokens,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// LegalAnalysisResponse represents the AI analysis response
type LegalAnalysisResponse struct {
	Analysis     string                 `json:"analysis"`
	Confidence   float64                `json:"confidence"`
	Summary      string                 `json:"summary,omitempty"`
	KeyFindings  []string               `json:"key_findings,omitempty"`
	Thinking     string                 `json:"thinking,omitempty"`
	Metadata     map[string]interface{} `json:"metadata"`
	ProcessingTime int64                `json:"processing_time_ms"`
	Model        string                 `json:"model"`
}

// OllamaRequest represents the Ollama API request format
type OllamaRequest struct {
	Model    string  `json:"model"`
	Prompt   string  `json:"prompt"`
	Stream   bool    `json:"stream"`
	Options  map[string]interface{} `json:"options"`
}

// OllamaResponse represents the Ollama API response format
type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Context  []int  `json:"context,omitempty"`
}

// ProcessLegalDocument analyzes legal documents using gemma3:legal model
func (ai *AIProcessor) ProcessLegalDocument(ctx context.Context, req *LegalAnalysisRequest) (*LegalAnalysisResponse, error) {
	startTime := time.Now()

	// Build enhanced legal analysis prompt
	prompt := ai.buildLegalAnalysisPrompt(req)

	// Configure request options
	options := map[string]interface{}{
		"temperature":      ai.getTemperature(req),
		"top_p":           0.9,
		"max_tokens":      ai.getMaxTokens(req),
		"num_ctx":         4096,
		"repeat_penalty":  1.1,
	}

	// Make request to Ollama
	ollamaReq := &OllamaRequest{
		Model:   ai.model,
		Prompt:  prompt,
		Stream:  false,
		Options: options,
	}

	response, err := ai.callOllama(ctx, ollamaReq)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %w", err)
	}

	// Parse and structure the response
	analysis := ai.parseAnalysisResponse(response.Response, req.UseThinking)
	
	processingTime := time.Since(startTime).Milliseconds()

	return &LegalAnalysisResponse{
		Analysis:       analysis.Analysis,
		Confidence:     analysis.Confidence,
		Summary:        analysis.Summary,
		KeyFindings:    analysis.KeyFindings,
		Thinking:       analysis.Thinking,
		Metadata: map[string]interface{}{
			"model":           ai.model,
			"document_type":   req.DocumentType,
			"analysis_type":   req.AnalysisType,
			"use_thinking":    req.UseThinking,
			"processing_time": processingTime,
			"timestamp":       time.Now().UTC().Format(time.RFC3339),
		},
		ProcessingTime: processingTime,
		Model:          ai.model,
	}, nil
}

// buildLegalAnalysisPrompt creates a specialized prompt for legal document analysis
func (ai *AIProcessor) buildLegalAnalysisPrompt(req *LegalAnalysisRequest) string {
	basePrompt := fmt.Sprintf(`You are an expert legal AI assistant specialized in %s analysis.

Document Type: %s
Analysis Type: %s
Context: %s

Document Content:
%s

Instructions:`,
		req.AnalysisType,
		req.DocumentType,
		req.AnalysisType,
		req.Context,
		req.Text,
	)

	if req.UseThinking {
		return basePrompt + `
Use <thinking> tags to show your reasoning process step-by-step, then provide your final analysis.

Your thinking should include:
1. Document type identification and validation
2. Key legal concepts and terminology analysis
3. Structure and format assessment
4. Content analysis for legal significance
5. Risk factors and compliance considerations
6. Confidence assessment and reasoning

After your thinking, provide a structured JSON analysis with:
- summary: Brief overview of the document
- key_findings: Array of important legal points
- confidence: Numerical confidence score (0-1)
- analysis: Detailed legal analysis
- recommendations: Suggested actions or considerations

Format: <thinking>...</thinking> followed by JSON response.`
	}

	return basePrompt + `
Provide a comprehensive legal analysis in JSON format with the following structure:
{
  "summary": "Brief document overview",
  "key_findings": ["Important legal point 1", "Important legal point 2", ...],
  "confidence": 0.95,
  "analysis": "Detailed legal analysis including applicable laws, regulations, and precedents",
  "recommendations": ["Recommended action 1", "Recommended action 2", ...]
}

Focus on legal accuracy, precedent relevance, and practical implications.`
}

// callOllama makes a request to the Ollama API
func (ai *AIProcessor) callOllama(ctx context.Context, req *OllamaRequest) (*OllamaResponse, error) {
	requestBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", 
		ai.ollamaURL+"/api/generate", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := ai.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
	}

	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &ollamaResp, nil
}

// StructuredAnalysis represents parsed analysis output
type StructuredAnalysis struct {
	Analysis     string   `json:"analysis"`
	Summary      string   `json:"summary"`
	KeyFindings  []string `json:"key_findings"`
	Confidence   float64  `json:"confidence"`
	Thinking     string   `json:"thinking,omitempty"`
}

// parseAnalysisResponse parses the AI response into structured format
func (ai *AIProcessor) parseAnalysisResponse(response string, useThinking bool) *StructuredAnalysis {
	analysis := &StructuredAnalysis{
		Confidence: 0.8, // Default confidence
	}

	if useThinking {
		// Extract thinking section
		if thinkingStart := bytes.Index([]byte(response), []byte("<thinking>")); thinkingStart != -1 {
			if thinkingEnd := bytes.Index([]byte(response), []byte("</thinking>")); thinkingEnd != -1 {
				analysis.Thinking = string(response[thinkingStart+10:thinkingEnd])
				response = response[thinkingEnd+11:] // Remove thinking section
			}
		}
	}

	// Try to parse JSON response
	var jsonResp map[string]interface{}
	if err := json.Unmarshal([]byte(response), &jsonResp); err == nil {
		if summary, ok := jsonResp["summary"].(string); ok {
			analysis.Summary = summary
		}
		if analysisText, ok := jsonResp["analysis"].(string); ok {
			analysis.Analysis = analysisText
		}
		if confidence, ok := jsonResp["confidence"].(float64); ok {
			analysis.Confidence = confidence
		}
		if findings, ok := jsonResp["key_findings"].([]interface{}); ok {
			for _, finding := range findings {
				if f, ok := finding.(string); ok {
					analysis.KeyFindings = append(analysis.KeyFindings, f)
				}
			}
		}
	} else {
		// Fallback: treat entire response as analysis
		analysis.Analysis = response
		analysis.Summary = "AI analysis completed"
	}

	return analysis
}

// getTemperature returns the temperature setting for the request
func (ai *AIProcessor) getTemperature(req *LegalAnalysisRequest) float32 {
	if req.Temperature > 0 {
		return req.Temperature
	}
	return ai.temperature
}

// getMaxTokens returns the max tokens setting for the request
func (ai *AIProcessor) getMaxTokens(req *LegalAnalysisRequest) int {
	if req.MaxTokens > 0 {
		return req.MaxTokens
	}
	return ai.maxTokens
}

// HealthCheck verifies the AI processor is working
func (ai *AIProcessor) HealthCheck() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := &OllamaRequest{
		Model:  ai.model,
		Prompt: "Health check: respond with 'OK'",
		Stream: false,
		Options: map[string]interface{}{
			"temperature": 0.1,
			"max_tokens":  10,
		},
	}

	_, err := ai.callOllama(ctx, req)
	return err
}