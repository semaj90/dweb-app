// typescript-error-types.go
// Shared types for TypeScript error processing system

package main

import (
	"context"
	"time"
)

// TypeScriptError represents a TypeScript compilation error
type TypeScriptError struct {
	File     string                 `json:"file"`
	Line     int                    `json:"line"`
	Column   int                    `json:"column"`
	Message  string                 `json:"message"`
	Code     string                 `json:"code"`
	Severity string                 `json:"severity"`
	Context  string                 `json:"context"`
	Source   string                 `json:"source"`
	Category string                 `json:"category"`
	Metadata map[string]interface{} `json:"metadata"`
}

// TypeScriptFixResult represents the result of an error fix attempt
type TypeScriptFixResult struct {
	Success        bool                   `json:"success"`
	FixedCode      string                 `json:"fixed_code"`
	Explanation    string                 `json:"explanation"`
	Confidence     float64                `json:"confidence"`
	ProcessingTime time.Duration          `json:"processing_time"`
	TokensUsed     int                    `json:"tokens_used"`
	ErrorType      string                 `json:"error_type"`
	FixType        string                 `json:"fix_type"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// GoLlamaEngine represents the interface to the Go-Llama inference engine
type GoLlamaEngine struct {
	ModelPath    string `json:"model_path"`
	IsInitialized bool   `json:"is_initialized"`
	ModelLoaded  bool   `json:"model_loaded"`
	MaxTokens    int    `json:"max_tokens"`
	Temperature  float64 `json:"temperature"`
}

// BatchProcessRequest represents a batch processing request
type BatchProcessRequest struct {
	Errors      []TypeScriptError `json:"errors"`
	Strategy    string            `json:"strategy"`
	MaxFixes    int               `json:"max_fixes"`
	UseThinking bool              `json:"use_thinking"`
	Temperature float64           `json:"temperature"`
	Concurrency int               `json:"concurrency"`
	MaxTokens   int               `json:"max_tokens"`
}

// BatchProcessResponse represents a batch processing response
type BatchProcessResponse struct {
	Success      bool                     `json:"success"`
	ProcessedCount int                    `json:"processed_count"`
	Results      []*TypeScriptFixResult   `json:"results"`
	ProcessingTime time.Duration          `json:"processing_time"`
	TokensUsed   int                      `json:"tokens_used"`
	Metadata     map[string]interface{}   `json:"metadata"`
}

// Mock methods for GoLlamaEngine to make the code compile
func (gle *GoLlamaEngine) IsLoaded() bool {
	return gle.ModelLoaded
}

func (gle *GoLlamaEngine) ProcessBatch(ctx context.Context, request *BatchProcessRequest) (*BatchProcessResponse, error) {
	// Mock implementation for compilation
	results := make([]*TypeScriptFixResult, len(request.Errors))
	
	for i, err := range request.Errors {
		results[i] = &TypeScriptFixResult{
			Success:        true,
			FixedCode:      "// Mock fix for: " + err.Message,
			Explanation:    "Mock template-based fix",
			Confidence:     0.8,
			ProcessingTime: time.Millisecond * 10,
			TokensUsed:     50,
			ErrorType:      err.Category,
			FixType:        "template",
			Metadata: map[string]interface{}{
				"source": "go_llama",
				"mock":   true,
			},
		}
	}
	
	return &BatchProcessResponse{
		Success:        true,
		ProcessedCount: len(request.Errors),
		Results:        results,
		ProcessingTime: time.Millisecond * time.Duration(len(request.Errors)) * 10,
		TokensUsed:     len(request.Errors) * 50,
		Metadata: map[string]interface{}{
			"strategy": request.Strategy,
			"mock":     true,
		},
	}, nil
}

// NewGoLlamaEngine creates a new Go-Llama engine instance
func NewGoLlamaEngine(modelPath string) *GoLlamaEngine {
	return &GoLlamaEngine{
		ModelPath:     modelPath,
		IsInitialized: true,
		ModelLoaded:   true,
		MaxTokens:     2048,
		Temperature:   0.7,
	}
}