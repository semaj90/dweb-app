//go:build legacy
// +build legacy

// model-config.go
// Optimized model configuration with Go best practices
// Add this file to your go-microservice directory

package main

import (
	"fmt"
	"os"
	"strconv"
	"sync"
)

// ModelConfiguration defines the AI models configuration with optimized access patterns
type ModelConfiguration struct {
	// Primary legal analysis model (your local model)
	LegalModel string
	
	// Embedding model (if you have it locally)
	EmbeddingModel string
	
	// Vision model for image analysis (optional)
	VisionModel string
	
	// Model parameters
	Temperature float32
	MaxTokens   int
	NumCtx      int
	
	// Performance settings
	BatchSize     int
	ConcurrentJobs int
	CacheSize     int
}

var (
	// ModelConfig is the global configuration instance
	ModelConfig *ModelConfiguration
	
	// configOnce ensures thread-safe initialization
	configOnce sync.Once
)

// GetModelConfig returns the singleton model configuration
func GetModelConfig() *ModelConfiguration {
	configOnce.Do(func() {
		ModelConfig = &ModelConfiguration{
			LegalModel:     getEnv("LEGAL_MODEL", "gemma3-legal:latest"),
			EmbeddingModel: getEnv("EMBEDDING_MODEL", "nomic-embed-text"),
			VisionModel:    getEnv("VISION_MODEL", "llava:7b"),
			Temperature:    getEnvFloat32("MODEL_TEMPERATURE", 0.3), // Lower temperature for consistent legal analysis
			MaxTokens:      getEnvInt("MODEL_MAX_TOKENS", 4096),
			NumCtx:         getEnvInt("MODEL_NUM_CTX", 4096),
			BatchSize:      getEnvInt("MODEL_BATCH_SIZE", 8),
			ConcurrentJobs: getEnvInt("MODEL_CONCURRENT_JOBS", 4),
			CacheSize:      getEnvInt("MODEL_CACHE_SIZE", 1000),
		}
	})
	return ModelConfig
}

// getEnv retrieves string environment variable with fallback
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvInt retrieves integer environment variable with fallback
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

// getEnvFloat32 retrieves float32 environment variable with fallback
func getEnvFloat32(key string, defaultValue float32) float32 {
	if value := os.Getenv(key); value != "" {
		if floatValue, err := strconv.ParseFloat(value, 32); err == nil {
			return float32(floatValue)
		}
	}
	return defaultValue
}

// ValidateConfig validates the model configuration
func (mc *ModelConfiguration) ValidateConfig() error {
	if mc.LegalModel == "" {
		return fmt.Errorf("legal model not configured")
	}
	if mc.Temperature < 0.0 || mc.Temperature > 2.0 {
		return fmt.Errorf("temperature must be between 0.0 and 2.0")
	}
	if mc.MaxTokens <= 0 {
		return fmt.Errorf("max_tokens must be positive")
	}
	return nil
}

// GetOptimalBatchSize returns the optimal batch size for processing
func (mc *ModelConfiguration) GetOptimalBatchSize(itemCount int) int {
	if itemCount <= mc.BatchSize {
		return itemCount
	}
	return mc.BatchSize
}
