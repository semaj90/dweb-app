// model-config.go
// Add this file to your go-microservice directory

package main

import "os"

// ModelConfig defines the AI models to use
var ModelConfig = struct {
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
}{
	LegalModel:     getEnv("LEGAL_MODEL", "gemma3-legal"),
	EmbeddingModel: getEnv("EMBEDDING_MODEL", "nomic-embed-text"),
	VisionModel:    getEnv("VISION_MODEL", "llava:7b"),
	Temperature:    0.3, // Lower temperature for more consistent legal analysis
	MaxTokens:      4096,
	NumCtx:         4096,
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
