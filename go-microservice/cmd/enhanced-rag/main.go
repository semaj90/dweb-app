package main

import (
	"log"
	"os"
)

func main() {
	log.Println("🚀 Starting Enhanced RAG Service with Context7 Integration")

	// Set environment variables if not set
	if os.Getenv("ENHANCED_RAG_PORT") == "" {
		os.Setenv("ENHANCED_RAG_PORT", "8095")
	}

	if os.Getenv("CONTEXT7_URL") == "" {
		os.Setenv("CONTEXT7_URL", "http://localhost:4100")
	}

	// Start the enhanced RAG service
	StartEnhancedRAGService()
}
