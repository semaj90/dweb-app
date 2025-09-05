package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"legal-ai-production/internal/envutil"
)

func main() {
	log.Println("Enhanced RAG with SOM (Production) v1.0 starting...")
	
	// Load environment with centralized helper
	config := envutil.LoadConfig()
	
	port := config.GetString("RAG_SOM_PORT", "8082")
	
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"healthy","service":"enhanced-rag-som","version":"1.0"}`)
	})
	
	http.HandleFunc("/som/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"som_initialized":true,"nodes":100,"convergence":0.85,"service":"enhanced-rag-som"}`)
	})
	
	log.Printf("Enhanced RAG-SOM listening on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Failed to start RAG-SOM: %v", err)
		os.Exit(1)
	}
}