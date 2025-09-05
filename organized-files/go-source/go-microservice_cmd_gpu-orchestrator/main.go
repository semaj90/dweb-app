package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"legal-ai-production/internal/envutil"
)

func main() {
	log.Println("GPU Orchestrator (Production) v1.0 starting...")
	
	// Load environment with centralized helper
	config := envutil.LoadConfig()
	
	port := config.GetString("GPU_ORCHESTRATOR_PORT", "8081")
	
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"healthy","service":"gpu-orchestrator","version":"1.0"}`)
	})
	
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"gpu_usage":0,"memory_usage":0,"service":"gpu-orchestrator"}`)
	})
	
	log.Printf("GPU Orchestrator listening on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Failed to start GPU orchestrator: %v", err)
		os.Exit(1)
	}
}