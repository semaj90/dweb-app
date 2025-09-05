package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"legal-ai-production/internal/envutil"
)

func main() {
	log.Println("Health Server (Production) v1.0 starting...")
	
	// Load environment with centralized helper
	config := envutil.LoadConfig()
	
	port := config.GetString("HEALTH_SERVER_PORT", "8083")
	
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"healthy","service":"health-server","version":"1.0","uptime":"running"}`)
	})
	
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "# HELP health_status Health check status\n# TYPE health_status gauge\nhealth_status 1\n")
	})
	
	log.Printf("Health Server listening on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Failed to start health server: %v", err)
		os.Exit(1)
	}
}