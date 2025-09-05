package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"legal-ai-production/internal/envutil"
)

func main() {
	log.Println("Multi-Protocol Gateway (Production) v1.0 starting...")
	
	// Load environment with centralized helper
	config := envutil.LoadConfig()
	
	port := config.GetString("GATEWAY_PORT", "8080")
	
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"healthy","service":"multi-protocol-gateway","version":"1.0"}`)
	})
	
	http.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"running","protocols":["http","grpc"],"port":"%s"}`, port)
	})
	
	log.Printf("Gateway listening on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatalf("Failed to start gateway: %v", err)
		os.Exit(1)
	}
}