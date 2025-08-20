//go:build legacy
// +build legacy

package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
)

// Health response structure
type HealthResponse struct {
    Status    string    `json:"status"`
    Service   string    `json:"service"`
    Version   string    `json:"version"`
    Timestamp time.Time `json:"timestamp"`
    Endpoints []string  `json:"endpoints"`
}

// Chat request/response structures
type ChatRequest struct {
    Message string `json:"message"`
    Model   string `json:"model,omitempty"`
}

type ChatResponse struct {
    Response  string    `json:"response"`
    Model     string    `json:"model"`
    Timestamp time.Time `json:"timestamp"`
}

func main() {
    // Setup routes
    http.HandleFunc("/api/health", healthHandler)
    http.HandleFunc("/health", healthHandler)
    http.HandleFunc("/api/chat", chatHandler)
    http.HandleFunc("/api/ai/summarize", summarizeHandler)
    http.HandleFunc("/api/metrics", metricsHandler)
    
    port := "8084"
    fmt.Printf(`
╔════════════════════════════════════════╗
║   Enhanced RAG V2 - Go Service        ║
║   GPU-Accelerated Legal AI            ║
╚════════════════════════════════════════╝

[✓] Starting server on port %s
[✓] Health check: http://localhost:%s/api/health
[✓] Chat API: http://localhost:%s/api/chat
[✓] Metrics: http://localhost:%s/api/metrics

Press Ctrl+C to stop the server
`, port, port, port, port)
    
    log.Fatal(http.ListenAndServe(":"+port, nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
    enableCORS(w)
    if r.Method == "OPTIONS" {
        return
    }
    
    response := HealthResponse{
        Status:    "healthy",
        Service:   "Enhanced RAG V2",
        Version:   "2.0.0",
        Timestamp: time.Now(),
        Endpoints: []string{
            "/api/health",
            "/api/chat",
            "/api/ai/summarize",
            "/api/metrics",
        },
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func chatHandler(w http.ResponseWriter, r *http.Request) {
    enableCORS(w)
    if r.Method == "OPTIONS" {
        return
    }
    
    if r.Method != "POST" {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var req ChatRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }
    
    response := ChatResponse{
        Response:  fmt.Sprintf("Enhanced RAG V2 response to: %s", req.Message),
        Model:     "enhanced-rag-v2",
        Timestamp: time.Now(),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func summarizeHandler(w http.ResponseWriter, r *http.Request) {
    enableCORS(w)
    if r.Method == "OPTIONS" {
        return
    }
    
    if r.Method != "POST" {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    
    var req map[string]interface{}
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request body", http.StatusBadRequest)
        return
    }
    
    text, _ := req["text"].(string)
    if len(text) > 100 {
        text = text[:100] + "..."
    }
    
    response := map[string]interface{}{
        "summary":   fmt.Sprintf("Summary: %s", text),
        "wordCount": len(text),
        "timestamp": time.Now(),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func metricsHandler(w http.ResponseWriter, r *http.Request) {
    enableCORS(w)
    if r.Method == "OPTIONS" {
        return
    }
    
    metrics := map[string]interface{}{
        "uptime_seconds": time.Since(startTime).Seconds(),
        "requests_total": requestCount,
        "service":        "Enhanced RAG V2",
        "timestamp":      time.Now(),
    }
    
    requestCount++
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(metrics)
}

func enableCORS(w http.ResponseWriter) {
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

var (
    startTime    = time.Now()
    requestCount = 0
)