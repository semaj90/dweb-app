// Minimal Mock Health and Metrics Server (std lib only)
// Exposes:
//   GET /health           -> JSON health status
//   GET /metrics          -> JSON metrics snapshot
//   GET /metrics/stream   -> SSE stream with periodic metrics
// Env:
//   PORT (default 8099)

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

var start = time.Now()

type healthResp struct {
	Status    string         `json:"status"`
	Timestamp int64          `json:"timestamp"`
	Uptime    string         `json:"uptime"`
	Metrics   map[string]any `json:"metrics"`
}

func getMetrics() map[string]any {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return map[string]any{
		"goroutines":  runtime.NumGoroutine(),
		"alloc_bytes": m.Alloc,
		"heap_alloc":  m.HeapAlloc,
		"sys_bytes":   m.Sys,
		"gc_num":      m.NumGC,
	}
}

func withCORS(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h.ServeHTTP(w, r)
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	resp := healthResp{
		Status:    "healthy",
		Timestamp: time.Now().Unix(),
		Uptime:    time.Since(start).String(),
		Metrics:   getMetrics(),
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func metricsHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(getMetrics())
}

// SSE stream without external deps
func metricsStreamHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
		return
	}

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	notify := w.(http.CloseNotifier).CloseNotify()

	for {
		select {
		case <-ticker.C:
			data, _ := json.Marshal(getMetrics())
			fmt.Fprintf(w, "event: metrics\n")
			fmt.Fprintf(w, "data: %s\n\n", strings.TrimSpace(string(data)))
			flusher.Flush()
		case <-notify:
			return
		}
	}
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/metrics", metricsHandler)
	mux.HandleFunc("/metrics/stream", metricsStreamHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8099"
	}
	if _, err := strconv.Atoi(port); err != nil {
		log.Printf("Invalid PORT %q, defaulting to 8099", port)
		port = "8099"
	}

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           withCORS(mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("Mock Health Server listening on http://localhost:%s", port)
	log.Printf("GET /health          -> JSON health")
	log.Printf("GET /metrics         -> JSON metrics snapshot")
	log.Printf("GET /metrics/stream  -> SSE metrics stream")
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}
