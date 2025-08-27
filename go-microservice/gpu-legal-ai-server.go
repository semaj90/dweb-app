//go:build legacy
// +build legacy

package main

import "fmt"

// GPU Analysis: GPU legal AI server - legacy implementation with Flash Attention
func main() {
	// Use port from environment to match deployment; default to catalog port 8202
	port := os.Getenv("PORT")
	if port == "" {
		port = "8202"
	}

	fmt.Printf("GPU legal AI server (legacy) starting on :%s...\n", port)

	mux := http.NewServeMux()

	// Health endpoint used by the GO_BINARIES_CATALOG health checks
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok","service":"gpu-legal-ai","legacy":true}`))
	})

	// Simple GPU legal analysis endpoint (placeholder / noop)
	type analyzeReq struct {
		JobID string                 `json:"jobId,omitempty"`
		Meta  map[string]interface{} `json:"metadata,omitempty"`
	}
	type analyzeResp struct {
		JobID     string `json:"jobId,omitempty"`
		Status    string `json:"status"`
		Timestamp int64  `json:"timestamp"`
		Message   string `json:"message,omitempty"`
	}

	mux.HandleFunc("/api/v1/gpu-legal/analyze", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req analyzeReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json", http.StatusBadRequest)
			return
		}

		resp := analyzeResp{
			JobID:     req.JobID,
			Status:    "accepted",
			Timestamp: time.Now().Unix(),
			Message:   "legacy GPU legal AI received the job (no GPU worker attached)",
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	srv := &http.Server{
		Addr:    ":" + port,
		Handler: mux,
	}

	// Start server
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	// Graceful shutdown on interrupt/terminate
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	fmt.Println("shutting down GPU legal AI server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("shutdown error: %v", err)
	}

	fmt.Println("server stopped")
}

