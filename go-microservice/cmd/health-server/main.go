package main

import (
	"encoding/json"
	"fmt"
	"legal-ai-production/internal/envutil"
	"log"
	"net/http"
	"runtime"
	"time"
)

type Health struct {
  Status string `json:"status"`
  Time string `json:"time"`
  GoVersion string `json:"goVersion"`
  Goroutines int `json:"goroutines"`
}

func main(){
  port := envutil.GetInt("HEALTH_SERVER_PORT", 8079)
  mux := http.NewServeMux()
  mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request){
    h := Health{Status:"ok", Time: time.Now().UTC().Format(time.RFC3339), GoVersion: runtime.Version(), Goroutines: runtime.NumGoroutine()}
    json.NewEncoder(w).Encode(h)
  })
  mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request){
    // Minimal placeholder metrics; real Prometheus integration can replace this.
    w.Header().Set("Content-Type", "text/plain; version=0.0.4")
    fmt.Fprintf(w, "# HELP health_server_uptime_seconds Uptime estimate in seconds\n")
    fmt.Fprintf(w, "# TYPE health_server_uptime_seconds counter\n")
    fmt.Fprintf(w, "health_server_goroutines %d\n", runtime.NumGoroutine())
  })
  srv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux, ReadHeaderTimeout:5*time.Second}
  log.Printf("health-server running on :%d (baseline)", port)
  if err := srv.ListenAndServe(); err != nil { log.Fatal(err) }
}
