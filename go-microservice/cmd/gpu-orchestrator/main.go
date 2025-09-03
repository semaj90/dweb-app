package main

import (
	"fmt"
	"legal-ai-production/internal/envutil"
	"log"
	"net/http"
	"time"
)

func main(){
  port := envutil.GetInt("GPU_ORCH_HTTP_PORT", 8231)
  mux := http.NewServeMux()
  mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request){ w.Write([]byte("ok")) })
  srv := &http.Server{ Addr:  fmt.Sprintf(":%d", port), Handler: mux, ReadHeaderTimeout: 5 * time.Second }
  log.Printf("gpu-orchestrator running on :%d (baseline)", port)
  if err := srv.ListenAndServe(); err != nil { log.Fatal(err) }
}
