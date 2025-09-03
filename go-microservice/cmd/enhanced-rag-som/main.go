package main

import (
	"encoding/json"
	"fmt"
	"legal-ai-production/internal/envutil"
	"log"
	"net/http"
	"time"
)

type SOMStatus struct { Ready bool `json:"ready"`; Models int `json:"models"`; Updated string `json:"updated"` }

func main(){
  port := envutil.GetInt("ENH_RAG_SOM_PORT", 8232)
  mux := http.NewServeMux()
  mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request){ json.NewEncoder(w).Encode(map[string]any{"status":"ok"}) })
  mux.HandleFunc("/som/status", func(w http.ResponseWriter, r *http.Request){ json.NewEncoder(w).Encode(SOMStatus{Ready:true, Models:1, Updated:time.Now().UTC().Format(time.RFC3339)}) })
  srv := &http.Server{Addr:  fmt.Sprintf(":%d", port), Handler: mux, ReadHeaderTimeout:5*time.Second}
  log.Printf("enhanced-rag-som running on :%d (baseline)", port)
  if err:=srv.ListenAndServe(); err!=nil { log.Fatal(err) }
}
