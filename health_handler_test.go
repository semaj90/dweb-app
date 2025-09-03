package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func healthHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusOK)
    _ = json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func TestHealthHandler(t *testing.T) {
    req := httptest.NewRequest(http.MethodGet, "/health", nil)
    rr := httptest.NewRecorder()
    healthHandler(rr, req)
    if rr.Code != http.StatusOK { t.Fatalf("expected 200 got %d", rr.Code) }
    var body map[string]any
    if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil { t.Fatalf("invalid json: %v", err) }
    if body["status"] != "healthy" { t.Fatalf("unexpected status %v", body["status"]) }
}
