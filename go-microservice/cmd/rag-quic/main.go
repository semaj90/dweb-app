package main

// Minimal embedding + RAG HTTP service with pluggable Ollama and Postgres pgvector.
// You can front this with QUIC/HTTP3 using quic-go (see repo's quic-server.go for patterns).

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	fastjson "legal-ai-production/internal/fastjson"
)

type EmbedRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model,omitempty"`
}

type EmbedResponse struct {
	Model   string      `json:"model"`
	Vectors [][]float32 `json:"vectors"`
}

type RAGRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"topK"`
}

type RAGDoc struct {
	ID    string  `json:"id"`
	Text  string  `json:"text"`
	Score float32 `json:"score"`
}

type RAGResponse struct {
	Results []RAGDoc `json:"results"`
}

// Globals wired from env
var (
	ollama     *OllamaClient
	embedModel string
	pg         *PGSearch
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	// Allow local dev cross-origin by default
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		writeJSON(w, 200, map[string]string{"ok": "true"})
		return
	}
	stream := r.URL.Query().Get("stream") == "1"
	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if len(req.Texts) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "texts required"})
		return
	}
	model := req.Model
	if model == "" {
		model = embedModel
	}

	// Fallback: simple zero-vectors if Ollama not configured
	if ollama == nil {
		out := make([][]float32, len(req.Texts))
		for i := range out {
			out[i] = make([]float32, 128)
		}
		writeJSON(w, 200, EmbedResponse{Model: model, Vectors: out})
		return
	}

	vecs, err := ollama.EmbedBatch(r.Context(), req.Texts, model)
	if err != nil {
		writeJSON(w, 502, map[string]string{"error": err.Error()})
		return
	}
	if !stream {
		// fast path aggregated
		if b, e := fastjson.Marshal(EmbedResponse{Model: model, Vectors: vecs}); e == nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(200)
			w.Write(b)
			return
		}
		writeJSON(w, 200, EmbedResponse{Model: model, Vectors: vecs})
		return
	}
	// streaming mode: one vector per line
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Transfer-Encoding", "chunked")
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, 500, map[string]string{"error": "stream unsupported"})
		return
	}
	meta := map[string]any{"model": model, "count": len(vecs), "stream": true, "ts": time.Now().Format(time.RFC3339)}
	if b, e := fastjson.Marshal(meta); e == nil { w.Write(b); w.Write([]byte("\n")) } else { mb, _ := json.Marshal(meta); w.Write(mb); w.Write([]byte("\n")) }
	flusher.Flush()
	for i, v := range vecs {
		line := map[string]any{"index": i, "vector": v}
		b, e := fastjson.Marshal(line)
		if e != nil { b, _ = json.Marshal(line) }
		w.Write(b)
		w.Write([]byte("\n"))
		flusher.Flush()
	}
	final := map[string]any{"complete": true, "total": len(vecs)}
	if b, e := fastjson.Marshal(final); e == nil { w.Write(b) } else { fb, _ := json.Marshal(final); w.Write(fb) }
	flusher.Flush()
}

func handleRAG(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		writeJSON(w, 200, map[string]string{"ok": "true"})
		return
	}
	var req RAGRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Query == "" {
		writeJSON(w, 400, map[string]string{"error": "query required"})
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}

	var vec []float32
	if ollama != nil {
		v, err := ollama.EmbedBatch(r.Context(), []string{req.Query}, embedModel)
		if err != nil || len(v) == 0 {
			writeJSON(w, 502, map[string]string{"error": fmt.Sprintf("embed failed: %v", err)})
			return
		}
		vec = v[0]
	} else {
		vec = make([]float32, 128) // fallback
	}

	if pg == nil {
		// No database configured; return stub
		writeJSON(w, 200, RAGResponse{Results: []RAGDoc{{ID: "demo", Text: "stub", Score: 0.9}}})
		return
	}

	results, err := pg.CosineSearch(r.Context(), vec, req.TopK)
	if err != nil {
		writeJSON(w, 502, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, 200, RAGResponse{Results: results})
}

func main() {
	// Env wiring
	ollamaBase := os.Getenv("OLLAMA_BASE_URL")
	if ollamaBase != "" {
		ollama = NewOllamaClient(ollamaBase)
	}
	if ollama == nil {
		// default to local ollama if reachable; no hard error if not
		ollama = NewOllamaClient("http://localhost:11434")
	}
	embedModel = env("EMBED_MODEL", "nomic-embed-text")

	// Optional Postgres
	if dsn := os.Getenv("PG_CONN_STRING"); dsn != "" {
		cfg := PGConfig{
			ConnString:   dsn,
			Table:        env("VECTOR_TABLE", "documents"),
			IDColumn:     env("ID_COLUMN", "id"),
			TextColumn:   env("TEXT_COLUMN", "text"),
			VectorColumn: env("VECTOR_COLUMN", "embedding"),
		}
		var err error
		pg, err = NewPGSearch(context.Background(), cfg)
		if err != nil {
			log.Printf("pg init failed: %v", err)
			pg = nil
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/embed", handleEmbed)
	mux.HandleFunc("/embed/stats", func(w http.ResponseWriter, r *http.Request) {
		st := fastjson.GetStats()
		w.Header().Set("X-JSON-Codec", st.CodecName)
		w.Header().Set("X-JSON-Encodes", strconv.FormatInt(st.Encodes, 10))
		w.Header().Set("X-JSON-Bytes", strconv.FormatInt(st.BytesProduced, 10))
		writeJSON(w, 200, st)
	})
	mux.HandleFunc("/rag", handleRAG)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		status := map[string]any{
			"time":   time.Now().Format(time.RFC3339),
			"ollama": ollama != nil,
			"pg":     pg != nil,
			"model":  embedModel,
		}
		writeJSON(w, 200, status)
	})

	addr := ":" + env("RAG_QUIC_PORT", "8092")
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("listen: %v", err)
	}
	fmt.Println("RAG service listening on", addr)
	fmt.Println("Env:")
	fmt.Println("  OLLAMA_BASE_URL=", ollamaBase)
	fmt.Println("  EMBED_MODEL=", embedModel)
	fmt.Println("  PG_CONN_STRING set:", os.Getenv("PG_CONN_STRING") != "")
	if pg != nil {
		fmt.Printf("  VECTOR_TABLE=%s, VECTOR_COLUMN=%s, TEXT_COLUMN=%s\n", pg.cfg.Table, pg.cfg.VectorColumn, pg.cfg.TextColumn)
	}

	if err := http.Serve(ln, mux); err != nil {
		log.Fatalf("serve error: %v", err)
	}
}

// --- Minimal Ollama client ---
type OllamaClient struct {
	baseURL string
	http    *http.Client
}

func NewOllamaClient(base string) *OllamaClient {
	return &OllamaClient{baseURL: base, http: &http.Client{Timeout: 60 * time.Second}}
}

func (o *OllamaClient) EmbedBatch(ctx context.Context, texts []string, model string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for _, t := range texts {
		vec, err := o.embedOne(ctx, t, model)
		if err != nil {
			return nil, err
		}
		out = append(out, vec)
	}
	return out, nil
}

func (o *OllamaClient) embedOne(ctx context.Context, text, model string) ([]float32, error) {
	req := map[string]any{"model": model, "input": text}
	body, _ := json.Marshal(req)
	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/embeddings", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := o.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("ollama embeddings status %d", resp.StatusCode)
	}
	var out struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Embedding, nil
}

// --- Minimal Postgres pgvector search stub ---
type PGConfig struct {
	ConnString   string
	Table        string
	IDColumn     string
	TextColumn   string
	VectorColumn string
}

type PGSearch struct{ cfg PGConfig }

func NewPGSearch(ctx context.Context, cfg PGConfig) (*PGSearch, error) {
	// In a future step, initialize pgxpool here and validate table/columns
	return &PGSearch{cfg: cfg}, nil
}

func (p *PGSearch) CosineSearch(ctx context.Context, vec []float32, k int) ([]RAGDoc, error) {
	// Placeholder: return empty results until wired to pgx + pgvector query
	return []RAGDoc{}, nil
}
