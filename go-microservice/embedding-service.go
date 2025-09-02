package main

// Embedding Service Skeleton
// Goal: Batch encode passages (text) into vectors and store into Postgres `passages.embedding`
// Features:
//  - Adaptive batch sizing (env EMB_BATCH_MAX, default 64)
//  - Dimension guard vs embedding_metadata.latest
//  - Redis caching (optional) for idempotency (skip if hash already has embedding)
//  - Prometheus metrics placeholders
//  - Queue ingestion stub (future: NATS / RabbitMQ)
//  - Graceful shutdown context

import (
	context "context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "github.com/lib/pq"
)

// Passage represents minimal fetch from DB
type Passage struct {
	ID        string
	Text      string
	Hash      string
	Embedding []float32 // nil until encoded
}

type EmbedModelSpec struct {
	ModelName    string
	ModelVersion string
	Dim          int
}

var (
	defaultDim = 768
)

func loadActiveModelSpec(db *sql.DB) (*EmbedModelSpec, error) {
	row := db.QueryRow(`SELECT model_name, model_version, dim FROM embedding_metadata ORDER BY created_at DESC LIMIT 1`)
	var s EmbedModelSpec
	if err := row.Scan(&s.ModelName, &s.ModelVersion, &s.Dim); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			// fallback to default spec if not yet inserted
			return &EmbedModelSpec{ModelName: "nomic-embed-text", ModelVersion: "bootstrap", Dim: defaultDim}, nil
		}
		return nil, err
	}
	return &s, nil
}

// fetchPendingPassages fetches passages without embeddings up to limit
func fetchPendingPassages(db *sql.DB, limit int) ([]Passage, error) {
	rows, err := db.Query(`SELECT id, text, hash FROM passages WHERE embedding IS NULL ORDER BY created_at ASC LIMIT $1`, limit)
	if err != nil { return nil, err }
	defer rows.Close()
	var list []Passage
	for rows.Next() {
		var p Passage
		if err := rows.Scan(&p.ID, &p.Text, &p.Hash); err != nil { return nil, err }
		list = append(list, p)
	}
	return list, rows.Err()
}

// mockEmbed simulates embedding generation (replace with real model binding)
func mockEmbed(texts []string, dim int) [][]float32 {
	out := make([][]float32, len(texts))
	for i := range texts {
		vec := make([]float32, dim)
		// simple deterministic hash-based filler (NOT semantic)
		s := 0
		for _, ch := range texts[i] {
			s += int(ch)
		}
		base := float32(s % 113)
		for j := 0; j < dim; j++ { vec[j] = (base + float32(j%7)) / 127.0 }
		out[i] = vec
	}
	return out
}

func storeEmbeddings(db *sql.DB, passages []Passage, vectors [][]float32) error {
	if len(passages) != len(vectors) { return fmt.Errorf("length mismatch passages vs vectors") }
	// Use transaction & parameterized updates
	tx, err := db.Begin()
	if err != nil { return err }
	stmt, err := tx.Prepare(`UPDATE passages SET embedding = $1, updated_at = NOW() WHERE id = $2`)
	if err != nil { tx.Rollback(); return err }
	defer stmt.Close()
	for i, p := range passages {
		// Serialize slice to pgvector input: e.g. '[1,2,3]'
		vals := make([]string, len(vectors[i]))
		for j, f := range vectors[i] { vals[j] = strconv.FormatFloat(float64(f), 'f', -1, 32) }
		pgLiteral := fmt.Sprintf("[%s]", strings.Join(vals, ","))
		if _, err := stmt.Exec(pgLiteral, p.ID); err != nil { tx.Rollback(); return err }
	}
	return tx.Commit()
}

// adaptiveBatchController naive implementation (placeholder)
func adaptiveBatchController(current int, util float64) int {
	if util < 0.70 && current < 256 { return current * 2 }
	if util > 0.85 && current > 8 { return current / 2 }
	return current
}

// Service holds state
type Service struct {
	DB      *sql.DB
	Model   *EmbedModelSpec
	BatchSz int
	Mu      sync.Mutex
}

func NewService(db *sql.DB) (*Service, error) {
	spec, err := loadActiveModelSpec(db)
	if err != nil { return nil, err }
	batch := 64
	if v := os.Getenv("EMB_BATCH_MAX"); v != "" { if n, _ := strconv.Atoi(v); n > 0 { batch = n } }
	return &Service{DB: db, Model: spec, BatchSz: batch}, nil
}

func (s *Service) processOnce(ctx context.Context) (int, error) {
	passages, err := fetchPendingPassages(s.DB, s.BatchSz)
	if err != nil { return 0, err }
	if len(passages) == 0 { return 0, nil }
	texts := make([]string, len(passages))
	for i, p := range passages { texts[i] = p.Text }
	vectors := mockEmbed(texts, s.Model.Dim)
	if err := storeEmbeddings(s.DB, passages, vectors); err != nil { return 0, err }
	return len(passages), nil
}

func (s *Service) runLoop(ctx context.Context) {
	idle := time.Second * 2
	for {
		select {
		case <-ctx.Done():
			log.Println("embedding service: shutdown")
			return
		default:
		}
		count, err := s.processOnce(ctx)
		if err != nil {
			log.Printf("embedding service error: %v", err)
			time.Sleep(time.Second * 5)
			continue
		}
		if count == 0 { time.Sleep(idle) }
	}
}

// HTTP endpoints: health, stats, enqueue stub
func (s *Service) registerHTTP(mux *http.ServeMux) {
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte("ok"))
	})
	mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{"model": s.Model, "batch_max": s.BatchSz}
		b, _ := json.Marshal(resp)
		w.Header().Set("Content-Type", "application/json")
		w.Write(b)
	})
	mux.HandleFunc("/enqueue", func(w http.ResponseWriter, r *http.Request) {
		// placeholder: accept JSON with passage_ids; for now just 202
		w.WriteHeader(http.StatusAccepted)
	})
}

func main() {
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" { dsn = "postgres://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable" }
	db, err := sql.Open("postgres", dsn)
	if err != nil { log.Fatalf("db open: %v", err) }
	if err := db.Ping(); err != nil { log.Fatalf("db ping: %v", err) }

	svc, err := NewService(db)
	if err != nil { log.Fatalf("init service: %v", err) }

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go svc.runLoop(ctx)

	mux := http.NewServeMux()
	svc.registerHTTP(mux)
	addr := ":8098"
	if v := os.Getenv("EMBED_SERVICE_PORT"); v != "" { addr = ":" + v }
	log.Printf("embedding service listening on %s (model=%s/%s dim=%d)", addr, svc.Model.ModelName, svc.Model.ModelVersion, svc.Model.Dim)
	if err := http.ListenAndServe(addr, mux); err != nil { log.Fatalf("http server: %v", err) }
}
