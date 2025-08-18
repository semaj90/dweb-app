package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	_ "github.com/lib/pq"
)

/*
#cgo CFLAGS: -IC:/Progra~1/NVIDIA~2/CUDA/v12.8/include -mavx2
#cgo LDFLAGS: -LC:/Progra~1/NVIDIA~2/CUDA/v12.8/lib/x64 -lcudart_static -lcublas

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <immintrin.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

// GPU-accelerated vector operations
void cuda_compute_embeddings(float* input, float* output, int batch_size, int dimensions);
void cuda_similarity_search(float* query_vector, float* database_vectors, float* similarities, int num_vectors, int dimensions);
int cuda_parallel_sort(float* scores, int* indices, int count);

// SIMD-accelerated text processing
int simd_tokenize(const char* text, char** tokens, int max_tokens);
void simd_normalize_text(const char* input, char* output, int length);

// Stub implementations for compilation
void cuda_compute_embeddings(float* input, float* output, int batch_size, int dimensions) {
    // GPU embedding computation
    for (int i = 0; i < batch_size * dimensions; i++) {
        output[i] = input[i] * 0.8f + 0.1f; // Placeholder transformation
    }
}

void cuda_similarity_search(float* query_vector, float* database_vectors, float* similarities, int num_vectors, int dimensions) {
    // GPU cosine similarity computation
    for (int i = 0; i < num_vectors; i++) {
        float dot_product = 0.0f;
        float norm_a = 0.0f, norm_b = 0.0f;
        
        for (int j = 0; j < dimensions; j++) {
            float a = query_vector[j];
            float b = database_vectors[i * dimensions + j];
            dot_product += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        
        similarities[i] = dot_product / (sqrt(norm_a) * sqrt(norm_b));
    }
}

int cuda_parallel_sort(float* scores, int* indices, int count) {
    // GPU parallel sorting (placeholder)
    return count;
}

int simd_tokenize(const char* text, char** tokens, int max_tokens) {
    // SIMD tokenization (simplified)
    return 0;
}

void simd_normalize_text(const char* input, char* output, int length) {
    // SIMD text normalization
    for (int i = 0; i < length; i++) {
        output[i] = tolower(input[i]);
    }
}
*/
import "C"

// Document represents an indexed document
type Document struct {
	ID          string                 `json:"id"`
	Content     string                 `json:"content"`
	Metadata    map[string]interface{} `json:"metadata"`
	Embeddings  []float32              `json:"embeddings"`
	TokenCount  int                    `json:"token_count"`
	Hash        string                 `json:"hash"`
	IndexedAt   time.Time              `json:"indexed_at"`
	SearchRank  float32                `json:"search_rank,omitempty"`
}

// SearchQuery represents a search request
type SearchQuery struct {
	Text           string                 `json:"text"`
	Filters        map[string]interface{} `json:"filters"`
	SortBy         string                 `json:"sort_by"`
	SortOrder      string                 `json:"sort_order"`
	Limit          int                    `json:"limit"`
	Offset         int                    `json:"offset"`
	MinSimilarity  float32                `json:"min_similarity"`
	IncludeContent bool                   `json:"include_content"`
}

// SearchResult represents search results
type SearchResult struct {
	Documents    []Document `json:"documents"`
	TotalCount   int        `json:"total_count"`
	SearchTime   string     `json:"search_time"`
	GPUUsed      bool       `json:"gpu_used"`
	ProcessedBy  string     `json:"processed_by"`
}

// GPUIndexer handles GPU-accelerated indexing and search
type GPUIndexer struct {
	db               *sql.DB
	embeddingDim     int
	maxBatchSize     int
	gpuEnabled       bool
	workerPool       *WorkerPool
	documentCache    sync.Map
	embeddingCache   sync.Map
	indexMutex       sync.RWMutex
}

// WorkerPool manages background indexing workers
type WorkerPool struct {
	workers      int
	taskQueue    chan IndexTask
	resultQueue  chan IndexResult
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
}

// IndexTask represents an indexing task
type IndexTask struct {
	Document Document
	Type     string // "index", "update", "delete"
}

// IndexResult represents an indexing result
type IndexResult struct {
	DocumentID string
	Success    bool
	Error      error
	Duration   time.Duration
}

func NewGPUIndexer(dbURL string) (*GPUIndexer, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %v", err)
	}

	// Test database connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %v", err)
	}

	// Initialize tables
	if err := initializeTables(db); err != nil {
		return nil, fmt.Errorf("failed to initialize tables: %v", err)
	}

	gpuEnabled := os.Getenv("CUDA_ENABLED") == "true"
	embeddingDim := 384 // nomic-embed-text dimensions
	if dim := os.Getenv("EMBEDDING_DIMENSIONS"); dim != "" {
		if d, err := strconv.Atoi(dim); err == nil {
			embeddingDim = d
		}
	}

	indexer := &GPUIndexer{
		db:           db,
		embeddingDim: embeddingDim,
		maxBatchSize: 32,
		gpuEnabled:   gpuEnabled,
	}

	// Initialize worker pool
	numWorkers := runtime.NumCPU()
	if gpuEnabled {
		numWorkers = numWorkers * 2 // More workers when GPU is available
	}
	
	indexer.workerPool = NewWorkerPool(numWorkers)
	indexer.workerPool.Start(indexer)

	log.Printf("🚀 GPU Indexer initialized (GPU: %v, Workers: %d, Dim: %d)", 
		gpuEnabled, numWorkers, embeddingDim)

	return indexer, nil
}

func NewWorkerPool(workers int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	return &WorkerPool{
		workers:     workers,
		taskQueue:   make(chan IndexTask, workers*10),
		resultQueue: make(chan IndexResult, workers*10),
		ctx:         ctx,
		cancel:      cancel,
	}
}

func (wp *WorkerPool) Start(indexer *GPUIndexer) {
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker(i, indexer)
	}

	// Result processor
	go wp.processResults()
}

func (wp *WorkerPool) worker(id int, indexer *GPUIndexer) {
	defer wp.wg.Done()
	
	log.Printf("🔧 Worker %d started", id)
	
	for {
		select {
		case task := <-wp.taskQueue:
			start := time.Now()
			
			var err error
			switch task.Type {
			case "index":
				err = indexer.indexDocument(task.Document)
			case "update":
				err = indexer.updateDocument(task.Document)
			case "delete":
				err = indexer.deleteDocument(task.Document.ID)
			}

			wp.resultQueue <- IndexResult{
				DocumentID: task.Document.ID,
				Success:    err == nil,
				Error:      err,
				Duration:   time.Since(start),
			}

		case <-wp.ctx.Done():
			log.Printf("🛑 Worker %d stopping", id)
			return
		}
	}
}

func (wp *WorkerPool) processResults() {
	for {
		select {
		case result := <-wp.resultQueue:
			if result.Error != nil {
				log.Printf("❌ Index task failed for %s: %v", result.DocumentID, result.Error)
			} else {
				log.Printf("✅ Indexed %s in %v", result.DocumentID, result.Duration)
			}

		case <-wp.ctx.Done():
			return
		}
	}
}

func (gi *GPUIndexer) IndexDocument(doc Document) error {
	// Add to worker queue
	gi.workerPool.taskQueue <- IndexTask{
		Document: doc,
		Type:     "index",
	}
	return nil
}

func (gi *GPUIndexer) indexDocument(doc Document) error {
	gi.indexMutex.Lock()
	defer gi.indexMutex.Unlock()

	// Generate embeddings
	embeddings, err := gi.generateEmbeddings(doc.Content)
	if err != nil {
		return fmt.Errorf("failed to generate embeddings: %v", err)
	}

	doc.Embeddings = embeddings
	doc.Hash = gi.generateHash(doc.Content)
	doc.IndexedAt = time.Now()

	// Store in database
	return gi.storeDocument(doc)
}

func (gi *GPUIndexer) generateEmbeddings(text string) ([]float32, error) {
	// Check cache first
	hash := gi.generateHash(text)
	if cached, ok := gi.embeddingCache.Load(hash); ok {
		return cached.([]float32), nil
	}

	// Normalize text using SIMD
	normalizedText := gi.normalizeText(text)
	
	// Tokenize using SIMD
	tokens := gi.tokenizeText(normalizedText)
	
	// Convert to input features (simplified)
	input := make([]float32, gi.embeddingDim)
	for i, token := range tokens {
		if i >= gi.embeddingDim {
			break
		}
		input[i] = float32(len(token)) / 10.0 // Simple feature extraction
	}

	// Generate embeddings
	embeddings := make([]float32, gi.embeddingDim)
	
	if gi.gpuEnabled {
		// Use GPU acceleration
		gi.computeEmbeddingsGPU(input, embeddings)
	} else {
		// CPU fallback
		gi.computeEmbeddingsCPU(input, embeddings)
	}

	// Cache the result
	gi.embeddingCache.Store(hash, embeddings)
	
	return embeddings, nil
}

func (gi *GPUIndexer) computeEmbeddingsGPU(input, output []float32) {
	// Call CUDA function
	C.cuda_compute_embeddings(
		(*C.float)(&input[0]),
		(*C.float)(&output[0]),
		1, // batch_size
		C.int(gi.embeddingDim),
	)
}

func (gi *GPUIndexer) computeEmbeddingsCPU(input, output []float32) {
	// CPU embedding computation
	for i := 0; i < gi.embeddingDim; i++ {
		output[i] = input[i]*0.8 + 0.1
		if i > 0 {
			output[i] += output[i-1] * 0.1 // Simple recurrence
		}
	}
}

func (gi *GPUIndexer) normalizeText(text string) string {
	// Use SIMD for text normalization
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	
	cNormalized := (*C.char)(C.malloc(C.size_t(len(text) + 1)))
	defer C.free(unsafe.Pointer(cNormalized))
	
	C.simd_normalize_text(cText, cNormalized, C.int(len(text)))
	
	return C.GoString(cNormalized)
}

func (gi *GPUIndexer) tokenizeText(text string) []string {
	// Simple tokenization for demo
	return strings.Fields(strings.ToLower(text))
}

func (gi *GPUIndexer) Search(query SearchQuery) (*SearchResult, error) {
	start := time.Now()

	// Generate query embeddings
	queryEmbeddings, err := gi.generateEmbeddings(query.Text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embeddings: %v", err)
	}

	// Search in database
	documents, err := gi.searchDatabase(queryEmbeddings, query)
	if err != nil {
		return nil, fmt.Errorf("database search failed: %v", err)
	}

	// Sort results
	gi.sortResults(documents, query.SortBy, query.SortOrder)

	// Apply pagination
	totalCount := len(documents)
	if query.Offset > len(documents) {
		documents = []Document{}
	} else {
		end := query.Offset + query.Limit
		if end > len(documents) {
			end = len(documents)
		}
		if query.Offset < len(documents) {
			documents = documents[query.Offset:end]
		}
	}

	return &SearchResult{
		Documents:   documents,
		TotalCount:  totalCount,
		SearchTime:  time.Since(start).String(),
		GPUUsed:     gi.gpuEnabled,
		ProcessedBy: fmt.Sprintf("worker-pool-%d", gi.workerPool.workers),
	}, nil
}

func (gi *GPUIndexer) searchDatabase(queryEmbeddings []float32, query SearchQuery) ([]Document, error) {
	// Build SQL query with metadata filters
	sqlQuery := `
		SELECT id, content, metadata, embeddings, token_count, hash, indexed_at
		FROM documents 
		WHERE 1=1
	`
	
	args := []interface{}{}
	argIndex := 1

	// Add metadata filters
	for key, value := range query.Filters {
		sqlQuery += fmt.Sprintf(" AND metadata ->> $%d = $%d", argIndex, argIndex+1)
		args = append(args, key, fmt.Sprintf("%v", value))
		argIndex += 2
	}

	rows, err := gi.db.Query(sqlQuery, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var documents []Document
	var allEmbeddings [][]float32

	for rows.Next() {
		var doc Document
		var metadataJSON string
		var embeddingsJSON string

		err := rows.Scan(
			&doc.ID, &doc.Content, &metadataJSON, &embeddingsJSON,
			&doc.TokenCount, &doc.Hash, &doc.IndexedAt,
		)
		if err != nil {
			continue
		}

		// Parse metadata
		if err := json.Unmarshal([]byte(metadataJSON), &doc.Metadata); err != nil {
			doc.Metadata = make(map[string]interface{})
		}

		// Parse embeddings
		if err := json.Unmarshal([]byte(embeddingsJSON), &doc.Embeddings); err != nil {
			continue
		}

		documents = append(documents, doc)
		allEmbeddings = append(allEmbeddings, doc.Embeddings)
	}

	// Compute similarities
	if len(documents) > 0 {
		similarities := gi.computeSimilarities(queryEmbeddings, allEmbeddings)
		
		// Filter by minimum similarity and assign search ranks
		filteredDocs := make([]Document, 0)
		for i, doc := range documents {
			if similarities[i] >= query.MinSimilarity {
				doc.SearchRank = similarities[i]
				if !query.IncludeContent {
					doc.Content = "" // Remove content if not needed
				}
				filteredDocs = append(filteredDocs, doc)
			}
		}
		documents = filteredDocs
	}

	return documents, nil
}

func (gi *GPUIndexer) computeSimilarities(query []float32, docEmbeddings [][]float32) []float32 {
	similarities := make([]float32, len(docEmbeddings))

	if gi.gpuEnabled && len(docEmbeddings) > 10 {
		// Use GPU for large batches
		gi.computeSimilaritiesGPU(query, docEmbeddings, similarities)
	} else {
		// CPU computation
		gi.computeSimilaritiesCPU(query, docEmbeddings, similarities)
	}

	return similarities
}

func (gi *GPUIndexer) computeSimilaritiesGPU(query []float32, docEmbeddings [][]float32, similarities []float32) {
	// Flatten document embeddings
	flattened := make([]float32, len(docEmbeddings)*gi.embeddingDim)
	for i, doc := range docEmbeddings {
		copy(flattened[i*gi.embeddingDim:], doc)
	}

	// Call CUDA similarity computation
	C.cuda_similarity_search(
		(*C.float)(&query[0]),
		(*C.float)(&flattened[0]),
		(*C.float)(&similarities[0]),
		C.int(len(docEmbeddings)),
		C.int(gi.embeddingDim),
	)
}

func (gi *GPUIndexer) computeSimilaritiesCPU(query []float32, docEmbeddings [][]float32, similarities []float32) {
	for i, docEmb := range docEmbeddings {
		similarities[i] = gi.cosineSimilarity(query, docEmb)
	}
}

func (gi *GPUIndexer) cosineSimilarity(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	
	for i := 0; i < len(a) && i < len(b); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func (gi *GPUIndexer) sortResults(documents []Document, sortBy, sortOrder string) {
	sort.Slice(documents, func(i, j int) bool {
		var less bool
		
		switch sortBy {
		case "similarity", "search_rank":
			less = documents[i].SearchRank > documents[j].SearchRank // Higher similarity first
		case "indexed_at", "timestamp":
			less = documents[i].IndexedAt.After(documents[j].IndexedAt) // Newer first
		case "content_length":
			less = len(documents[i].Content) > len(documents[j].Content)
		case "token_count":
			less = documents[i].TokenCount > documents[j].TokenCount
		default:
			less = documents[i].SearchRank > documents[j].SearchRank
		}
		
		if sortOrder == "asc" {
			less = !less
		}
		
		return less
	})
}

func (gi *GPUIndexer) generateHash(content string) string {
	hash := sha256.Sum256([]byte(content))
	return hex.EncodeToString(hash[:])
}

func (gi *GPUIndexer) storeDocument(doc Document) error {
	metadataJSON, _ := json.Marshal(doc.Metadata)
	embeddingsJSON, _ := json.Marshal(doc.Embeddings)

	_, err := gi.db.Exec(`
		INSERT INTO documents (id, content, metadata, embeddings, token_count, hash, indexed_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (id) DO UPDATE SET
			content = EXCLUDED.content,
			metadata = EXCLUDED.metadata,
			embeddings = EXCLUDED.embeddings,
			token_count = EXCLUDED.token_count,
			hash = EXCLUDED.hash,
			indexed_at = EXCLUDED.indexed_at
	`, doc.ID, doc.Content, metadataJSON, embeddingsJSON, doc.TokenCount, doc.Hash, doc.IndexedAt)

	return err
}

func (gi *GPUIndexer) updateDocument(doc Document) error {
	return gi.indexDocument(doc)
}

func (gi *GPUIndexer) deleteDocument(id string) error {
	_, err := gi.db.Exec("DELETE FROM documents WHERE id = $1", id)
	return err
}

// HTTP Handlers
func (gi *GPUIndexer) ServeHTTP() {
	http.HandleFunc("/index", gi.handleIndex)
	http.HandleFunc("/search", gi.handleSearch)
	http.HandleFunc("/status", gi.handleStatus)
	http.HandleFunc("/health", gi.handleHealth)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8097"
	}

	log.Printf("🌐 GPU Indexer API starting on port %s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func (gi *GPUIndexer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var doc Document
	if err := json.NewDecoder(r.Body).Decode(&doc); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if doc.ID == "" {
		doc.ID = gi.generateHash(doc.Content)
	}

	if err := gi.IndexDocument(doc); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status":      "queued",
		"document_id": doc.ID,
	})
}

func (gi *GPUIndexer) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var query SearchQuery
	if err := json.NewDecoder(r.Body).Decode(&query); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Set defaults
	if query.Limit <= 0 {
		query.Limit = 10
	}
	if query.MinSimilarity == 0 {
		query.MinSimilarity = 0.3
	}
	if query.SortBy == "" {
		query.SortBy = "similarity"
	}

	result, err := gi.Search(query)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (gi *GPUIndexer) handleStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"service":        "gpu-indexer",
		"gpu_enabled":    gi.gpuEnabled,
		"embedding_dim":  gi.embeddingDim,
		"workers":        gi.workerPool.workers,
		"cache_size":     gi.getCacheSize(),
		"uptime":         time.Since(time.Now()).String(),
	})
}

func (gi *GPUIndexer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func (gi *GPUIndexer) getCacheSize() int {
	count := 0
	gi.embeddingCache.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

func initializeTables(db *sql.DB) error {
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS documents (
		id VARCHAR(255) PRIMARY KEY,
		content TEXT NOT NULL,
		metadata JSONB,
		embeddings JSONB,
		token_count INTEGER,
		hash VARCHAR(64),
		indexed_at TIMESTAMP,
		UNIQUE(hash)
	);
	
	CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);
	CREATE INDEX IF NOT EXISTS idx_documents_indexed_at ON documents(indexed_at);
	CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
	`

	_, err := db.Exec(createTableSQL)
	return err
}

func main() {
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		databaseURL = "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"
	}

	indexer, err := NewGPUIndexer(databaseURL)
	if err != nil {
		log.Fatal("Failed to initialize GPU indexer:", err)
	}

	indexer.ServeHTTP()
}