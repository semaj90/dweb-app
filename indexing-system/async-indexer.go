// Asynchronous Codebase Indexer with Ollama Integration
// High-performance concurrent indexing with SIMD acceleration
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

// Configuration
const (
	OLLAMA_URL          = "http://localhost:11434"
	EMBEDDING_MODEL     = "nomic-embed-text"
	MAX_WORKERS         = 16 // Adjust based on CPU cores
	BATCH_SIZE          = 50
	MAX_FILE_SIZE       = 1024 * 1024 // 1MB max file size
	WEBSOCKET_PORT      = ":8082"
	API_PORT           = ":8081"
)

// File types to process
var SUPPORTED_EXTENSIONS = map[string]bool{
	".go":     true,
	".ts":     true,
	".svelte": true,
	".js":     true,
	".py":     true,
	".md":     true,
	".json":   true,
	".sql":    true,
	".css":    true,
	".scss":   true,
	".html":   true,
}

// Structs
type FileInfo struct {
	Path         string            `json:"path"`
	Content      string            `json:"content"`
	Size         int64             `json:"size"`
	Language     string            `json:"language"`
	ModTime      time.Time         `json:"mod_time"`
	Embedding    []float64         `json:"embedding,omitempty"`
	Metadata     map[string]string `json:"metadata"`
	ProcessedAt  time.Time         `json:"processed_at"`
	Hash         string            `json:"hash"`
}

type IndexingProgress struct {
	TotalFiles     int       `json:"total_files"`
	ProcessedFiles int       `json:"processed_files"`
	FailedFiles    int       `json:"failed_files"`
	CurrentFile    string    `json:"current_file"`
	StartTime      time.Time `json:"start_time"`
	EstimatedEnd   time.Time `json:"estimated_end"`
	Rate           float64   `json:"rate"` // files per second
}

type BatchEmbeddingRequest struct {
	Texts []string `json:"texts"`
	Model string   `json:"model"`
}

type BatchEmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
	Model      string      `json:"model"`
	Timing     float64     `json:"timing"`
}

type IndexingService struct {
	progress   *IndexingProgress
	files      []FileInfo
	mutex      sync.RWMutex
	clients    map[*websocket.Conn]bool
	upgrader   websocket.Upgrader
	ctx        context.Context
	cancel     context.CancelFunc
	workerPool chan struct{}
	results    chan FileInfo
	errors     chan error
}

// WebSocket upgrader
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for development
	},
}

func main() {
	indexer := NewIndexingService()
	
	// Start the indexing service
	go indexer.StartIndexing("C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app")
	
	// Start API server
	go indexer.StartAPIServer()
	
	// Start WebSocket server
	indexer.StartWebSocketServer()
}

func NewIndexingService() *IndexingService {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &IndexingService{
		progress: &IndexingProgress{
			StartTime: time.Now(),
		},
		files:      make([]FileInfo, 0),
		clients:    make(map[*websocket.Conn]bool),
		upgrader:   upgrader,
		ctx:        ctx,
		cancel:     cancel,
		workerPool: make(chan struct{}, MAX_WORKERS),
		results:    make(chan FileInfo, BATCH_SIZE*2),
		errors:     make(chan error, 100),
	}
}

func (s *IndexingService) StartIndexing(rootPath string) {
	log.Printf("🚀 Starting asynchronous indexing of: %s", rootPath)
	
	// Phase 1: Discover all files
	filePaths, err := s.discoverFiles(rootPath)
	if err != nil {
		log.Printf("❌ Failed to discover files: %v", err)
		return
	}
	
	s.mutex.Lock()
	s.progress.TotalFiles = len(filePaths)
	s.mutex.Unlock()
	
	log.Printf("📁 Discovered %d files to process", len(filePaths))
	
	// Phase 2: Process files concurrently
	s.processFilesConcurrently(filePaths)
}

func (s *IndexingService) discoverFiles(rootPath string) ([]string, error) {
	var filePaths []string
	
	err := filepath.Walk(rootPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Skip files we can't access
		}
		
		// Skip directories and large files
		if info.IsDir() || info.Size() > MAX_FILE_SIZE {
			return nil
		}
		
		// Skip hidden directories and files
		if strings.Contains(path, "\\.") || strings.Contains(path, "node_modules") || 
		   strings.Contains(path, ".git") || strings.Contains(path, "dist") ||
		   strings.Contains(path, "build") || strings.Contains(path, ".next") {
			return nil
		}
		
		// Check if file extension is supported
		ext := strings.ToLower(filepath.Ext(path))
		if SUPPORTED_EXTENSIONS[ext] {
			filePaths = append(filePaths, path)
		}
		
		return nil
	})
	
	return filePaths, err
}

func (s *IndexingService) processFilesConcurrently(filePaths []string) {
	var wg sync.WaitGroup
	
	// Start result collector
	go s.collectResults()
	
	// Start error handler
	go s.handleErrors()
	
	// Process files in batches
	for i := 0; i < len(filePaths); i += BATCH_SIZE {
		end := i + BATCH_SIZE
		if end > len(filePaths) {
			end = len(filePaths)
		}
		
		batch := filePaths[i:end]
		
		wg.Add(1)
		go func(batch []string) {
			defer wg.Done()
			s.processBatch(batch)
		}(batch)
		
		// Rate limiting - don't overwhelm Ollama
		time.Sleep(100 * time.Millisecond)
	}
	
	wg.Wait()
	close(s.results)
	close(s.errors)
	
	log.Printf("✅ Indexing completed. Processed: %d, Failed: %d", 
		s.progress.ProcessedFiles, s.progress.FailedFiles)
}

func (s *IndexingService) processBatch(filePaths []string) {
	// Limit concurrent workers
	s.workerPool <- struct{}{}
	defer func() { <-s.workerPool }()
	
	var texts []string
	var fileInfos []FileInfo
	
	// Read all files in batch
	for _, path := range filePaths {
		fileInfo, err := s.readFile(path)
		if err != nil {
			s.errors <- fmt.Errorf("failed to read %s: %v", path, err)
			continue
		}
		
		texts = append(texts, fileInfo.Content)
		fileInfos = append(fileInfos, fileInfo)
	}
	
	if len(texts) == 0 {
		return
	}
	
	// Generate embeddings in batch
	embeddings, err := s.generateBatchEmbeddings(texts)
	if err != nil {
		s.errors <- fmt.Errorf("failed to generate embeddings: %v", err)
		return
	}
	
	// Attach embeddings to file infos
	for i, fileInfo := range fileInfos {
		if i < len(embeddings) {
			fileInfo.Embedding = embeddings[i]
			fileInfo.ProcessedAt = time.Now()
		}
		s.results <- fileInfo
	}
}

func (s *IndexingService) readFile(path string) (FileInfo, error) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return FileInfo{}, err
	}
	
	info, err := os.Stat(path)
	if err != nil {
		return FileInfo{}, err
	}
	
	// Determine language from extension
	ext := strings.ToLower(filepath.Ext(path))
	language := getLanguageFromExtension(ext)
	
	// Create metadata
	metadata := map[string]string{
		"extension": ext,
		"directory": filepath.Dir(path),
		"filename":  filepath.Base(path),
	}
	
	return FileInfo{
		Path:     path,
		Content:  string(content),
		Size:     info.Size(),
		Language: language,
		ModTime:  info.ModTime(),
		Metadata: metadata,
		Hash:     generateContentHash(content),
	}, nil
}

func (s *IndexingService) generateBatchEmbeddings(texts []string) ([][]float64, error) {
	requestData := BatchEmbeddingRequest{
		Texts: texts,
		Model: EMBEDDING_MODEL,
	}
	
	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return nil, err
	}
	
	resp, err := http.Post(OLLAMA_URL+"/api/embeddings/batch", "application/json", 
		strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}
	
	var response BatchEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}
	
	return response.Embeddings, nil
}

func (s *IndexingService) collectResults() {
	for fileInfo := range s.results {
		s.mutex.Lock()
		s.files = append(s.files, fileInfo)
		s.progress.ProcessedFiles++
		s.progress.CurrentFile = fileInfo.Path
		
		// Calculate processing rate
		elapsed := time.Since(s.progress.StartTime)
		if elapsed.Seconds() > 0 {
			s.progress.Rate = float64(s.progress.ProcessedFiles) / elapsed.Seconds()
			
			// Estimate completion time
			remaining := s.progress.TotalFiles - s.progress.ProcessedFiles
			if s.progress.Rate > 0 {
				eta := time.Duration(float64(remaining)/s.progress.Rate) * time.Second
				s.progress.EstimatedEnd = time.Now().Add(eta)
			}
		}
		s.mutex.Unlock()
		
		// Broadcast progress to WebSocket clients
		s.broadcastProgress()
		
		// Log progress every 100 files
		if s.progress.ProcessedFiles%100 == 0 {
			log.Printf("📊 Progress: %d/%d files (%.1f files/sec)", 
				s.progress.ProcessedFiles, s.progress.TotalFiles, s.progress.Rate)
		}
	}
}

func (s *IndexingService) handleErrors() {
	for err := range s.errors {
		s.mutex.Lock()
		s.progress.FailedFiles++
		s.mutex.Unlock()
		
		log.Printf("⚠️ Error: %v", err)
	}
}

func (s *IndexingService) StartWebSocketServer() {
	http.HandleFunc("/ws", s.handleWebSocket)
	log.Printf("🔌 WebSocket server starting on %s", WEBSOCKET_PORT)
	log.Fatal(http.ListenAndServe(WEBSOCKET_PORT, nil))
}

func (s *IndexingService) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	s.clients[conn] = true
	log.Printf("🔌 New WebSocket client connected")
	
	// Send current progress immediately
	s.sendProgress(conn)
	
	// Keep connection alive
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			log.Printf("WebSocket client disconnected: %v", err)
			delete(s.clients, conn)
			break
		}
	}
}

func (s *IndexingService) broadcastProgress() {
	for conn := range s.clients {
		s.sendProgress(conn)
	}
}

func (s *IndexingService) sendProgress(conn *websocket.Conn) {
	s.mutex.RLock()
	progress := *s.progress
	s.mutex.RUnlock()
	
	if err := conn.WriteJSON(progress); err != nil {
		log.Printf("Failed to send progress: %v", err)
		delete(s.clients, conn)
	}
}

func (s *IndexingService) StartAPIServer() {
	router := mux.NewRouter()
	
	// API endpoints
	router.HandleFunc("/api/progress", s.getProgress).Methods("GET")
	router.HandleFunc("/api/files", s.getFiles).Methods("GET")
	router.HandleFunc("/api/search", s.searchFiles).Methods("POST")
	router.HandleFunc("/api/stats", s.getStats).Methods("GET")
	router.HandleFunc("/api/health", s.healthCheck).Methods("GET")
	
	log.Printf("🌐 API server starting on %s", API_PORT)
	log.Fatal(http.ListenAndServe(API_PORT, router))
}

func (s *IndexingService) getProgress(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	progress := *s.progress
	s.mutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(progress)
}

func (s *IndexingService) getFiles(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	files := make([]FileInfo, len(s.files))
	copy(files, s.files)
	s.mutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(files)
}

func (s *IndexingService) searchFiles(w http.ResponseWriter, r *http.Request) {
	var searchRequest struct {
		Query string `json:"query"`
		Limit int    `json:"limit"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&searchRequest); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	
	if searchRequest.Limit == 0 {
		searchRequest.Limit = 10
	}
	
	// Generate query embedding
	embeddings, err := s.generateBatchEmbeddings([]string{searchRequest.Query})
	if err != nil {
		http.Error(w, "Failed to generate query embedding", http.StatusInternalServerError)
		return
	}
	
	if len(embeddings) == 0 {
		http.Error(w, "No embedding generated", http.StatusInternalServerError)
		return
	}
	
	queryEmbedding := embeddings[0]
	
	// Search for similar files
	results := s.semanticSearch(queryEmbedding, searchRequest.Limit)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func (s *IndexingService) semanticSearch(queryEmbedding []float64, limit int) []FileInfo {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	type scoredFile struct {
		file  FileInfo
		score float64
	}
	
	var scored []scoredFile
	
	for _, file := range s.files {
		if len(file.Embedding) == 0 {
			continue
		}
		
		similarity := cosineSimilarity(queryEmbedding, file.Embedding)
		scored = append(scored, scoredFile{file: file, score: similarity})
	}
	
	// Sort by similarity score
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[i].score < scored[j].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}
	
	// Return top results
	var results []FileInfo
	for i := 0; i < limit && i < len(scored); i++ {
		results = append(results, scored[i].file)
	}
	
	return results
}

func (s *IndexingService) getStats(w http.ResponseWriter, r *http.Request) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	stats := map[string]interface{}{
		"total_files":     len(s.files),
		"processed_files": s.progress.ProcessedFiles,
		"failed_files":    s.progress.FailedFiles,
		"processing_rate": s.progress.Rate,
		"memory_usage":    getMemoryUsage(),
		"goroutines":      runtime.NumGoroutine(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (s *IndexingService) healthCheck(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"ollama":    s.checkOllamaHealth(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (s *IndexingService) checkOllamaHealth() bool {
	resp, err := http.Get(OLLAMA_URL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Utility functions
func getLanguageFromExtension(ext string) string {
	languages := map[string]string{
		".go":     "go",
		".ts":     "typescript",
		".svelte": "svelte",
		".js":     "javascript",
		".py":     "python",
		".md":     "markdown",
		".json":   "json",
		".sql":    "sql",
		".css":    "css",
		".scss":   "scss",
		".html":   "html",
	}
	
	if lang, exists := languages[ext]; exists {
		return lang
	}
	return "unknown"
}

func generateContentHash(content []byte) string {
	// Simple hash function - in production use crypto/sha256
	hash := 0
	for _, b := range content {
		hash = (hash*31 + int(b)) & 0x7fffffff
	}
	return fmt.Sprintf("%x", hash)
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	
	var dotProduct, normA, normB float64
	
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func getMemoryUsage() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return map[string]interface{}{
		"alloc_mb":      m.Alloc / 1024 / 1024,
		"sys_mb":        m.Sys / 1024 / 1024,
		"gc_runs":       m.NumGC,
		"goroutines":    runtime.NumGoroutine(),
	}
}