//go:build legacy
// +build legacy

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
)

// Automated indexing service with GPU processing integration
type AutoIndexer struct {
	dbPool      *pgxpool.Pool
	redisClient *redis.Client
	watcher     *fsnotify.Watcher
	processor   *DocumentProcessor
	config      *IndexerConfig
}

type IndexerConfig struct {
	WatchDirs       []string `json:"watch_directories"`
	SupportedExts   []string `json:"supported_extensions"`
	BatchSize       int      `json:"batch_size"`
	ProcessingDelay time.Duration `json:"processing_delay"`
	UseGPU          bool     `json:"use_gpu"`
	WorkerCount     int      `json:"worker_count"`
}

type DocumentProcessor struct {
	jobQueue    chan IndexingJob
	resultQueue chan IndexingResult
	workers     int
	wg          sync.WaitGroup
}

type IndexingJob struct {
	FilePath    string    `json:"file_path"`
	Content     string    `json:"content"`
	Timestamp   time.Time `json:"timestamp"`
	Priority    int       `json:"priority"`
}

type IndexingResult struct {
	FilePath        string    `json:"file_path"`
	Success         bool      `json:"success"`
	Embedding       []float32 `json:"embedding,omitempty"`
	Summary         string    `json:"summary,omitempty"`
	ProcessingTime  int64     `json:"processing_time_ms"`
	Error           string    `json:"error,omitempty"`
	Method          string    `json:"method"`
}

func NewAutoIndexer(config *IndexerConfig) (*AutoIndexer, error) {
	// Database connection
	dbPool, err := pgxpool.New(context.Background(), "postgres://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db")
	if err != nil {
		return nil, fmt.Errorf("database connection failed: %v", err)
	}

	// Redis connection
	redisClient := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   0,
	})

	// File watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("watcher creation failed: %v", err)
	}

	// Document processor
	processor := &DocumentProcessor{
		jobQueue:    make(chan IndexingJob, config.BatchSize*2),
		resultQueue: make(chan IndexingResult, config.BatchSize*2),
		workers:     config.WorkerCount,
	}

	return &AutoIndexer{
		dbPool:      dbPool,
		redisClient: redisClient,
		watcher:     watcher,
		processor:   processor,
		config:      config,
	}, nil
}

func (ai *AutoIndexer) Start() error {
	log.Println("🚀 Starting automated indexer service...")

	// Start document processor workers
	ai.processor.startWorkers(ai.config.UseGPU)

	// Start result handler
	go ai.handleResults()

	// Setup file watching
	for _, dir := range ai.config.WatchDirs {
		if err := ai.setupWatcher(dir); err != nil {
			return fmt.Errorf("watcher setup failed for %s: %v", dir, err)
		}
	}

	// Process existing files
	go ai.processExistingFiles()

	// Start event loop
	ai.eventLoop()

	return nil
}

func (ai *AutoIndexer) setupWatcher(dir string) error {
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return ai.watcher.Add(path)
		}
		return nil
	})
	return err
}

func (ai *AutoIndexer) eventLoop() {
	for {
		select {
		case event, ok := <-ai.watcher.Events:
			if !ok {
				return
			}
			if event.Has(fsnotify.Write) || event.Has(fsnotify.Create) {
				ai.handleFileEvent(event.Name)
			}

		case err, ok := <-ai.watcher.Errors:
			if !ok {
				return
			}
			log.Printf("Watcher error: %v", err)
		}
	}
}

func (ai *AutoIndexer) handleFileEvent(filePath string) {
	if !ai.isSupported(filePath) {
		return
	}

	// Rate limiting with delay
	time.Sleep(ai.config.ProcessingDelay)

	content, err := os.ReadFile(filePath)
	if err != nil {
		log.Printf("Failed to read file %s: %v", filePath, err)
		return
	}

	job := IndexingJob{
		FilePath:  filePath,
		Content:   string(content),
		Timestamp: time.Now(),
		Priority:  1,
	}

	select {
	case ai.processor.jobQueue <- job:
		log.Printf("Queued file for processing: %s", filePath)
	default:
		log.Printf("Job queue full, dropping file: %s", filePath)
	}
}

func (ai *AutoIndexer) processExistingFiles() {
	for _, dir := range ai.config.WatchDirs {
		filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() || !ai.isSupported(path) {
				return nil
			}

			// Check if already indexed
			exists, err := ai.isAlreadyIndexed(path)
			if err != nil || exists {
				return nil
			}

			ai.handleFileEvent(path)
			return nil
		})
	}
}

func (ai *AutoIndexer) isSupported(filePath string) bool {
	ext := strings.ToLower(filepath.Ext(filePath))
	for _, supportedExt := range ai.config.SupportedExts {
		if ext == supportedExt {
			return true
		}
	}
	return false
}

func (ai *AutoIndexer) isAlreadyIndexed(filePath string) (bool, error) {
	var exists bool
	err := ai.dbPool.QueryRow(context.Background(),
		"SELECT EXISTS(SELECT 1 FROM indexed_files WHERE file_path = $1)", filePath).Scan(&exists)
	return exists, err
}

func (dp *DocumentProcessor) startWorkers(useGPU bool) {
	for i := 0; i < dp.workers; i++ {
		dp.wg.Add(1)
		go dp.worker(useGPU)
	}
}

func (dp *DocumentProcessor) worker(useGPU bool) {
	defer dp.wg.Done()

	for job := range dp.jobQueue {
		startTime := time.Now()
		result := IndexingResult{
			FilePath: job.FilePath,
		}

		// Call GPU processing service
		if useGPU {
			embedding, summary, err := dp.processWithGPU(job.Content)
			if err != nil {
				result.Success = false
				result.Error = err.Error()
				result.Method = "gpu_failed"
			} else {
				result.Success = true
				result.Embedding = embedding
				result.Summary = summary
				result.Method = "gpu"
			}
		} else {
			// CPU fallback
			embedding, summary, err := dp.processWithCPU(job.Content)
			if err != nil {
				result.Success = false
				result.Error = err.Error()
				result.Method = "cpu_failed"
			} else {
				result.Success = true
				result.Embedding = embedding
				result.Summary = summary
				result.Method = "cpu"
			}
		}

		result.ProcessingTime = time.Since(startTime).Milliseconds()
		dp.resultQueue <- result
	}
}

func (dp *DocumentProcessor) processWithGPU(content string) ([]float32, string, error) {
	// Call our GPU microservice
	payload := map[string]interface{}{
		"endpoint": "analyze-document",
		"content":  content,
		"llmProvider": "ollama",
	}

	jsonData, _ := json.Marshal(payload)
	resp, err := http.Post("http://localhost:8080/llm-request", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	// Parse response and extract embedding + summary
	// Implementation depends on GPU service response format
	embedding := make([]float32, 768) // Placeholder
	summary := "GPU-generated summary"
	
	return embedding, summary, nil
}

func (dp *DocumentProcessor) processWithCPU(content string) ([]float32, string, error) {
	// CPU-based processing fallback
	embedding := make([]float32, 768) // Simple placeholder
	summary := fmt.Sprintf("CPU summary of %d character document", len(content))
	return embedding, summary, nil
}

func (ai *AutoIndexer) handleResults() {
	for result := range ai.processor.resultQueue {
		if result.Success {
			ai.storeResult(result)
			ai.cacheResult(result)
		}
		ai.logResult(result)
	}
}

func (ai *AutoIndexer) storeResult(result IndexingResult) {
	embeddingStr := fmt.Sprintf("[%s]", strings.Trim(fmt.Sprint(result.Embedding), "[]"))
	
	_, err := ai.dbPool.Exec(context.Background(),
		`INSERT INTO indexed_files (file_path, content, embedding, summary, processing_method, gpu_processing_time_ms) 
		 VALUES ($1, $2, $3, $4, $5, $6)
		 ON CONFLICT (file_path) DO UPDATE SET 
		 embedding = EXCLUDED.embedding, 
		 summary = EXCLUDED.summary, 
		 processing_method = EXCLUDED.processing_method,
		 gpu_processing_time_ms = EXCLUDED.gpu_processing_time_ms,
		 indexed_at = NOW()`,
		result.FilePath, "", embeddingStr, result.Summary, result.Method, result.ProcessingTime)
	
	if err != nil {
		log.Printf("Database storage failed for %s: %v", result.FilePath, err)
	}
}

func (ai *AutoIndexer) cacheResult(result IndexingResult) {
	cacheKey := fmt.Sprintf("indexed:%s", result.FilePath)
	cacheData, _ := json.Marshal(result)
	
	ai.redisClient.Set(context.Background(), cacheKey, cacheData, 24*time.Hour)
}

func (ai *AutoIndexer) logResult(result IndexingResult) {
	logEntry := map[string]interface{}{
		"timestamp":       time.Now().Unix(),
		"file_path":       result.FilePath,
		"success":         result.Success,
		"processing_time": result.ProcessingTime,
		"method":          result.Method,
		"error":           result.Error,
	}
	
	logJSON, _ := json.Marshal(logEntry)
	log.Printf("INDEXING_RESULT: %s", logJSON)
}

func (ai *AutoIndexer) Stop() {
	close(ai.processor.jobQueue)
	ai.processor.wg.Wait()
	ai.watcher.Close()
	ai.dbPool.Close()
	ai.redisClient.Close()
}

func main() {
	config := &IndexerConfig{
		WatchDirs:       []string{"./uploads", "./documents", "./evidence"},
		SupportedExts:   []string{".pdf", ".txt", ".doc", ".docx", ".md"},
		BatchSize:       100,
		ProcessingDelay: 2 * time.Second,
		UseGPU:          true,
		WorkerCount:     4,
	}

	indexer, err := NewAutoIndexer(config)
	if err != nil {
		log.Fatalf("Indexer initialization failed: %v", err)
	}

	log.Println("🔍 Auto-indexer starting...")
	if err := indexer.Start(); err != nil {
		log.Fatalf("Indexer start failed: %v", err)
	}
}
