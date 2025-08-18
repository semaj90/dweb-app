package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
)

type Context7Worker struct {
	ID       string `json:"id"`
	Port     int    `json:"port"`
	Endpoint string `json:"endpoint"`
	Health   string `json:"health_path"`
	Capacity int    `json:"capacity"`
	Active   bool
	Queue    chan ErrorTask
}

type ErrorTask struct {
	ID          string      `json:"id"`
	ErrorData   SvelteError `json:"error_data"`
	Timestamp   time.Time   `json:"timestamp"`
	ProcessedBy string      `json:"processed_by"`
}

type SvelteError struct {
	File        string `json:"file"`
	Line        int    `json:"line"`
	Column      int    `json:"column"`
	Message     string `json:"message"`
	Code        string `json:"code"`
	Severity    string `json:"severity"`
	Context     string `json:"context"`
}

type EmbeddingRequest struct {
	Text   string            `json:"text"`
	Model  string            `json:"model"`
	Params map[string]interface{} `json:"params"`
}

type EmbeddingResponse struct {
	Vector    []float32 `json:"vector"`
	Dimension int       `json:"dimension"`
	Error     string    `json:"error,omitempty"`
}

type Context7Pipeline struct {
	Workers     []*Context7Worker
	WorkerIndex int
	Mutex       sync.RWMutex
	Watcher     *fsnotify.Watcher
	Config      Config
}

type Config struct {
	Context7Workers struct {
		Enabled        bool              `json:"enabled"`
		LoadBalancer   string            `json:"load_balancer"`
		Timeout        string            `json:"timeout"`
		RetryAttempts  int               `json:"retry_attempts"`
		Workers        []Context7Worker  `json:"workers"`
	} `json:"context7_workers"`
	ErrorPipeline struct {
		WatchFiles         []string `json:"watch_files"`
		ParallelProcessing bool     `json:"parallel_processing"`
		BatchSize          int      `json:"batch_size"`
		EmbeddingModel     string   `json:"embedding_model"`
		VectorDimensions   int      `json:"vector_dimensions"`
	} `json:"error_pipeline"`
}

func NewContext7Pipeline(configPath string) (*Context7Pipeline, error) {
	pipeline := &Context7Pipeline{
		Workers: make([]*Context7Worker, 0),
	}

	// Load configuration
	configData, err := ioutil.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %v", err)
	}

	if err := json.Unmarshal(configData, &pipeline.Config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %v", err)
	}

	// Initialize workers
	for _, workerConfig := range pipeline.Config.Context7Workers.Workers {
		worker := &Context7Worker{
			ID:       workerConfig.ID,
			Port:     workerConfig.Port,
			Endpoint: workerConfig.Endpoint,
			Health:   workerConfig.Health,
			Capacity: workerConfig.Capacity,
			Queue:    make(chan ErrorTask, workerConfig.Capacity),
		}
		
		// Health check
		if pipeline.healthCheck(worker) {
			worker.Active = true
			pipeline.Workers = append(pipeline.Workers, worker)
			log.Printf("✅ Context7 Worker %s active on port %d", worker.ID, worker.Port)
		} else {
			log.Printf("⚠️ Context7 Worker %s unavailable on port %d", worker.ID, worker.Port)
		}
	}

	// Initialize file watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create file watcher: %v", err)
	}
	pipeline.Watcher = watcher

	return pipeline, nil
}

func (p *Context7Pipeline) healthCheck(worker *Context7Worker) bool {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(worker.Endpoint + worker.Health)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func (p *Context7Pipeline) getNextWorker() *Context7Worker {
	p.Mutex.Lock()
	defer p.Mutex.Unlock()

	if len(p.Workers) == 0 {
		return nil
	}

	// Round robin load balancing
	worker := p.Workers[p.WorkerIndex]
	p.WorkerIndex = (p.WorkerIndex + 1) % len(p.Workers)
	return worker
}

func (p *Context7Pipeline) processErrorBatch(errors []SvelteError) error {
	log.Printf("📊 Processing batch of %d errors across %d workers", len(errors), len(p.Workers))
	
	var wg sync.WaitGroup
	errorChan := make(chan SvelteError, len(errors))
	
	// Fill error channel
	for _, err := range errors {
		errorChan <- err
	}
	close(errorChan)

	// Start worker goroutines
	for _, worker := range p.Workers {
		if !worker.Active {
			continue
		}
		
		wg.Add(1)
		go p.workerProcessor(worker, errorChan, &wg)
	}

	wg.Wait()
	log.Printf("✅ Batch processing completed")
	return nil
}

func (p *Context7Pipeline) workerProcessor(worker *Context7Worker, errorChan <-chan SvelteError, wg *sync.WaitGroup) {
	defer wg.Done()

	for svlErr := range errorChan {
		// Create descriptive text for embedding
		errorText := fmt.Sprintf("SvelteKit error in %s at line %d: %s. Context: %s", 
			svlErr.File, svlErr.Line, svlErr.Message, svlErr.Context)

		// Request embedding from Context7 worker
		embedding, err := p.requestEmbedding(worker, errorText)
		if err != nil {
			log.Printf("❌ Worker %s failed to process error: %v", worker.ID, err)
			continue
		}

		// Store in vector database (placeholder - integrate with your vector store)
		err = p.storeVector(svlErr, embedding)
		if err != nil {
			log.Printf("❌ Failed to store vector for error in %s: %v", svlErr.File, err)
			continue
		}

		log.Printf("✅ Worker %s processed error from %s:%d", worker.ID, svlErr.File, svlErr.Line)
	}
}

func (p *Context7Pipeline) requestEmbedding(worker *Context7Worker, text string) ([]float32, error) {
	reqData := EmbeddingRequest{
		Text:  text,
		Model: p.Config.ErrorPipeline.EmbeddingModel,
		Params: map[string]interface{}{
			"dimensions": p.Config.ErrorPipeline.VectorDimensions,
		},
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(worker.Endpoint+"/embed", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var embeddingResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
		return nil, err
	}

	if embeddingResp.Error != "" {
		return nil, fmt.Errorf("embedding error: %s", embeddingResp.Error)
	}

	return embeddingResp.Vector, nil
}

func (p *Context7Pipeline) storeVector(svlErr SvelteError, vector []float32) error {
	// Integration point with your vector database (Qdrant/pgvector)
	// This would make a request to your Enhanced RAG service at port 8094
	
	vectorData := map[string]interface{}{
		"id":          fmt.Sprintf("error_%s_%d_%d", filepath.Base(svlErr.File), svlErr.Line, time.Now().Unix()),
		"vector":      vector,
		"metadata": map[string]interface{}{
			"file":     svlErr.File,
			"line":     svlErr.Line,
			"column":   svlErr.Column,
			"message":  svlErr.Message,
			"severity": svlErr.Severity,
			"context":  svlErr.Context,
			"timestamp": time.Now().Format(time.RFC3339),
		},
	}

	jsonData, _ := json.Marshal(vectorData)
	resp, err := http.Post("http://localhost:8094/api/vectors/store", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

func (p *Context7Pipeline) startFileWatcher() error {
	log.Printf("👁️ Starting file watcher for error pipeline")
	
	// Watch for error files
	for _, filename := range p.Config.ErrorPipeline.WatchFiles {
		err := p.Watcher.Add(filename)
		if err != nil {
			log.Printf("⚠️ Failed to watch %s: %v", filename, err)
		} else {
			log.Printf("👀 Watching %s for changes", filename)
		}
	}

	go func() {
		for {
			select {
			case event, ok := <-p.Watcher.Events:
				if !ok {
					return
				}
				
				if event.Op&fsnotify.Write == fsnotify.Write {
					log.Printf("🔥 File modified: %s", event.Name)
					go p.handleFileChange(event.Name)
				}
				
			case err, ok := <-p.Watcher.Errors:
				if !ok {
					return
				}
				log.Printf("❌ File watcher error: %v", err)
			}
		}
	}()

	return nil
}

func (p *Context7Pipeline) handleFileChange(filename string) {
	log.Printf("📄 Processing file change: %s", filename)
	
	// Read and parse error file
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Printf("❌ Failed to read %s: %v", filename, err)
		return
	}

	var errors []SvelteError
	if err := json.Unmarshal(data, &errors); err != nil {
		log.Printf("❌ Failed to parse JSON from %s: %v", filename, err)
		return
	}

	// Process errors through Context7 workers
	if err := p.processErrorBatch(errors); err != nil {
		log.Printf("❌ Failed to process error batch: %v", err)
	}
}

func (p *Context7Pipeline) Start() error {
	log.Printf("🚀 Starting Context7 Error-to-Vector Pipeline")
	log.Printf("📊 %d active workers", len(p.Workers))
	
	if err := p.startFileWatcher(); err != nil {
		return fmt.Errorf("failed to start file watcher: %v", err)
	}

	// HTTP API for manual processing
	http.HandleFunc("/process", p.handleManualProcess)
	http.HandleFunc("/health", p.handleHealth)
	http.HandleFunc("/workers/status", p.handleWorkersStatus)

	log.Printf("🌐 API server starting on :8095")
	return http.ListenAndServe(":8095", nil)
}

func (p *Context7Pipeline) handleManualProcess(w http.ResponseWriter, r *http.Request) {
	// Manual trigger for processing
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "processing triggered"})
}

func (p *Context7Pipeline) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"workers": len(p.Workers),
		"active_workers": func() int {
			count := 0
			for _, w := range p.Workers {
				if w.Active { count++ }
			}
			return count
		}(),
	})
}

func (p *Context7Pipeline) handleWorkersStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"workers": p.Workers,
		"load_balancer": p.Config.Context7Workers.LoadBalancer,
		"current_index": p.WorkerIndex,
	})
}

func main() {
	pipeline, err := NewContext7Pipeline("context7-workers-config.json")
	if err != nil {
		log.Fatalf("❌ Failed to initialize pipeline: %v", err)
	}
	defer pipeline.Watcher.Close()

	log.Fatal(pipeline.Start())
}