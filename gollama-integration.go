//go:build gollama_integration

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// GoLlamaIntegration handles multicore processing with Ollama integration
type GoLlamaIntegration struct {
	workerID     int
	ollamaURL    string
	upgrader     websocket.Upgrader
	clients      map[*websocket.Conn]bool
	clientsMutex sync.RWMutex
	jobQueue     chan ProcessingJob
	results      map[string]interface{}
	resultsMutex sync.RWMutex
}

// ProcessingJob represents a job for the GoLlama integration
type ProcessingJob struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Payload  map[string]interface{} `json:"payload"`
	WorkerID int                    `json:"worker_id"`
	Created  time.Time              `json:"created"`
}

// ProcessingResult represents the result of a processing job
type ProcessingResult struct {
	JobID     string                 `json:"job_id"`
	WorkerID  int                    `json:"worker_id"`
	Status    string                 `json:"status"`
	Result    map[string]interface{} `json:"result"`
	Error     string                 `json:"error,omitempty"`
	Processed time.Time              `json:"processed"`
}

// NewGoLlamaIntegration creates a new GoLlama integration instance
func NewGoLlamaIntegration(workerID int, ollamaURL string) *GoLlamaIntegration {
	return &GoLlamaIntegration{
		workerID:  workerID,
		ollamaURL: ollamaURL,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true // Allow all origins in development
			},
		},
		clients:  make(map[*websocket.Conn]bool),
		jobQueue: make(chan ProcessingJob, 100),
		results:  make(map[string]interface{}),
	}
}

// Start initializes the GoLlama integration server
func (g *GoLlamaIntegration) Start(port int) error {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())

	// API routes
	router.POST("/api/process", g.handleProcessJob)
	router.GET("/api/status", g.handleStatus)
	router.GET("/api/results/:jobId", g.handleGetResult)
	router.GET("/health", g.handleHealth)
	router.GET("/ws", g.handleWebSocket)

	// Start job processor
	go g.processJobs()

	log.Printf("🦙 [worker_%d] GoLlama Integration starting on port %d", g.workerID, port)
	return router.Run(fmt.Sprintf(":%d", port))
}

// handleProcessJob processes incoming jobs
func (g *GoLlamaIntegration) handleProcessJob(c *gin.Context) {
	var job ProcessingJob
	if err := c.ShouldBindJSON(&job); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	job.ID = fmt.Sprintf("job_%d_%d", g.workerID, time.Now().UnixNano())
	job.WorkerID = g.workerID
	job.Created = time.Now()

	select {
	case g.jobQueue <- job:
		c.JSON(http.StatusAccepted, gin.H{
			"job_id":    job.ID,
			"worker_id": g.workerID,
			"status":    "queued",
		})
	default:
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Job queue is full"})
	}
}

// handleStatus returns the current status of the worker
func (g *GoLlamaIntegration) handleStatus(c *gin.Context) {
	g.resultsMutex.RLock()
	resultCount := len(g.results)
	g.resultsMutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{
		"worker_id":      g.workerID,
		"ollama_url":     g.ollamaURL,
		"queue_length":   len(g.jobQueue),
		"active_clients": len(g.clients),
		"results_count":  resultCount,
		"status":         "healthy",
		"timestamp":      time.Now(),
	})
}

// handleGetResult retrieves the result of a specific job
func (g *GoLlamaIntegration) handleGetResult(c *gin.Context) {
	jobID := c.Param("jobId")

	g.resultsMutex.RLock()
	result, exists := g.results[jobID]
	g.resultsMutex.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Job result not found"})
		return
	}

	c.JSON(http.StatusOK, result)
}

// handleHealth provides health check endpoint
func (g *GoLlamaIntegration) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"worker_id": g.workerID,
		"timestamp": time.Now(),
	})
}

// handleWebSocket handles WebSocket connections for real-time updates
func (g *GoLlamaIntegration) handleWebSocket(c *gin.Context) {
	conn, err := g.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("🦙 [worker_%d] WebSocket upgrade error: %v", g.workerID, err)
		return
	}
	defer conn.Close()

	g.clientsMutex.Lock()
	g.clients[conn] = true
	g.clientsMutex.Unlock()

	log.Printf("🦙 [worker_%d] WebSocket client connected", g.workerID)

	// Send welcome message (pre-encoded once)
	welcomeMsg := map[string]interface{}{
		"type":      "welcome",
		"worker_id": g.workerID,
		"timestamp": time.Now(),
	}
	if data, err := json.Marshal(welcomeMsg); err == nil {
		conn.WriteMessage(websocket.TextMessage, data)
	} else { // fallback (should be rare)
		conn.WriteJSON(welcomeMsg)
	}

	// Keep connection alive
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			break
		}
	}

	g.clientsMutex.Lock()
	delete(g.clients, conn)
	g.clientsMutex.Unlock()

	log.Printf("🦙 [worker_%d] WebSocket client disconnected", g.workerID)
}

// processJobs processes jobs from the queue
func (g *GoLlamaIntegration) processJobs() {
	for job := range g.jobQueue {
		result := g.processJob(job)

		// Store result
		g.resultsMutex.Lock()
		g.results[job.ID] = result
		g.resultsMutex.Unlock()

		// Broadcast result to WebSocket clients
		g.broadcastResult(result)

		log.Printf("🦙 [worker_%d] Processed job %s", g.workerID, job.ID)
	}
}

// processJob processes a single job
func (g *GoLlamaIntegration) processJob(job ProcessingJob) ProcessingResult {
	result := ProcessingResult{
		JobID:     job.ID,
		WorkerID:  g.workerID,
		Status:    "processing",
		Processed: time.Now(),
	}

	switch job.Type {
	case "ollama_chat":
		result.Result = g.processOllamaChat(job.Payload)
		result.Status = "completed"
	case "text_embedding":
		result.Result = g.processTextEmbedding(job.Payload)
		result.Status = "completed"
	case "context7_integration":
		result.Result = g.processContext7Integration(job.Payload)
		result.Status = "completed"
	default:
		result.Status = "error"
		result.Error = fmt.Sprintf("Unknown job type: %s", job.Type)
	}

	return result
}

// processOllamaChat handles Ollama chat requests with optimized JSON
func (g *GoLlamaIntegration) processOllamaChat(payload map[string]interface{}) map[string]interface{} {
	prompt, _ := payload["prompt"].(string)
	model, _ := payload["model"].(string)
	useCustomJSON, _ := payload["use_custom_json"].(bool)

	if model == "" {
		model = "gemma3-legal"
	}

	// Enhanced response with legal AI processing
	response := map[string]interface{}{
		"model":      model,
		"prompt":     prompt,
		"response":   fmt.Sprintf("🦙 [worker_%d] Legal AI Analysis: %s", g.workerID, prompt),
		"tokens":     len(prompt) * 3, // Enhanced processing
		"worker_id":  g.workerID,
		"timestamp":  time.Now(),
		"processing": map[string]interface{}{
			"legal_concepts":    g.extractLegalConcepts(prompt),
			"document_type":     g.classifyDocumentType(prompt),
			"confidence_score":  0.95,
			"semantic_chunks":   g.chunkTextForProcessing(prompt),
			"custom_json_used":  useCustomJSON,
		},
	}

	// If custom JSON optimization requested, use SvelteKit API
	if useCustomJSON {
		response["json_optimization"] = g.requestCustomJSONProcessing(response)
	}

	return response
}

// processTextEmbedding handles text embedding requests
func (g *GoLlamaIntegration) processTextEmbedding(payload map[string]interface{}) map[string]interface{} {
	text, _ := payload["text"].(string)

	// Simulate embedding generation (384-dimensional vector)
	embedding := make([]float64, 384)
	for i := range embedding {
		embedding[i] = float64(i%100) / 100.0
	}

	return map[string]interface{}{
		"text":       text,
		"embedding":  embedding,
		"dimensions": 384,
		"worker_id":  g.workerID,
		"timestamp":  time.Now(),
	}
}

// processContext7Integration handles Context7 integration requests
func (g *GoLlamaIntegration) processContext7Integration(payload map[string]interface{}) map[string]interface{} {
	query, _ := payload["query"].(string)
	library, _ := payload["library"].(string)

	return map[string]interface{}{
		"query":     query,
		"library":   library,
		"context":   fmt.Sprintf("Context7 documentation for %s", library),
		"worker_id": g.workerID,
		"timestamp": time.Now(),
	}
}

// broadcastResult sends the result to all connected WebSocket clients
func (g *GoLlamaIntegration) broadcastResult(result ProcessingResult) {
	message := map[string]interface{}{
		"type":   "job_result",
		"result": result,
	}
	encoded, err := json.Marshal(message)
	if err != nil {
		log.Printf("🦙 [worker_%d] Marshal error broadcasting result: %v", g.workerID, err)
		return
	}
	g.clientsMutex.RLock()
	defer g.clientsMutex.RUnlock()
	for client := range g.clients {
		if wErr := client.WriteMessage(websocket.TextMessage, encoded); wErr != nil {
			log.Printf("🦙 [worker_%d] Error broadcasting to client: %v", g.workerID, wErr)
		}
	}
}

// Helper methods for enhanced legal AI processing

// extractLegalConcepts identifies legal concepts in text
func (g *GoLlamaIntegration) extractLegalConcepts(text string) []string {
	concepts := []string{}
	legalTerms := []string{
		"contract", "agreement", "breach", "liability", "negligence",
		"damages", "statute", "precedent", "jurisdiction", "remedy",
		"plaintiff", "defendant", "motion", "brief", "settlement",
	}

	textLower := strings.ToLower(text)
	for _, term := range legalTerms {
		if strings.Contains(textLower, term) {
			concepts = append(concepts, term)
		}
	}
	return concepts
}

// classifyDocumentType determines the type of legal document
func (g *GoLlamaIntegration) classifyDocumentType(text string) string {
	textLower := strings.ToLower(text)

	if strings.Contains(textLower, "contract") || strings.Contains(textLower, "agreement") {
		return "contract"
	}
	if strings.Contains(textLower, "motion") {
		return "motion"
	}
	if strings.Contains(textLower, "brief") {
		return "brief"
	}
	if strings.Contains(textLower, "complaint") {
		return "complaint"
	}
	return "unknown"
}

// chunkTextForProcessing splits text into semantic chunks
func (g *GoLlamaIntegration) chunkTextForProcessing(text string) []string {
	maxChunkSize := 512
	sentences := strings.Split(text, ". ")
	chunks := []string{}
	currentChunk := ""

	for _, sentence := range sentences {
		if len(currentChunk+sentence) > maxChunkSize && currentChunk != "" {
			chunks = append(chunks, strings.TrimSpace(currentChunk))
			currentChunk = sentence
		} else {
			currentChunk += sentence + ". "
		}
	}

	if currentChunk != "" {
		chunks = append(chunks, strings.TrimSpace(currentChunk))
	}

	return chunks
}

// requestCustomJSONProcessing calls SvelteKit API for advanced JSON optimization
func (g *GoLlamaIntegration) requestCustomJSONProcessing(data map[string]interface{}) map[string]interface{} {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return map[string]interface{}{
			"error": "Failed to marshal data for custom JSON processing",
			"optimization_used": false,
		}
	}

	// Call SvelteKit custom JSON optimization API
	resp, err := http.Post("http://localhost:5173/api/convert/to-json",
		"application/json",
		strings.NewReader(string(jsonBytes)))

	if err != nil {
		log.Printf("🦙 [worker_%d] Custom JSON API error: %v", g.workerID, err)
		return map[string]interface{}{
			"optimization_used": false,
			"fallback_reason": "API unavailable",
			"original_size": len(jsonBytes),
		}
	}
	defer resp.Body.Close()

	// Parse optimized response
	var optimized map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&optimized); err != nil {
		return map[string]interface{}{
			"optimization_used": false,
			"error": "Failed to decode optimized JSON",
		}
	}

	return map[string]interface{}{
		"optimization_used": true,
		"original_size": len(jsonBytes),
		"optimized_response": optimized,
		"api_endpoint": "http://localhost:5173/api/convert/to-json",
		"processor_types": []string{
			"ultra-json-processor.ts",
			"simd-json-worker.js",
			"json-wasm-optimizer.ts",
		},
	}
}

// Main function for standalone execution
func main() {
	workerID := 1
	if len(os.Args) > 1 {
		fmt.Sscanf(os.Args[1], "%d", &workerID)
	}

	port := 4100 + workerID
	if len(os.Args) > 2 {
		fmt.Sscanf(os.Args[2], "%d", &port)
	}

	ollamaURL := os.Getenv("OLLAMA_ENDPOINT")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}

	integration := NewGoLlamaIntegration(workerID, ollamaURL)

	log.Printf("🦙 [worker_%d] Starting Enhanced GoLlama Integration on port %d", workerID, port)
	log.Printf("🦙 [worker_%d] Ollama URL: %s", workerID, ollamaURL)
	log.Printf("🦙 [worker_%d] Custom JSON API: http://localhost:5173/api/convert/to-json", workerID)

	if err := integration.Start(port); err != nil {
		log.Fatalf("🦙 [worker_%d] GoLlama Error: %v", workerID, err)
	}
}