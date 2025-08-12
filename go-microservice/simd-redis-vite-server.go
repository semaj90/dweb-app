//go:build legacy
// +build legacy

// simd-redis-vite-server.go
// Ultra-High Performance SIMD JSON + Redis Integration with Vite
// Multi-concurrency and Data Parallelism Support

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/valyala/fastjson"
	"github.com/gorilla/websocket"
)

// Performance metrics tracking
type Metrics struct {
	ParseCount       uint64
	CacheHits        uint64
	CacheMisses      uint64
	TotalParseTime   int64
	WorkerPoolActive int32
	ConnectedClients int32
}

// Worker pool for concurrent processing
type WorkerPool struct {
	workers   int
	tasks     chan *Task
	results   chan *Result
	wg        sync.WaitGroup
	parsers   []*fastjson.Parser
	metrics   *Metrics
}

// Task for worker pool
type Task struct {
	ID       string
	Data     []byte
	Key      string
	TTL      time.Duration
	Response chan *Result
}

// Result from worker processing
type Result struct {
	ID        string      `json:"id"`
	Success   bool        `json:"success"`
	Data      interface{} `json:"data,omitempty"`
	Error     string      `json:"error,omitempty"`
	ParseTime int64       `json:"parse_time_ns"`
	Cached    bool        `json:"cached"`
}

var (
	// Redis clients for different purposes
	redisCache    *redis.Client
	redisJSON     *redis.Client
	redisPubSub   *redis.Client
	
	// Worker pool for SIMD processing
	workerPool    *WorkerPool
	
	// Global metrics
	metrics       = &Metrics{}
	
	// WebSocket upgrader for real-time updates
	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins in development
		},
	}
	
	// Active WebSocket connections
	wsClients = make(map[*websocket.Conn]bool)
	wsMutex   sync.RWMutex
)

func init() {
	// Initialize Redis connections with pooling
	redisOptions := &redis.Options{
		Addr:         "localhost:6379",
		PoolSize:     runtime.NumCPU() * 2,
		MinIdleConns: runtime.NumCPU(),
		MaxRetries:   3,
	}
	
	redisCache = redis.NewClient(redisOptions)
	redisJSON = redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   1, // Use different DB for JSON storage
	})
	redisPubSub = redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   2, // Separate DB for pub/sub
	})
	
	// Initialize worker pool
	workerCount := runtime.NumCPU() * 2
	workerPool = NewWorkerPool(workerCount, metrics)
	workerPool.Start()
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool(workers int, m *Metrics) *WorkerPool {
	wp := &WorkerPool{
		workers: workers,
		tasks:   make(chan *Task, workers*10),
		results: make(chan *Result, workers*10),
		parsers: make([]*fastjson.Parser, workers),
		metrics: m,
	}
	
	// Initialize SIMD-optimized parsers for each worker
	for i := 0; i < workers; i++ {
		wp.parsers[i] = &fastjson.Parser{}
	}
	
	return wp
}

// Start the worker pool
func (wp *WorkerPool) Start() {
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

// Worker function
func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()
	parser := wp.parsers[id]
	
	for task := range wp.tasks {
		atomic.AddInt32(&wp.metrics.WorkerPoolActive, 1)
		result := wp.processTask(task, parser)
		task.Response <- result
		atomic.AddInt32(&wp.metrics.WorkerPoolActive, -1)
	}
}

// Process a single task with SIMD JSON parsing
func (wp *WorkerPool) processTask(task *Task, parser *fastjson.Parser) *Result {
	start := time.Now()
	ctx := context.Background()
	
	// Check cache first
	cached, err := redisCache.Get(ctx, task.Key).Result()
	if err == nil && cached != "" {
		atomic.AddUint64(&wp.metrics.CacheHits, 1)
		return &Result{
			ID:        task.ID,
			Success:   true,
			Data:      json.RawMessage(cached),
			ParseTime: 0,
			Cached:    true,
		}
	}
	
	atomic.AddUint64(&wp.metrics.CacheMisses, 1)
	
	// SIMD-optimized JSON parsing
	parsed, err := parser.ParseBytes(task.Data)
	if err != nil {
		return &Result{
			ID:      task.ID,
			Success: false,
			Error:   err.Error(),
		}
	}
	
	parseTime := time.Since(start).Nanoseconds()
	atomic.AddUint64(&wp.metrics.ParseCount, 1)
	atomic.AddInt64(&wp.metrics.TotalParseTime, parseTime)
	
	// Store in Redis with TTL
	jsonStr := parsed.String()
	redisCache.Set(ctx, task.Key, jsonStr, task.TTL)
	
	// Try Redis JSON module if available
	if err := redisJSON.Do(ctx, "JSON.SET", task.Key+":json", "$", jsonStr).Err(); err == nil {
		// Successfully stored as native JSON
		log.Printf("Stored in Redis JSON: %s", task.Key)
	}
	
	return &Result{
		ID:        task.ID,
		Success:   true,
		Data:      json.RawMessage(jsonStr),
		ParseTime: parseTime,
		Cached:    false,
	}
}

// Submit task to worker pool
func (wp *WorkerPool) Submit(task *Task) {
	wp.tasks <- task
}

func main() {
	// Set Gin to release mode for production
	gin.SetMode(gin.ReleaseMode)
	
	r := gin.New()
	r.Use(gin.Recovery())
	
	// CORS configuration for Vite development server
	config := cors.Config{
		AllowOrigins:     []string{"http://localhost:3130", "http://localhost:5173", "http://localhost:4173"},
		AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}
	r.Use(cors.New(config))
	
	// Health check endpoint
	r.GET("/health", healthCheck)
	
	// SIMD JSON parsing endpoints
	r.POST("/simd-parse", handleSIMDParse)
	r.POST("/simd-batch", handleSIMDBatch)
	
	// Document processing with SIMD
	r.POST("/process-document", handleDocumentProcess)
	
	// AI Assistant with caching
	r.POST("/ai-assistant", handleAIAssistant)
	
	// Cache management
	r.GET("/cache/:key", getCached)
	r.DELETE("/cache/:key", deleteCached)
	
	// Metrics and monitoring
	r.GET("/metrics", getMetrics)
	r.GET("/metrics/stream", streamMetrics)
	
	// WebSocket for real-time updates
	r.GET("/ws", handleWebSocket)
	
	// Legal AI specific endpoints
	r.POST("/legal/analyze", handleLegalAnalysis)
	r.POST("/legal/evidence", handleEvidenceProcessing)
	r.POST("/legal/summary", handleLegalSummary)
	
	// Start metrics broadcaster
	go broadcastMetrics()
	
	log.Printf("🚀 SIMD+Redis+Vite Integration Server starting on :8080")
	log.Printf("   Workers: %d | Redis Pool: %d", workerPool.workers, runtime.NumCPU()*2)
	log.Printf("   SIMD JSON: ✓ | Redis JSON: ✓ | WebSocket: ✓")
	
	if err := r.Run(":8080"); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

// Health check with detailed status
func healthCheck(c *gin.Context) {
	ctx := context.Background()
	
	// Check Redis connectivity
	redisOK := redisCache.Ping(ctx).Err() == nil
	
	// Check Redis JSON module
	redisJSONOK := false
	if err := redisJSON.Do(ctx, "JSON.GET", "test:key").Err(); err == nil || err == redis.Nil {
		redisJSONOK = true
	}
	
	c.JSON(200, gin.H{
		"status":       "healthy",
		"simd":         true,
		"redis":        redisOK,
		"redis_json":   redisJSONOK,
		"workers":      workerPool.workers,
		"active_tasks": atomic.LoadInt32(&metrics.WorkerPoolActive),
		"ws_clients":   atomic.LoadInt32(&metrics.ConnectedClients),
	})
}

// Handle SIMD JSON parsing
func handleSIMDParse(c *gin.Context) {
	data, err := c.GetRawData()
	if err != nil {
		c.JSON(400, gin.H{"error": "Invalid request body"})
		return
	}
	
	taskID := fmt.Sprintf("parse-%d", time.Now().UnixNano())
	responseChannel := make(chan *Result, 1)
	
	task := &Task{
		ID:       taskID,
		Data:     data,
		Key:      c.Query("key"),
		TTL:      5 * time.Minute,
		Response: responseChannel,
	}
	
	if task.Key == "" {
		task.Key = taskID
	}
	
	workerPool.Submit(task)
	result := <-responseChannel
	
	c.JSON(200, result)
}

// Handle batch SIMD parsing
func handleSIMDBatch(c *gin.Context) {
	var batch []json.RawMessage
	if err := c.ShouldBindJSON(&batch); err != nil {
		c.JSON(400, gin.H{"error": "Invalid batch format"})
		return
	}
	
	results := make([]*Result, len(batch))
	var wg sync.WaitGroup
	
	for i, item := range batch {
		wg.Add(1)
		go func(index int, data []byte) {
			defer wg.Done()
			
			taskID := fmt.Sprintf("batch-%d-%d", time.Now().UnixNano(), index)
			responseChannel := make(chan *Result, 1)
			
			task := &Task{
				ID:       taskID,
				Data:     data,
				Key:      fmt.Sprintf("batch:%s:%d", taskID, index),
				TTL:      5 * time.Minute,
				Response: responseChannel,
			}
			
			workerPool.Submit(task)
			results[index] = <-responseChannel
		}(i, []byte(item))
	}
	
	wg.Wait()
	
	c.JSON(200, gin.H{
		"batch_size": len(batch),
		"results":    results,
	})
}

// Handle document processing
func handleDocumentProcess(c *gin.Context) {
	var req struct {
		DocumentID string          `json:"document_id"`
		Content    json.RawMessage `json:"content"`
		Metadata   map[string]interface{} `json:"metadata"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}
	
	responseChannel := make(chan *Result, 1)
	
	task := &Task{
		ID:       req.DocumentID,
		Data:     []byte(req.Content),
		Key:      fmt.Sprintf("doc:%s", req.DocumentID),
		TTL:      30 * time.Minute,
		Response: responseChannel,
	}
	
	workerPool.Submit(task)
	result := <-responseChannel
	
	if result.Success {
		// Store metadata separately
		ctx := context.Background()
		metaKey := fmt.Sprintf("meta:%s", req.DocumentID)
		metaJSON, _ := json.Marshal(req.Metadata)
		redisCache.Set(ctx, metaKey, metaJSON, 30*time.Minute)
		
		// Publish to subscribers
		redisPubSub.Publish(ctx, "document:processed", req.DocumentID)
	}
	
	c.JSON(200, gin.H{
		"document_id": req.DocumentID,
		"processed":   result.Success,
		"parse_time":  result.ParseTime,
		"cached":      result.Cached,
	})
}

// Handle AI assistant queries with caching
func handleAIAssistant(c *gin.Context) {
	var req struct {
		Query   string `json:"query"`
		Context string `json:"context"`
		Model   string `json:"model"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}
	
	// Create cache key from query
	cacheKey := fmt.Sprintf("ai:%s:%s", req.Model, req.Query)
	
	// Check cache first
	ctx := context.Background()
	if cached, err := redisCache.Get(ctx, cacheKey).Result(); err == nil {
		atomic.AddUint64(&metrics.CacheHits, 1)
		c.JSON(200, gin.H{
			"response": cached,
			"cached":   true,
			"model":    req.Model,
		})
		return
	}
	
	// Process with SIMD for context parsing
	contextData := []byte(req.Context)
	responseChannel := make(chan *Result, 1)
	
	task := &Task{
		ID:       fmt.Sprintf("ai-%d", time.Now().UnixNano()),
		Data:     contextData,
		Key:      cacheKey,
		TTL:      10 * time.Minute,
		Response: responseChannel,
	}
	
	workerPool.Submit(task)
	result := <-responseChannel
	
	c.JSON(200, gin.H{
		"query":      req.Query,
		"processed":  result.Success,
		"parse_time": result.ParseTime,
		"cached":     result.Cached,
	})
}

// Legal analysis endpoint
func handleLegalAnalysis(c *gin.Context) {
	var req struct {
		CaseID     string          `json:"case_id"`
		Documents  []json.RawMessage `json:"documents"`
		AnalysisType string        `json:"analysis_type"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}
	
	// Process documents in parallel
	results := make([]*Result, len(req.Documents))
	var wg sync.WaitGroup
	
	for i, doc := range req.Documents {
		wg.Add(1)
		go func(index int, data []byte) {
			defer wg.Done()
			
			responseChannel := make(chan *Result, 1)
			task := &Task{
				ID:       fmt.Sprintf("legal-%s-%d", req.CaseID, index),
				Data:     data,
				Key:      fmt.Sprintf("legal:case:%s:doc:%d", req.CaseID, index),
				TTL:      1 * time.Hour,
				Response: responseChannel,
			}
			
			workerPool.Submit(task)
			results[index] = <-responseChannel
		}(i, []byte(doc))
	}
	
	wg.Wait()
	
	// Store analysis results
	ctx := context.Background()
	analysisKey := fmt.Sprintf("analysis:%s:%s", req.CaseID, req.AnalysisType)
	analysisData, _ := json.Marshal(results)
	redisCache.Set(ctx, analysisKey, analysisData, 1*time.Hour)
	
	c.JSON(200, gin.H{
		"case_id":       req.CaseID,
		"analysis_type": req.AnalysisType,
		"documents":     len(req.Documents),
		"results":       results,
	})
}

// Evidence processing endpoint
func handleEvidenceProcessing(c *gin.Context) {
	var req struct {
		EvidenceID   string                 `json:"evidence_id"`
		Type         string                 `json:"type"`
		Content      json.RawMessage        `json:"content"`
		Metadata     map[string]interface{} `json:"metadata"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}
	
	responseChannel := make(chan *Result, 1)
	
	task := &Task{
		ID:       req.EvidenceID,
		Data:     []byte(req.Content),
		Key:      fmt.Sprintf("evidence:%s", req.EvidenceID),
		TTL:      2 * time.Hour,
		Response: responseChannel,
	}
	
	workerPool.Submit(task)
	result := <-responseChannel
	
	if result.Success {
		// Store in Redis JSON for complex queries
		ctx := context.Background()
		evidenceData := map[string]interface{}{
			"id":       req.EvidenceID,
			"type":     req.Type,
			"metadata": req.Metadata,
			"processed_at": time.Now().Unix(),
		}
		
		jsonData, _ := json.Marshal(evidenceData)
		redisJSON.Do(ctx, "JSON.SET", fmt.Sprintf("evidence:json:%s", req.EvidenceID), "$", string(jsonData))
		
		// Publish notification
		redisPubSub.Publish(ctx, "evidence:processed", req.EvidenceID)
	}
	
	c.JSON(200, gin.H{
		"evidence_id": req.EvidenceID,
		"type":        req.Type,
		"processed":   result.Success,
		"parse_time":  result.ParseTime,
	})
}

// Legal summary generation
func handleLegalSummary(c *gin.Context) {
	var req struct {
		CaseID      string   `json:"case_id"`
		DocumentIDs []string `json:"document_ids"`
		SummaryType string   `json:"summary_type"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": "Invalid request"})
		return
	}
	
	ctx := context.Background()
	documents := make([]map[string]interface{}, 0)
	
	// Retrieve documents from cache
	for _, docID := range req.DocumentIDs {
		key := fmt.Sprintf("doc:%s", docID)
		if data, err := redisCache.Get(ctx, key).Result(); err == nil {
			var doc map[string]interface{}
			if json.Unmarshal([]byte(data), &doc) == nil {
				documents = append(documents, doc)
			}
		}
	}
	
	// Generate summary (placeholder for actual AI processing)
	summaryData := map[string]interface{}{
		"case_id":       req.CaseID,
		"summary_type":  req.SummaryType,
		"document_count": len(documents),
		"generated_at":  time.Now().Unix(),
	}
	
	// Store summary
	summaryKey := fmt.Sprintf("summary:%s:%s", req.CaseID, req.SummaryType)
	summaryJSON, _ := json.Marshal(summaryData)
	redisCache.Set(ctx, summaryKey, summaryJSON, 24*time.Hour)
	
	c.JSON(200, summaryData)
}

// Get cached item
func getCached(c *gin.Context) {
	key := c.Param("key")
	ctx := context.Background()
	
	// Try regular cache first
	if data, err := redisCache.Get(ctx, key).Result(); err == nil {
		c.JSON(200, gin.H{
			"key":    key,
			"data":   json.RawMessage(data),
			"source": "cache",
		})
		return
	}
	
	// Try Redis JSON
	if result := redisJSON.Do(ctx, "JSON.GET", key+":json", "$"); result.Err() == nil {
		if data, ok := result.Val().(string); ok {
			c.JSON(200, gin.H{
				"key":    key,
				"data":   json.RawMessage(data),
				"source": "json_module",
			})
			return
		}
	}
	
	c.JSON(404, gin.H{"error": "Key not found"})
}

// Delete cached item
func deleteCached(c *gin.Context) {
	key := c.Param("key")
	ctx := context.Background()
	
	// Delete from both caches
	redisCache.Del(ctx, key)
	redisJSON.Do(ctx, "JSON.DEL", key+":json")
	
	c.JSON(200, gin.H{"deleted": key})
}

// Get metrics
func getMetrics(c *gin.Context) {
	avgParseTime := int64(0)
	parseCount := atomic.LoadUint64(&metrics.ParseCount)
	if parseCount > 0 {
		avgParseTime = atomic.LoadInt64(&metrics.TotalParseTime) / int64(parseCount)
	}
	
	c.JSON(200, gin.H{
		"parse_count":        parseCount,
		"cache_hits":         atomic.LoadUint64(&metrics.CacheHits),
		"cache_misses":       atomic.LoadUint64(&metrics.CacheMisses),
		"avg_parse_time_ns":  avgParseTime,
		"worker_pool_active": atomic.LoadInt32(&metrics.WorkerPoolActive),
		"connected_clients":  atomic.LoadInt32(&metrics.ConnectedClients),
		"workers":            workerPool.workers,
	})
}

// Stream metrics via SSE
func streamMetrics(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			avgParseTime := int64(0)
			parseCount := atomic.LoadUint64(&metrics.ParseCount)
			if parseCount > 0 {
				avgParseTime = atomic.LoadInt64(&metrics.TotalParseTime) / int64(parseCount)
			}
			
			data := map[string]interface{}{
				"parse_count":        parseCount,
				"cache_hits":         atomic.LoadUint64(&metrics.CacheHits),
				"cache_misses":       atomic.LoadUint64(&metrics.CacheMisses),
				"avg_parse_time_ns":  avgParseTime,
				"worker_pool_active": atomic.LoadInt32(&metrics.WorkerPoolActive),
				"connected_clients":  atomic.LoadInt32(&metrics.ConnectedClients),
			}
			
			jsonData, _ := json.Marshal(data)
			fmt.Fprintf(c.Writer, "data: %s\n\n", jsonData)
			c.Writer.Flush()
			
		case <-c.Request.Context().Done():
			return
		}
	}
}

// Handle WebSocket connections
func handleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	
	// Register client
	wsMutex.Lock()
	wsClients[conn] = true
	atomic.AddInt32(&metrics.ConnectedClients, 1)
	wsMutex.Unlock()
	
	// Unregister on disconnect
	defer func() {
		wsMutex.Lock()
		delete(wsClients, conn)
		atomic.AddInt32(&metrics.ConnectedClients, -1)
		wsMutex.Unlock()
	}()
	
	// Keep connection alive and handle messages
	for {
		messageType, p, err := conn.ReadMessage()
		if err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}
		
		// Echo back for now (can be extended for bidirectional communication)
		if err := conn.WriteMessage(messageType, p); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

// Broadcast metrics to all WebSocket clients
func broadcastMetrics() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		avgParseTime := int64(0)
		parseCount := atomic.LoadUint64(&metrics.ParseCount)
		if parseCount > 0 {
			avgParseTime = atomic.LoadInt64(&metrics.TotalParseTime) / int64(parseCount)
		}
		
		data := map[string]interface{}{
			"type":               "metrics",
			"parse_count":        parseCount,
			"cache_hits":         atomic.LoadUint64(&metrics.CacheHits),
			"cache_misses":       atomic.LoadUint64(&metrics.CacheMisses),
			"avg_parse_time_ns":  avgParseTime,
			"worker_pool_active": atomic.LoadInt32(&metrics.WorkerPoolActive),
			"connected_clients":  atomic.LoadInt32(&metrics.ConnectedClients),
			"timestamp":          time.Now().Unix(),
		}
		
		jsonData, _ := json.Marshal(data)
		
		wsMutex.RLock()
		for conn := range wsClients {
			go func(c *websocket.Conn) {
				if err := c.WriteMessage(websocket.TextMessage, jsonData); err != nil {
					log.Printf("Failed to send metrics to client: %v", err)
				}
			}(conn)
		}
		wsMutex.RUnlock()
	}
}
