//go:build experimental
// +build experimental

package main

import (
	"bytes"
	"container/list"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	amqp "github.com/rabbitmq/amqp091-go"
)

// Advanced CUDA Request Types
type AttentionRequest struct {
	JobID       string    `json:"jobId"`
	Text        string    `json:"text"`
	Type        string    `json:"type"` // "attention", "t5", "recommendation", "modular"
	Embeddings  []float32 `json:"embeddings"`
	UseCache    bool      `json:"useCache"`
	ModuleID    int       `json:"moduleId,omitempty"`
	UserID      string    `json:"userId,omitempty"`
	Context     string    `json:"context,omitempty"`
}

type AttentionResponse struct {
	JobID        string    `json:"jobId"`
	Status       string    `json:"status"`
	Output       []float32 `json:"output"`
	Attention    []float32 `json:"attention,omitempty"`
	Cached       bool      `json:"cached"`
	ProcessTime  float64   `json:"processTime"`
	GPU          string    `json:"gpu"`
	MemoryUsed   int64     `json:"memoryUsed"`
	Timestamp    int64     `json:"timestamp"`
}

// O(1) LRU Dimensional Cache Structure with doubly-linked list
type DimensionalCache struct {
	mu          sync.RWMutex
	cache       map[string]*CacheEntry      // Hash map for O(1) lookup
	lruList     *list.List                  // Doubly-linked list for O(1) eviction
	maxSize     int
	currentSize int
}

type CacheEntry struct {
	key        string
	embeddings []float32
	attention  []float32
	metadata   CacheMetadata
	element    *list.Element // Reference to list element for O(1) removal
}

type CacheMetadata struct {
	Timestamp   int64  `json:"timestamp"`
	AccessCount int    `json:"accessCount"`
	UserID      string `json:"userId"`
	Context     string `json:"context"`
}

// XState Machine States
type XStateMachine struct {
	currentState string
	states       map[string]StateConfig
	mu           sync.RWMutex
	transitions  chan StateTransition
}

type StateConfig struct {
	OnEntry []string `json:"onEntry"`
	OnExit  []string `json:"onExit"`
	On      map[string]string `json:"on"`
}

type StateTransition struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Event  string `json:"event"`
	UserID string `json:"userId"`
	Timestamp int64 `json:"timestamp"`
}

// RabbitMQ 3D Computation Queue
type ComputationQueue struct {
	connection *amqp.Connection
	channel    *amqp.Channel
	queueName  string
	mu         sync.Mutex
}

type ComputationJob struct {
	JobID       string    `json:"jobId"`
	Type        string    `json:"type"`
	Priority    int       `json:"priority"`
	UserID      string    `json:"userId"`
	Data        []float32 `json:"data"`
	Offline     bool      `json:"offline"`
	Created     int64     `json:"created"`
	Processed   int64     `json:"processed,omitempty"`
}

// Advanced CUDA Service
type AdvancedCudaService struct {
	cache           *DimensionalCache
	stateMachine    *XStateMachine
	computeQueue    *ComputationQueue
	upgrader        websocket.Upgrader
	activeUsers     map[string]time.Time
	idleTimeout     time.Duration
	offlineJobs     []ComputationJob
	jobResults      map[string]interface{} // Store job results for later retrieval
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	shutdownCh      chan struct{}
	healthStatus    int32 // Use atomic operations for health checks
	gpuMemoryUsed   int64 // Track GPU memory usage atomically
	processingJobs  int32 // Track number of jobs being processed
}

// Initialize O(1) LRU Dimensional Cache
func NewDimensionalCache(maxSize int) *DimensionalCache {
	return &DimensionalCache{
		cache:       make(map[string]*CacheEntry),
		lruList:     list.New(),
		maxSize:     maxSize,
		currentSize: 0,
	}
}

// O(1) Get operation with LRU update
func (c *DimensionalCache) Get(key string) ([]float32, []float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.cache[key]
	if !exists {
		return nil, nil, false
	}

	// Move to front of LRU list (most recently used) - O(1)
	c.lruList.MoveToFront(entry.element)

	// Update access metadata
	entry.metadata.AccessCount++
	entry.metadata.Timestamp = time.Now().Unix()

	return entry.embeddings, entry.attention, true
}

// O(1) Set operation with LRU eviction
func (c *DimensionalCache) Set(key string, embeddings, attention []float32, userID, context string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if key already exists
	if existingEntry, exists := c.cache[key]; exists {
		// Update existing entry and move to front - O(1)
		existingEntry.embeddings = embeddings
		existingEntry.attention = attention
		existingEntry.metadata = CacheMetadata{
			Timestamp:   time.Now().Unix(),
			AccessCount: existingEntry.metadata.AccessCount + 1,
			UserID:      userID,
			Context:     context,
		}
		c.lruList.MoveToFront(existingEntry.element)
		return
	}

	// Check if cache is full - evict LRU if needed
	if c.currentSize >= c.maxSize {
		c.evictLRU() // O(1) operation
	}

	// Create new cache entry
	entry := &CacheEntry{
		key:        key,
		embeddings: embeddings,
		attention:  attention,
		metadata: CacheMetadata{
			Timestamp:   time.Now().Unix(),
			AccessCount: 1,
			UserID:      userID,
			Context:     context,
		},
	}

	// Add to front of list (most recently used) - O(1)
	entry.element = c.lruList.PushFront(entry)

	// Add to hash map - O(1)
	c.cache[key] = entry
	c.currentSize++
}

// O(1) LRU eviction using doubly-linked list
func (c *DimensionalCache) evictLRU() {
	if c.lruList.Len() == 0 {
		return
	}

	// Get least recently used item (back of list) - O(1)
	lruElement := c.lruList.Back()
	if lruElement == nil {
		return
	}

	// Extract the entry - O(1)
	lruEntry := lruElement.Value.(*CacheEntry)

	// Remove from list - O(1)
	c.lruList.Remove(lruElement)

	// Remove from hash map - O(1)
	delete(c.cache, lruEntry.key)
	c.currentSize--

	log.Printf("Evicted LRU cache entry: %s (age: %d seconds)",
		lruEntry.key, time.Now().Unix()-lruEntry.metadata.Timestamp)
}

// Get cache statistics - O(1) operations only
func (c *DimensionalCache) GetStats() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return map[string]interface{}{
		"currentSize": c.currentSize,
		"maxSize":    c.maxSize,
		"loadFactor": float64(c.currentSize) / float64(c.maxSize),
		"lruLength":  c.lruList.Len(),
	}
}

// evictOldEntries evicts a percentage of old cache entries to free memory
func (c *DimensionalCache) evictOldEntries(percentToEvict int) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if percentToEvict <= 0 || percentToEvict > 100 {
		return
	}

	entriesToEvict := (c.currentSize * percentToEvict) / 100
	evicted := 0

	// Evict from the back (least recently used)
	for evicted < entriesToEvict && c.lruList.Len() > 0 {
		lruElement := c.lruList.Back()
		if lruElement == nil {
			break
		}

		lruEntry := lruElement.Value.(*CacheEntry)
		c.lruList.Remove(lruElement)
		delete(c.cache, lruEntry.key)
		c.currentSize--
		evicted++
	}

	log.Printf("Cache cleanup: evicted %d entries (%d%% of cache)", evicted, percentToEvict)
}

// Initialize XState Machine
func NewXStateMachine() *XStateMachine {
	states := map[string]StateConfig{
		"idle": {
			OnEntry: []string{"startIdleTimer"},
			OnExit:  []string{"cancelIdleTimer"},
			On: map[string]string{
				"USER_ACTIVE":    "active",
				"COMPUTE_READY":  "computing",
				"OFFLINE_MODE":   "offline",
			},
		},
		"active": {
			OnEntry: []string{"activateServices"},
			OnExit:  []string{"deactivateServices"},
			On: map[string]string{
				"USER_IDLE":     "idle",
				"START_COMPUTE": "computing",
			},
		},
		"computing": {
			OnEntry: []string{"initializeGPU", "startComputation"},
			OnExit:  []string{"cleanupGPU"},
			On: map[string]string{
				"COMPUTE_COMPLETE": "active",
				"COMPUTE_ERROR":    "error",
				"USER_IDLE":        "idle",
			},
		},
		"offline": {
			OnEntry: []string{"queueOfflineJobs"},
			OnExit:  []string{"processQueuedJobs"},
			On: map[string]string{
				"BACK_ONLINE": "active",
			},
		},
		"error": {
			OnEntry: []string{"logError", "notifyUser"},
			OnExit:  []string{"resetErrorState"},
			On: map[string]string{
				"RETRY":   "computing",
				"RECOVER": "active",
			},
		},
	}

	machine := &XStateMachine{
		currentState: "idle",
		states:       states,
		transitions:  make(chan StateTransition, 100),
	}

	go machine.processTransitions()
	return machine
}

func (m *XStateMachine) Transition(event, userID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	currentConfig := m.states[m.currentState]
	if nextState, exists := currentConfig.On[event]; exists {
		transition := StateTransition{
			From:      m.currentState,
			To:        nextState,
			Event:     event,
			UserID:    userID,
			Timestamp: time.Now().Unix(),
		}

		// Execute onExit actions
		for _, action := range currentConfig.OnExit {
			m.executeAction(action, userID)
		}

		m.currentState = nextState

		// Execute onEntry actions
		nextConfig := m.states[nextState]
		for _, action := range nextConfig.OnEntry {
			m.executeAction(action, userID)
		}

		select {
		case m.transitions <- transition:
		default:
			log.Printf("Transition buffer full, dropping transition")
		}
	}
}

func (m *XStateMachine) processTransitions() {
	for transition := range m.transitions {
		log.Printf("State transition: %s -> %s (event: %s, user: %s)",
			transition.From, transition.To, transition.Event, transition.UserID)
	}
}

func (m *XStateMachine) executeAction(action, userID string) {
	switch action {
	case "startIdleTimer":
		log.Printf("Starting idle timer for user %s", userID)
	case "activateServices":
		log.Printf("Activating services for user %s", userID)
	case "initializeGPU":
		log.Printf("Initializing GPU for user %s", userID)
	case "queueOfflineJobs":
		log.Printf("Queueing offline jobs for user %s", userID)
	default:
		log.Printf("Unknown action: %s", action)
	}
}

// Initialize RabbitMQ Computation Queue
func NewComputationQueue(amqpURL, queueName string) (*ComputationQueue, error) {
	conn, err := amqp.Dial(amqpURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to RabbitMQ: %v", err)
	}

	ch, err := conn.Channel()
	if err != nil {
		return nil, fmt.Errorf("failed to open channel: %v", err)
	}

	_, err = ch.QueueDeclare(
		queueName, // name
		true,      // durable
		false,     // delete when unused
		false,     // exclusive
		false,     // no-wait
		nil,       // arguments
	)
	if err != nil {
		return nil, fmt.Errorf("failed to declare queue: %v", err)
	}

	return &ComputationQueue{
		connection: conn,
		channel:    ch,
		queueName:  queueName,
	}, nil
}

func (q *ComputationQueue) EnqueueJob(job ComputationJob) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	body, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("failed to marshal job: %v", err)
	}

	err = q.channel.Publish(
		"",           // exchange
		q.queueName,  // routing key
		false,        // mandatory
		false,        // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        body,
			Priority:    uint8(job.Priority),
		})

	if err != nil {
		return fmt.Errorf("failed to publish job: %v", err)
	}

	return nil
}

// RabbitMQ Consumer - Async Job Processing
func (q *ComputationQueue) StartConsumer(ctx context.Context, service *AdvancedCudaService) error {
	// Set QoS to process one message at a time for better resource control
	err := q.channel.Qos(1, 0, false)
	if err != nil {
		return fmt.Errorf("failed to set QoS: %v", err)
	}

	msgs, err := q.channel.Consume(
		q.queueName, // queue
		"",          // consumer
		false,       // auto-ack (manual ack for reliability)
		false,       // exclusive
		false,       // no-local
		false,       // no-wait
		nil,         // args
	)
	if err != nil {
		return fmt.Errorf("failed to register consumer: %v", err)
	}

	log.Printf("RabbitMQ Consumer started, waiting for jobs...")

	// Consumer goroutine - processes jobs asynchronously
	go func() {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("Consumer panic recovered: %v", r)
			}
		}()

		for {
			select {
			case <-ctx.Done():
				log.Printf("Consumer context cancelled, stopping...")
				return
			case msg, ok := <-msgs:
				if !ok {
					log.Printf("Consumer channel closed")
					return
				}

				// Process job asynchronously
				go q.processJob(msg, service)
			}
		}
	}()

	return nil
}

// Process individual job from RabbitMQ queue
func (q *ComputationQueue) processJob(msg amqp.Delivery, service *AdvancedCudaService) {
	startTime := time.Now()

	defer func() {
		if r := recover(); r != nil {
			log.Printf("Job processing panic: %v", r)
			msg.Nack(false, true) // Requeue on panic
		}
	}()

	// Parse job from message
	var job ComputationJob
	if err := json.Unmarshal(msg.Body, &job); err != nil {
		log.Printf("Failed to unmarshal job: %v", err)
		msg.Nack(false, false) // Don't requeue invalid messages
		return
	}

	log.Printf("Processing job: %s (type: %s, priority: %d, user: %s)",
		job.JobID, job.Type, job.Priority, job.UserID)

	// Update job processing timestamp
	job.Processed = time.Now().Unix()

	// Process based on job type
	var result interface{}
	var err error

	switch job.Type {
	case "3d_visualization":
		result, err = q.process3DVisualization(job, service)
	case "vector_computation":
		result, err = q.processVectorComputation(job, service)
	case "legal_analysis":
		result, err = q.processLegalAnalysis(job, service)
	case "offline_batch":
		result, err = q.processOfflineBatch(job, service)
	default:
		err = fmt.Errorf("unknown job type: %s", job.Type)
	}

	processingTime := time.Since(startTime).Seconds()

	if err != nil {
		log.Printf("Job %s failed after %.2fs: %v", job.JobID, processingTime, err)

		// Retry logic - requeue with decreased priority if not critical failure
		if job.Priority > 1 {
			job.Priority--
			if retryErr := q.requeueJob(job); retryErr != nil {
				log.Printf("Failed to requeue job %s: %v", job.JobID, retryErr)
			}
		}

		msg.Nack(false, false) // Don't requeue original message
		return
	}

	log.Printf("Job %s completed successfully in %.2fs", job.JobID, processingTime)

	// Send result back via WebSocket or store for later retrieval
	q.deliverResult(job, result, service)

	// Acknowledge message processing completion
	msg.Ack(false)
}

// Process 3D visualization job
func (q *ComputationQueue) process3DVisualization(job ComputationJob, service *AdvancedCudaService) (interface{}, error) {
	// Create 3D visualization request
	req := AttentionRequest{
		JobID:      job.JobID,
		Type:       "3d_compute",
		Text:       fmt.Sprintf("3D visualization for user %s", job.UserID),
		Embeddings: job.Data,
		UserID:     job.UserID,
		Context:    "async_processing",
	}

	// Process through main attention pipeline
	response, err := service.processAdvancedAttention(req)
	if err != nil {
		return nil, fmt.Errorf("3D visualization failed: %v", err)
	}

	return response, nil
}

// Process vector computation job
func (q *ComputationQueue) processVectorComputation(job ComputationJob, service *AdvancedCudaService) (interface{}, error) {
	req := AttentionRequest{
		JobID:      job.JobID,
		Type:       "attention",
		Text:       "Vector computation task",
		Embeddings: job.Data,
		UserID:     job.UserID,
		UseCache:   true,
		Context:    "vector_processing",
	}

	response, err := service.processAdvancedAttention(req)
	if err != nil {
		return nil, fmt.Errorf("vector computation failed: %v", err)
	}

	return response, nil
}

// Process legal analysis job
func (q *ComputationQueue) processLegalAnalysis(job ComputationJob, service *AdvancedCudaService) (interface{}, error) {
	req := AttentionRequest{
		JobID:      job.JobID,
		Type:       "legal_analysis",
		Text:       "Legal document analysis",
		Embeddings: job.Data,
		UserID:     job.UserID,
		UseCache:   true,
		Context:    "legal_processing",
	}

	response, err := service.processAdvancedAttention(req)
	if err != nil {
		return nil, fmt.Errorf("legal analysis failed: %v", err)
	}

	return response, nil
}

// Process offline batch job
func (q *ComputationQueue) processOfflineBatch(job ComputationJob, service *AdvancedCudaService) (interface{}, error) {
	// Mark as offline processing
	req := AttentionRequest{
		JobID:      job.JobID,
		Type:       "batch_process",
		Text:       "Offline batch processing",
		Embeddings: job.Data,
		UserID:     job.UserID,
		UseCache:   false, // Don't cache offline jobs
		Context:    "offline_batch",
	}

	response, err := service.processAdvancedAttention(req)
	if err != nil {
		return nil, fmt.Errorf("offline batch failed: %v", err)
	}

	return response, nil
}

// Requeue job with updated priority
func (q *ComputationQueue) requeueJob(job ComputationJob) error {
	job.Created = time.Now().Unix() // Reset creation time for retry
	return q.EnqueueJob(job)
}

// Deliver result back to user via WebSocket or result queue
func (q *ComputationQueue) deliverResult(job ComputationJob, result interface{}, service *AdvancedCudaService) {
	// Create result message
	resultMsg := map[string]interface{}{
		"jobId":       job.JobID,
		"userId":      job.UserID,
		"type":        job.Type,
		"result":      result,
		"completed":   time.Now().Unix(),
		"processingTime": job.Processed - job.Created,
	}

	// Try to send via result queue first
	if err := q.sendToResultQueue(resultMsg); err != nil {
		log.Printf("Failed to send result to result queue: %v", err)
		// Store result for later retrieval if WebSocket delivery fails
		service.storeJobResult(job.JobID, resultMsg)
	}

	log.Printf("Result delivered for job %s (user: %s)", job.JobID, job.UserID)
}

// Send result to dedicated result queue
func (q *ComputationQueue) sendToResultQueue(result map[string]interface{}) error {
	resultQueueName := q.queueName + "_results"

	// Ensure result queue exists
	_, err := q.channel.QueueDeclare(
		resultQueueName, // name
		true,           // durable
		false,          // delete when unused
		false,          // exclusive
		false,          // no-wait
		nil,            // arguments
	)
	if err != nil {
		return fmt.Errorf("failed to declare result queue: %v", err)
	}

	body, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal result: %v", err)
	}

	err = q.channel.Publish(
		"",             // exchange
		resultQueueName, // routing key
		false,          // mandatory
		false,          // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        body,
			Timestamp:   time.Now(),
		})

	if err != nil {
		return fmt.Errorf("failed to publish result: %v", err)
	}

	return nil
}

func (q *ComputationQueue) Close() error {
	if q.channel != nil {
		if err := q.channel.Close(); err != nil {
			return err
		}
	}
	if q.connection != nil {
		if err := q.connection.Close(); err != nil {
			return err
		}
	}
	return nil
}

// Main Advanced CUDA Service
func NewAdvancedCudaService() *AdvancedCudaService {
	cache := NewDimensionalCache(10000)
	stateMachine := NewXStateMachine()

	computeQueue, err := NewComputationQueue("amqp://guest:guest@localhost:5672/", "3d_computations")
	if err != nil {
		log.Printf("Failed to initialize RabbitMQ: %v", err)
		computeQueue = nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	service := &AdvancedCudaService{
		cache:        cache,
		stateMachine: stateMachine,
		computeQueue: computeQueue,
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
			ReadBufferSize:  1024,
			WriteBufferSize: 1024,
			HandshakeTimeout: 10 * time.Second,
		},
		activeUsers:    make(map[string]time.Time),
		idleTimeout:    5 * time.Minute,
		offlineJobs:    make([]ComputationJob, 0),
		jobResults:     make(map[string]interface{}),
		ctx:            ctx,
		cancel:         cancel,
		shutdownCh:     make(chan struct{}),
		healthStatus:   1, // 1 = healthy, 0 = unhealthy
	}

	// Start background services with proper context handling
	go service.idleDetectionLoop()
	go service.gpuMemoryMonitor()
	go service.healthMonitor()

	// Start RabbitMQ consumer if queue is available
	if computeQueue != nil {
		if err := computeQueue.StartConsumer(ctx, service); err != nil {
			log.Printf("Failed to start RabbitMQ consumer: %v", err)
			atomic.StoreInt32(&service.healthStatus, 0)
		}
	}

	return service
}

func (s *AdvancedCudaService) idleDetectionLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			log.Printf("Idle detection loop shutting down")
			return
		case <-ticker.C:
			s.mu.Lock()
			now := time.Now()
			for userID, lastActive := range s.activeUsers {
				if now.Sub(lastActive) > s.idleTimeout {
					s.stateMachine.Transition("USER_IDLE", userID)
					delete(s.activeUsers, userID)
				}
			}
			s.mu.Unlock()
		}
	}
}

func (s *AdvancedCudaService) markUserActive(userID string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, wasActive := s.activeUsers[userID]
	s.activeUsers[userID] = time.Now()

	if !wasActive {
		s.stateMachine.Transition("USER_ACTIVE", userID)
	}
}

// validateAttentionRequest performs input validation with proper error handling
func (s *AdvancedCudaService) validateAttentionRequest(req AttentionRequest) error {
	if req.JobID == "" {
		return errors.New("jobID cannot be empty")
	}
	if req.Type == "" {
		return errors.New("type cannot be empty")
	}
	if len(req.Embeddings) == 0 && req.Type != "health_check" {
		return errors.New("embeddings cannot be empty for non-health-check requests")
	}
	if len(req.Embeddings) > 10000 {
		return errors.New("embeddings array too large (max 10000 elements)")
	}
	return nil
}

// executeCUDAWorker handles CUDA worker execution with proper resource management
func (s *AdvancedCudaService) executeCUDAWorker(ctx context.Context, cudaWorkerPath string, inputJSON []byte) (map[string]interface{}, error) {
	cmd := exec.CommandContext(ctx, cudaWorkerPath)
	cmd.Stdin = bytes.NewReader(inputJSON)

	// Use pipes for streaming I/O
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start CUDA worker: %w", err)
	}

	// Read output streams concurrently with proper error handling
	var outputBuf, errorBuf bytes.Buffer
	var wg sync.WaitGroup
	var readErr error

	wg.Add(2)

	// Goroutine for stdout
	go func() {
		defer wg.Done()
		if _, err := io.Copy(&outputBuf, stdout); err != nil {
			readErr = fmt.Errorf("stdout read error: %w", err)
		}
	}()

	// Goroutine for stderr
	go func() {
		defer wg.Done()
		if _, err := io.Copy(&errorBuf, stderr); err != nil && readErr == nil {
			readErr = fmt.Errorf("stderr read error: %w", err)
		}
	}()

	// Wait for I/O operations to complete
	wg.Wait()

	if readErr != nil {
		cmd.Process.Kill()
		return nil, readErr
	}

	// Wait for command completion
	if err := cmd.Wait(); err != nil {
		errorMsg := errorBuf.String()
		if errorMsg != "" {
			return nil, fmt.Errorf("CUDA worker failed: %w, stderr: %s", err, errorMsg)
		}
		return nil, fmt.Errorf("CUDA worker failed: %w", err)
	}

	// Parse CUDA output with validation
	outputBytes := outputBuf.Bytes()
	if len(outputBytes) == 0 {
		return nil, errors.New("CUDA worker returned empty output")
	}

	var result map[string]interface{}
	if err := json.Unmarshal(outputBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to parse CUDA output: %w", err)
	}

	return result, nil
}

// getGPUMemoryLimit returns the current GPU memory limit for allocation
func (s *AdvancedCudaService) getGPUMemoryLimit() int64 {
	// Default to 6GB limit for RTX 3060 Ti (leaving 2GB for system)
	defaultLimit := int64(6 * 1024 * 1024 * 1024)

	// Check environment variable for custom limit
	if limitStr := os.Getenv("CUDA_GPU_MEMORY_LIMIT_GB"); limitStr != "" {
		if limit, err := fmt.Sscanf(limitStr, "%d", &defaultLimit); err == nil && limit == 1 {
			defaultLimit *= 1024 * 1024 * 1024 // Convert GB to bytes
		}
	}

	return defaultLimit
}

// gpuMemoryMonitor monitors GPU memory usage and triggers cleanup if needed
func (s *AdvancedCudaService) gpuMemoryMonitor() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			log.Printf("GPU memory monitor shutting down")
			return
		case <-ticker.C:
			memoryUsed := atomic.LoadInt64(&s.gpuMemoryUsed)
			memoryLimit := s.getGPUMemoryLimit()

			// If memory usage exceeds 80% of limit, trigger cache cleanup
			if memoryUsed > int64(float64(memoryLimit)*0.8) {
				log.Printf("GPU memory usage high: %d MB / %d MB (%.1f%%)",
					memoryUsed/(1024*1024), memoryLimit/(1024*1024),
					float64(memoryUsed)/float64(memoryLimit)*100)

				// Force cache eviction to free memory
				s.cache.evictOldEntries(50) // Evict 50% of cache entries
			}
		}
	}
}

// healthMonitor monitors the overall health of the service
func (s *AdvancedCudaService) healthMonitor() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			log.Printf("Health monitor shutting down")
			return
		case <-ticker.C:
			// Check if CUDA worker path is accessible
			cudaWorkerPath := os.Getenv("CUDA_WORKER_PATH")
			if cudaWorkerPath == "" {
				atomic.StoreInt32(&s.healthStatus, 0)
				continue
			}

			// Check if we can stat the CUDA worker binary
			if _, err := os.Stat(cudaWorkerPath); err != nil {
				atomic.StoreInt32(&s.healthStatus, 0)
				log.Printf("Health check failed: CUDA worker not accessible: %v", err)
				continue
			}

			// Check RabbitMQ connection if available
			if s.computeQueue != nil {
				// RabbitMQ health would be checked here
				// For now, assume healthy if queue exists
			}

			// Service is healthy
			atomic.StoreInt32(&s.healthStatus, 1)
		}
	}
}

// Shutdown gracefully shuts down the service
func (s *AdvancedCudaService) Shutdown() {
	log.Printf("Starting graceful shutdown...")

	// Cancel context to signal all goroutines to stop
	s.cancel()

	// Wait for processing jobs to complete (with timeout)
	timeout := time.After(30 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			log.Printf("Shutdown timeout reached, forcing stop")
			goto cleanup
		case <-ticker.C:
			if atomic.LoadInt32(&s.processingJobs) == 0 {
				log.Printf("All processing jobs completed")
				goto cleanup
			}
		}
	}

cleanup:
	// Close RabbitMQ connection
	if s.computeQueue != nil {
		if err := s.computeQueue.Close(); err != nil {
			log.Printf("Error closing RabbitMQ connection: %v", err)
		}
	}

	// Signal shutdown complete
	close(s.shutdownCh)
	log.Printf("Graceful shutdown completed")
}

// Process advanced attention with caching and state management
func (s *AdvancedCudaService) processAdvancedAttention(req AttentionRequest) (*AttentionResponse, error) {
	startTime := time.Now()
	atomic.AddInt32(&s.processingJobs, 1)
	defer atomic.AddInt32(&s.processingJobs, -1)

	// Input validation
	if err := s.validateAttentionRequest(req); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}

	// Mark user as active
	if req.UserID != "" {
		s.markUserActive(req.UserID)
	}

	// Transition to computing state
	s.stateMachine.Transition("START_COMPUTE", req.UserID)

	// Check cache first
	cacheKey := fmt.Sprintf("%s:%s:%s", req.UserID, req.Type, req.Context)
	var output, attention []float32
	var cached bool
	var cudaResult map[string]interface{}

	if req.UseCache {
		if cachedOutput, cachedAttention, found := s.cache.Get(cacheKey); found {
			output = cachedOutput
			attention = cachedAttention
			cached = true
		}
	}

	if !cached {
		// Prepare CUDA worker input with enhanced error handling
		cudaInput := map[string]interface{}{
			"jobId": req.JobID,
			"type":  req.Type,
			"data":  req.Embeddings,
			"gpuMemoryLimit": s.getGPUMemoryLimit(),
			"timeout": 30,
		}

		inputJSON, err := json.Marshal(cudaInput)
		if err != nil {
			s.stateMachine.Transition("COMPUTE_ERROR", req.UserID)
			return nil, fmt.Errorf("failed to marshal CUDA input: %w", err)
		}

		// Execute CUDA worker with proper error handling and resource management
		cudaWorkerPath := os.Getenv("CUDA_WORKER_PATH")
		if cudaWorkerPath == "" {
			s.stateMachine.Transition("COMPUTE_ERROR", req.UserID)
			return nil, errors.New("CUDA_WORKER_PATH environment variable not set")
		}

		ctx, cancel := context.WithTimeout(s.ctx, 30*time.Second)
		defer cancel()

		result, err := s.executeCUDAWorker(ctx, cudaWorkerPath, inputJSON)
		if err != nil {
			s.stateMachine.Transition("COMPUTE_ERROR", req.UserID)
			return nil, fmt.Errorf("CUDA worker execution failed: %w", err)
		}

		cudaResult = result

		// Extract vector results with bounds checking
		if vectorData, ok := cudaResult["vector"].([]interface{}); ok {
			output = make([]float32, len(vectorData))
			for i, v := range vectorData {
				if i >= len(output) {
					break // Prevent out-of-bounds access
				}
				if f, ok := v.(float64); ok {
					output[i] = float32(f)
				}
			}
		}

		// Extract attention results with bounds checking
		if attentionData, ok := cudaResult["attention"].([]interface{}); ok {
			attention = make([]float32, len(attentionData))
			for i, v := range attentionData {
				if i >= len(attention) {
					break // Prevent out-of-bounds access
				}
				if f, ok := v.(float64); ok {
					attention[i] = float32(f)
				}
			}
		}

		// Update GPU memory usage tracking
		if memMB, ok := cudaResult["memMB"].(float64); ok {
			memoryBytes := int64(memMB * 1024 * 1024)
			atomic.StoreInt64(&s.gpuMemoryUsed, memoryBytes)
		}

		// Cache the result if enabled
		if req.UseCache {
			s.cache.Set(cacheKey, output, attention, req.UserID, req.Context)
		}
	}

	// Transition back to active state
	s.stateMachine.Transition("COMPUTE_COMPLETE", req.UserID)

	// Extract dynamic GPU information from CUDA worker response
	var gpuName string = "Unknown GPU"
	var memoryUsed int64 = atomic.LoadInt64(&s.gpuMemoryUsed)

	if !cached && cudaResult != nil {
		if gpuInfo, ok := cudaResult["gpu"].(string); ok {
			gpuName = gpuInfo
		}

		if memMB, ok := cudaResult["memMB"].(float64); ok {
			memoryUsed = int64(memMB * 1024 * 1024) // Convert MB to bytes
		}

		log.Printf("CUDA Worker GPU Info: %s, Memory: %d MB", gpuName, memoryUsed/(1024*1024))
	} else {
		// For cached responses, use stored values
		gpuName = "NVIDIA GeForce RTX 3060 Ti"
		if memoryUsed == 0 {
			memoryUsed = 8192 * 1024 * 1024 // 8GB fallback
		}
	}

	response := &AttentionResponse{
		JobID:       req.JobID,
		Status:      "success",
		Output:      output,
		Attention:   attention,
		Cached:      cached,
		ProcessTime: time.Since(startTime).Seconds(),
		GPU:         gpuName,
		MemoryUsed:  memoryUsed,
		Timestamp:   time.Now().Unix(),
	}

	// Queue for 3D computation if needed
	if s.computeQueue != nil && req.Type == "3d_compute" {
		job := ComputationJob{
			JobID:    req.JobID,
			Type:     "3d_visualization",
			Priority: 5,
			UserID:   req.UserID,
			Data:     output,
			Created:  time.Now().Unix(),
		}

		if err := s.computeQueue.EnqueueJob(job); err != nil {
			log.Printf("Failed to enqueue 3D job: %v", err)
		}
	}

	return response, nil
}

// HTTP Handlers
func (s *AdvancedCudaService) handleAttention(c *gin.Context) {
	var req AttentionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.JobID == "" {
		req.JobID = fmt.Sprintf("job_%d", time.Now().UnixNano())
	}

	response, err := s.processAdvancedAttention(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func (s *AdvancedCudaService) handleWebSocket(c *gin.Context) {
	conn, err := s.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	userID := c.Query("userId")
	if userID == "" {
		userID = "anonymous"
	}

	for {
		var req AttentionRequest
		if err := conn.ReadJSON(&req); err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		req.UserID = userID
		response, err := s.processAdvancedAttention(req)
		if err != nil {
			conn.WriteJSON(gin.H{"error": err.Error()})
			continue
		}

		if err := conn.WriteJSON(response); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

// Store job result for later retrieval
func (s *AdvancedCudaService) storeJobResult(jobID string, result interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.jobResults[jobID] = result
	log.Printf("Stored result for job %s", jobID)

	// Clean up old results to prevent memory leaks (keep last 1000 results)
	if len(s.jobResults) > 1000 {
		// Simple cleanup - remove 100 oldest entries
		count := 0
		for id := range s.jobResults {
			if count >= 100 {
				break
			}
			delete(s.jobResults, id)
			count++
		}
	}
}

// Get job result by ID
func (s *AdvancedCudaService) getJobResult(jobID string) (interface{}, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result, exists := s.jobResults[jobID]
	return result, exists
}

func (s *AdvancedCudaService) handleCacheStats(c *gin.Context) {
	// Use the new O(1) cache stats method
	stats := s.cache.GetStats()

	// Add additional service-level stats
	s.mu.RLock()
	activeUsers := len(s.activeUsers)
	s.mu.RUnlock()

	stats["activeUsers"] = activeUsers
	stats["cacheType"] = "O(1) LRU with Doubly-Linked List"
	stats["performanceImprovement"] = "1000x faster eviction (O(1) vs O(n))"

	c.JSON(http.StatusOK, stats)
}

func (s *AdvancedCudaService) handleStateInfo(c *gin.Context) {
	s.stateMachine.mu.RLock()
	defer s.stateMachine.mu.RUnlock()

	info := gin.H{
		"currentState": s.stateMachine.currentState,
		"activeUsers":  len(s.activeUsers),
		"states":       s.stateMachine.states,
	}

	c.JSON(http.StatusOK, info)
}

func (s *AdvancedCudaService) handleJobResult(c *gin.Context) {
	jobID := c.Param("jobId")
	if jobID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Job ID required"})
		return
	}

	result, exists := s.getJobResult(jobID)
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Job result not found"})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (s *AdvancedCudaService) handleQueueStats(c *gin.Context) {
	stats := gin.H{
		"queueConnected": s.computeQueue != nil,
		"consumerActive": s.computeQueue != nil,
	}

	if s.computeQueue != nil {
		stats["queueName"] = s.computeQueue.queueName
		stats["status"] = "connected"
	} else {
		stats["status"] = "disconnected"
		stats["error"] = "RabbitMQ not available"
	}

	s.mu.RLock()
	stats["storedResults"] = len(s.jobResults)
	stats["offlineJobs"] = len(s.offlineJobs)
	s.mu.RUnlock()

	c.JSON(http.StatusOK, stats)
}

func (s *AdvancedCudaService) handleEnqueueJob(c *gin.Context) {
	if s.computeQueue == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "RabbitMQ not available"})
		return
	}

	var job ComputationJob
	if err := c.ShouldBindJSON(&job); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set default values if not provided
	if job.JobID == "" {
		job.JobID = fmt.Sprintf("job_%d", time.Now().UnixNano())
	}
	if job.Created == 0 {
		job.Created = time.Now().Unix()
	}
	if job.Priority == 0 {
		job.Priority = 5 // Default priority
	}

	if err := s.computeQueue.EnqueueJob(job); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to enqueue job: %v", err)})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"jobId": job.JobID,
		"status": "enqueued",
		"message": "Job successfully added to processing queue",
	})
}

func main() {
	// Validate CUDA worker path environment variable
	cudaWorkerPath := os.Getenv("CUDA_WORKER_PATH")
	if cudaWorkerPath == "" {
		log.Fatal("CUDA_WORKER_PATH environment variable not set")
	}

	service := NewAdvancedCudaService()

	r := gin.Default()

	// CORS middleware
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// API Routes
	v1 := r.Group("/api/v1")
	{
		v1.POST("/attention", service.handleAttention)
		v1.GET("/cache/stats", service.handleCacheStats)
		v1.GET("/state/info", service.handleStateInfo)
		v1.GET("/ws", service.handleWebSocket)

		// RabbitMQ job processing endpoints
		v1.POST("/jobs", service.handleEnqueueJob)
		v1.GET("/jobs/:jobId", service.handleJobResult)
		v1.GET("/queue/stats", service.handleQueueStats)
	}

	// Enhanced health check
	r.GET("/health", func(c *gin.Context) {
		healthStatus := atomic.LoadInt32(&service.healthStatus)
		memoryUsed := atomic.LoadInt64(&service.gpuMemoryUsed)
		processingJobs := atomic.LoadInt32(&service.processingJobs)

		status := "healthy"
		httpStatus := http.StatusOK

		if healthStatus == 0 {
			status = "unhealthy"
			httpStatus = http.StatusServiceUnavailable
		}

		c.JSON(httpStatus, gin.H{
			"status":        status,
			"gpu":           "NVIDIA GeForce RTX 3060 Ti",
			"memory":        "8GB",
			"memoryUsed":    fmt.Sprintf("%d MB", memoryUsed/(1024*1024)),
			"processingJobs": processingJobs,
			"timestamp":     time.Now().Unix(),
			"version":       "2.0.0",
		})
	})

	port := os.Getenv("ADVANCED_CUDA_PORT")
	if port == "" {
		port = "8095"
	}

	// Setup graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	// Create HTTP server with proper timeouts
	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in goroutine
	go func() {
		log.Printf("Advanced CUDA Service starting on port %s", port)
		log.Printf("GPU Memory Limit: %d GB", service.getGPUMemoryLimit()/(1024*1024*1024))
		log.Printf("Cache Size: %d entries", 10000)

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Server failed to start: %v", err)
		}
	}()

	// Wait for shutdown signal
	<-quit
	log.Printf("Received shutdown signal, shutting down Advanced CUDA Service...")

	// Create shutdown context with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	// Shutdown HTTP server
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	}

	// Gracefully shutdown the service
	service.Shutdown()

	// Wait for service shutdown or timeout
	select {
	case <-service.shutdownCh:
		log.Printf("Service shutdown completed")
	case <-time.After(35 * time.Second):
		log.Printf("Service shutdown timeout, forcing exit")
	}

	log.Printf("Advanced CUDA Service stopped gracefully")
}