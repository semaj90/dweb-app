// Vector Processing Service with Redis Streams + CUDA Worker Integration
// Replaces BullMQ with Redis Streams for vector job processing
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/gorilla/mux"
	"github.com/jackc/pgx/v4/pgxpool"
	amqp "github.com/rabbitmq/amqp091-go"
)

type VectorService struct {
	redisClient  *redis.Client
	pgPool       *pgxpool.Pool
	rabbitConn   *amqp.Connection
	rabbitCh     *amqp.Channel
	cudaWorker   *CUDAWorker
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
}

type CUDAWorker struct {
	executablePath string
	mu             sync.Mutex
}

type VectorJob struct {
	ID        string                 `json:"id"`
	OwnerType string                 `json:"owner_type"` // 'evidence' | 'report' | 'case' | 'document'
	OwnerID   string                 `json:"owner_id"`
	Event     string                 `json:"event"`      // 'upsert' | 'delete' | 'reembed'
	Vector    []float64              `json:"vector,omitempty"`
	Payload   map[string]interface{} `json:"payload,omitempty"`
	Priority  string                 `json:"priority"`   // 'high' | 'medium' | 'low'
	CreatedAt time.Time              `json:"created_at"`
	Status    string                 `json:"status"`     // 'pending' | 'processing' | 'completed' | 'failed'
}

type CUDARequest struct {
	JobID string    `json:"jobId"`
	Type  string    `json:"type"` // 'embedding' | 'similarity' | 'autoindex' | 'som_train'
	Data  []float64 `json:"data"`
}

type CUDAResponse struct {
	JobID     string    `json:"jobId"`
	Type      string    `json:"type"`
	Vector    []float64 `json:"vector"`
	Status    string    `json:"status"`
	Timestamp int64     `json:"timestamp"`
	Error     string    `json:"error,omitempty"`
}

func NewVectorService() (*VectorService, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Initialize Redis client
	redisClient := redis.NewClient(&redis.Options{
		Addr:     getEnv("REDIS_URL", "localhost:6379"),
		Password: "",
		DB:       0,
	})

	// Test Redis connection
	_, err := redisClient.Ping(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %v", err)
	}

	// Initialize PostgreSQL connection
	dbURL := getEnv("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/legal_ai_db")
	pgPool, err := pgxpool.Connect(ctx, dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to PostgreSQL: %v", err)
	}

	// Initialize RabbitMQ connection
	rabbitURL := getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
	rabbitConn, err := amqp.Dial(rabbitURL)
	if err != nil {
		log.Printf("Warning: RabbitMQ connection failed: %v. Continuing without fanout.", err)
	}

	var rabbitCh *amqp.Channel
	if rabbitConn != nil {
		rabbitCh, err = rabbitConn.Channel()
		if err != nil {
			log.Printf("Warning: RabbitMQ channel creation failed: %v", err)
		}
	}

	// Initialize CUDA Worker
	cudaPath := getEnv("CUDA_WORKER_PATH", "../cuda-worker/cuda-worker.exe")
	if _, err := os.Stat(cudaPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("CUDA worker not found at %s", cudaPath)
	}

	cudaWorker := &CUDAWorker{
		executablePath: cudaPath,
	}

	service := &VectorService{
		redisClient: redisClient,
		pgPool:      pgPool,
		rabbitConn:  rabbitConn,
		rabbitCh:    rabbitCh,
		cudaWorker:  cudaWorker,
		ctx:         ctx,
		cancel:      cancel,
	}

	return service, nil
}

func (s *VectorService) Start() error {
	// Create Redis Streams for vector processing
	streams := []string{
		"vector:embeddings",    // New embeddings to generate
		"vector:similarities",  // Similarity calculations
		"vector:indexing",     // Auto-indexing operations
		"vector:clustering",   // SOM clustering operations
	}

	for _, stream := range streams {
		// Create consumer group if it doesn't exist
		s.redisClient.XGroupCreate(s.ctx, stream, "vector-processors", "0")
	}

	// Setup RabbitMQ exchanges and queues if available
	if s.rabbitCh != nil {
		s.setupRabbitMQ()
	}

	// Start processing workers for each stream
	for _, stream := range streams {
		s.wg.Add(1)
		go s.processStream(stream)
	}

	// Start outbox processor
	s.wg.Add(1)
	go s.processOutbox()

	// Start HTTP API server
	s.wg.Add(1)
	go s.startAPIServer()

	log.Println("🚀 Vector Redis Service started successfully")
	log.Println("📊 Processing streams:", strings.Join(streams, ", "))
	log.Println("🔥 CUDA worker ready at:", s.cudaWorker.executablePath)

	return nil
}

func (s *VectorService) setupRabbitMQ() error {
	// Declare exchange for vector updates
	err := s.rabbitCh.ExchangeDeclare(
		"vector_updates", // name
		"fanout",        // type
		true,            // durable
		false,           // auto-deleted
		false,           // internal
		false,           // no-wait
		nil,             // arguments
	)
	if err != nil {
		return fmt.Errorf("failed to declare exchange: %v", err)
	}

	log.Println("✅ RabbitMQ exchange 'vector_updates' ready")
	return nil
}

func (s *VectorService) processStream(streamName string) {
	defer s.wg.Done()

	for {
		select {
		case <-s.ctx.Done():
			return
		default:
			// Read from Redis Stream with consumer group
			results, err := s.redisClient.XReadGroup(s.ctx, &redis.XReadGroupArgs{
				Group:    "vector-processors",
				Consumer: fmt.Sprintf("worker-%s", streamName),
				Streams:  []string{streamName, ">"},
				Count:    1,
				Block:    time.Second * 5,
			}).Result()

			if err != nil {
				if err != redis.Nil {
					log.Printf("Error reading from stream %s: %v", streamName, err)
				}
				continue
			}

			for _, result := range results {
				for _, message := range result.Messages {
					s.processVectorJob(streamName, message.ID, message.Values)
				}
			}
		}
	}
}

func (s *VectorService) processVectorJob(streamName, messageID string, values map[string]interface{}) {
	jobData, _ := json.Marshal(values)
	var job VectorJob
	if err := json.Unmarshal(jobData, &job); err != nil {
		log.Printf("Error unmarshaling job: %v", err)
		return
	}

	log.Printf("🔄 Processing job %s from stream %s", job.ID, streamName)

	// Update job status to processing
	s.updateJobStatus(job.ID, "processing")

	var result *CUDAResponse
	var err error

	// Route to appropriate CUDA operation based on stream
	switch streamName {
	case "vector:embeddings":
		result, err = s.processEmbedding(job)
	case "vector:similarities":
		result, err = s.processSimilarity(job)
	case "vector:indexing":
		result, err = s.processAutoIndex(job)
	case "vector:clustering":
		result, err = s.processClustering(job)
	default:
		err = fmt.Errorf("unknown stream: %s", streamName)
	}

	if err != nil {
		log.Printf("❌ Job %s failed: %v", job.ID, err)
		s.updateJobStatus(job.ID, "failed")
		return
	}

	// Store result in PostgreSQL
	if err := s.storeVectorResult(job, result); err != nil {
		log.Printf("❌ Failed to store result for job %s: %v", job.ID, err)
		s.updateJobStatus(job.ID, "failed")
		return
	}

	// Fanout to RabbitMQ if available
	if s.rabbitCh != nil {
		s.fanoutUpdate(job, result)
	}

	// Mark job as completed
	s.updateJobStatus(job.ID, "completed")
	
	// ACK the message in Redis Stream
	s.redisClient.XAck(s.ctx, streamName, "vector-processors", messageID)

	log.Printf("✅ Job %s completed successfully", job.ID)
}

func (s *VectorService) processEmbedding(job VectorJob) (*CUDAResponse, error) {
	// Extract text data from payload and convert to vector
	textData, ok := job.Payload["text"].(string)
	if !ok {
		return nil, fmt.Errorf("missing text data in payload")
	}

	// Simple text-to-vector conversion (in production, use proper embedding model)
	vector := s.textToVector(textData)

	cudaReq := CUDARequest{
		JobID: job.ID,
		Type:  "embedding",
		Data:  vector,
	}

	return s.cudaWorker.Execute(cudaReq)
}

func (s *VectorService) processSimilarity(job VectorJob) (*CUDAResponse, error) {
	vec1, ok1 := job.Payload["vector1"].([]interface{})
	vec2, ok2 := job.Payload["vector2"].([]interface{})
	
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("missing vector data in payload")
	}

	// Combine vectors for similarity calculation
	combinedData := make([]float64, len(vec1)+len(vec2))
	for i, v := range vec1 {
		combinedData[i] = v.(float64)
	}
	for i, v := range vec2 {
		combinedData[len(vec1)+i] = v.(float64)
	}

	cudaReq := CUDARequest{
		JobID: job.ID,
		Type:  "similarity",
		Data:  combinedData,
	}

	return s.cudaWorker.Execute(cudaReq)
}

func (s *VectorService) processAutoIndex(job VectorJob) (*CUDAResponse, error) {
	vector := job.Vector

	cudaReq := CUDARequest{
		JobID: job.ID,
		Type:  "autoindex",
		Data:  vector,
	}

	return s.cudaWorker.Execute(cudaReq)
}

func (s *VectorService) processClustering(job VectorJob) (*CUDAResponse, error) {
	points, ok := job.Payload["points"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("missing points data for clustering")
	}

	k, _ := job.Payload["k"].(float64)
	dim, _ := job.Payload["dim"].(float64)

	// Prepare data for SOM training
	data := []float64{k, dim}
	for _, point := range points {
		if pointData, ok := point.([]interface{}); ok {
			for _, val := range pointData {
				data = append(data, val.(float64))
			}
		}
	}

	cudaReq := CUDARequest{
		JobID: job.ID,
		Type:  "som_train",
		Data:  data,
	}

	return s.cudaWorker.Execute(cudaReq)
}

func (c *CUDAWorker) Execute(req CUDARequest) (*CUDAResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Prepare JSON input for CUDA worker
	jsonInput, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal CUDA request: %v", err)
	}

	// Execute CUDA worker
	cmd := exec.Command(c.executablePath)
	cmd.Stdin = strings.NewReader(string(jsonInput))

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("CUDA execution failed: %v", err)
	}

	// Parse CUDA response
	var response CUDAResponse
	if err := json.Unmarshal(output, &response); err != nil {
		return nil, fmt.Errorf("failed to parse CUDA response: %v", err)
	}

	if response.Status != "success" {
		return nil, fmt.Errorf("CUDA processing failed: %s", response.Error)
	}

	return &response, nil
}

func (s *VectorService) storeVectorResult(job VectorJob, result *CUDAResponse) error {
	// Store in vectors table (unified source of truth)
	query := `
		INSERT INTO vectors (id, owner_type, owner_id, embedding, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
		ON CONFLICT (owner_type, owner_id) 
		DO UPDATE SET 
			embedding = EXCLUDED.embedding,
			metadata = EXCLUDED.metadata,
			updated_at = NOW()
	`

	metadata := map[string]interface{}{
		"job_id":       job.ID,
		"cuda_type":    result.Type,
		"timestamp":    result.Timestamp,
		"vector_dim":   len(result.Vector),
		"processing_ms": time.Since(job.CreatedAt).Milliseconds(),
	}

	metadataJSON, _ := json.Marshal(metadata)

	_, err := s.pgPool.Exec(s.ctx, query, 
		result.JobID, job.OwnerType, job.OwnerID, 
		result.Vector, metadataJSON,
	)

	return err
}

func (s *VectorService) processOutbox() {
	defer s.wg.Done()

	ticker := time.NewTicker(time.Second * 2) // Process outbox every 2 seconds
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.processOutboxEntries()
		}
	}
}

func (s *VectorService) processOutboxEntries() {
	// Query unprocessed outbox entries
	query := `
		SELECT id, owner_type, owner_id, event, vector, payload, attempts
		FROM vector_outbox 
		WHERE processed_at IS NULL AND attempts < 5
		ORDER BY created_at ASC
		LIMIT 50
	`

	rows, err := s.pgPool.Query(s.ctx, query)
	if err != nil {
		log.Printf("Error querying outbox: %v", err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var id, ownerType, ownerID, event string
		var vector []float64
		var payload map[string]interface{}
		var attempts int

		err := rows.Scan(&id, &ownerType, &ownerID, &event, &vector, &payload, &attempts)
		if err != nil {
			log.Printf("Error scanning outbox row: %v", err)
			continue
		}

		// Create vector job from outbox entry
		job := VectorJob{
			ID:        id,
			OwnerType: ownerType,
			OwnerID:   ownerID,
			Event:     event,
			Vector:    vector,
			Payload:   payload,
			Priority:  "medium",
			CreatedAt: time.Now(),
			Status:    "pending",
		}

		// Route to appropriate Redis Stream based on event type
		var streamName string
		switch event {
		case "upsert":
			if len(vector) > 0 {
				streamName = "vector:similarities"
			} else {
				streamName = "vector:embeddings"
			}
		case "reembed":
			streamName = "vector:embeddings"
		case "cluster":
			streamName = "vector:clustering"
		default:
			streamName = "vector:indexing"
		}

		// Add to Redis Stream
		err = s.addJobToStream(streamName, job)
		if err != nil {
			log.Printf("Failed to add job %s to stream: %v", id, err)
			s.incrementOutboxAttempts(id)
			continue
		}

		// Mark as processed
		s.markOutboxProcessed(id)
	}
}

func (s *VectorService) addJobToStream(streamName string, job VectorJob) error {
	jobData, _ := json.Marshal(job)
	var jobMap map[string]interface{}
	json.Unmarshal(jobData, &jobMap)

	_, err := s.redisClient.XAdd(s.ctx, &redis.XAddArgs{
		Stream: streamName,
		Values: jobMap,
	}).Result()

	return err
}

func (s *VectorService) markOutboxProcessed(id string) {
	query := `UPDATE vector_outbox SET processed_at = NOW() WHERE id = $1`
	s.pgPool.Exec(s.ctx, query, id)
}

func (s *VectorService) incrementOutboxAttempts(id string) {
	query := `UPDATE vector_outbox SET attempts = attempts + 1 WHERE id = $1`
	s.pgPool.Exec(s.ctx, query, id)
}

func (s *VectorService) fanoutUpdate(job VectorJob, result *CUDAResponse) {
	if s.rabbitCh == nil {
		return
	}

	updateMessage := map[string]interface{}{
		"job_id":     job.ID,
		"owner_type": job.OwnerType,
		"owner_id":   job.OwnerID,
		"event":      job.Event,
		"vector":     result.Vector,
		"timestamp":  time.Now(),
	}

	messageBody, _ := json.Marshal(updateMessage)

	err := s.rabbitCh.Publish(
		"vector_updates", // exchange
		"",              // routing key (ignored for fanout)
		false,           // mandatory
		false,           // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        messageBody,
		},
	)

	if err != nil {
		log.Printf("Failed to publish update to RabbitMQ: %v", err)
	}
}

func (s *VectorService) updateJobStatus(jobID, status string) {
	query := `
		UPDATE vector_jobs 
		SET status = $2, updated_at = NOW() 
		WHERE id = $1
	`
	s.pgPool.Exec(s.ctx, query, jobID, status)
}

func (s *VectorService) startAPIServer() {
	defer s.wg.Done()

	r := mux.NewRouter()
	
	// Health check endpoint
	r.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		status := map[string]interface{}{
			"service": "Vector Redis Service",
			"status":  "healthy",
			"cuda":    s.cudaWorker != nil,
			"redis":   s.redisClient != nil,
			"postgres": s.pgPool != nil,
			"rabbitmq": s.rabbitCh != nil,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(status)
	}).Methods("GET")

	// Vector job creation endpoint
	r.HandleFunc("/api/vector/jobs", func(w http.ResponseWriter, r *http.Request) {
		var job VectorJob
		if err := json.NewDecoder(r.Body).Decode(&job); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		job.ID = fmt.Sprintf("job_%d", time.Now().UnixNano())
		job.CreatedAt = time.Now()
		job.Status = "pending"

		// Add to outbox for reliable processing
		if err := s.addToOutbox(job); err != nil {
			http.Error(w, "Failed to queue job", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"job_id": job.ID,
			"status": "queued",
		})
	}).Methods("POST")

	port := getEnv("VECTOR_SERVICE_PORT", "8095")
	server := &http.Server{
		Addr:    ":" + port,
		Handler: r,
	}

	log.Printf("🌐 Vector API server starting on port %s", port)
	
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("API server error: %v", err)
	}
}

func (s *VectorService) addToOutbox(job VectorJob) error {
	query := `
		INSERT INTO vector_outbox (id, owner_type, owner_id, event, vector, payload, attempts, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, 0, NOW())
	`

	payloadJSON, _ := json.Marshal(job.Payload)
	
	_, err := s.pgPool.Exec(s.ctx, query,
		job.ID, job.OwnerType, job.OwnerID, job.Event, job.Vector, payloadJSON,
	)
	
	return err
}

func (s *VectorService) textToVector(text string) []float64 {
	// Simple text-to-vector conversion for demo
	// In production, this would use a proper embedding model
	vector := make([]float64, 384) // Match nomic-embed-text dimensions
	
	for i, char := range text {
		if i >= len(vector) {
			break
		}
		vector[i] = float64(char) / 1000.0 // Normalize
	}
	
	return vector
}

func (s *VectorService) Stop() {
	s.cancel()
	s.wg.Wait()
	
	if s.pgPool != nil {
		s.pgPool.Close()
	}
	if s.redisClient != nil {
		s.redisClient.Close()
	}
	if s.rabbitConn != nil {
		s.rabbitConn.Close()
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func main() {
	service, err := NewVectorService()
	if err != nil {
		log.Fatalf("Failed to initialize service: %v", err)
	}

	if err := service.Start(); err != nil {
		log.Fatalf("Failed to start service: %v", err)
	}

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	<-sigChan
	log.Println("🛑 Shutting down Vector Redis Service...")
	
	service.Stop()
	log.Println("✅ Vector Redis Service stopped")
}