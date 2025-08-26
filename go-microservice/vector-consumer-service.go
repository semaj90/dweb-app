// vector-consumer-service.go
// Production Go service that consumes Redis Streams, processes vectors with CUDA worker, and updates databases
// Build: go build -o vector-consumer-service.exe vector-consumer-service.go

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/lib/pq"
	_ "github.com/lib/pq"
	amqp "github.com/rabbitmq/amqp091-go"
)

// Configuration
type Config struct {
	RedisURL         string
	DatabaseURL      string
	RabbitMQURL      string
	StreamName       string
	ConsumerGroup    string
	ConsumerName     string
	CUDAWorkerPath   string
	EmbedServiceURL  string
	QdrantURL        string
	BatchSize        int
	WorkerCount      int
	ProcessTimeout   time.Duration
}

// Vector processing job
type VectorJob struct {
	ID         string                 `json:"id"`
	JobType    string                 `json:"job_type"`    // "rotate", "embed", "similarity"
	OwnerType  string                 `json:"owner_type"`
	OwnerID    string                 `json:"owner_id"`
	Data       map[string]interface{} `json:"data"`
	Quaternion *Quaternion            `json:"quaternion,omitempty"`
	Points     []float32              `json:"points,omitempty"`
	Text       string                 `json:"text,omitempty"`
	CreatedAt  time.Time              `json:"created_at"`
}

type Quaternion struct {
	W float32 `json:"w"`
	X float32 `json:"x"`
	Y float32 `json:"y"`
	Z float32 `json:"z"`
}

// CUDA worker response
type CUDAResponse struct {
	JobID    string    `json:"jobId"`
	Status   string    `json:"status"`
	Rotated  []float32 `json:"rotated,omitempty"`
	Error    string    `json:"error,omitempty"`
}

// Vector Consumer Service
type VectorConsumerService struct {
	config        *Config
	redisClient   *redis.Client
	db            *sql.DB
	rabbitConn    *amqp.Connection
	rabbitChannel *amqp.Channel
	ctx           context.Context
	cancel        context.CancelFunc
	wg            sync.WaitGroup
}

func NewConfig() *Config {
	return &Config{
		RedisURL:         getEnv("REDIS_URL", "redis://localhost:6379"),
		DatabaseURL:      getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		RabbitMQURL:      getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		StreamName:       getEnv("REDIS_STREAM", "vector:jobs"),
		ConsumerGroup:    getEnv("CONSUMER_GROUP", "vector-processors"),
		ConsumerName:     getEnv("CONSUMER_NAME", fmt.Sprintf("worker-%d", os.Getpid())),
		CUDAWorkerPath:   getEnv("CUDA_WORKER_PATH", "./cuda-worker/cuda-quaternion-worker.exe"),
		EmbedServiceURL:  getEnv("EMBED_SERVICE_URL", "http://localhost:9001"),
		QdrantURL:        getEnv("QDRANT_URL", "http://localhost:6333"),
		BatchSize:        getEnvInt("BATCH_SIZE", 10),
		WorkerCount:      getEnvInt("WORKER_COUNT", 4),
		ProcessTimeout:   time.Duration(getEnvInt("PROCESS_TIMEOUT_SEC", 30)) * time.Second,
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func NewVectorConsumerService(config *Config) *VectorConsumerService {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &VectorConsumerService{
		config: config,
		ctx:    ctx,
		cancel: cancel,
	}
}

func (s *VectorConsumerService) Initialize() error {
	// Initialize Redis client
	opts, err := redis.ParseURL(s.config.RedisURL)
	if err != nil {
		return fmt.Errorf("failed to parse Redis URL: %w", err)
	}
	s.redisClient = redis.NewClient(opts)

	// Test Redis connection
	if err := s.redisClient.Ping(s.ctx).Err(); err != nil {
		return fmt.Errorf("failed to connect to Redis: %w", err)
	}
	log.Println("✅ Connected to Redis")

	// Initialize PostgreSQL connection
	s.db, err = sql.Open("postgres", s.config.DatabaseURL)
	if err != nil {
		return fmt.Errorf("failed to connect to PostgreSQL: %w", err)
	}

	if err := s.db.Ping(); err != nil {
		return fmt.Errorf("failed to ping PostgreSQL: %w", err)
	}
	log.Println("✅ Connected to PostgreSQL")

	// Initialize RabbitMQ connection
	s.rabbitConn, err = amqp.Dial(s.config.RabbitMQURL)
	if err != nil {
		return fmt.Errorf("failed to connect to RabbitMQ: %w", err)
	}

	s.rabbitChannel, err = s.rabbitConn.Channel()
	if err != nil {
		return fmt.Errorf("failed to open RabbitMQ channel: %w", err)
	}
	log.Println("✅ Connected to RabbitMQ")

	// Create consumer group if it doesn't exist
	err = s.redisClient.XGroupCreateMkStream(s.ctx, s.config.StreamName, s.config.ConsumerGroup, "0").Err()
	if err != nil && err.Error() != "BUSYGROUP Consumer Group name already exists" {
		return fmt.Errorf("failed to create consumer group: %w", err)
	}
	log.Printf("✅ Consumer group '%s' ready", s.config.ConsumerGroup)

	return nil
}

func (s *VectorConsumerService) Start() {
	log.Printf("🚀 Starting Vector Consumer Service with %d workers", s.config.WorkerCount)
	
	// Start worker goroutines
	for i := 0; i < s.config.WorkerCount; i++ {
		s.wg.Add(1)
		go s.worker(i)
	}
	
	// Start health check endpoint (simple HTTP server)
	go s.startHealthServer()
	
	log.Println("✅ All workers started successfully")
}

func (s *VectorConsumerService) worker(workerID int) {
	defer s.wg.Done()
	
	workerName := fmt.Sprintf("%s-worker-%d", s.config.ConsumerName, workerID)
	log.Printf("Worker %d started as %s", workerID, workerName)
	
	for {
		select {
		case <-s.ctx.Done():
			log.Printf("Worker %d shutting down", workerID)
			return
		default:
			if err := s.processMessages(workerName); err != nil {
				log.Printf("Worker %d error: %v", workerID, err)
				time.Sleep(5 * time.Second)
			}
		}
	}
}

func (s *VectorConsumerService) processMessages(consumerName string) error {
	// Read messages from Redis Stream
	streams, err := s.redisClient.XReadGroup(s.ctx, &redis.XReadGroupArgs{
		Group:    s.config.ConsumerGroup,
		Consumer: consumerName,
		Streams:  []string{s.config.StreamName, ">"},
		Count:    int64(s.config.BatchSize),
		Block:    time.Second * 5,
	}).Result()

	if err != nil {
		if err == redis.Nil {
			return nil // No messages available
		}
		return fmt.Errorf("failed to read from stream: %w", err)
	}

	for _, stream := range streams {
		for _, message := range stream.Messages {
			if err := s.processMessage(message); err != nil {
				log.Printf("Failed to process message %s: %v", message.ID, err)
				// Continue processing other messages
			} else {
				// Acknowledge successful processing
				s.redisClient.XAck(s.ctx, s.config.StreamName, s.config.ConsumerGroup, message.ID)
			}
		}
	}

	return nil
}

func (s *VectorConsumerService) processMessage(message redis.XMessage) error {
	// Parse job from message
	jobData, exists := message.Values["job"]
	if !exists {
		return fmt.Errorf("missing job data in message")
	}

	var job VectorJob
	if err := json.Unmarshal([]byte(jobData.(string)), &job); err != nil {
		return fmt.Errorf("failed to unmarshal job: %w", err)
	}

	log.Printf("Processing job %s (type: %s, owner: %s/%s)", job.ID, job.JobType, job.OwnerType, job.OwnerID)

	// Process based on job type
	switch job.JobType {
	case "rotate":
		return s.processRotationJob(&job)
	case "embed":
		return s.processEmbeddingJob(&job)
	case "similarity":
		return s.processSimilarityJob(&job)
	default:
		return fmt.Errorf("unknown job type: %s", job.JobType)
	}
}

func (s *VectorConsumerService) processRotationJob(job *VectorJob) error {
	if job.Quaternion == nil || len(job.Points) == 0 {
		return fmt.Errorf("rotation job missing quaternion or points")
	}

	// Prepare CUDA worker input
	cudaInput := map[string]interface{}{
		"jobId": job.ID,
		"quat": map[string]float32{
			"w": job.Quaternion.W,
			"x": job.Quaternion.X,
			"y": job.Quaternion.Y,
			"z": job.Quaternion.Z,
		},
		"points": job.Points,
	}

	// Execute CUDA worker
	result, err := s.executeCUDAWorker(cudaInput)
	if err != nil {
		return fmt.Errorf("CUDA worker failed: %w", err)
	}

	if result.Status != "success" {
		return fmt.Errorf("CUDA worker error: %s", result.Error)
	}

	// Update database with rotated points
	return s.updateRotatedPoints(job.OwnerType, job.OwnerID, result.Rotated)
}

func (s *VectorConsumerService) processEmbeddingJob(job *VectorJob) error {
	// Call embedding microservice
	// Implementation for calling the FastAPI embedding service
	log.Printf("Processing embedding job for text: %.50s...", job.Text)
	
	// This would call the embedding microservice we created
	// and store the result in PostgreSQL and Qdrant
	return nil
}

func (s *VectorConsumerService) processSimilarityJob(job *VectorJob) error {
	// Process similarity computation
	log.Printf("Processing similarity job for %s/%s", job.OwnerType, job.OwnerID)
	return nil
}

func (s *VectorConsumerService) executeCUDAWorker(input map[string]interface{}) (*CUDAResponse, error) {
	// Marshal input to JSON
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal input: %w", err)
	}

	// Execute CUDA worker process
	ctx, cancel := context.WithTimeout(s.ctx, s.config.ProcessTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, s.config.CUDAWorkerPath)
	cmd.Stdin = strings.NewReader(string(inputJSON))

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("CUDA worker execution failed: %w", err)
	}

	// Parse response
	var response CUDAResponse
	if err := json.Unmarshal(output, &response); err != nil {
		return nil, fmt.Errorf("failed to parse CUDA response: %w", err)
	}

	return &response, nil
}

func (s *VectorConsumerService) updateRotatedPoints(ownerType, ownerID string, rotatedPoints []float32) error {
	// Convert to JSON array for JSONB storage
	pointsJSON, err := json.Marshal(rotatedPoints)
	if err != nil {
		return fmt.Errorf("failed to marshal rotated points: %w", err)
	}

	// Update based on owner type
	var query string
	switch ownerType {
	case "chunk":
		query = `
			UPDATE chunks 
			SET rotated_points = $1::jsonb, updated_at = NOW()
			WHERE id = $2::uuid`
	case "document":
		query = `
			UPDATE documents 
			SET rotated_points = $1::jsonb, updated_at = NOW()
			WHERE id = $2::uuid`
	default:
		return fmt.Errorf("unsupported owner type for rotation: %s", ownerType)
	}

	_, err = s.db.Exec(query, string(pointsJSON), ownerID)
	if err != nil {
		return fmt.Errorf("failed to update rotated points: %w", err)
	}

	// Publish completion event to RabbitMQ
	return s.publishCompletionEvent("rotation", ownerType, ownerID, map[string]interface{}{
		"points_count": len(rotatedPoints) / 3,
	})
}

func (s *VectorConsumerService) publishCompletionEvent(eventType, ownerType, ownerID string, data map[string]interface{}) error {
	event := map[string]interface{}{
		"event_type":  eventType,
		"owner_type":  ownerType,
		"owner_id":    ownerID,
		"data":        data,
		"timestamp":   time.Now().UTC(),
		"service":     "vector-consumer",
	}

	eventJSON, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	return s.rabbitChannel.Publish(
		"vector.events", // exchange
		"completed",     // routing key
		false,           // mandatory
		false,           // immediate
		amqp.Publishing{
			ContentType: "application/json",
			Body:        eventJSON,
		},
	)
}

func (s *VectorConsumerService) startHealthServer() {
	// Simple HTTP health check server
	// This would be implemented with net/http for production
	log.Println("Health check server would start on :8080/health")
}

func (s *VectorConsumerService) Shutdown() {
	log.Println("🔄 Shutting down Vector Consumer Service...")
	
	s.cancel()
	s.wg.Wait()
	
	if s.redisClient != nil {
		s.redisClient.Close()
	}
	
	if s.db != nil {
		s.db.Close()
	}
	
	if s.rabbitConn != nil {
		s.rabbitConn.Close()
	}
	
	log.Println("✅ Vector Consumer Service shutdown complete")
}

func main() {
	// Load configuration
	config := NewConfig()
	
	// Create and initialize service
	service := NewVectorConsumerService(config)
	
	if err := service.Initialize(); err != nil {
		log.Fatalf("Failed to initialize service: %v", err)
	}
	
	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	// Start service
	service.Start()
	
	// Wait for shutdown signal
	<-sigChan
	
	// Graceful shutdown
	service.Shutdown()
}