package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/streadway/amqp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	pb "github.com/deeds-evidence/evidence-processor/proto"
)

// GPUInfo holds GPU acceleration information
type GPUInfo struct {
	Available bool   `json:"available"`
	Device    string `json:"device"`
	Memory    int64  `json:"memory"`
	Cores     int    `json:"cores"`
}

// EvidenceProcessor implements the gRPC service
type EvidenceProcessor struct {
	pb.UnimplementedEvidenceProcessorServer
	redis    *redis.Client
	rabbitmq *amqp.Connection
	gpu      *GPUInfo
	mu       sync.RWMutex
	jobs     map[string]*ProcessingJob
}

type ProcessingJob struct {
	ID       string                 `json:"id"`
	Status   string                 `json:"status"`
	Progress float32                `json:"progress"`
	Steps    []string               `json:"steps"`
	Results  map[string]interface{} `json:"results"`
	Error    string                 `json:"error,omitempty"`
	Started  time.Time              `json:"started"`
	Updated  time.Time              `json:"updated"`
}

// NewEvidenceProcessor creates a new processor instance
func NewEvidenceProcessor() *EvidenceProcessor {
	// Initialize Redis
	rdb := redis.NewClient(&redis.Options{
		Addr:     getEnv("REDIS_URL", "localhost:6379"),
		Password: "",
		DB:       0,
	})

	// Initialize RabbitMQ
	conn, err := amqp.Dial(getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"))
	if err != nil {
		log.Printf("Failed to connect to RabbitMQ: %v", err)
	}

	// Detect GPU capabilities
	gpu := detectGPU()

	processor := &EvidenceProcessor{
		redis:    rdb,
		rabbitmq: conn,
		gpu:      gpu,
		jobs:     make(map[string]*ProcessingJob),
	}

	// Start background workers
	go processor.startWorkers()

	return processor
}

// ProcessEvidence implements the main processing method
func (ep *EvidenceProcessor) ProcessEvidence(ctx context.Context, req *pb.ProcessRequest) (*pb.ProcessResponse, error) {
	jobID := generateJobID()
	
	job := &ProcessingJob{
		ID:      jobID,
		Status:  "queued",
		Steps:   req.Steps,
		Results: make(map[string]interface{}),
		Started: time.Now(),
		Updated: time.Now(),
	}

	ep.mu.Lock()
	ep.jobs[jobID] = job
	ep.mu.Unlock()

	// Queue the job for processing
	go ep.processJobAsync(job, req)

	return &pb.ProcessResponse{
		JobId:   jobID,
		Status:  "queued",
		Message: "Evidence processing started",
	}, nil
}

// GetJobStatus returns the current status of a processing job
func (ep *EvidenceProcessor) GetJobStatus(ctx context.Context, req *pb.StatusRequest) (*pb.StatusResponse, error) {
	ep.mu.RLock()
	job, exists := ep.jobs[req.JobId]
	ep.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("job not found: %s", req.JobId)
	}

	return &pb.StatusResponse{
		JobId:    job.ID,
		Status:   job.Status,
		Progress: job.Progress,
		Steps:    job.Steps,
		Error:    job.Error,
	}, nil
}

// processJobAsync handles the actual evidence processing
func (ep *EvidenceProcessor) processJobAsync(job *ProcessingJob, req *pb.ProcessRequest) {
	defer func() {
		if r := recover(); r != nil {
			job.Status = "failed"
			job.Error = fmt.Sprintf("panic: %v", r)
			job.Updated = time.Now()
		}
	}()

	job.Status = "processing"
	job.Updated = time.Now()

	totalSteps := len(job.Steps)
	
	for i, step := range job.Steps {
		job.Progress = float32(i) / float32(totalSteps) * 100
		job.Updated = time.Now()

		// Publish progress update
		ep.publishProgress(job)

		switch step {
		case "ocr":
			err := ep.processOCR(job, req)
			if err != nil {
				job.Status = "failed"
				job.Error = err.Error()
				return
			}
		case "langextract":
			err := ep.processLangExtract(job, req)
			if err != nil {
				job.Status = "failed"
				job.Error = err.Error()
				return
			}
		case "embedding":
			err := ep.processEmbedding(job, req)
			if err != nil {
				job.Status = "failed"
				job.Error = err.Error()
				return
			}
		case "analysis":
			err := ep.processAnalysis(job, req)
			if err != nil {
				job.Status = "failed"
				job.Error = err.Error()
				return
			}
		}
	}

	job.Status = "completed"
	job.Progress = 100
	job.Updated = time.Now()
	ep.publishProgress(job)
}

// processOCR handles GPU-accelerated OCR processing
func (ep *EvidenceProcessor) processOCR(job *ProcessingJob, req *pb.ProcessRequest) error {
	log.Printf("Processing OCR for job %s with GPU: %v", job.ID, ep.gpu.Available)
	
	// Simulate GPU-accelerated OCR processing
	time.Sleep(time.Second * 2)
	
	job.Results["ocr"] = map[string]interface{}{
		"text":       "Extracted text content...",
		"confidence": 0.95,
		"pages":      []int{1, 2, 3},
		"gpu_used":   ep.gpu.Available,
	}
	
	return nil
}

// processLangExtract handles structured information extraction
func (ep *EvidenceProcessor) processLangExtract(job *ProcessingJob, req *pb.ProcessRequest) error {
	log.Printf("Processing LangExtract for job %s", job.ID)
	
	// Simulate LangExtract processing with GPU acceleration
	time.Sleep(time.Second * 3)
	
	job.Results["langextract"] = map[string]interface{}{
		"entities": []map[string]interface{}{
			{"type": "PERSON", "text": "John Doe", "confidence": 0.98},
			{"type": "DATE", "text": "2024-08-18", "confidence": 0.95},
			{"type": "LOCATION", "text": "San Francisco", "confidence": 0.92},
		},
		"relations": []map[string]interface{}{
			{"subject": "John Doe", "predicate": "lives in", "object": "San Francisco"},
		},
		"structured_data": map[string]interface{}{
			"case_number": "2024-CV-001234",
			"parties":     []string{"John Doe", "Jane Smith"},
			"court":       "Superior Court of California",
		},
		"gpu_accelerated": ep.gpu.Available,
	}
	
	return nil
}

// processEmbedding handles vector embedding generation
func (ep *EvidenceProcessor) processEmbedding(job *ProcessingJob, req *pb.ProcessRequest) error {
	log.Printf("Processing embeddings for job %s with GPU: %v", job.ID, ep.gpu.Available)
	
	// Simulate GPU-accelerated embedding generation
	time.Sleep(time.Second * 4)
	
	job.Results["embedding"] = map[string]interface{}{
		"vectors":    [][]float32{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}},
		"dimensions": 384,
		"model":      "sentence-transformers/all-MiniLM-L6-v2",
		"gpu_used":   ep.gpu.Available,
	}
	
	return nil
}

// processAnalysis handles final AI analysis
func (ep *EvidenceProcessor) processAnalysis(job *ProcessingJob, req *pb.ProcessRequest) error {
	log.Printf("Processing analysis for job %s", job.ID)
	
	// Simulate comprehensive analysis
	time.Sleep(time.Second * 3)
	
	job.Results["analysis"] = map[string]interface{}{
		"summary":     "Comprehensive legal document analysis completed",
		"key_points":  []string{"Point 1", "Point 2", "Point 3"},
		"risk_score":  0.3,
		"confidence":  0.89,
		"processing_time_ms": 10000,
	}
	
	return nil
}

// publishProgress publishes job progress to Redis and RabbitMQ
func (ep *EvidenceProcessor) publishProgress(job *ProcessingJob) {
	ctx := context.Background()
	
	// Publish to Redis for real-time updates
	jobData, _ := json.Marshal(job)
	ep.redis.Publish(ctx, fmt.Sprintf("evidence:progress:%s", job.ID), jobData)
	
	// Publish to RabbitMQ for persistent messaging
	if ep.rabbitmq != nil {
		ch, err := ep.rabbitmq.Channel()
		if err == nil {
			defer ch.Close()
			
			ch.Publish(
				"evidence.progress", // exchange
				job.ID,             // routing key
				false,              // mandatory
				false,              // immediate
				amqp.Publishing{
					ContentType: "application/json",
					Body:        jobData,
				},
			)
		}
	}
}

// detectGPU detects available GPU acceleration
func detectGPU() *GPUInfo {
	// This is a simplified GPU detection
	// In production, you'd use CUDA/OpenCL libraries
	gpu := &GPUInfo{
		Available: false,
		Device:    "none",
		Memory:    0,
		Cores:     0,
	}

	// Check for NVIDIA GPU (simplified)
	if _, err := os.Stat("C:\\Program Files\\NVIDIA GPU Computing Toolkit"); err == nil {
		gpu.Available = true
		gpu.Device = "NVIDIA CUDA"
		gpu.Memory = 8192 // Mock 8GB VRAM
		gpu.Cores = 2048  // Mock CUDA cores
	}

	// Check for AMD GPU (simplified)
	if _, err := os.Stat("C:\\Program Files\\AMD"); err == nil && !gpu.Available {
		gpu.Available = true
		gpu.Device = "AMD ROCm"
		gpu.Memory = 8192
		gpu.Cores = 1024
	}

	log.Printf("GPU Detection: Available=%v, Device=%s", gpu.Available, gpu.Device)
	return gpu
}

// startWorkers starts background processing workers
func (ep *EvidenceProcessor) startWorkers() {
	// Start Redis subscriber for coordination
	go func() {
		ctx := context.Background()
		pubsub := ep.redis.Subscribe(ctx, "evidence:commands:*")
		defer pubsub.Close()

		for msg := range pubsub.Channel() {
			log.Printf("Received command: %s", msg.Payload)
			// Handle commands (pause, resume, cancel, etc.)
		}
	}()

	// Start cleanup worker
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()

		for range ticker.C {
			ep.cleanupOldJobs()
		}
	}()
}

// cleanupOldJobs removes completed jobs older than 1 hour
func (ep *EvidenceProcessor) cleanupOldJobs() {
	ep.mu.Lock()
	defer ep.mu.Unlock()

	cutoff := time.Now().Add(-1 * time.Hour)
	for id, job := range ep.jobs {
		if job.Status == "completed" || job.Status == "failed" {
			if job.Updated.Before(cutoff) {
				delete(ep.jobs, id)
				log.Printf("Cleaned up old job: %s", id)
			}
		}
	}
}

// Health check method
func (ep *EvidenceProcessor) HealthCheck(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	status := "healthy"
	
	// Check Redis connection
	if err := ep.redis.Ping(ctx).Err(); err != nil {
		status = "unhealthy"
	}
	
	// Check system resources
	var m runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m)
	
	return &pb.HealthResponse{
		Status: status,
		Gpu:    ep.gpu,
		Metrics: map[string]float32{
			"memory_mb":    float32(m.Alloc) / 1024 / 1024,
			"goroutines":   float32(runtime.NumGoroutine()),
			"active_jobs":  float32(len(ep.jobs)),
		},
	}, nil
}

// Utility functions
func generateJobID() string {
	return fmt.Sprintf("job_%d", time.Now().UnixNano())
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func main() {
	port := getEnv("GRPC_PORT", "50051")
	
	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	s := grpc.NewServer()
	processor := NewEvidenceProcessor()
	
	pb.RegisterEvidenceProcessorServer(s, processor)
	reflection.Register(s)

	log.Printf("🚀 Evidence Processor gRPC server starting on port %s", port)
	log.Printf("GPU Acceleration: %v (%s)", processor.gpu.Available, processor.gpu.Device)
	
	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
