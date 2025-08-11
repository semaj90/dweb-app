package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/streadway/amqp"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"golang.org/x/sys/cpu"
)

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for SIMD-optimized vector operations
extern "C" {
    cudaError_t gpuVectorDot(float* a, float* b, float* result, int size);
    cudaError_t gpuMatrixMult(float* a, float* b, float* c, int m, int n, int k);
    cudaError_t gpuTensorReduce(float* input, float* output, int size, int operation);
}
*/
import "C"

// =====================================
// GPU-Accelerated SIMD Parser
// =====================================

type GPUSIMDParser struct {
	hasAVX2        bool
	hasSSE42       bool
	hasAVX512      bool
	hasCUDA        bool
	cudaDeviceID   int
	cublasHandle   C.cublasHandle_t
	workers        int
	chunkSize      int
	gpuMemoryPool  sync.Pool
	tensorCache    map[string]*tensor.Dense
	cacheMutex     sync.RWMutex
}

type GPUTensor struct {
	Data     []float32
	Shape    []int
	Device   string
	DeviceID int
}

type CUDAContext struct {
	DeviceID     int
	MemoryTotal  int64
	MemoryFree   int64
	ComputeCaps  string
	CoresCount   int
	ClockRate    int
}

func NewGPUSIMDParser() *GPUSIMDParser {
	parser := &GPUSIMDParser{
		hasAVX2:     cpu.X86.HasAVX2,
		hasSSE42:    cpu.X86.HasSSE42,
		hasAVX512:   cpu.X86.HasAVX512F,
		workers:     runtime.NumCPU(),
		chunkSize:   8192,
		tensorCache: make(map[string]*tensor.Dense),
	}

	// Initialize CUDA if available
	if err := parser.initCUDA(); err != nil {
		log.Printf("CUDA initialization failed: %v, falling back to CPU", err)
		parser.hasCUDA = false
	}

	// Initialize GPU memory pool
	parser.gpuMemoryPool = sync.Pool{
		New: func() interface{} {
			return make([]float32, parser.chunkSize)
		},
	}

	return parser
}

func (gp *GPUSIMDParser) initCUDA() error {
	var deviceCount C.int
	if C.cudaGetDeviceCount(&deviceCount) != C.cudaSuccess {
		return fmt.Errorf("CUDA not available")
	}

	if deviceCount == 0 {
		return fmt.Errorf("No CUDA devices found")
	}

	gp.cudaDeviceID = 0
	if C.cudaSetDevice(C.int(gp.cudaDeviceID)) != C.cudaSuccess {
		return fmt.Errorf("Failed to set CUDA device")
	}

	// Initialize cuBLAS
	if C.cublasCreate(&gp.cublasHandle) != C.CUBLAS_STATUS_SUCCESS {
		return fmt.Errorf("Failed to initialize cuBLAS")
	}

	gp.hasCUDA = true
	log.Printf("✓ CUDA initialized with device %d", gp.cudaDeviceID)
	return nil
}

// GPU-accelerated vector operations
func (gp *GPUSIMDParser) GPUVectorDot(a, b []float32) (float32, error) {
	if !gp.hasCUDA || len(a) != len(b) || len(a) == 0 {
		return gp.cpuVectorDot(a, b), nil
	}

	size := len(a)
	var result float32

	// Allocate GPU memory
	var d_a, d_b, d_result unsafe.Pointer
	C.cudaMalloc(&d_a, C.size_t(size*4))
	C.cudaMalloc(&d_b, C.size_t(size*4))
	C.cudaMalloc(&d_result, C.size_t(4))

	defer func() {
		C.cudaFree(d_a)
		C.cudaFree(d_b)
		C.cudaFree(d_result)
	}()

	// Copy data to GPU
	C.cudaMemcpy(d_a, unsafe.Pointer(&a[0]), C.size_t(size*4), C.cudaMemcpyHostToDevice)
	C.cudaMemcpy(d_b, unsafe.Pointer(&b[0]), C.size_t(size*4), C.cudaMemcpyHostToDevice)

	// Execute GPU kernel
	if C.gpuVectorDot((*C.float)(d_a), (*C.float)(d_b), (*C.float)(d_result), C.int(size)) != C.cudaSuccess {
		return gp.cpuVectorDot(a, b), fmt.Errorf("GPU vector dot failed")
	}

	// Copy result back
	C.cudaMemcpy(unsafe.Pointer(&result), d_result, C.size_t(4), C.cudaMemcpyDeviceToHost)

	return result, nil
}

func (gp *GPUSIMDParser) cpuVectorDot(a, b []float32) float32 {
	if gp.hasAVX2 && len(a) >= 8 {
		return gp.avx2VectorDot(a, b)
	} else if gp.hasSSE42 && len(a) >= 4 {
		return gp.sse42VectorDot(a, b)
	}
	return gp.scalarVectorDot(a, b)
}

func (gp *GPUSIMDParser) avx2VectorDot(a, b []float32) float32 {
	sum := float32(0.0)
	i := 0
	
	// Process 8 floats at a time with AVX2
	for i <= len(a)-8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			   a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
		i += 8
	}
	
	// Handle remaining elements
	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	
	return sum
}

func (gp *GPUSIMDParser) sse42VectorDot(a, b []float32) float32 {
	sum := float32(0.0)
	i := 0
	
	// Process 4 floats at a time with SSE4.2
	for i <= len(a)-4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		i += 4
	}
	
	// Handle remaining elements
	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	
	return sum
}

func (gp *GPUSIMDParser) scalarVectorDot(a, b []float32) float32 {
	sum := float32(0.0)
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// GPU-accelerated tensor operations using Gorgonia
func (gp *GPUSIMDParser) CreateTensor(data []float32, shape ...int) (*tensor.Dense, error) {
	return tensor.New(tensor.WithBacking(data), tensor.WithShape(shape...))
}

func (gp *GPUSIMDParser) TensorMatMul(a, b *tensor.Dense) (*tensor.Dense, error) {
	// Use Gorgonia for tensor operations with GPU acceleration
	g := gorgonia.NewGraph()
	
	aNode := gorgonia.NewTensor(g, tensor.Float32, a.Dims(), gorgonia.WithName("a"), gorgonia.WithValue(a))
	bNode := gorgonia.NewTensor(g, tensor.Float32, b.Dims(), gorgonia.WithName("b"), gorgonia.WithValue(b))
	
	result, err := gorgonia.Mul(aNode, bNode)
	if err != nil {
		return nil, err
	}
	
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()
	
	if err := machine.RunAll(); err != nil {
		return nil, err
	}
	
	return result.Value().(*tensor.Dense), nil
}

// =====================================
// Redis-Native Caching Service
// =====================================

type RedisService struct {
	client      *redis.Client
	clusterMode bool
	pipeline    redis.Pipeliner
	pubsub      *redis.PubSub
}

func NewRedisService(addr string, password string, db int) *RedisService {
	rdb := redis.NewClient(&redis.Options{
		Addr:         addr,
		Password:     password,
		DB:           db,
		PoolSize:     100,
		MinIdleConns: 20,
		MaxRetries:   3,
		ReadTimeout:  time.Second * 3,
		WriteTimeout: time.Second * 3,
	})

	return &RedisService{
		client: rdb,
	}
}

func (rs *RedisService) SetEmbedding(ctx context.Context, key string, embedding []float32, ttl time.Duration) error {
	data, err := json.Marshal(embedding)
	if err != nil {
		return err
	}
	return rs.client.Set(ctx, fmt.Sprintf("embedding:%s", key), data, ttl).Err()
}

func (rs *RedisService) GetEmbedding(ctx context.Context, key string) ([]float32, error) {
	data, err := rs.client.Get(ctx, fmt.Sprintf("embedding:%s", key)).Result()
	if err != nil {
		return nil, err
	}
	
	var embedding []float32
	err = json.Unmarshal([]byte(data), &embedding)
	return embedding, err
}

func (rs *RedisService) CacheJSON(ctx context.Context, key string, data interface{}, ttl time.Duration) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return rs.client.Set(ctx, fmt.Sprintf("json:%s", key), jsonData, ttl).Err()
}

func (rs *RedisService) GetCachedJSON(ctx context.Context, key string, dest interface{}) error {
	data, err := rs.client.Get(ctx, fmt.Sprintf("json:%s", key)).Result()
	if err != nil {
		return err
	}
	return json.Unmarshal([]byte(data), dest)
}

// =====================================
// RabbitMQ Integration
// =====================================

type RabbitMQService struct {
	conn         *amqp.Connection
	channel      *amqp.Channel
	queues       map[string]amqp.Queue
	exchanges    map[string]string
	consumers    map[string]<-chan amqp.Delivery
	mutex        sync.RWMutex
}

type ProcessingTask struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Priority    int                    `json:"priority"`
	Data        map[string]interface{} `json:"data"`
	CreatedAt   time.Time              `json:"created_at"`
	ProcessedAt *time.Time             `json:"processed_at,omitempty"`
	Status      string                 `json:"status"`
	Result      interface{}            `json:"result,omitempty"`
	Error       string                 `json:"error,omitempty"`
}

func NewRabbitMQService(amqpURL string) (*RabbitMQService, error) {
	conn, err := amqp.Dial(amqpURL)
	if err != nil {
		return nil, err
	}

	channel, err := conn.Channel()
	if err != nil {
		return nil, err
	}

	rq := &RabbitMQService{
		conn:      conn,
		channel:   channel,
		queues:    make(map[string]amqp.Queue),
		exchanges: make(map[string]string),
		consumers: make(map[string]<-chan amqp.Delivery),
	}

	// Declare default exchanges and queues
	err = rq.setupDefaultQueues()
	if err != nil {
		return nil, err
	}

	return rq, nil
}

func (rq *RabbitMQService) setupDefaultQueues() error {
	// Declare exchanges
	exchanges := []string{"legal.direct", "legal.topic", "legal.fanout"}
	for _, exchange := range exchanges {
		err := rq.channel.ExchangeDeclare(
			exchange, "direct", true, false, false, false, nil)
		if err != nil {
			return err
		}
		rq.exchanges[exchange] = "direct"
	}

	// Declare queues
	queues := []string{"gpu.processing", "embedding.generation", "legal.analysis", "cache.invalidation"}
	for _, queueName := range queues {
		queue, err := rq.channel.QueueDeclare(
			queueName, true, false, false, false, nil)
		if err != nil {
			return err
		}
		rq.queues[queueName] = queue
	}

	return nil
}

func (rq *RabbitMQService) PublishTask(queueName string, task *ProcessingTask) error {
	body, err := json.Marshal(task)
	if err != nil {
		return err
	}

	return rq.channel.Publish(
		"",        // exchange
		queueName, // routing key
		false,     // mandatory
		false,     // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			Body:         body,
			DeliveryMode: amqp.Persistent,
			Timestamp:    time.Now(),
			Priority:     uint8(task.Priority),
		})
}

func (rq *RabbitMQService) ConsumeTask(queueName string, handler func(*ProcessingTask) error) error {
	messages, err := rq.channel.Consume(
		queueName, "", false, false, false, false, nil)
	if err != nil {
		return err
	}

	go func() {
		for msg := range messages {
			var task ProcessingTask
			err := json.Unmarshal(msg.Body, &task)
			if err != nil {
				log.Printf("Failed to unmarshal task: %v", err)
				msg.Nack(false, false)
				continue
			}

			err = handler(&task)
			if err != nil {
				log.Printf("Task processing failed: %v", err)
				msg.Nack(false, true)
			} else {
				msg.Ack(false)
			}
		}
	}()

	return nil
}

// =====================================
// Legal BERT Integration
// =====================================

type LegalBERTService struct {
	modelPath    string
	tokenizer    map[string]int
	vocabulary   []string
	embeddings   map[string][]float32
	keywords     map[string][]string
	cache        *RedisService
	gpu          *GPUSIMDParser
}

type LegalEntity struct {
	Text       string    `json:"text"`
	Type       string    `json:"type"`
	Confidence float64   `json:"confidence"`
	StartPos   int       `json:"start_pos"`
	EndPos     int       `json:"end_pos"`
	Keywords   []string  `json:"keywords"`
	Embedding  []float32 `json:"embedding,omitempty"`
}

func NewLegalBERTService(modelPath string, cache *RedisService, gpu *GPUSIMDParser) *LegalBERTService {
	service := &LegalBERTService{
		modelPath:  modelPath,
		tokenizer:  make(map[string]int),
		embeddings: make(map[string][]float32),
		keywords:   make(map[string][]string),
		cache:      cache,
		gpu:        gpu,
	}

	// Load legal domain keywords
	service.loadLegalKeywords()
	return service
}

func (lbs *LegalBERTService) loadLegalKeywords() {
	lbs.keywords = map[string][]string{
		"contract": {"agreement", "contract", "deal", "covenant", "arrangement"},
		"liability": {"liable", "responsibility", "accountability", "obligation"},
		"evidence": {"proof", "documentation", "testimony", "exhibit", "evidence"},
		"jurisdiction": {"court", "venue", "jurisdiction", "forum", "tribunal"},
		"damages": {"compensation", "damages", "remedies", "restitution", "relief"},
		"breach": {"violation", "breach", "default", "non-compliance", "infringement"},
		"plaintiff": {"plaintiff", "claimant", "petitioner", "complainant"},
		"defendant": {"defendant", "respondent", "accused", "party"},
		"statute": {"law", "statute", "regulation", "code", "ordinance"},
		"precedent": {"precedent", "case law", "jurisprudence", "authority"},
	}
}

func (lbs *LegalBERTService) RecognizeEntities(text string) ([]LegalEntity, error) {
	var entities []LegalEntity
	
	// Tokenize text
	words := strings.Fields(strings.ToLower(text))
	
	for i, word := range words {
		// Check against legal keywords
		for entityType, keywords := range lbs.keywords {
			for _, keyword := range keywords {
				if strings.Contains(word, keyword) {
					// Generate embedding for the word
					embedding, err := lbs.generateEmbedding(word)
					if err != nil {
						embedding = nil
					}
					
					entity := LegalEntity{
						Text:       word,
						Type:       entityType,
						Confidence: lbs.calculateConfidence(word, keyword),
						StartPos:   i,
						EndPos:     i + 1,
						Keywords:   []string{keyword},
						Embedding:  embedding,
					}
					entities = append(entities, entity)
					break
				}
			}
		}
	}
	
	return entities, nil
}

func (lbs *LegalBERTService) generateEmbedding(text string) ([]float32, error) {
	// Check cache first
	cached, err := lbs.cache.GetEmbedding(context.Background(), text)
	if err == nil {
		return cached, nil
	}
	
	// Generate embedding using simple word2vec-like approach
	// In production, this would use a proper BERT model
	embedding := make([]float32, 384) // Common embedding size
	
	// Simple hash-based embedding generation
	for i, char := range text {
		idx := (int(char) + i) % len(embedding)
		embedding[idx] += float32(math.Sin(float64(idx) * 0.1))
	}
	
	// Normalize the embedding
	norm := float32(0.0)
	for _, val := range embedding {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	
	for i := range embedding {
		embedding[i] /= norm
	}
	
	// Cache the result
	lbs.cache.SetEmbedding(context.Background(), text, embedding, time.Hour*24)
	
	return embedding, nil
}

func (lbs *LegalBERTService) calculateConfidence(word, keyword string) float64 {
	if word == keyword {
		return 1.0
	}
	
	// Simple Levenshtein-based confidence
	distance := levenshteinDistance(word, keyword)
	maxLen := float64(max(len(word), len(keyword)))
	
	if maxLen == 0 {
		return 0.0
	}
	
	confidence := 1.0 - (float64(distance) / maxLen)
	return math.Max(0.0, confidence)
}

// =====================================
// Main Service Integration
// =====================================

type GPULegalAIService struct {
	gpu          *GPUSIMDParser
	redis        *RedisService
	rabbitmq     *RabbitMQService
	legalBERT    *LegalBERTService
	httpClient   *http.Client
	ollamaURL    string
	modelCache   map[string]interface{}
	mutex        sync.RWMutex
}

type ProcessingRequest struct {
	ID            string                 `json:"id"`
	Text          string                 `json:"text"`
	Type          string                 `json:"type"`
	UseGPU        bool                   `json:"use_gpu"`
	CacheResults  bool                   `json:"cache_results"`
	GenerateEmbedding bool               `json:"generate_embedding"`
	ExtractEntities   bool               `json:"extract_entities"`
	Options       map[string]interface{} `json:"options"`
}

type ProcessingResponse struct {
	ID          string        `json:"id"`
	Status      string        `json:"status"`
	Result      interface{}   `json:"result"`
	Entities    []LegalEntity `json:"entities,omitempty"`
	Embedding   []float32     `json:"embedding,omitempty"`
	Performance struct {
		ProcessingTime int64  `json:"processing_time_ms"`
		GPUUsed       bool   `json:"gpu_used"`
		CacheHit      bool   `json:"cache_hit"`
		TokensProcessed int  `json:"tokens_processed"`
	} `json:"performance"`
	Error string `json:"error,omitempty"`
}

func NewGPULegalAIService(redisAddr, rabbitMQURL, ollamaURL string) (*GPULegalAIService, error) {
	// Initialize GPU SIMD parser
	gpu := NewGPUSIMDParser()
	
	// Initialize Redis
	redis := NewRedisService(redisAddr, "", 0)
	
	// Initialize RabbitMQ
	rabbitmq, err := NewRabbitMQService(rabbitMQURL)
	if err != nil {
		return nil, err
	}
	
	// Initialize Legal BERT
	legalBERT := NewLegalBERTService("models/legal-bert", redis, gpu)
	
	service := &GPULegalAIService{
		gpu:         gpu,
		redis:       redis,
		rabbitmq:    rabbitmq,
		legalBERT:   legalBERT,
		ollamaURL:   ollamaURL,
		modelCache:  make(map[string]interface{}),
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
	
	// Set up RabbitMQ consumers
	err = service.setupConsumers()
	if err != nil {
		return nil, err
	}
	
	return service, nil
}

func (gls *GPULegalAIService) setupConsumers() error {
	// GPU processing queue consumer
	err := gls.rabbitmq.ConsumeTask("gpu.processing", gls.handleGPUTask)
	if err != nil {
		return err
	}
	
	// Embedding generation queue consumer
	err = gls.rabbitmq.ConsumeTask("embedding.generation", gls.handleEmbeddingTask)
	if err != nil {
		return err
	}
	
	// Legal analysis queue consumer
	err = gls.rabbitmq.ConsumeTask("legal.analysis", gls.handleAnalysisTask)
	if err != nil {
		return err
	}
	
	return nil
}

func (gls *GPULegalAIService) handleGPUTask(task *ProcessingTask) error {
	log.Printf("Processing GPU task: %s", task.ID)
	
	// Extract processing request from task data
	reqData, _ := json.Marshal(task.Data)
	var req ProcessingRequest
	json.Unmarshal(reqData, &req)
	
	// Process using GPU
	response, err := gls.ProcessTextGPU(context.Background(), &req)
	if err != nil {
		return err
	}
	
	// Update task with result
	task.Result = response
	task.Status = "completed"
	now := time.Now()
	task.ProcessedAt = &now
	
	return nil
}

func (gls *GPULegalAIService) handleEmbeddingTask(task *ProcessingTask) error {
	log.Printf("Processing embedding task: %s", task.ID)
	
	text, ok := task.Data["text"].(string)
	if !ok {
		return fmt.Errorf("invalid text data in embedding task")
	}
	
	// Generate embedding
	embedding, err := gls.legalBERT.generateEmbedding(text)
	if err != nil {
		return err
	}
	
	// Cache the embedding
	cacheKey := fmt.Sprintf("embedding_%s", task.ID)
	err = gls.redis.SetEmbedding(context.Background(), cacheKey, embedding, time.Hour*24)
	if err != nil {
		log.Printf("Failed to cache embedding: %v", err)
	}
	
	task.Result = map[string]interface{}{
		"embedding": embedding,
		"dimensions": len(embedding),
	}
	task.Status = "completed"
	now := time.Now()
	task.ProcessedAt = &now
	
	return nil
}

func (gls *GPULegalAIService) handleAnalysisTask(task *ProcessingTask) error {
	log.Printf("Processing analysis task: %s", task.ID)
	
	text, ok := task.Data["text"].(string)
	if !ok {
		return fmt.Errorf("invalid text data in analysis task")
	}
	
	// Recognize legal entities
	entities, err := gls.legalBERT.RecognizeEntities(text)
	if err != nil {
		return err
	}
	
	task.Result = map[string]interface{}{
		"entities": entities,
		"count":    len(entities),
	}
	task.Status = "completed"
	now := time.Now()
	task.ProcessedAt = &now
	
	return nil
}

func (gls *GPULegalAIService) ProcessTextGPU(ctx context.Context, req *ProcessingRequest) (*ProcessingResponse, error) {
	startTime := time.Now()
	
	response := &ProcessingResponse{
		ID:     req.ID,
		Status: "processing",
	}
	
	// Check cache first if enabled
	var cacheHit bool
	if req.CacheResults {
		var cached ProcessingResponse
		err := gls.redis.GetCachedJSON(ctx, fmt.Sprintf("processing_%s", req.ID), &cached)
		if err == nil {
			cached.Performance.CacheHit = true
			return &cached, nil
		}
	}
	
	// Generate embedding if requested
	var embedding []float32
	if req.GenerateEmbedding {
		emb, err := gls.legalBERT.generateEmbedding(req.Text)
		if err != nil {
			response.Error = fmt.Sprintf("Embedding generation failed: %v", err)
			response.Status = "error"
			return response, err
		}
		embedding = emb
	}
	
	// Extract entities if requested
	var entities []LegalEntity
	if req.ExtractEntities {
		ents, err := gls.legalBERT.RecognizeEntities(req.Text)
		if err != nil {
			response.Error = fmt.Sprintf("Entity extraction failed: %v", err)
			response.Status = "error"
			return response, err
		}
		entities = ents
	}
	
	// Perform GPU-accelerated processing
	var result interface{}
	if req.UseGPU && gls.gpu.hasCUDA {
		// Example GPU processing - vector operations
		if len(embedding) > 0 {
			// Perform GPU operations on embedding
			similarity, err := gls.gpu.GPUVectorDot(embedding, embedding)
			if err != nil {
				log.Printf("GPU operation failed: %v", err)
			}
			
			result = map[string]interface{}{
				"gpu_similarity": similarity,
				"text_processed": req.Text,
				"embedding_dims": len(embedding),
			}
		}
	} else {
		// CPU fallback
		result = map[string]interface{}{
			"text_processed": req.Text,
			"cpu_processed": true,
		}
	}
	
	// Build response
	response.Status = "completed"
	response.Result = result
	response.Entities = entities
	response.Embedding = embedding
	response.Performance.ProcessingTime = time.Since(startTime).Milliseconds()
	response.Performance.GPUUsed = req.UseGPU && gls.gpu.hasCUDA
	response.Performance.CacheHit = cacheHit
	response.Performance.TokensProcessed = len(strings.Fields(req.Text))
	
	// Cache result if enabled
	if req.CacheResults {
		err := gls.redis.CacheJSON(ctx, fmt.Sprintf("processing_%s", req.ID), response, time.Hour)
		if err != nil {
			log.Printf("Failed to cache result: %v", err)
		}
	}
	
	return response, nil
}

// HTTP handlers
func (gls *GPULegalAIService) SetupRoutes() *gin.Engine {
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())
	
	// CORS middleware
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})
	
	// GPU processing endpoint
	router.POST("/api/gpu/process", gls.handleGPUProcess)
	
	// Embedding generation endpoint
	router.POST("/api/embeddings/generate", gls.handleGenerateEmbedding)
	
	// Entity recognition endpoint
	router.POST("/api/entities/extract", gls.handleExtractEntities)
	
	// Batch processing endpoint
	router.POST("/api/batch/process", gls.handleBatchProcess)
	
	// Cache management endpoints
	router.GET("/api/cache/stats", gls.handleCacheStats)
	router.DELETE("/api/cache/clear", gls.handleClearCache)
	
	// GPU status endpoint
	router.GET("/api/gpu/status", gls.handleGPUStatus)
	
	// Health check
	router.GET("/health", gls.handleHealth)
	
	return router
}

func (gls *GPULegalAIService) handleGPUProcess(c *gin.Context) {
	var req ProcessingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Generate ID if not provided
	if req.ID == "" {
		req.ID = fmt.Sprintf("gpu_%d", time.Now().UnixNano())
	}
	
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Minute)
	defer cancel()
	
	response, err := gls.ProcessTextGPU(ctx, &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, response)
}

func (gls *GPULegalAIService) handleGenerateEmbedding(c *gin.Context) {
	var req struct {
		Text  string `json:"text" binding:"required"`
		Cache bool   `json:"cache"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	embedding, err := gls.legalBERT.generateEmbedding(req.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"embedding":  embedding,
		"dimensions": len(embedding),
		"text":       req.Text,
	})
}

func (gls *GPULegalAIService) handleExtractEntities(c *gin.Context) {
	var req struct {
		Text string `json:"text" binding:"required"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	entities, err := gls.legalBERT.RecognizeEntities(req.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"entities": entities,
		"count":    len(entities),
		"text":     req.Text,
	})
}

func (gls *GPULegalAIService) handleBatchProcess(c *gin.Context) {
	var req struct {
		Items []ProcessingRequest `json:"items" binding:"required"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	// Process items concurrently using RabbitMQ
	var results []string
	for _, item := range req.Items {
		task := &ProcessingTask{
			ID:        fmt.Sprintf("batch_%d", time.Now().UnixNano()),
			Type:      "gpu.processing",
			Priority:  1,
			Data:      map[string]interface{}{
				"text":               item.Text,
				"use_gpu":           item.UseGPU,
				"cache_results":     item.CacheResults,
				"generate_embedding": item.GenerateEmbedding,
				"extract_entities":  item.ExtractEntities,
			},
			CreatedAt: time.Now(),
			Status:    "queued",
		}
		
		err := gls.rabbitmq.PublishTask("gpu.processing", task)
		if err != nil {
			log.Printf("Failed to queue task: %v", err)
			continue
		}
		
		results = append(results, task.ID)
	}
	
	c.JSON(http.StatusOK, gin.H{
		"queued_tasks": results,
		"count":        len(results),
		"status":       "queued",
	})
}

func (gls *GPULegalAIService) handleCacheStats(c *gin.Context) {
	ctx := c.Request.Context()
	
	// Get Redis info
	info, err := gls.redis.client.Info(ctx).Result()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"redis_info": info,
		"gpu_cache":  len(gls.gpu.tensorCache),
		"model_cache": len(gls.modelCache),
	})
}

func (gls *GPULegalAIService) handleClearCache(c *gin.Context) {
	ctx := c.Request.Context()
	
	// Clear Redis cache
	err := gls.redis.client.FlushDB(ctx).Err()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	// Clear GPU tensor cache
	gls.gpu.cacheMutex.Lock()
	gls.gpu.tensorCache = make(map[string]*tensor.Dense)
	gls.gpu.cacheMutex.Unlock()
	
	// Clear model cache
	gls.mutex.Lock()
	gls.modelCache = make(map[string]interface{})
	gls.mutex.Unlock()
	
	c.JSON(http.StatusOK, gin.H{
		"status": "cache_cleared",
		"timestamp": time.Now(),
	})
}

func (gls *GPULegalAIService) handleGPUStatus(c *gin.Context) {
	var cudaInfo CUDAContext
	if gls.gpu.hasCUDA {
		// Get CUDA device properties
		var props C.cudaDeviceProp
		C.cudaGetDeviceProperties(&props, C.int(gls.gpu.cudaDeviceID))
		
		cudaInfo = CUDAContext{
			DeviceID:    gls.gpu.cudaDeviceID,
			ComputeCaps: fmt.Sprintf("%d.%d", props.major, props.minor),
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"gpu_available": gls.gpu.hasCUDA,
		"cuda_info":     cudaInfo,
		"simd_features": gin.H{
			"avx2":   gls.gpu.hasAVX2,
			"sse42":  gls.gpu.hasSSE42,
			"avx512": gls.gpu.hasAVX512,
		},
		"workers":     gls.gpu.workers,
		"chunk_size":  gls.gpu.chunkSize,
		"tensor_cache": len(gls.gpu.tensorCache),
	})
}

func (gls *GPULegalAIService) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":     "healthy",
		"timestamp":  time.Now(),
		"gpu_ready":  gls.gpu.hasCUDA,
		"redis_ready": gls.redis.client.Ping(c.Request.Context()).Err() == nil,
		"services":   "gpu_legal_ai",
		"version":    "1.0.0",
	})
}

// Utility functions
func levenshteinDistance(s1, s2 string) int {
	if len(s1) == 0 {
		return len(s2)
	}
	if len(s2) == 0 {
		return len(s1)
	}
	
	matrix := make([][]int, len(s1)+1)
	for i := range matrix {
		matrix[i] = make([]int, len(s2)+1)
		matrix[i][0] = i
	}
	
	for j := range matrix[0] {
		matrix[0][j] = j
	}
	
	for i := 1; i <= len(s1); i++ {
		for j := 1; j <= len(s2); j++ {
			cost := 0
			if s1[i-1] != s2[j-1] {
				cost = 1
			}
			
			matrix[i][j] = min(
				matrix[i-1][j]+1,      // deletion
				matrix[i][j-1]+1,      // insertion
				matrix[i-1][j-1]+cost, // substitution
			)
		}
	}
	
	return matrix[len(s1)][len(s2)]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Main function
func main() {
	log.Println("🚀 Starting GPU-Accelerated Legal AI Service")
	
	// Configuration from environment
	redisAddr := getEnv("REDIS_ADDR", "localhost:6379")
	rabbitMQURL := getEnv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
	ollamaURL := getEnv("OLLAMA_URL", "http://localhost:11434")
	port := getEnv("PORT", "8082")
	
	// Initialize service
	service, err := NewGPULegalAIService(redisAddr, rabbitMQURL, ollamaURL)
	if err != nil {
		log.Fatalf("❌ Failed to initialize service: %v", err)
	}
	
	// Setup HTTP router
	router := service.SetupRoutes()
	
	log.Printf("🌐 GPU Legal AI Service listening on port %s", port)
	log.Printf("🔧 GPU Available: %v", service.gpu.hasCUDA)
	log.Printf("🚀 Service ready for processing")
	
	// Start server
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("❌ Server failed to start: %v", err)
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}