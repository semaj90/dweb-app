package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"runtime"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
	
	pb "github.com/legal-ai/gpu-service/proto"
	"github.com/bytedance/sonic"
	"github.com/klauspost/compress/zstd"
)

// GPUServiceClient wraps the gRPC client with connection pooling
type GPUServiceClient struct {
	conn   *grpc.ClientConn
	client pb.GPUServiceClient
	mu     sync.RWMutex
}

// SIMDProcessor handles high-performance JSON processing
type SIMDProcessor struct {
	gpuClient    *GPUServiceClient
	encoder      *zstd.Encoder
	decoder      *zstd.Decoder
	workerPool   chan struct{}
	resultCache  sync.Map
	metrics      *ProcessingMetrics
}

// ProcessingMetrics tracks performance data
type ProcessingMetrics struct {
	DocumentsProcessed int64
	TotalProcessingTime time.Duration
	CacheHits          int64
	CacheMisses        int64
	ErrorCount         int64
	mu                 sync.RWMutex
}

// DocumentChunk represents a processed document chunk
type DocumentChunk struct {
	ID          string                 `json:"id"`
	Content     string                 `json:"content"`
	Metadata    map[string]interface{} `json:"metadata"`
	Timestamp   time.Time             `json:"timestamp"`
	ChunkIndex  int                   `json:"chunk_index"`
	TotalChunks int                   `json:"total_chunks"`
}

// ProcessingRequest represents incoming processing requests
type ProcessingRequest struct {
	DocumentID   string                 `json:"document_id"`
	Content      string                 `json:"content"`
	ProcessType  string                 `json:"process_type"`
	Options      map[string]interface{} `json:"options"`
	Priority     int                   `json:"priority"`
}

// ProcessingResult represents the final processing result
type ProcessingResult struct {
	DocumentID       string                 `json:"document_id"`
	Embeddings       [][]float32           `json:"embeddings"`
	Clusters         []int32               `json:"clusters"`
	Similarities     []float32             `json:"similarities"`
	ProcessingTime   time.Duration         `json:"processing_time"`
	Metadata         map[string]interface{} `json:"metadata"`
	BoostTransforms  [][]float32           `json:"boost_transforms"`
}

func main() {
	// Initialize SIMD processor
	processor, err := NewSIMDProcessor()
	if err != nil {
		log.Fatalf("Failed to initialize SIMD processor: %v", err)
	}
	defer processor.Close()

	// Start gRPC server
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen on port 50051: %v", err)
	}

	s := grpc.NewServer(
		grpc.MaxRecvMsgSize(32*1024*1024), // 32MB max message size
		grpc.MaxSendMsgSize(32*1024*1024),
	)

	// Register services
	pb.RegisterGPUServiceServer(s, processor)

	log.Printf("Go SIMD Service starting on port 50051")
	log.Printf("Using %d CPU cores for parallel processing", runtime.NumCPU())
	log.Printf("Worker pool size: %d", cap(processor.workerPool))

	// Start metrics reporting
	go processor.reportMetrics()

	if err := s.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

// NewSIMDProcessor creates a new SIMD processor instance
func NewSIMDProcessor() (*SIMDProcessor, error) {
	// Initialize GPU client connection
	gpuConn, err := grpc.Dial("localhost:50052", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to GPU service: %v", err)
	}

	gpuClient := &GPUServiceClient{
		conn:   gpuConn,
		client: pb.NewGPUServiceClient(gpuConn),
	}

	// Initialize compression
	encoder, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedFastest))
	if err != nil {
		return nil, fmt.Errorf("failed to create encoder: %v", err)
	}

	decoder, err := zstd.NewReader(nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder: %v", err)
	}

	// Create worker pool based on CPU count
	workerPoolSize := runtime.NumCPU() * 2
	workerPool := make(chan struct{}, workerPoolSize)

	processor := &SIMDProcessor{
		gpuClient:   gpuClient,
		encoder:     encoder,
		decoder:     decoder,
		workerPool:  workerPool,
		metrics:     &ProcessingMetrics{},
	}

	return processor, nil
}

// ProcessDocument implements the gRPC service method
func (s *SIMDProcessor) ProcessDocument(ctx context.Context, req *pb.DocumentRequest) (*pb.DocumentResponse, error) {
	startTime := time.Now()

	// Acquire worker slot
	select {
	case s.workerPool <- struct{}{}:
		defer func() { <-s.workerPool }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Check cache first
	cacheKey := fmt.Sprintf("doc:%s:%s", req.DocumentId, req.ProcessType)
	if cached, ok := s.resultCache.Load(cacheKey); ok {
		s.metrics.mu.Lock()
		s.metrics.CacheHits++
		s.metrics.mu.Unlock()
		
		if result, ok := cached.(*pb.DocumentResponse); ok {
			return result, nil
		}
	}

	s.metrics.mu.Lock()
	s.metrics.CacheMisses++
	s.metrics.mu.Unlock()

	// Parse JSON using SIMD-optimized sonic
	var processingReq ProcessingRequest
	if err := sonic.UnmarshalString(req.JsonData, &processingReq); err != nil {
		s.updateErrorMetrics()
		return nil, fmt.Errorf("failed to parse JSON: %v", err)
	}

	// Chunk document for processing
	chunks := s.chunkDocument(processingReq.Content, 512)
	
	// Process chunks in parallel
	embeddingReqs := make([]*pb.EmbeddingRequest, len(chunks))
	for i, chunk := range chunks {
		embeddingReqs[i] = &pb.EmbeddingRequest{
			Text:      chunk.Content,
			ModelName: "nomic-embed-text",
			ChunkId:   chunk.ID,
			Options: map[string]string{
				"normalize": "true",
				"pooling":   "mean",
			},
		}
	}

	// Send to GPU service for processing
	embeddingResp, err := s.gpuClient.client.ProcessEmbeddings(ctx, &pb.EmbeddingBatchRequest{
		Requests: embeddingReqs,
		BatchSize: int32(len(embeddingReqs)),
	})
	if err != nil {
		s.updateErrorMetrics()
		return nil, fmt.Errorf("GPU processing failed: %v", err)
	}

	// Perform clustering if requested
	var clusterResp *pb.ClusterResponse
	if processingReq.ProcessType == "cluster" || processingReq.ProcessType == "full" {
		clusterReq := &pb.ClusterRequest{
			Embeddings:    embeddingResp.Embeddings,
			NumClusters:   8, // Default cluster count
			Algorithm:     "kmeans",
			MaxIterations: 100,
		}

		clusterResp, err = s.gpuClient.client.PerformClustering(ctx, clusterReq)
		if err != nil {
			log.Printf("Clustering failed, continuing without: %v", err)
		}
	}

	// Compute similarities if requested
	var similarityResp *pb.SimilarityResponse
	if processingReq.ProcessType == "similarity" || processingReq.ProcessType == "full" {
		if len(embeddingResp.Embeddings) >= 2 {
			similarityReq := &pb.SimilarityRequest{
				EmbeddingsA: embeddingResp.Embeddings[:len(embeddingResp.Embeddings)/2],
				EmbeddingsB: embeddingResp.Embeddings[len(embeddingResp.Embeddings)/2:],
				Metric:      "cosine",
			}

			similarityResp, err = s.gpuClient.client.ComputeSimilarity(ctx, similarityReq)
			if err != nil {
				log.Printf("Similarity computation failed, continuing without: %v", err)
			}
		}
	}

	// Apply 4D boost transforms if requested
	var boostResp *pb.BoostTransformResponse
	if processingReq.ProcessType == "boost" || processingReq.ProcessType == "full" {
		boostReq := &pb.BoostTransformRequest{
			Embeddings: embeddingResp.Embeddings,
			BoostFactors: []float32{1.2, 1.1, 1.0, 0.9}, // Example boost factors
			Dimensions:   4,
		}

		boostResp, err = s.gpuClient.client.ApplyBoostTransform(ctx, boostReq)
		if err != nil {
			log.Printf("Boost transform failed, continuing without: %v", err)
		}
	}

	// Compile results
	result := &pb.DocumentResponse{
		DocumentId:     req.DocumentId,
		ProcessingTime: float32(time.Since(startTime).Milliseconds()),
		Success:        true,
	}

	// Add embeddings
	if embeddingResp != nil {
		result.Embeddings = embeddingResp.Embeddings
		result.EmbeddingDimensions = embeddingResp.Dimensions
	}

	// Add cluster assignments
	if clusterResp != nil {
		result.ClusterAssignments = clusterResp.Assignments
		result.ClusterCenters = clusterResp.Centers
	}

	// Add similarity scores
	if similarityResp != nil {
		result.SimilarityScores = similarityResp.Scores
	}

	// Add boost transforms
	if boostResp != nil {
		result.BoostTransforms = boostResp.TransformedEmbeddings
	}

	// Add metadata
	metadataJSON, _ := sonic.MarshalString(map[string]interface{}{
		"chunks_processed": len(chunks),
		"processing_type":  processingReq.ProcessType,
		"timestamp":        time.Now(),
		"cpu_cores_used":   runtime.NumCPU(),
	})
	result.Metadata = metadataJSON

	// Cache result
	s.resultCache.Store(cacheKey, result)

	// Update metrics
	s.updateSuccessMetrics(time.Since(startTime))

	return result, nil
}

// StreamDocuments implements streaming document processing
func (s *SIMDProcessor) StreamDocuments(stream pb.GPUService_StreamDocumentsServer) error {
	for {
		req, err := stream.Recv()
		if err != nil {
			return err
		}

		// Process document
		resp, err := s.ProcessDocument(stream.Context(), req)
		if err != nil {
			return err
		}

		// Send response
		if err := stream.Send(resp); err != nil {
			return err
		}
	}
}

// GetHealthStatus implements health check
func (s *SIMDProcessor) GetHealthStatus(ctx context.Context, req *pb.HealthRequest) (*pb.HealthResponse, error) {
	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()

	return &pb.HealthResponse{
		Status: "healthy",
		Uptime: float32(time.Since(time.Now()).Seconds()),
		Metrics: map[string]string{
			"documents_processed":  fmt.Sprintf("%d", s.metrics.DocumentsProcessed),
			"cache_hits":          fmt.Sprintf("%d", s.metrics.CacheHits),
			"cache_misses":        fmt.Sprintf("%d", s.metrics.CacheMisses),
			"error_count":         fmt.Sprintf("%d", s.metrics.ErrorCount),
			"avg_processing_time": fmt.Sprintf("%.2fms", float64(s.metrics.TotalProcessingTime.Nanoseconds())/float64(s.metrics.DocumentsProcessed)/1e6),
		},
	}, nil
}

// chunkDocument splits document into smaller chunks for processing
func (s *SIMDProcessor) chunkDocument(content string, chunkSize int) []DocumentChunk {
	var chunks []DocumentChunk
	runes := []rune(content)
	
	for i := 0; i < len(runes); i += chunkSize {
		end := i + chunkSize
		if end > len(runes) {
			end = len(runes)
		}
		
		chunk := DocumentChunk{
			ID:          fmt.Sprintf("chunk_%d_%d", i, end),
			Content:     string(runes[i:end]),
			ChunkIndex:  i / chunkSize,
			TotalChunks: (len(runes) + chunkSize - 1) / chunkSize,
			Timestamp:   time.Now(),
		}
		
		chunks = append(chunks, chunk)
	}
	
	return chunks
}

// updateSuccessMetrics updates metrics for successful processing
func (s *SIMDProcessor) updateSuccessMetrics(duration time.Duration) {
	s.metrics.mu.Lock()
	defer s.metrics.mu.Unlock()
	
	s.metrics.DocumentsProcessed++
	s.metrics.TotalProcessingTime += duration
}

// updateErrorMetrics updates metrics for failed processing
func (s *SIMDProcessor) updateErrorMetrics() {
	s.metrics.mu.Lock()
	defer s.metrics.mu.Unlock()
	
	s.metrics.ErrorCount++
}

// reportMetrics periodically reports performance metrics
func (s *SIMDProcessor) reportMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.metrics.mu.RLock()
			log.Printf("Metrics - Docs: %d, Cache Hit Rate: %.2f%%, Avg Processing: %.2fms, Errors: %d",
				s.metrics.DocumentsProcessed,
				float64(s.metrics.CacheHits)/float64(s.metrics.CacheHits+s.metrics.CacheMisses)*100,
				float64(s.metrics.TotalProcessingTime.Nanoseconds())/float64(s.metrics.DocumentsProcessed)/1e6,
				s.metrics.ErrorCount,
			)
			s.metrics.mu.RUnlock()
		}
	}
}

// Close cleans up resources
func (s *SIMDProcessor) Close() error {
	if s.gpuClient != nil && s.gpuClient.conn != nil {
		s.gpuClient.conn.Close()
	}
	if s.encoder != nil {
		s.encoder.Close()
	}
	if s.decoder != nil {
		s.decoder.Close()
	}
	return nil
}