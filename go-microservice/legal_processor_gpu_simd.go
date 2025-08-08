package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/bytedance/sonic"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/tidwall/gjson"
)

/*
#cgo LDFLAGS: -lcudart -lcublas
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <immintrin.h>

// CUDA kernel for parallel similarity calculation
__global__ void batch_similarity_kernel(float* queries, float* documents, float* results, 
                                       int num_queries, int num_docs, int embedding_dim) {
    int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (doc_idx < num_docs && query_idx < num_queries) {
        float sum = 0.0f;
        for (int i = 0; i < embedding_dim; i++) {
            sum += queries[query_idx * embedding_dim + i] * 
                   documents[doc_idx * embedding_dim + i];
        }
        results[query_idx * num_docs + doc_idx] = sum;
    }
}

// CPU SIMD fallback for non-GPU systems
void simd_dot_product_avx2(float* a, float* b, float* result, int len) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < len; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(sum_high, sum_low);
    
    __m128 temp = _mm_hadd_ps(sum128, sum128);
    temp = _mm_hadd_ps(temp, temp);
    
    *result = _mm_cvtss_f32(temp);
}

// GPU Context Management
typedef struct {
    cublasHandle_t handle;
    float* d_queries;
    float* d_documents;
    float* d_results;
    int max_docs;
    int embedding_dim;
    bool initialized;
} GPUContext;

GPUContext* init_gpu_context(int max_docs, int embedding_dim) {
    GPUContext* ctx = (GPUContext*)malloc(sizeof(GPUContext));
    
    if (cudaSetDevice(0) != cudaSuccess) {
        ctx->initialized = false;
        return ctx;
    }
    
    if (cublasCreate(&ctx->handle) != CUBLAS_STATUS_SUCCESS) {
        ctx->initialized = false;
        return ctx;
    }
    
    // Allocate GPU memory
    if (cudaMalloc(&ctx->d_queries, embedding_dim * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ctx->d_documents, max_docs * embedding_dim * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ctx->d_results, max_docs * sizeof(float)) != cudaSuccess) {
        ctx->initialized = false;
        return ctx;
    }
    
    ctx->max_docs = max_docs;
    ctx->embedding_dim = embedding_dim;
    ctx->initialized = true;
    
    return ctx;
}

void cleanup_gpu_context(GPUContext* ctx) {
    if (ctx->initialized) {
        cudaFree(ctx->d_queries);
        cudaFree(ctx->d_documents);
        cudaFree(ctx->d_results);
        cublasDestroy(ctx->handle);
    }
    free(ctx);
}

// High-performance GPU similarity calculation
int gpu_similarity_search(GPUContext* ctx, float* query_embedding, 
                         float* document_embeddings, float* results, int num_docs) {
    if (!ctx->initialized) return -1;
    
    // Copy data to GPU
    if (cudaMemcpy(ctx->d_queries, query_embedding, 
                   ctx->embedding_dim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return -2;
    }
    
    if (cudaMemcpy(ctx->d_documents, document_embeddings, 
                   num_docs * ctx->embedding_dim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return -3;
    }
    
    // Use cuBLAS SGEMV for optimized matrix-vector multiplication (Tensor Core acceleration)
    const float alpha = 1.0f, beta = 0.0f;
    if (cublasSgemv(ctx->handle, CUBLAS_OP_T, ctx->embedding_dim, num_docs,
                    &alpha, ctx->d_documents, ctx->embedding_dim,
                    ctx->d_queries, 1, &beta, ctx->d_results, 1) != CUBLAS_STATUS_SUCCESS) {
        return -4;
    }
    
    // Copy results back to CPU
    if (cudaMemcpy(results, ctx->d_results, num_docs * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -5;
    }
    
    return 0;
}

// CPU SIMD fallback
void cpu_simd_similarity_search(float* query_embedding, float* document_embeddings, 
                               float* results, int num_docs, int embedding_dim) {
    for (int i = 0; i < num_docs; i++) {
        simd_dot_product_avx2(query_embedding, 
                             &document_embeddings[i * embedding_dim], 
                             &results[i], embedding_dim);
    }
}
*/
import "C"

// Configuration
const (
	postgresURL       = "postgres://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db"
	neo4jURI          = "bolt://localhost:7687"
	neo4jUser         = "neo4j"
	neo4jPassword     = "legalai123"
	analysisOutputDir = "./generated_reports"
	
	// Performance tuning
	maxDocuments     = 10000  // Adjust based on GPU VRAM
	embeddingDim     = 768    // Standard embedding dimension
	maxWorkers       = 8      // CPU workers for non-GPU tasks
	batchSize        = 512    // GPU batch processing size
)

// LLM Endpoint Configuration
type LLMEndpoint struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	APIKey   string `json:"apiKey,omitempty"`
	Model    string `json:"model"`
	Headers  map[string]string `json:"headers,omitempty"`
}

var llmEndpoints = map[string]LLMEndpoint{
	"ollama": {
		Name:  "Ollama",
		URL:   "http://localhost:11434/api",
		Model: "gemma3-legal",
	},
	"claude": {
		Name:  "Claude",
		URL:   "https://api.anthropic.com/v1/messages",
		Model: "claude-3-sonnet-20240229",
	},
	"gemini": {
		Name:  "Gemini",
		URL:   "https://generativelanguage.googleapis.com/v1beta/models",
		Model: "gemini-pro",
	},
	"llamacpp": {
		Name:  "llama.cpp",
		URL:   "http://localhost:8081/completion",
		Model: "local",
	},
}

// Core Structs
type FilePathsPayload struct {
	FilePaths []string `json:"filePaths"`
}

type SimilaritySearchRequest struct {
	QueryEmbedding     []float32   `json:"queryEmbedding"`
	DocumentEmbeddings [][]float32 `json:"documentEmbeddings"`
	DocumentIDs        []string    `json:"documentIds"`
	TopK               int         `json:"topK"`
	UseGPU             bool        `json:"useGPU"`
}

type SimilarityResult struct {
	DocumentID string  `json:"documentId"`
	Score      float32 `json:"score"`
	Rank       int     `json:"rank"`
}

type SimilaritySearchResponse struct {
	Results   []SimilarityResult `json:"results"`
	TimingMs  int64              `json:"timingMs"`
	Method    string             `json:"method"`
	GPUUsed   bool               `json:"gpuUsed"`
}

type LLMRequest struct {
	Provider string      `json:"provider"`
	Model    string      `json:"model,omitempty"`
	Prompt   string      `json:"prompt"`
	Format   string      `json:"format,omitempty"`
	Options  interface{} `json:"options,omitempty"`
}

type LLMResponse struct {
	Provider string      `json:"provider"`
	Model    string      `json:"model"`
	Response string      `json:"response"`
	TimingMs int64       `json:"timingMs"`
	Error    string      `json:"error,omitempty"`
}

type RAGSearchRequest struct {
	Query      string `json:"query"`
	TopK       int    `json:"topK"`
	UseGPU     bool   `json:"useGPU"`
	LLMProvider string `json:"llmProvider"`
}

type AnalysisReport struct {
	FilePath        string   `json:"filePath"`
	Severity        string   `json:"severity"`
	IssueSummary    string   `json:"issueSummary"`
	Recommendations []string `json:"recommendations"`
	TodoList        []string `json:"todoList"`
	ProcessingTime  int64    `json:"processingTimeMs"`
	Method          string   `json:"method"`
}

// Global GPU context and worker pool
var (
	gpuCtx     *C.GPUContext
	gpuEnabled bool
	workerPool *WorkerPool
)

type WorkerPool struct {
	workers int
	jobs    chan func()
	wg      sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
	return &WorkerPool{
		workers: workers,
		jobs:    make(chan func(), workers*2),
	}
}

func (wp *WorkerPool) Start() {
	for i := 0; i < wp.workers; i++ {
		wp.wg.Add(1)
		go wp.worker()
	}
}

func (wp *WorkerPool) worker() {
	defer wp.wg.Done()
	for job := range wp.jobs {
		job()
	}
}

func (wp *WorkerPool) Submit(job func()) {
	wp.jobs <- job
}

func (wp *WorkerPool) Stop() {
	close(wp.jobs)
	wp.wg.Wait()
}

func main() {
	// Initialize logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	
	// Ensure output directory exists
	if err := os.MkdirAll(analysisOutputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Initialize GPU context
	log.Println("🚀 Initializing GPU context...")
	gpuCtx = C.init_gpu_context(C.int(maxDocuments), C.int(embeddingDim))
	gpuEnabled = bool(gpuCtx.initialized)
	if gpuEnabled {
		log.Println("✅ GPU acceleration enabled")
	} else {
		log.Println("⚠️  GPU not available, using CPU SIMD fallback")
	}
	defer C.cleanup_gpu_context(gpuCtx)

	// Initialize worker pool
	workerPool = NewWorkerPool(maxWorkers)
	workerPool.Start()
	defer workerPool.Stop()

	// Database connections
	ctx := context.Background()
	dbpool, err := pgxpool.New(ctx, postgresURL)
	if err != nil {
		log.Fatalf("Unable to connect to PostgreSQL: %v\n", err)
	}
	defer dbpool.Close()
	log.Println("✅ Connected to PostgreSQL")

	driver, err := neo4j.NewDriverWithContext(neo4jURI, neo4j.BasicAuth(neo4jUser, neo4jPassword, ""))
	if err != nil {
		log.Fatalf("Unable to connect to Neo4j: %v\n", err)
	}
	defer driver.Close(ctx)
	log.Println("✅ Connected to Neo4j")

	// Setup Gin router
	router := gin.Default()
	
	// Enable CORS for SvelteKit integration
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

	// Health check with system status
	router.GET("/health", func(c *gin.Context) {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		
		c.JSON(http.StatusOK, gin.H{
			"status":        "ok",
			"gpu_enabled":   gpuEnabled,
			"goroutines":    runtime.NumGoroutine(),
			"heap_alloc":    m.HeapAlloc,
			"sys_memory":    m.Sys,
			"gc_runs":       m.NumGC,
			"embedding_dim": embeddingDim,
			"max_docs":     maxDocuments,
		})
	})

	// Original batch processing endpoint
	router.POST("/batch-process-files", func(c *gin.Context) {
		var payload FilePathsPayload
		if err := c.ShouldBindJSON(&payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		go processFiles(payload.FilePaths, dbpool, driver)
		c.JSON(http.StatusAccepted, gin.H{
			"status":     "processing_started",
			"file_count": len(payload.FilePaths),
			"gpu_enabled": gpuEnabled,
		})
	})

	// Enhanced GPU/SIMD similarity search
	router.POST("/similarity-search", func(c *gin.Context) {
		var req SimilaritySearchRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
			return
		}

		startTime := time.Now()
		results, method, err := performSimilaritySearch(req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		response := SimilaritySearchResponse{
			Results:  results,
			TimingMs: time.Since(startTime).Milliseconds(),
			Method:   method,
			GPUUsed:  method == "GPU",
		}

		c.JSON(http.StatusOK, response)
	})

	// Multi-LLM endpoint
	router.POST("/llm-request", func(c *gin.Context) {
		var req LLMRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
			return
		}

		startTime := time.Now()
		response, err := callLLMEndpoint(req)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		response.TimingMs = time.Since(startTime).Milliseconds()
		c.JSON(http.StatusOK, response)
	})

	// Enhanced RAG search with GPU acceleration and multi-LLM support
	router.POST("/rag-search", func(c *gin.Context) {
		handleEnhancedRAGSearch(c, dbpool)
	})

	// Document analysis endpoint
	router.POST("/analyze-document", func(c *gin.Context) {
		handleDocumentAnalysis(c)
	})

	// LLM endpoints list
	router.GET("/llm-endpoints", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"endpoints": llmEndpoints})
	})

	log.Printf("🚀 Enhanced GPU+SIMD Legal Processor listening on :8080")
	log.Printf("📊 GPU Enabled: %v | Max Documents: %d | Workers: %d", gpuEnabled, maxDocuments, maxWorkers)
	router.Run(":8080")
}

// Enhanced similarity search with GPU/SIMD fallback
func performSimilaritySearch(req SimilaritySearchRequest) ([]SimilarityResult, string, error) {
	numDocs := len(req.DocumentEmbeddings)
	if numDocs == 0 {
		return []SimilarityResult{}, "NONE", nil
	}

	if numDocs > maxDocuments {
		return nil, "", fmt.Errorf("too many documents: %d (max: %d)", numDocs, maxDocuments)
	}

	// Validate embedding dimensions
	for i, embedding := range req.DocumentEmbeddings {
		if len(embedding) != embeddingDim {
			return nil, "", fmt.Errorf("document %d embedding dimension mismatch: expected %d, got %d", i, embeddingDim, len(embedding))
		}
	}

	if len(req.QueryEmbedding) != embeddingDim {
		return nil, "", fmt.Errorf("query embedding dimension mismatch: expected %d, got %d", embeddingDim, len(req.QueryEmbedding))
	}

	scores := make([]float32, numDocs)
	method := "CPU SIMD"

	// Try GPU first if enabled and requested
	if gpuEnabled && req.UseGPU {
		flatEmbeddings := make([]float32, numDocs*embeddingDim)
		for i, embedding := range req.DocumentEmbeddings {
			copy(flatEmbeddings[i*embeddingDim:(i+1)*embeddingDim], embedding)
		}

		result := C.gpu_similarity_search(
			gpuCtx,
			(*C.float)(unsafe.Pointer(&req.QueryEmbedding[0])),
			(*C.float)(unsafe.Pointer(&flatEmbeddings[0])),
			(*C.float)(unsafe.Pointer(&scores[0])),
			C.int(numDocs),
		)

		if result == 0 {
			method = "GPU"
		} else {
			log.Printf("GPU search failed with code %d, falling back to CPU SIMD", result)
			// Fallback to CPU SIMD
			performCPUSIMDSearch(req.QueryEmbedding, req.DocumentEmbeddings, scores)
		}
	} else {
		// Use CPU SIMD
		performCPUSIMDSearch(req.QueryEmbedding, req.DocumentEmbeddings, scores)
	}

	// Create and sort results
	results := make([]SimilarityResult, numDocs)
	for i := 0; i < numDocs; i++ {
		results[i] = SimilarityResult{
			DocumentID: req.DocumentIDs[i],
			Score:      scores[i],
			Rank:       i + 1,
		}
	}

	// Sort by score (descending)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Update ranks and return top K
	for i := range results {
		results[i].Rank = i + 1
	}

	topK := req.TopK
	if topK > len(results) || topK <= 0 {
		topK = len(results)
	}

	return results[:topK], method, nil
}

// CPU SIMD fallback implementation
func performCPUSIMDSearch(queryEmbedding []float32, documentEmbeddings [][]float32, scores []float32) {
	numDocs := len(documentEmbeddings)
	flatDocs := make([]float32, numDocs*embeddingDim)
	
	for i, embedding := range documentEmbeddings {
		copy(flatDocs[i*embeddingDim:(i+1)*embeddingDim], embedding)
	}

	C.cpu_simd_similarity_search(
		(*C.float)(unsafe.Pointer(&queryEmbedding[0])),
		(*C.float)(unsafe.Pointer(&flatDocs[0])),
		(*C.float)(unsafe.Pointer(&scores[0])),
		C.int(numDocs),
		C.int(embeddingDim),
	)
}

// Multi-LLM endpoint handler
func callLLMEndpoint(req LLMRequest) (*LLMResponse, error) {
	endpoint, exists := llmEndpoints[req.Provider]
	if !exists {
		return nil, fmt.Errorf("unknown LLM provider: %s", req.Provider)
	}

	model := req.Model
	if model == "" {
		model = endpoint.Model
	}

	var requestBody []byte
	var url string
	var headers map[string]string

	switch req.Provider {
	case "ollama":
		reqData := map[string]interface{}{
			"model":  model,
			"prompt": req.Prompt,
			"stream": false,
		}
		if req.Format != "" {
			reqData["format"] = req.Format
		}
		requestBody, _ = sonic.Marshal(reqData)
		url = fmt.Sprintf("%s/generate", endpoint.URL)
		headers = map[string]string{"Content-Type": "application/json"}

	case "claude":
		reqData := map[string]interface{}{
			"model":      model,
			"max_tokens": 4000,
			"messages":   []map[string]string{{"role": "user", "content": req.Prompt}},
		}
		requestBody, _ = sonic.Marshal(reqData)
		url = endpoint.URL
		headers = map[string]string{
			"Content-Type":    "application/json",
			"x-api-key":       endpoint.APIKey,
			"anthropic-version": "2023-06-01",
		}

	case "gemini":
		reqData := map[string]interface{}{
			"contents": []map[string]interface{}{
				{"parts": []map[string]string{{"text": req.Prompt}}},
			},
		}
		requestBody, _ = sonic.Marshal(reqData)
		url = fmt.Sprintf("%s/%s:generateContent?key=%s", endpoint.URL, model, endpoint.APIKey)
		headers = map[string]string{"Content-Type": "application/json"}

	case "llamacpp":
		reqData := map[string]interface{}{
			"prompt": req.Prompt,
			"n_predict": 4000,
			"stream": false,
		}
		requestBody, _ = sonic.Marshal(reqData)
		url = endpoint.URL
		headers = map[string]string{"Content-Type": "application/json"}

	default:
		return nil, fmt.Errorf("unsupported provider: %s", req.Provider)
	}

	client := &http.Client{Timeout: 120 * time.Second}
	httpReq, err := http.NewRequest("POST", url, bytes.NewBuffer(requestBody))
	if err != nil {
		return nil, err
	}

	for key, value := range headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return &LLMResponse{
			Provider: req.Provider,
			Model:    model,
			Error:    fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBody)),
		}, nil
	}

	// Parse response based on provider
	var responseText string
	switch req.Provider {
	case "ollama":
		responseText = gjson.Get(string(respBody), "response").String()
	case "claude":
		responseText = gjson.Get(string(respBody), "content.0.text").String()
	case "gemini":
		responseText = gjson.Get(string(respBody), "candidates.0.content.parts.0.text").String()
	case "llamacpp":
		responseText = gjson.Get(string(respBody), "content").String()
	}

	return &LLMResponse{
		Provider: req.Provider,
		Model:    model,
		Response: responseText,
	}, nil
}

// Enhanced RAG search handler
func handleEnhancedRAGSearch(c *gin.Context, dbpool *pgxpool.Pool) {
	var req RAGSearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	startTime := time.Now()

	// Step 1: Generate query embedding using specified LLM provider
	llmReq := LLMRequest{
		Provider: req.LLMProvider,
		Prompt:   req.Query,
	}
	if req.LLMProvider == "" {
		llmReq.Provider = "ollama"
	}

	// Get embedding (simplified - in production you'd call a specialized embedding endpoint)
	queryEmbedding, err := getEmbedding(req.Query)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate query embedding"})
		return
	}

	// Step 2: Retrieve candidate documents from PostgreSQL
	rows, err := dbpool.Query(context.Background(),
		`SELECT file_path, embedding, summary FROM indexed_files LIMIT 1000`)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Database query failed"})
		return
	}
	defer rows.Close()

	var documentEmbeddings [][]float32
	var documentIDs []string
	var summaries []string

	for rows.Next() {
		var filePath, embeddingStr, summary string
		if err := rows.Scan(&filePath, &embeddingStr, &summary); err != nil {
			continue
		}

		embedding := parseEmbeddingString(embeddingStr)
		if len(embedding) == embeddingDim {
			documentEmbeddings = append(documentEmbeddings, embedding)
			documentIDs = append(documentIDs, filePath)
			summaries = append(summaries, summary)
		}
	}

	// Step 3: GPU/SIMD accelerated similarity search
	searchReq := SimilaritySearchRequest{
		QueryEmbedding:     queryEmbedding,
		DocumentEmbeddings: documentEmbeddings,
		DocumentIDs:        documentIDs,
		TopK:               req.TopK,
		UseGPU:             req.UseGPU,
	}

	results, method, err := performSimilaritySearch(searchReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Similarity search failed"})
		return
	}

	// Step 4: Enhance results with summaries and context
	enhancedResults := make([]map[string]interface{}, len(results))
	for i, result := range results {
		var summary string
		for j, docID := range documentIDs {
			if docID == result.DocumentID {
				summary = summaries[j]
				break
			}
		}

		enhancedResults[i] = map[string]interface{}{
			"documentId": result.DocumentID,
			"score":      result.Score,
			"rank":       result.Rank,
			"summary":    summary,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"query":        req.Query,
		"results":      enhancedResults,
		"method":       method,
		"gpu_used":     method == "GPU",
		"timing_ms":    time.Since(startTime).Milliseconds(),
		"total_docs":   len(documentEmbeddings),
		"llm_provider": req.LLMProvider,
	})
}

// Document analysis handler
func handleDocumentAnalysis(c *gin.Context) {
	type AnalysisRequest struct {
		FilePath    string `json:"filePath"`
		Content     string `json:"content"`
		LLMProvider string `json:"llmProvider"`
	}

	var req AnalysisRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	startTime := time.Now()

	prompt := fmt.Sprintf(`Analyze the following legal document and provide a structured assessment:

File: %s

Content:
%s

Provide a JSON response with:
{
  "severity": "high|medium|low",
  "issueSummary": "Brief summary of key issues",
  "recommendations": ["recommendation1", "recommendation2"],
  "todoList": ["task1", "task2"]
}`, req.FilePath, req.Content)

	llmReq := LLMRequest{
		Provider: req.LLMProvider,
		Prompt:   prompt,
		Format:   "json",
	}
	if req.LLMProvider == "" {
		llmReq.Provider = "ollama"
	}

	llmResp, err := callLLMEndpoint(llmReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "LLM analysis failed"})
		return
	}

	if llmResp.Error != "" {
		c.JSON(http.StatusInternalServerError, gin.H{"error": llmResp.Error})
		return
	}

	var report AnalysisReport
	if err := sonic.Unmarshal([]byte(llmResp.Response), &report); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse analysis response"})
		return
	}

	report.FilePath = req.FilePath
	report.ProcessingTime = time.Since(startTime).Milliseconds()
	report.Method = fmt.Sprintf("%s via %s", llmResp.Model, llmResp.Provider)

	// Save reports to files
	go saveAnalysisReports(report, llmResp.Response)

	c.JSON(http.StatusOK, report)
}

// Helper functions
func parseEmbeddingString(embStr string) []float32 {
	embStr = strings.Trim(embStr, "[]")
	jsonStr := "[" + embStr + "]"
	
	result := gjson.Get(jsonStr, "@this")
	if !result.IsArray() {
		return nil
	}

	embedding := make([]float32, 0, embeddingDim)
	result.ForEach(func(_, value gjson.Result) bool {
		embedding = append(embedding, float32(value.Float()))
		return true
	})

	return embedding
}

func getEmbedding(text string) ([]float32, error) {
	// This would call your embedding model (e.g., nomic-embed-text)
	// Simplified implementation - replace with actual embedding call
	embedding := make([]float32, embeddingDim)
	for i := range embedding {
		embedding[i] = 0.1 // Placeholder
	}
	return embedding, nil
}

func saveAnalysisReports(report AnalysisReport, jsonResponse string) {
	baseName := filepath.Base(report.FilePath)
	
	// JSON report
	os.WriteFile(filepath.Join(analysisOutputDir, baseName+".json"), []byte(jsonResponse), 0644)
	
	// Markdown report
	mdContent := fmt.Sprintf(`# Analysis Report: %s

**Severity**: %s  
**Processing Time**: %dms  
**Method**: %s

## Issue Summary
%s

## Recommendations
%s

## Todo List
%s
`, report.FilePath, report.Severity, report.ProcessingTime, report.Method,
		report.IssueSummary,
		"- "+strings.Join(report.Recommendations, "\n- "),
		"- [ ] "+strings.Join(report.TodoList, "\n- [ ] "))
	
	os.WriteFile(filepath.Join(analysisOutputDir, baseName+".md"), []byte(mdContent), 0644)
}

// Simplified file processing for compatibility
func processFiles(paths []string, dbpool *pgxpool.Pool, driver neo4j.DriverWithContext) {
	log.Printf("Processing %d files...", len(paths))
	// Implementation would process files using the worker pool
	// This is a simplified version for space constraints
}
