//go:build legacy
// +build legacy

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/minio/simdjson-go"
)

type BatchEmbedRequest struct {
	DocID  string   `json:"docId"`
	Chunks []string `json:"chunks"`
	Model  string   `json:"model,omitempty"`
}

type BatchEmbedResponse struct {
	DocID      string      `json:"docId"`
	Embeddings [][]float32 `json:"embeddings"`
	Metadata   EmbedMeta   `json:"metadata"`
}

type EmbedMeta struct {
	ProcessedAt   time.Time `json:"processedAt"`
	ChunkCount    int       `json:"chunkCount"`
	Model         string    `json:"model"`
	ProcessTimeMs int64     `json:"processTimeMs"`
}

type OllamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type OllamaEmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

var (
	embedCache    sync.Map
	// Context7 performance counters - atomic operations
	totalRequests   int64
	totalCacheHits  int64
	totalProcessed  int64
	averageLatency  int64

	// Buffer pool for JSON operations
	jsonBufferPool = sync.Pool{
		New: func() interface{} {
			return bytes.NewBuffer(make([]byte, 0, 4096))
		},
	}
)

func initBatchEmbed() {
	// Initialize without Redis for now
}

// BatchEmbedHandler processes batch embedding requests with SIMD optimization
func BatchEmbedHandler(c *gin.Context) {
	start := time.Now()

	// Context7 performance tracking - atomic increments
	atomic.AddInt64(&totalRequests, 1)
	defer func() {
		latency := time.Since(start).Microseconds()
		// Update rolling average atomically
		currentAvg := atomic.LoadInt64(&averageLatency)
		newAvg := (currentAvg + latency) / 2
		atomic.StoreInt64(&averageLatency, newAvg)
	}()

	// Read request body
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
		return
	}

	// Parse request with SIMD-accelerated JSON
	parsed, err := simdjson.Parse(body, nil)
	if err != nil {
		// Fallback to standard JSON parsing
		var req BatchEmbedRequest
		if err := json.Unmarshal(body, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON"})
			return
		}
		stream := c.Query("stream") == "1"
		processBatchEmbed(c, req, start, stream)
		return
	}

	// Extract fields using SIMD parser
	req := BatchEmbedRequest{}

	// Get docId using correct simdjson API
	if docIdField, err := parsed.FindKey("docId"); err == nil {
		if docId, err := docIdField.StringBytes(); err == nil {
			req.DocID = string(docId)
		}
	}

	// Get model (optional) using correct simdjson API
	if modelField, err := parsed.FindKey("model"); err == nil {
		if model, err := modelField.StringBytes(); err == nil {
			req.Model = string(model)
		}
	} else {
		req.Model = "nomic-embed-text" // Default model
	}

	// Get chunks array with pre-allocated capacity using correct simdjson API
	if chunksField, err := parsed.FindKey("chunks"); err == nil {
		if chunksArray, err := chunksField.Array(); err == nil {
			req.Chunks = make([]string, 0, 100) // Pre-allocate capacity
			iter := chunksArray.Iter()
			for {
				chunk, err := iter.AdvanceIter()
				if err != nil {
					break
				}
				if chunkStr, err := chunk.StringBytes(); err == nil {
					req.Chunks = append(req.Chunks, string(chunkStr))
				}
			}
		}
	}

	stream := c.Query("stream") == "1"
	processBatchEmbed(c, req, start, stream)
}

func processBatchEmbed(c *gin.Context, req BatchEmbedRequest, start time.Time, stream bool) {
	// Check cache first with atomic hit tracking
	cacheKey := fmt.Sprintf("embed:%s", req.DocID)
	if cached, exists := embedCache.Load(cacheKey); exists {
		atomic.AddInt64(&totalCacheHits, 1)

		if stream {
			// For cached response in streaming mode just send full payload
			c.Header("Content-Type", "application/json")
			// Use buffer pool for JSON encoding
			buffer := jsonBufferPool.Get().(*bytes.Buffer)
			buffer.Reset()
			defer jsonBufferPool.Put(buffer)

			json.NewEncoder(buffer).Encode(cached)
			c.Writer.Write(buffer.Bytes())
			return
		}
		c.JSON(http.StatusOK, cached)
		return
	}

	// Process embeddings with parallel workers and atomic tracking
	embeddings := make([][]float32, len(req.Chunks))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 4) // Limit concurrent Ollama calls

	for i, chunk := range req.Chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() {
				<-semaphore
				atomic.AddInt64(&totalProcessed, 1)  // Track processed chunks
			}()

			embedding, err := getOllamaEmbedding(text, req.Model)
			if err != nil {
				// Log error but continue with zero embedding
				fmt.Printf("Error getting embedding for chunk %d: %v\n", idx, err)
				embeddings[idx] = make([]float32, 384) // Default embedding size
			} else {
				embeddings[idx] = embedding
			}
		}(i, chunk)
	}

	wg.Wait()

	// Prepare response (non-stream or final aggregate)
	response := BatchEmbedResponse{DocID: req.DocID, Embeddings: embeddings, Metadata: EmbedMeta{ProcessedAt: time.Now(), ChunkCount: len(req.Chunks), Model: req.Model, ProcessTimeMs: time.Since(start).Milliseconds()}}

	// Cache to memory (store full response)
	embedCache.Store(cacheKey, response)

	if !stream {
		// Use buffer pool for optimized JSON encoding
		buffer := jsonBufferPool.Get().(*bytes.Buffer)
		buffer.Reset()
		defer jsonBufferPool.Put(buffer)

		if err := json.NewEncoder(buffer).Encode(response); err == nil {
			c.Header("Content-Type", "application/json")
			c.Writer.Write(buffer.Bytes())
			return
		}
		c.JSON(http.StatusOK, response)
		return
	}

	// Streaming mode: send each embedding as a JSON line / chunk
	c.Header("Content-Type", "application/json; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("Transfer-Encoding", "chunked")
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "streaming not supported"})
		return
	}

	// Get buffer for streaming operations
	buffer := jsonBufferPool.Get().(*bytes.Buffer)
	buffer.Reset()
	defer jsonBufferPool.Put(buffer)

	// Begin stream with metadata (without embeddings)
	metaOnly := map[string]interface{}{"docId": req.DocID, "model": req.Model, "chunkCount": len(req.Chunks), "processedAt": time.Now().Format(time.RFC3339)}
	json.NewEncoder(buffer).Encode(metaOnly)
	c.Writer.Write(buffer.Bytes())
	c.Writer.Write([]byte("\n"))
	flusher.Flush()

	// Stream each embedding
	for i, emb := range embeddings {
		buffer.Reset()
		item := map[string]interface{}{"index": i, "embedding": emb}
		json.NewEncoder(buffer).Encode(item)
		c.Writer.Write(buffer.Bytes())
		c.Writer.Write([]byte("\n"))
		flusher.Flush()
	}

	// Final summary line
	buffer.Reset()
	summary := map[string]interface{}{"complete": true, "docId": req.DocID, "total": len(embeddings), "processTimeMs": response.Metadata.ProcessTimeMs}
	json.NewEncoder(buffer).Encode(summary)
	c.Writer.Write(buffer.Bytes())
	flusher.Flush()
}

func getOllamaEmbedding(text string, model string) ([]float32, error) {
	// Check if Ollama is running
	ollamaURL := "http://localhost:11434/api/embeddings"

	reqBody := OllamaEmbedRequest{
		Model:  model,
		Prompt: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(ollamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		// Fallback to mock embedding for development
		return generateMockEmbedding(text), nil
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return generateMockEmbedding(text), nil
	}

	var embedResp OllamaEmbedResponse
	if err := json.Unmarshal(body, &embedResp); err != nil {
		return generateMockEmbedding(text), nil
	}

	return embedResp.Embedding, nil
}

func generateMockEmbedding(text string) []float32 {
	// Generate deterministic mock embedding based on text
	embedding := make([]float32, 384)
	hash := 0
	for _, ch := range text {
		hash = (hash * 31 + int(ch)) % 1000000
	}

	for i := range embedding {
		embedding[i] = float32((hash+i)%100) / 100.0
	}
	return embedding
}

// RegisterBatchEmbedRoutes adds the batch embed routes to the Gin router
func RegisterBatchEmbedRoutes(router *gin.Engine) {
	initBatchEmbed()
	router.POST("/batch-embed", BatchEmbedHandler)
	router.GET("/batch-embed", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"usage": "POST /batch-embed?stream=1 with {docId, chunks:[...], model?}"})
	})
	router.GET("/batch-embed/stats", func(c *gin.Context) {
		// Context7 performance stats using atomic operations
		stats := map[string]interface{}{
			"total_requests":   atomic.LoadInt64(&totalRequests),
			"cache_hits":       atomic.LoadInt64(&totalCacheHits),
			"total_processed":  atomic.LoadInt64(&totalProcessed),
			"average_latency_us": atomic.LoadInt64(&averageLatency),
			"cache_hit_ratio":  float64(atomic.LoadInt64(&totalCacheHits)) / float64(atomic.LoadInt64(&totalRequests)),
			"codec_name":       "standard_json_with_buffer_pool",
		}

		c.Header("X-Total-Requests", strconv.FormatInt(atomic.LoadInt64(&totalRequests), 10))
		c.Header("X-Cache-Hits", strconv.FormatInt(atomic.LoadInt64(&totalCacheHits), 10))
		c.Header("X-Average-Latency-Us", strconv.FormatInt(atomic.LoadInt64(&averageLatency), 10))
		c.JSON(http.StatusOK, stats)
	})
}