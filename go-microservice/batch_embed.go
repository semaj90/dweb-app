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
	"time"

	"github.com/gin-gonic/gin"
	"github.com/minio/simdjson-go"

	fastjson "legal-ai-production/internal/fastjson"
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
	embedCache sync.Map
)

func initBatchEmbed() {
	// Initialize without Redis for now
}

// BatchEmbedHandler processes batch embedding requests with SIMD optimization
func BatchEmbedHandler(c *gin.Context) {
	start := time.Now()

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
		processBatchEmbed(c, req, start)
		return
	}

	// Extract fields using SIMD parser
	req := BatchEmbedRequest{}

	// Get docId
	docIdIter, _ := parsed.Iter.Lookup("docId")
	if docId, _ := docIdIter.StringCvt(); docId != "" {
		req.DocID = docId
	}

	// Get model (optional)
	modelIter, _ := parsed.Iter.Lookup("model")
	if model, _ := modelIter.StringCvt(); model != "" {
		req.Model = model
	} else {
		req.Model = "nomic-embed-text" // Default model
	}

	// Get chunks array
	chunksIter, _ := parsed.Iter.Lookup("chunks")
	if chunksArray, err := chunksIter.Array(); err == nil {
		req.Chunks = make([]string, 0)
		iter := chunksArray.Iter()
		for {
			chunk, iterErr := iter.AdvanceIter()
			if iterErr != nil {
				break
			}
			if chunkStr, err := chunk.StringCvt(); err == nil {
				req.Chunks = append(req.Chunks, chunkStr)
			}
		}
	}

    stream := c.Query("stream") == "1"
	processBatchEmbed(c, req, start, stream)
}

func processBatchEmbed(c *gin.Context, req BatchEmbedRequest, start time.Time, stream bool) {
	// Check cache first
	cacheKey := fmt.Sprintf("embed:%s", req.DocID)
	if cached, exists := embedCache.Load(cacheKey); exists {
		if stream {
			// For cached response in streaming mode just send full payload
			c.Header("Content-Type", "application/json")
			b, _ := fastjson.Marshal(cached)
			c.Writer.Write(b)
			return
		}
		c.JSON(http.StatusOK, cached)
		return
	}

	// Process embeddings with parallel workers
	embeddings := make([][]float32, len(req.Chunks))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 4) // Limit concurrent Ollama calls

	for i, chunk := range req.Chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

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
		// Use fastjson for encoding if possible
		if buf, err := fastjson.EncodeToBuffer(response); err == nil {
			c.Header("Content-Type", "application/json")
			c.Writer.Write(buf.Bytes())
			fastjson.ReleaseBuffer(buf)
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

	// Begin stream with metadata (without embeddings)
	metaOnly := map[string]any{"docId": req.DocID, "model": req.Model, "chunkCount": len(req.Chunks), "processedAt": time.Now().Format(time.RFC3339)}
	if b, err := fastjson.Marshal(metaOnly); err == nil { c.Writer.Write(b); c.Writer.Write([]byte("\n")) } else { mb, _ := json.Marshal(metaOnly); c.Writer.Write(mb); c.Writer.Write([]byte("\n")) }
	flusher.Flush()

	// Stream each embedding
	for i, emb := range embeddings {
		item := map[string]any{"index": i, "embedding": emb}
		b, err := fastjson.Marshal(item)
		if err != nil { b, _ = json.Marshal(item) }
		c.Writer.Write(b)
		c.Writer.Write([]byte("\n"))
		flusher.Flush()
	}

	// Final summary line
	summary := map[string]any{"complete": true, "docId": req.DocID, "total": len(embeddings), "processTimeMs": response.Metadata.ProcessTimeMs}
	if b, err := fastjson.Marshal(summary); err == nil { c.Writer.Write(b) } else { sb, _ := json.Marshal(summary); c.Writer.Write(sb) }
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
		stats := fastjson.GetStats()
		c.Header("X-JSON-Codec", stats.CodecName)
		c.Header("X-JSON-Encodes", strconv.FormatInt(stats.Encodes, 10))
		c.Header("X-JSON-Bytes", strconv.FormatInt(stats.BytesProduced, 10))
		c.JSON(http.StatusOK, stats)
	})
}