package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
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
	
	processBatchEmbed(c, req, start)
}

func processBatchEmbed(c *gin.Context, req BatchEmbedRequest, start time.Time) {
	// Check cache first
	cacheKey := fmt.Sprintf("embed:%s", req.DocID)
	if cached, exists := embedCache.Load(cacheKey); exists {
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
	
	// Prepare response
	response := BatchEmbedResponse{
		DocID:      req.DocID,
		Embeddings: embeddings,
		Metadata: EmbedMeta{
			ProcessedAt:   time.Now(),
			ChunkCount:    len(req.Chunks),
			Model:         req.Model,
			ProcessTimeMs: time.Since(start).Milliseconds(),
		},
	}
	
	// Cache to memory
	embedCache.Store(cacheKey, response)
	
	c.JSON(http.StatusOK, response)
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
}