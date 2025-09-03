// vector_ranking_integration.go
// Integrates PostgreSQL pgvector search with MinIO document storage and QUIC ranking cache
// Provides ultra-fast vector search with bit-packed result caching

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/minio/minio-go/v7"
	"github.com/pgvector/pgvector-go"
)

// VectorSearchService integrates vector search with MinIO and ranking cache
type VectorSearchService struct {
	pgPool      *pgxpool.Pool
	minioClient *minio.Client
	rankCache   *RankingCache
}

// VectorDocument represents a document with vector embedding
type VectorDocument struct {
	ID           uint64    `json:"id" db:"id"`
	Title        string    `json:"title" db:"title"`
	Content      string    `json:"content" db:"content"`
	DocumentType string    `json:"document_type" db:"document_type"`
	Embedding    []float32 `json:"embedding" db:"embedding"`
	MinIOPath    string    `json:"minio_path" db:"minio_path"`
	Summary      string    `json:"summary" db:"summary"`
	URL          string    `json:"url" db:"url"`
	Score        float32   `json:"score,omitempty"`
	CreatedAt    time.Time `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time `json:"updated_at" db:"updated_at"`
}

// VectorSearchRequest enhanced with caching options
type VectorSearchRequest struct {
	Query          string    `json:"query"`
	Embedding      []float32 `json:"embedding,omitempty"`
	Limit          int       `json:"limit"`
	Threshold      float32   `json:"threshold"`
	DocumentTypes  []string  `json:"document_types,omitempty"`
	UseCache       bool      `json:"use_cache"`
	CacheKey       string    `json:"cache_key,omitempty"`
	MinIOBuckets   []string  `json:"minio_buckets,omitempty"`
	IncludeContent bool      `json:"include_content"`
}

// VectorSearchResponse with ranking cache metadata
type VectorSearchResponse struct {
	Results     []VectorDocument `json:"results"`
	Query       string           `json:"query"`
	TotalFound  int              `json:"total_found"`
	ProcessTime float64          `json:"process_time_ms"`
	CacheKey    string           `json:"cache_key,omitempty"`
	CacheHit    bool             `json:"cache_hit"`
	MinIOPaths  []string         `json:"minio_paths,omitempty"`
	RankingHash uint64           `json:"ranking_hash,omitempty"`
}

// NewVectorSearchService creates a new integrated vector search service
func NewVectorSearchService(pgPool *pgxpool.Pool, minioClient *minio.Client, rankCache *RankingCache) *VectorSearchService {
	return &VectorSearchService{
		pgPool:      pgPool,
		minioClient: minioClient,
		rankCache:   rankCache,
	}
}

// SearchVectors performs vector similarity search with ranking cache integration
func (v *VectorSearchService) SearchVectors(ctx context.Context, req *VectorSearchRequest) (*VectorSearchResponse, error) {
	startTime := time.Now()
	
	// Check ranking cache first if cache key provided
	if req.UseCache && req.CacheKey != "" && len(req.CacheKey) == 1 {
		if cachedResults := v.tryGetCachedResults(req.CacheKey); cachedResults != nil {
			log.Printf("🔥 QUIC cache HIT for key: %s", req.CacheKey)
			return cachedResults, nil
		}
	}

	// Generate embedding from query if not provided
	var queryEmbedding []float32
	if len(req.Embedding) == 0 && req.Query != "" {
		var err error
		queryEmbedding, err = v.generateQueryEmbedding(req.Query)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding: %w", err)
		}
	} else {
		queryEmbedding = req.Embedding
	}

	// Build PostgreSQL vector search query
	query := `
		SELECT 
			id, title, content, document_type, embedding, 
			minio_path, summary, COALESCE(url, '') as url, 
			created_at, updated_at,
			1 - (embedding <=> $1) as similarity_score
		FROM legal_documents 
		WHERE 1 - (embedding <=> $1) > $2
	`
	
	params := []interface{}{pgvector.NewVector(queryEmbedding), req.Threshold}
	paramCount := 2

	// Add document type filters
	if len(req.DocumentTypes) > 0 {
		paramCount++
		query += fmt.Sprintf(" AND document_type = ANY($%d)", paramCount)
		params = append(params, req.DocumentTypes)
	}

	// Add MinIO bucket filters
	if len(req.MinIOBuckets) > 0 {
		bucketConditions := ""
		for i, bucket := range req.MinIOBuckets {
			if i > 0 {
				bucketConditions += " OR "
			}
			paramCount++
			bucketConditions += fmt.Sprintf("minio_path LIKE $%d", paramCount)
			params = append(params, bucket+"/%")
		}
		query += " AND (" + bucketConditions + ")"
	}

	// Order and limit
	query += " ORDER BY similarity_score DESC"
	if req.Limit > 0 {
		paramCount++
		query += fmt.Sprintf(" LIMIT $%d", paramCount)
		params = append(params, req.Limit)
	}

	// Execute vector search
	rows, err := v.pgPool.Query(ctx, query, params...)
	if err != nil {
		return nil, fmt.Errorf("vector search query failed: %w", err)
	}
	defer rows.Close()

	var results []VectorDocument
	var minIOPaths []string
	
	for rows.Next() {
		var doc VectorDocument
		var embeddingData []byte
		
		err := rows.Scan(
			&doc.ID, &doc.Title, &doc.Content, &doc.DocumentType,
			&embeddingData, &doc.MinIOPath, &doc.Summary, &doc.URL,
			&doc.CreatedAt, &doc.UpdatedAt, &doc.Score,
		)
		if err != nil {
			log.Printf("Error scanning row: %v", err)
			continue
		}

		// Decode embedding if needed for further processing
		if len(embeddingData) > 0 {
			doc.Embedding, err = pgvector.NewVectorFromBytes(embeddingData).AsSlice()
			if err != nil {
				log.Printf("Error decoding embedding: %v", err)
			}
		}

		// Collect MinIO paths for bulk operations
		if doc.MinIOPath != "" {
			minIOPaths = append(minIOPaths, doc.MinIOPath)
		}

		// Optionally fetch full content from MinIO
		if req.IncludeContent && doc.MinIOPath != "" {
			content, err := v.fetchMinIOContent(ctx, doc.MinIOPath)
			if err != nil {
				log.Printf("Warning: Failed to fetch MinIO content for %s: %v", doc.MinIOPath, err)
			} else {
				doc.Content = content
			}
		}

		results = append(results, doc)
	}

	processTime := float64(time.Since(startTime)) / float64(time.Millisecond)

	// Create response
	response := &VectorSearchResponse{
		Results:     results,
		Query:       req.Query,
		TotalFound:  len(results),
		ProcessTime: processTime,
		MinIOPaths:  minIOPaths,
		CacheHit:    false,
	}

	// Store results in ranking cache if requested
	if req.UseCache && len(results) > 0 {
		cacheKey, rankingHash := v.storeInRankingCache(results)
		response.CacheKey = cacheKey
		response.RankingHash = rankingHash
		log.Printf("🚀 Stored vector search results in QUIC cache: %s (hash: %d)", cacheKey, rankingHash)
	}

	return response, nil
}

// generateQueryEmbedding calls Ollama to generate embeddings for text queries
func (v *VectorSearchService) generateQueryEmbedding(query string) ([]float32, error) {
	// Call your existing Ollama embedding service
	// This is a placeholder - integrate with your Ollama service
	resp, err := http.Post("http://localhost:11434/api/embeddings", "application/json", 
		nil) // Add proper request body
	if err != nil {
		return nil, fmt.Errorf("ollama embedding request failed: %w", err)
	}
	defer resp.Body.Close()
	
	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode ollama response: %w", err)
	}
	
	return result.Embedding, nil
}

// fetchMinIOContent retrieves document content from MinIO
func (v *VectorSearchService) fetchMinIOContent(ctx context.Context, path string) (string, error) {
	// Parse MinIO path (format: bucket/object)
	// This is a simplified implementation
	bucket := "legal-documents" // Default bucket
	objectName := path
	
	if path == "" {
		return "", fmt.Errorf("empty MinIO path")
	}

	// Get object from MinIO
	object, err := v.minioClient.GetObject(ctx, bucket, objectName, minio.GetObjectOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to get object from MinIO: %w", err)
	}
	defer object.Close()

	// Read content
	var content []byte
	buffer := make([]byte, 1024)
	for {
		n, err := object.Read(buffer)
		if n > 0 {
			content = append(content, buffer[:n]...)
		}
		if err != nil {
			break
		}
	}

	return string(content), nil
}

// storeInRankingCache converts vector search results to ranking format and caches them
func (v *VectorSearchService) storeInRankingCache(results []VectorDocument) (string, uint64) {
	// Convert to RankingInput format
	rankingInputs := make([]RankingInput, len(results))
	for i, doc := range results {
		rankingInputs[i] = RankingInput{
			DocID:   doc.ID,
			Score:   doc.Score,
			Flags:   v.getDocumentFlags(doc.DocumentType),
			Summary: doc.Summary,
			URL:     doc.URL,
		}
	}

	// Hash the results
	hash := v.rankCache.hashInputs(rankingInputs)
	
	// Pack results
	blob, err := v.rankCache.packRankings(rankingInputs, hash)
	if err != nil {
		log.Printf("Error packing rankings: %v", err)
		return "", 0
	}

	// Store in cache
	key, meta := v.rankCache.getOrAssignSlot(hash, blob, len(rankingInputs))
	
	return string(key), meta.Hash
}

// getDocumentFlags converts document type to bit flags
func (v *VectorSearchService) getDocumentFlags(docType string) uint8 {
	flags := uint8(0)
	switch docType {
	case "contract":
		flags |= 0x01
	case "legal_brief":
		flags |= 0x02
	case "case_law":
		flags |= 0x04
	case "regulation":
		flags |= 0x08
	}
	return flags
}

// tryGetCachedResults attempts to retrieve results from ranking cache
func (v *VectorSearchService) tryGetCachedResults(cacheKey string) *VectorSearchResponse {
	if len(cacheKey) != 1 {
		return nil
	}

	blob, meta, found := v.rankCache.fetchByKey(rune(cacheKey[0]))
	if !found {
		return nil
	}

	// For now, return basic cache hit response
	// In production, you'd decode the blob back to results
	return &VectorSearchResponse{
		Results:     []VectorDocument{}, // Placeholder - implement decoding
		Query:       "cached_query",
		TotalFound:  meta.Count,
		ProcessTime: 0.1, // Cache hit is super fast
		CacheKey:    cacheKey,
		CacheHit:    true,
		RankingHash: meta.Hash,
	}
}

// RegisterVectorSearchHandlers adds vector search endpoints with caching
func RegisterVectorSearchHandlers(router *gin.Engine, vectorService *VectorSearchService) {
	v1 := router.Group("/api/v1")
	{
		// Enhanced vector search with caching
		v1.POST("/vector/search", func(c *gin.Context) {
			var req VectorSearchRequest
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
				return
			}

			// Set defaults
			if req.Limit <= 0 {
				req.Limit = 10
			}
			if req.Threshold <= 0 {
				req.Threshold = 0.7
			}

			results, err := vectorService.SearchVectors(c.Request.Context(), &req)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, results)
		})

		// Fast cached search by ranking key
		v1.GET("/vector/cached/:key", func(c *gin.Context) {
			key := c.Param("key")
			if len(key) != 1 {
				c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid cache key format"})
				return
			}

			results := vectorService.tryGetCachedResults(key)
			if results == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Cache key not found"})
				return
			}

			c.JSON(http.StatusOK, results)
		})

		// MinIO content retrieval
		v1.GET("/vector/content/:path", func(c *gin.Context) {
			path := c.Param("path")
			content, err := vectorService.fetchMinIOContent(c.Request.Context(), path)
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{
				"path":    path,
				"content": content,
				"length":  len(content),
			})
		})

		// Bulk MinIO content retrieval
		v1.POST("/vector/content/bulk", func(c *gin.Context) {
			var req struct {
				Paths []string `json:"paths"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
				return
			}

			results := make(map[string]interface{})
			for _, path := range req.Paths {
				content, err := vectorService.fetchMinIOContent(c.Request.Context(), path)
				if err != nil {
					results[path] = gin.H{"error": err.Error()}
				} else {
					results[path] = gin.H{
						"content": content,
						"length":  len(content),
					}
				}
			}

			c.JSON(http.StatusOK, gin.H{"results": results})
		})
	}

	// QUIC-specific endpoints (ultra-fast)
	quic := router.Group("/quic/vector")
	{
		// Ultra-fast vector search (QUIC only)
		quic.POST("/search", func(c *gin.Context) {
			var req VectorSearchRequest
			req.UseCache = true // Always use cache for QUIC
			
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
				return
			}

			results, err := vectorService.SearchVectors(c.Request.Context(), &req)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.Header("X-Cache-Key", results.CacheKey)
			c.Header("X-Ranking-Hash", strconv.FormatUint(results.RankingHash, 10))
			c.JSON(http.StatusOK, results)
		})

		// Ultra-fast cached retrieval
		quic.GET("/cached/:key", func(c *gin.Context) {
			key := c.Param("key")
			results := vectorService.tryGetCachedResults(key)
			if results == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Not found"})
				return
			}

			c.Header("X-Cache-Hit", "true")
			c.JSON(http.StatusOK, results)
		})
	}
}

// Global instance for easy access
var globalVectorSearchService *VectorSearchService

func initVectorSearchService(pgPool *pgxpool.Pool, minioClient *minio.Client, rankCache *RankingCache) {
	globalVectorSearchService = NewVectorSearchService(pgPool, minioClient, rankCache)
	log.Println("🔍 Vector Search Service initialized with MinIO and QUIC ranking cache")
}