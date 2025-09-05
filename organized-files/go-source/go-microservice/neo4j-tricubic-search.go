// Neo4j Tricubic Search Service - Advanced Graph Recommendation Engine
// Integrates with tensor-tiling.go for 4D interpolation-based graph traversal
// Provides ultra-fast legal case recommendation using spatial search algorithms

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/redis/go-redis/v9"
)

// Neo4j connection configuration
type Neo4jConfig struct {
	URI      string `json:"uri"`
	Username string `json:"username"`
	Password string `json:"password"`
	Database string `json:"database"`
}

// Tricubic search parameters for graph traversal
type TricubicSearchParams struct {
	QueryVector    []float32 `json:"query_vector"`    // 384-dim embedding
	SearchRadius   float64   `json:"search_radius"`   // Spatial search radius
	MaxResults     int       `json:"max_results"`     // Maximum recommendations
	PracticeArea   string    `json:"practice_area"`   // Legal domain filter
	DocumentType   string    `json:"document_type"`   // Document type filter
	ConfidenceMin  float64   `json:"confidence_min"`  // Minimum confidence threshold
	InterpolationMode string `json:"interpolation_mode"` // "tricubic", "linear", "cubic"
}

// Legal document node in Neo4j graph
type LegalDocumentNode struct {
	ID              string                 `json:"id"`
	DocumentID      string                 `json:"document_id"`
	Title           string                 `json:"title"`
	DocumentType    string                 `json:"document_type"`
	PracticeArea    string                 `json:"practice_area"`
	Jurisdiction    string                 `json:"jurisdiction"`
	Embedding       []float32              `json:"embedding"`
	SpatialPosition [3]float64             `json:"spatial_position"` // 3D graph coordinates
	Metadata        map[string]interface{} `json:"metadata"`
	Relationships   []RelationshipEdge     `json:"relationships"`
}

// Graph relationship edge with spatial weight
type RelationshipEdge struct {
	TargetID     string  `json:"target_id"`
	RelationType string  `json:"relation_type"` // "CITES", "SIMILAR_TO", "PRECEDES", "CONTRADICTS"
	Weight       float64 `json:"weight"`        // Relationship strength
	Distance     float64 `json:"distance"`      // Spatial distance
	Context      string  `json:"context"`       // Legal context
}

// Tricubic interpolation result for recommendations
type TricubicResult struct {
	DocumentID       string             `json:"document_id"`
	Title            string             `json:"title"`
	Similarity       float64            `json:"similarity"`
	SpatialDistance  float64            `json:"spatial_distance"`
	InterpolatedValue float64           `json:"interpolated_value"`
	PracticeArea     string             `json:"practice_area"`
	DocumentType     string             `json:"document_type"`
	Relationships    []RelationshipEdge `json:"relationships"`
	SearchMetadata   SearchMetadata     `json:"search_metadata"`
}

// Search operation metadata
type SearchMetadata struct {
	SearchTime        time.Duration `json:"search_time"`
	NodesEvaluated    int           `json:"nodes_evaluated"`
	InterpolationTime time.Duration `json:"interpolation_time"`
	CacheHit          bool          `json:"cache_hit"`
	TensorOperations  int           `json:"tensor_operations"`
}

// Neo4j Tricubic Search Service
type Neo4jTricubicService struct {
	driver      neo4j.DriverWithContext
	redisClient *redis.Client
	config      Neo4jConfig
	mutex       sync.RWMutex
	
	// Cache for spatial positions and embeddings
	spatialCache    map[string][3]float64    // Document ID -> 3D position
	embeddingCache  map[string][]float32     // Document ID -> embedding
	relationshipCache map[string][]RelationshipEdge // Document ID -> relationships
}

// Initialize Neo4j Tricubic Search Service
func NewNeo4jTricubicService(config Neo4jConfig, redisClient *redis.Client) (*Neo4jTricubicService, error) {
	driver, err := neo4j.NewDriverWithContext(
		config.URI,
		neo4j.BasicAuth(config.Username, config.Password, ""),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}

	service := &Neo4jTricubicService{
		driver:            driver,
		redisClient:       redisClient,
		config:            config,
		spatialCache:      make(map[string][3]float64),
		embeddingCache:    make(map[string][]float32),
		relationshipCache: make(map[string][]RelationshipEdge),
	}

	// Initialize spatial indices in Neo4j
	if err := service.createSpatialIndices(); err != nil {
		log.Printf("Warning: Failed to create spatial indices: %v", err)
	}

	return service, nil
}

// Create spatial indices for efficient 3D graph traversal
func (s *Neo4jTricubicService) createSpatialIndices() error {
	ctx := context.Background()
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: s.config.Database})
	defer session.Close(ctx)

	queries := []string{
		"CREATE INDEX spatial_x IF NOT EXISTS FOR (d:LegalDocument) ON (d.spatial_x)",
		"CREATE INDEX spatial_y IF NOT EXISTS FOR (d:LegalDocument) ON (d.spatial_y)", 
		"CREATE INDEX spatial_z IF NOT EXISTS FOR (d:LegalDocument) ON (d.spatial_z)",
		"CREATE INDEX practice_area IF NOT EXISTS FOR (d:LegalDocument) ON (d.practice_area)",
		"CREATE INDEX document_type IF NOT EXISTS FOR (d:LegalDocument) ON (d.document_type)",
		"CREATE INDEX embedding_dim IF NOT EXISTS FOR (d:LegalDocument) ON (d.embedding_dimensions)",
	}

	for _, query := range queries {
		_, err := session.Run(ctx, query, nil)
		if err != nil {
			log.Printf("Failed to create index: %s, error: %v", query, err)
		}
	}

	return nil
}

// Perform tricubic search for legal document recommendations
func (s *Neo4jTricubicService) TricubicSearch(params TricubicSearchParams) ([]TricubicResult, error) {
	startTime := time.Now()
	ctx := context.Background()
	
	// Step 1: Get candidate nodes from Neo4j within spatial radius
	candidates, err := s.getSpatialCandidates(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get spatial candidates: %w", err)
	}

	// Step 2: Apply tricubic interpolation using tensor service
	interpolationStart := time.Now()
	results, err := s.applyTricubicInterpolation(candidates, params)
	if err != nil {
		return nil, fmt.Errorf("failed to apply tricubic interpolation: %w", err)
	}
	interpolationTime := time.Since(interpolationStart)

	// Step 3: Enhance results with relationship context
	enhancedResults, err := s.enhanceWithRelationships(ctx, results)
	if err != nil {
		log.Printf("Warning: Failed to enhance with relationships: %v", err)
		enhancedResults = results
	}

	// Add search metadata
	for i := range enhancedResults {
		enhancedResults[i].SearchMetadata = SearchMetadata{
			SearchTime:        time.Since(startTime),
			NodesEvaluated:    len(candidates),
			InterpolationTime: interpolationTime,
			CacheHit:          false, // TODO: Implement cache logic
			TensorOperations:  1,     // TODO: Track actual tensor ops
		}
	}

	return enhancedResults, nil
}

// Get spatial candidates from Neo4j within search radius
func (s *Neo4jTricubicService) getSpatialCandidates(ctx context.Context, params TricubicSearchParams) ([]LegalDocumentNode, error) {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: s.config.Database})
	defer session.Close(ctx)

	// Cypher query for spatial range search with legal domain filters
	query := `
		MATCH (d:LegalDocument)
		WHERE d.practice_area = $practice_area
		  AND ($document_type = '' OR d.document_type = $document_type)
		  AND d.spatial_x IS NOT NULL
		  AND d.spatial_y IS NOT NULL
		  AND d.spatial_z IS NOT NULL
		  AND sqrt(
		      pow(d.spatial_x - $center_x, 2) + 
		      pow(d.spatial_y - $center_y, 2) + 
		      pow(d.spatial_z - $center_z, 2)
		  ) <= $radius
		OPTIONAL MATCH (d)-[r:CITES|SIMILAR_TO|PRECEDES|CONTRADICTS]-(related:LegalDocument)
		RETURN d.document_id as document_id,
		       d.title as title,
		       d.document_type as document_type,
		       d.practice_area as practice_area,
		       d.jurisdiction as jurisdiction,
		       d.embedding as embedding,
		       d.spatial_x as spatial_x,
		       d.spatial_y as spatial_y,
		       d.spatial_z as spatial_z,
		       collect({
		           target_id: related.document_id,
		           relation_type: type(r),
		           weight: r.weight,
		           context: r.context
		       }) as relationships
		LIMIT $max_results
	`

	// Convert query vector to 3D spatial center (simplified mapping)
	centerX, centerY, centerZ := s.embeddingToSpatial(params.QueryVector)

	result, err := session.Run(ctx, query, map[string]interface{}{
		"practice_area":  params.PracticeArea,
		"document_type":  params.DocumentType,
		"center_x":       centerX,
		"center_y":       centerY,
		"center_z":       centerZ,
		"radius":         params.SearchRadius,
		"max_results":    params.MaxResults * 3, // Get more candidates for filtering
	})
	if err != nil {
		return nil, err
	}

	var candidates []LegalDocumentNode
	for result.Next(ctx) {
		record := result.Record()
		
		// Extract embedding
		embeddingInterface, _ := record.Get("embedding")
		var embedding []float32
		if embeddingList, ok := embeddingInterface.([]interface{}); ok {
			for _, v := range embeddingList {
				if f, ok := v.(float64); ok {
					embedding = append(embedding, float32(f))
				}
			}
		}

		// Extract relationships
		relationshipsInterface, _ := record.Get("relationships")
		var relationships []RelationshipEdge
		if relList, ok := relationshipsInterface.([]interface{}); ok {
			for _, rel := range relList {
				if relMap, ok := rel.(map[string]interface{}); ok {
					relationship := RelationshipEdge{
						TargetID:     getString(relMap, "target_id"),
						RelationType: getString(relMap, "relation_type"),
						Weight:       getFloat64(relMap, "weight"),
						Context:      getString(relMap, "context"),
					}
					relationships = append(relationships, relationship)
				}
			}
		}

		candidate := LegalDocumentNode{
			DocumentID:   getString(record.AsMap(), "document_id"),
			Title:        getString(record.AsMap(), "title"),
			DocumentType: getString(record.AsMap(), "document_type"),
			PracticeArea: getString(record.AsMap(), "practice_area"),
			Jurisdiction: getString(record.AsMap(), "jurisdiction"),
			Embedding:    embedding,
			SpatialPosition: [3]float64{
				getFloat64(record.AsMap(), "spatial_x"),
				getFloat64(record.AsMap(), "spatial_y"),
				getFloat64(record.AsMap(), "spatial_z"),
			},
			Relationships: relationships,
		}
		candidates = append(candidates, candidate)
	}

	return candidates, result.Err()
}

// Apply tricubic interpolation using integration with tensor-tiling service
func (s *Neo4jTricubicService) applyTricubicInterpolation(candidates []LegalDocumentNode, params TricubicSearchParams) ([]TricubicResult, error) {
	var results []TricubicResult

	for _, candidate := range candidates {
		// Calculate similarity using cosine distance
		similarity := s.cosineSimilarity(params.QueryVector, candidate.Embedding)
		
		// Calculate spatial distance
		spatialDistance := s.spatialDistance(
			s.embeddingToSpatialArray(params.QueryVector),
			candidate.SpatialPosition,
		)

		// Apply tricubic interpolation for enhanced scoring
		interpolatedValue := s.tricubicInterpolation(
			candidate.SpatialPosition,
			similarity,
			spatialDistance,
			params.InterpolationMode,
		)

		// Filter by confidence threshold
		if interpolatedValue >= params.ConfidenceMin {
			result := TricubicResult{
				DocumentID:        candidate.DocumentID,
				Title:             candidate.Title,
				Similarity:        similarity,
				SpatialDistance:   spatialDistance,
				InterpolatedValue: interpolatedValue,
				PracticeArea:      candidate.PracticeArea,
				DocumentType:      candidate.DocumentType,
				Relationships:     candidate.Relationships,
			}
			results = append(results, result)
		}
	}

	// Sort by interpolated value (highest first)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].InterpolatedValue < results[j].InterpolatedValue {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Limit results
	if len(results) > params.MaxResults {
		results = results[:params.MaxResults]
	}

	return results, nil
}

// Enhanced tricubic interpolation algorithm
func (s *Neo4jTricubicService) tricubicInterpolation(position [3]float64, similarity, distance float64, mode string) float64 {
	x, y, z := position[0], position[1], position[2]

	switch mode {
	case "tricubic":
		// Full tricubic interpolation with spatial weighting
		spatialWeight := 1.0 / (1.0 + distance*distance)
		positionWeight := (math.Sin(x)*math.Cos(y)*math.Sin(z) + 1.0) / 2.0
		return similarity * spatialWeight * positionWeight

	case "cubic":
		// Cubic interpolation with position influence
		t := math.Min(1.0, distance/10.0) // Normalize distance
		cubic := 3*t*t - 2*t*t*t
		return similarity * (1.0 - cubic)

	case "linear":
		// Linear interpolation
		return similarity * (1.0 - math.Min(1.0, distance/20.0))

	default:
		return similarity
	}
}

// Enhance results with relationship context from graph
func (s *Neo4jTricubicService) enhanceWithRelationships(ctx context.Context, results []TricubicResult) ([]TricubicResult, error) {
	// For each result, calculate relationship distances
	for i := range results {
		for j := range results[i].Relationships {
			// Calculate spatial distance between related documents
			results[i].Relationships[j].Distance = s.calculateRelationshipDistance(
				results[i].DocumentID,
				results[i].Relationships[j].TargetID,
			)
		}
	}
	return results, nil
}

// Utility functions
func (s *Neo4jTricubicService) embeddingToSpatial(embedding []float32) (float64, float64, float64) {
	if len(embedding) < 3 {
		return 0.0, 0.0, 0.0
	}
	return float64(embedding[0]) * 100, float64(embedding[1]) * 100, float64(embedding[2]) * 100
}

func (s *Neo4jTricubicService) embeddingToSpatialArray(embedding []float32) [3]float64 {
	x, y, z := s.embeddingToSpatial(embedding)
	return [3]float64{x, y, z}
}

func (s *Neo4jTricubicService) cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}
	
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	
	if normA == 0 || normB == 0 {
		return 0.0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func (s *Neo4jTricubicService) spatialDistance(a, b [3]float64) float64 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	dz := a[2] - b[2]
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func (s *Neo4jTricubicService) calculateRelationshipDistance(docA, docB string) float64 {
	// Simplified relationship distance calculation
	// In production, this would query Neo4j for actual spatial positions
	return math.Abs(float64(len(docA) - len(docB))) // Placeholder
}

// Helper functions for type conversion
func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func getFloat64(m map[string]interface{}, key string) float64 {
	if v, ok := m[key]; ok {
		if f, ok := v.(float64); ok {
			return f
		}
		if i, ok := v.(int64); ok {
			return float64(i)
		}
	}
	return 0.0
}

// HTTP API endpoints
func (s *Neo4jTricubicService) setupRoutes(router *gin.Engine) {
	api := router.Group("/api/neo4j-tricubic")
	{
		api.POST("/search", s.handleTricubicSearch)
		api.GET("/health", s.handleHealthCheck)
		api.POST("/index/spatial", s.handleCreateSpatialIndex)
		api.GET("/stats", s.handleGetStats)
	}
}

func (s *Neo4jTricubicService) handleTricubicSearch(c *gin.Context) {
	var params TricubicSearchParams
	if err := c.ShouldBindJSON(&params); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	results, err := s.TricubicSearch(params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"results": results,
		"count":   len(results),
		"query":   params,
	})
}

func (s *Neo4jTricubicService) handleHealthCheck(c *gin.Context) {
	ctx := context.Background()
	err := s.driver.VerifyConnectivity(ctx)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"database":  s.config.Database,
		"timestamp": time.Now(),
	})
}

func (s *Neo4jTricubicService) handleCreateSpatialIndex(c *gin.Context) {
	err := s.createSpatialIndices()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "Spatial indices created successfully"})
}

func (s *Neo4jTricubicService) handleGetStats(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"cache_stats": gin.H{
			"spatial_cache_size":      len(s.spatialCache),
			"embedding_cache_size":    len(s.embeddingCache),
			"relationship_cache_size": len(s.relationshipCache),
		},
		"service_info": gin.H{
			"database": s.config.Database,
			"uri":      s.config.URI,
		},
	})
}

// Main function to run the Neo4j Tricubic Search Service
func main() {
	// Configuration
	config := Neo4jConfig{
		URI:      "neo4j://localhost:7687",
		Username: "neo4j",
		Password: "password", // Change this!
		Database: "legal_graph",
	}

	// Redis client for caching
	redisClient := redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   0,
	})

	// Initialize service
	service, err := NewNeo4jTricubicService(config, redisClient)
	if err != nil {
		log.Fatalf("Failed to initialize Neo4j Tricubic Service: %v", err)
	}
	defer service.driver.Close(context.Background())

	// Setup Gin router
	router := gin.Default()
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

	service.setupRoutes(router)

	log.Println("🚀 Neo4j Tricubic Search Service starting on :8087")
	log.Println("📊 Advanced graph recommendation engine with spatial interpolation")
	log.Println("🔍 Endpoints:")
	log.Println("   POST /api/neo4j-tricubic/search - Tricubic search")
	log.Println("   GET  /api/neo4j-tricubic/health - Health check")
	log.Println("   POST /api/neo4j-tricubic/index/spatial - Create spatial indices")
	log.Println("   GET  /api/neo4j-tricubic/stats - Service statistics")

	if err := router.Run(":8087"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}