//go:build dblayer
// +build dblayer

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Database Integration Layer with Context7 performance optimizations
// Comprehensive integration with PostgreSQL (pgvector), Neo4j, and Redis
// Provides unified data access for the legal AI platform

type DatabaseLayer struct {
	config     *DatabaseConfig
	pgPool     *pgxpool.Pool
	neo4jDriver neo4j.DriverWithContext
	redisClient *redis.Client
	metrics     *DatabaseMetrics
	cache       *DatabaseCache
	mutex       sync.RWMutex

	// Context7 performance optimization fields
	totalRequests        int64       // atomic counter
	totalCacheHits       int64       // atomic counter
	totalCacheMisses     int64       // atomic counter
	totalDatabaseOps     int64       // atomic counter
	averageResponseTime  int64       // atomic average (microseconds)
	startTime            time.Time   // Service start time
	bufferPool           *sync.Pool  // Buffer pool for JSON operations
}

type DatabaseConfig struct {
	PostgresURL    string
	Neo4jURI       string
	Neo4jUser      string
	Neo4jPassword  string
	RedisURL       string
	MaxConnections int
	QueryTimeout   time.Duration
	CacheEnabled   bool
	CacheTTL       time.Duration
	EnableMetrics  bool
	Port           string
}

type DatabaseMetrics struct {
	// Atomic counters for Context7 performance tracking
	PostgreSQLQueries    int64   `json:"postgresql_queries"`
	Neo4jQueries         int64   `json:"neo4j_queries"`
	RedisOperations      int64   `json:"redis_operations"`
	AverageQueryTime     int64   `json:"average_query_time_us"` // Changed to atomic int64
	CacheHitRate         float64 `json:"cache_hit_rate"`
	ActiveConnections    int64   `json:"active_connections"`
	ErrorCount           int64   `json:"error_count"`
	LastError            string  `json:"last_error,omitempty"`
	VectorOperations     int64   `json:"vector_operations"`
	GraphTraversals      int64   `json:"graph_traversals"`
	TotalOperations      int64   `json:"total_operations"`
	UptimeSeconds        int64   `json:"uptime_seconds"`
	StartTime            time.Time `json:"-"`
	mutex                sync.RWMutex
}

type DatabaseCache struct {
	enabled     bool
	ttl         time.Duration
	localCache  map[string]CacheEntry
	mutex       sync.RWMutex
}

type CacheEntry struct {
	Data      interface{}
	ExpiresAt time.Time
	Hits      int64
}

// Data Models
type LegalDocument struct {
	ID           int                    `json:"id" db:"id"`
	CaseID       string                 `json:"case_id" db:"case_id"`
	Title        string                 `json:"title" db:"title"`
	Content      string                 `json:"content" db:"content"`
	DocumentType string                 `json:"document_type" db:"document_type"`
	Embedding    []float64              `json:"embedding,omitempty"`
	Metadata     map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt    time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" db:"updated_at"`
}

type LegalCase struct {
	ID          string                 `json:"id" db:"id"`
	Title       string                 `json:"title" db:"title"`
	Description string                 `json:"description" db:"description"`
	Status      string                 `json:"status" db:"status"`
	Priority    int                    `json:"priority" db:"priority"`
	CreatedBy   string                 `json:"created_by" db:"created_by"`
	AssignedTo  []string               `json:"assigned_to"`
	Tags        []string               `json:"tags"`
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	CreatedAt   time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at" db:"updated_at"`
}

type EntityRelation struct {
	ID           string                 `json:"id"`
	FromEntity   string                 `json:"from_entity"`
	ToEntity     string                 `json:"to_entity"`
	Relationship string                 `json:"relationship"`
	Properties   map[string]interface{} `json:"properties"`
	Confidence   float64                `json:"confidence"`
	CreatedAt    time.Time              `json:"created_at"`
}

type VectorSearchRequest struct {
	Query      string    `json:"query"`
	Embedding  []float64 `json:"embedding"`
	Limit      int       `json:"limit"`
	MinScore   float64   `json:"min_score"`
	CaseID     string    `json:"case_id,omitempty"`
	DocType    string    `json:"doc_type,omitempty"`
}

type VectorSearchResult struct {
	Documents      []LegalDocument `json:"documents"`
	TotalResults   int             `json:"total_results"`
	QueryTime      float64         `json:"query_time_ms"`
	MaxScore       float64         `json:"max_score"`
	AverageScore   float64         `json:"average_score"`
}

func NewDatabaseLayer() *DatabaseLayer {
	config := &DatabaseConfig{
		PostgresURL:    getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		Neo4jURI:       getEnv("NEO4J_URI", "bolt://localhost:7687"),
		Neo4jUser:      getEnv("NEO4J_USER", "neo4j"),
		Neo4jPassword:  getEnv("NEO4J_PASSWORD", "password"),
		RedisURL:       getEnv("REDIS_URL", "redis://localhost:6379"),
		MaxConnections: getEnvInt("DB_MAX_CONNECTIONS", 25),
		QueryTimeout:   time.Duration(getEnvInt("DB_QUERY_TIMEOUT", 30)) * time.Second,
		CacheEnabled:   getEnvBool("DB_CACHE_ENABLED", true),
		CacheTTL:       time.Duration(getEnvInt("DB_CACHE_TTL", 300)) * time.Second,
		EnableMetrics:  getEnvBool("DB_ENABLE_METRICS", true),
		Port:           getEnv("DB_LAYER_PORT", "8098"),
	}

	cache := &DatabaseCache{
		enabled:    config.CacheEnabled,
		ttl:        config.CacheTTL,
		localCache: make(map[string]CacheEntry),
	}

	return &DatabaseLayer{
		config:    config,
		cache:     cache,
		startTime: time.Now(),
		metrics: &DatabaseMetrics{
			StartTime: time.Now(),
		},
		bufferPool: &sync.Pool{
			New: func() interface{} {
				// Pre-allocate 8KB buffer for JSON operations
				return make([]byte, 0, 8192)
			},
		},
	}
}

func (dl *DatabaseLayer) Initialize() error {
	log.Println("🗄️  Initializing Database Integration Layer...")

	var err error

	// Initialize PostgreSQL connection pool
	if err = dl.initializePostgreSQL(); err != nil {
		return fmt.Errorf("PostgreSQL initialization failed: %w", err)
	}

	// Initialize Neo4j driver
	if err = dl.initializeNeo4j(); err != nil {
		return fmt.Errorf("Neo4j initialization failed: %w", err)
	}

	// Initialize Redis client
	if err = dl.initializeRedis(); err != nil {
		return fmt.Errorf("Redis initialization failed: %w", err)
	}

	// Verify database schemas and extensions
	if err = dl.verifySchemas(); err != nil {
		return fmt.Errorf("Schema verification failed: %w", err)
	}

	// Start metrics collection
	if dl.config.EnableMetrics {
		go dl.collectMetrics()
	}

	log.Println("✅ Database Integration Layer initialized successfully")
	return nil
}

func (dl *DatabaseLayer) initializePostgreSQL() error {
	log.Println("📊 Connecting to PostgreSQL with pgvector...")

	config, err := pgxpool.ParseConfig(dl.config.PostgresURL)
	if err != nil {
		return err
	}

	config.MaxConns = int32(dl.config.MaxConnections)
	config.MinConns = 5
	config.MaxConnLifetime = time.Hour
	config.MaxConnIdleTime = time.Minute * 30

	dl.pgPool, err = pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		return err
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err = dl.pgPool.Ping(ctx); err != nil {
		return err
	}

	log.Println("✅ PostgreSQL connection established")
	return nil
}

func (dl *DatabaseLayer) initializeNeo4j() error {
	log.Println("🕸️  Connecting to Neo4j...")

	var err error
	dl.neo4jDriver, err = neo4j.NewDriverWithContext(
		dl.config.Neo4jURI,
		neo4j.BasicAuth(dl.config.Neo4jUser, dl.config.Neo4jPassword, ""),
		func(config *neo4j.Config) {
			config.MaxConnectionLifetime = time.Hour
			config.MaxConnectionPoolSize = dl.config.MaxConnections
			config.ConnectionAcquisitionTimeout = 30 * time.Second
		},
	)

	if err != nil {
		return err
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err = dl.neo4jDriver.VerifyConnectivity(ctx); err != nil {
		return err
	}

	log.Println("✅ Neo4j connection established")
	return nil
}

func (dl *DatabaseLayer) initializeRedis() error {
	log.Println("🔴 Connecting to Redis...")

	opt, err := redis.ParseURL(dl.config.RedisURL)
	if err != nil {
		return err
	}

	opt.PoolSize = dl.config.MaxConnections
	opt.MinIdleConns = 5
	opt.PoolTimeout = 30 * time.Second
	opt.IdleTimeout = 10 * time.Minute

	dl.redisClient = redis.NewClient(opt)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err = dl.redisClient.Ping(ctx).Err(); err != nil {
		return err
	}

	log.Println("✅ Redis connection established")
	return nil
}

func (dl *DatabaseLayer) verifySchemas() error {
	log.Println("🔍 Verifying database schemas and extensions...")

	// Check pgvector extension
	var hasVector bool
	err := dl.pgPool.QueryRow(context.Background(),
		"SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')").Scan(&hasVector)
	if err != nil {
		return err
	}

	if !hasVector {
		log.Println("⚠️  pgvector extension not found, attempting to create...")
		_, err = dl.pgPool.Exec(context.Background(), "CREATE EXTENSION IF NOT EXISTS vector")
		if err != nil {
			return fmt.Errorf("failed to create vector extension: %w", err)
		}
	}

	// Verify legal_documents table exists with vector column
	var hasTable bool
	err = dl.pgPool.QueryRow(context.Background(),
		`SELECT EXISTS(SELECT 1 FROM information_schema.tables
		 WHERE table_name = 'legal_documents' AND table_schema = 'public')`).Scan(&hasTable)
	if err != nil {
		return err
	}

	if !hasTable {
		log.Println("⚠️  legal_documents table not found, creating...")
		if err = dl.createLegalDocumentsTable(); err != nil {
			return err
		}
	}

	// Verify Neo4j constraints
	if err = dl.createNeo4jConstraints(); err != nil {
		return err
	}

	log.Println("✅ Database schemas verified")
	return nil
}

func (dl *DatabaseLayer) createLegalDocumentsTable() error {
	query := `
		CREATE TABLE legal_documents (
			id SERIAL PRIMARY KEY,
			case_id VARCHAR(255) NOT NULL,
			title TEXT NOT NULL,
			content TEXT,
			document_type VARCHAR(100),
			embedding vector(384),
			metadata JSONB DEFAULT '{}',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
		);

		CREATE INDEX idx_legal_documents_case_id ON legal_documents(case_id);
		CREATE INDEX idx_legal_documents_type ON legal_documents(document_type);
		CREATE INDEX idx_legal_documents_embedding ON legal_documents USING ivfflat (embedding vector_cosine_ops);
		CREATE INDEX idx_legal_documents_metadata ON legal_documents USING gin(metadata);
	`

	_, err := dl.pgPool.Exec(context.Background(), query)
	return err
}

func (dl *DatabaseLayer) createNeo4jConstraints() error {
	ctx := context.Background()
	session := dl.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	constraints := []string{
		"CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:LegalCase) REQUIRE c.id IS UNIQUE",
		"CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
		"CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
		"CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
	}

	for _, constraint := range constraints {
		if _, err := session.Run(ctx, constraint, nil); err != nil {
			log.Printf("Warning: Neo4j constraint creation failed: %v", err)
		}
	}

	return nil
}

// Vector Search Operations with Context7 performance tracking
func (dl *DatabaseLayer) VectorSearch(req *VectorSearchRequest) (*VectorSearchResult, error) {
	start := time.Now()

	// Context7 performance tracking - atomic increments
	atomic.AddInt64(&dl.totalRequests, 1)
	atomic.AddInt64(&dl.totalDatabaseOps, 1)
	defer func() {
		latency := time.Since(start).Microseconds()
		// Update rolling average atomically
		currentAvg := atomic.LoadInt64(&dl.averageResponseTime)
		newAvg := (currentAvg + latency) / 2
		atomic.StoreInt64(&dl.averageResponseTime, newAvg)
	}()

	// Check cache first
	cacheKey := fmt.Sprintf("vector_search:%x", req)
	if dl.cache.enabled {
		if cached := dl.getFromCache(cacheKey); cached != nil {
			if result, ok := cached.(*VectorSearchResult); ok {
				atomic.AddInt64(&dl.totalCacheHits, 1)
				dl.updateMetrics("vector_search_cached", time.Since(start), nil)
				return result, nil
			}
		}
		atomic.AddInt64(&dl.totalCacheMisses, 1)
	}

	// Build query
	query := `
		SELECT id, case_id, title, content, document_type,
			   1 - (embedding <=> $1::vector) as score,
			   metadata, created_at
		FROM legal_documents
		WHERE 1 - (embedding <=> $1::vector) > $2
	`
	args := []interface{}{vectorToString(req.Embedding), req.MinScore}
	argIndex := 3

	// Add filters
	if req.CaseID != "" {
		query += fmt.Sprintf(" AND case_id = $%d", argIndex)
		args = append(args, req.CaseID)
		argIndex++
	}

	if req.DocType != "" {
		query += fmt.Sprintf(" AND document_type = $%d", argIndex)
		args = append(args, req.DocType)
		argIndex++
	}

	query += fmt.Sprintf(" ORDER BY score DESC LIMIT $%d", argIndex)
	args = append(args, req.Limit)

	// Execute query
	rows, err := dl.pgPool.Query(context.Background(), query, args...)
	if err != nil {
		dl.updateMetrics("vector_search", time.Since(start), err)
		return nil, err
	}
	defer rows.Close()

	var documents []LegalDocument
	var scores []float64

	for rows.Next() {
		var doc LegalDocument
		var score float64
		var metadataJSON []byte

		err := rows.Scan(&doc.ID, &doc.CaseID, &doc.Title, &doc.Content,
			&doc.DocumentType, &score, &metadataJSON, &doc.CreatedAt)
		if err != nil {
			continue
		}

		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &doc.Metadata)
		}

		documents = append(documents, doc)
		scores = append(scores, score)
	}

	// Calculate statistics
	result := &VectorSearchResult{
		Documents:    documents,
		TotalResults: len(documents),
		QueryTime:    float64(time.Since(start).Nanoseconds()) / 1e6,
	}

	if len(scores) > 0 {
		result.MaxScore = scores[0] // First is highest due to ORDER BY
		sum := 0.0
		for _, score := range scores {
			sum += score
		}
		result.AverageScore = sum / float64(len(scores))
	}

	// Cache result
	if dl.cache.enabled {
		dl.setCache(cacheKey, result)
	}

	dl.updateMetrics("vector_search", time.Since(start), nil)
	return result, nil
}

func (dl *DatabaseLayer) StoreDocument(doc *LegalDocument) error {
	start := time.Now()

	// Context7 performance tracking
	atomic.AddInt64(&dl.totalRequests, 1)
	atomic.AddInt64(&dl.totalDatabaseOps, 1)
	defer func() {
		latency := time.Since(start).Microseconds()
		currentAvg := atomic.LoadInt64(&dl.averageResponseTime)
		newAvg := (currentAvg + latency) / 2
		atomic.StoreInt64(&dl.averageResponseTime, newAvg)
	}()

	query := `
		INSERT INTO legal_documents (case_id, title, content, document_type, embedding, metadata)
		VALUES ($1, $2, $3, $4, $5::vector, $6)
		RETURNING id, created_at
	`

	metadataJSON, _ := json.Marshal(doc.Metadata)
	embeddingStr := vectorToString(doc.Embedding)

	err := dl.pgPool.QueryRow(context.Background(), query,
		doc.CaseID, doc.Title, doc.Content, doc.DocumentType, embeddingStr, metadataJSON).
		Scan(&doc.ID, &doc.CreatedAt)

	dl.updateMetrics("store_document", time.Since(start), err)
	return err
}

// Neo4j Graph Operations with Context7 performance tracking
func (dl *DatabaseLayer) CreateEntityRelation(relation *EntityRelation) error {
	start := time.Now()

	// Context7 performance tracking
	atomic.AddInt64(&dl.totalRequests, 1)
	atomic.AddInt64(&dl.totalDatabaseOps, 1)
	defer func() {
		latency := time.Since(start).Microseconds()
		currentAvg := atomic.LoadInt64(&dl.averageResponseTime)
		newAvg := (currentAvg + latency) / 2
		atomic.StoreInt64(&dl.averageResponseTime, newAvg)
	}()
	ctx := context.Background()
	session := dl.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	query := `
		MERGE (from:Entity {id: $fromEntity})
		MERGE (to:Entity {id: $toEntity})
		CREATE (from)-[r:RELATES {
			type: $relationship,
			properties: $properties,
			confidence: $confidence,
			created_at: datetime()
		}]->(to)
		RETURN r
	`

	propertiesJSON, _ := json.Marshal(relation.Properties)

	_, err := session.Run(ctx, query, map[string]interface{}{
		"fromEntity":   relation.FromEntity,
		"toEntity":     relation.ToEntity,
		"relationship": relation.Relationship,
		"properties":   string(propertiesJSON),
		"confidence":   relation.Confidence,
	})

	dl.updateMetrics("create_relation", time.Since(start), err)
	return err
}

func (dl *DatabaseLayer) FindRelatedEntities(entityID string, maxDepth int) ([]EntityRelation, error) {
	start := time.Now()

	// Context7 performance tracking
	atomic.AddInt64(&dl.totalRequests, 1)
	atomic.AddInt64(&dl.totalDatabaseOps, 1)
	defer func() {
		latency := time.Since(start).Microseconds()
		currentAvg := atomic.LoadInt64(&dl.averageResponseTime)
		newAvg := (currentAvg + latency) / 2
		atomic.StoreInt64(&dl.averageResponseTime, newAvg)
	}()
	ctx := context.Background()
	session := dl.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	query := fmt.Sprintf(`
		MATCH (start:Entity {id: $entityId})-[r:RELATES*1..%d]-(related:Entity)
		RETURN DISTINCT related.id as entity, r, type(r) as rel_type
		LIMIT 100
	`, maxDepth)

	result, err := session.Run(ctx, query, map[string]interface{}{
		"entityId": entityID,
	})

	if err != nil {
		dl.updateMetrics("find_relations", time.Since(start), err)
		return nil, err
	}

	var relations []EntityRelation
	for result.Next(ctx) {
		record := result.Record()
		// Process Neo4j result into EntityRelation struct
		relation := EntityRelation{
			FromEntity:   entityID,
			ToEntity:     record.Values[0].(string),
			Relationship: record.Values[2].(string),
			Properties:   make(map[string]interface{}),
		}
		relations = append(relations, relation)
	}

	dl.updateMetrics("find_relations", time.Since(start), nil)
	return relations, nil
}

// Cache Operations
func (dl *DatabaseLayer) getFromCache(key string) interface{} {
	dl.cache.mutex.RLock()
	defer dl.cache.mutex.RUnlock()

	if entry, exists := dl.cache.localCache[key]; exists {
		if time.Now().Before(entry.ExpiresAt) {
			entry.Hits++
			dl.cache.localCache[key] = entry
			return entry.Data
		}
		delete(dl.cache.localCache, key)
	}

	return nil
}

func (dl *DatabaseLayer) setCache(key string, data interface{}) {
	if !dl.cache.enabled {
		return
	}

	dl.cache.mutex.Lock()
	defer dl.cache.mutex.Unlock()

	dl.cache.localCache[key] = CacheEntry{
		Data:      data,
		ExpiresAt: time.Now().Add(dl.cache.ttl),
		Hits:      0,
	}
}

// Metrics and monitoring
func (dl *DatabaseLayer) updateMetrics(operation string, duration time.Duration, err error) {
	dl.metrics.mutex.Lock()
	defer dl.metrics.mutex.Unlock()

	atomic.AddInt64(&dl.metrics.TotalOperations, 1)

	switch operation {
	case "vector_search", "vector_search_cached":
		atomic.AddInt64(&dl.metrics.VectorOperations, 1)
		if strings.Contains(operation, "cached") {
			// Update cache hit rate
			dl.metrics.CacheHitRate = (dl.metrics.CacheHitRate + 1.0) / 2.0
		}
	case "store_document":
		atomic.AddInt64(&dl.metrics.PostgreSQLQueries, 1)
	case "create_relation", "find_relations":
		atomic.AddInt64(&dl.metrics.Neo4jQueries, 1)
		if strings.Contains(operation, "find") {
			atomic.AddInt64(&dl.metrics.GraphTraversals, 1)
		}
	}

	if err != nil {
		atomic.AddInt64(&dl.metrics.ErrorCount, 1)
		dl.metrics.LastError = err.Error()
	}

	// Update average query time with atomic operation
	queryTimeUs := duration.Microseconds()
	currentAvg := atomic.LoadInt64(&dl.metrics.AverageQueryTime)
	newAvg := (currentAvg + queryTimeUs) / 2
	atomic.StoreInt64(&dl.metrics.AverageQueryTime, newAvg)
}

func (dl *DatabaseLayer) collectMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		dl.metrics.mutex.Lock()

		// Update uptime
		uptime := time.Since(dl.metrics.StartTime)
		dl.metrics.UptimeSeconds = int64(uptime.Seconds())

		// Update active connections (approximation)
		dl.metrics.ActiveConnections = int64(dl.pgPool.Stat().AcquiredConns())

		dl.metrics.mutex.Unlock()
	}
}

// HTTP API Handlers
func (dl *DatabaseLayer) setupAPI() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// CORS
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

	api := router.Group("/api")
	{
		api.POST("/vector-search", dl.handleVectorSearch)
		api.POST("/documents", dl.handleStoreDocument)
		api.POST("/relations", dl.handleCreateRelation)
		api.GET("/relations/:entity", dl.handleFindRelations)
		api.GET("/metrics", dl.handleMetrics)
		api.GET("/status", dl.handleStatus)
		api.GET("/performance", dl.handlePerformanceStats)
	}

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Database Integration Layer",
			"status":  "healthy",
			"version": "2.0.0",
			"databases": gin.H{
				"postgresql": "connected",
				"neo4j":      "connected",
				"redis":      "connected",
			},
		})
	})

	return router
}

func (dl *DatabaseLayer) handleVectorSearch(c *gin.Context) {
	var req VectorSearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Limit == 0 {
		req.Limit = 10
	}
	if req.MinScore == 0 {
		req.MinScore = 0.3
	}

	result, err := dl.VectorSearch(&req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (dl *DatabaseLayer) handleStoreDocument(c *gin.Context) {
	var doc LegalDocument
	if err := c.ShouldBindJSON(&doc); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := dl.StoreDocument(&doc); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, doc)
}

func (dl *DatabaseLayer) handleCreateRelation(c *gin.Context) {
	var relation EntityRelation
	if err := c.ShouldBindJSON(&relation); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := dl.CreateEntityRelation(&relation); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{"status": "created"})
}

func (dl *DatabaseLayer) handleFindRelations(c *gin.Context) {
	entityID := c.Param("entity")
	maxDepth := 2

	if depth := c.Query("depth"); depth != "" {
		if d, err := strconv.Atoi(depth); err == nil {
			maxDepth = d
		}
	}

	relations, err := dl.FindRelatedEntities(entityID, maxDepth)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"entity":    entityID,
		"relations": relations,
		"count":     len(relations),
	})
}

func (dl *DatabaseLayer) handleMetrics(c *gin.Context) {
	dl.metrics.mutex.RLock()
	metrics := *dl.metrics
	dl.metrics.mutex.RUnlock()

	c.JSON(http.StatusOK, metrics)
}

func (dl *DatabaseLayer) handleStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":     "Database Integration Layer",
		"version":     "2.0.0",
		"status":      "running",
		"databases": gin.H{
			"postgresql": "connected",
			"neo4j":      "connected",
			"redis":      "connected",
		},
		"cache_enabled": dl.cache.enabled,
		"metrics_enabled": dl.config.EnableMetrics,
		"timestamp": time.Now(),
	})
}

// Context7 Performance Statistics Handler
func (dl *DatabaseLayer) handlePerformanceStats(c *gin.Context) {
	uptime := time.Since(dl.startTime)

	// Calculate rates and performance metrics
	totalOps := atomic.LoadInt64(&dl.totalRequests)
	totalDbOps := atomic.LoadInt64(&dl.totalDatabaseOps)
	cacheHits := atomic.LoadInt64(&dl.totalCacheHits)
	cacheMisses := atomic.LoadInt64(&dl.totalCacheMisses)

	var cacheHitRatio float64
	if cacheHits+cacheMisses > 0 {
		cacheHitRatio = float64(cacheHits) / float64(cacheHits+cacheMisses) * 100.0
	}

	stats := gin.H{
		"service_name": "DatabaseIntegrationLayer",
		"codec_name":  "context7_database_layer",
		"uptime": gin.H{
			"seconds":   int64(uptime.Seconds()),
			"formatted": uptime.String(),
		},
		"performance": gin.H{
			"total_requests":        totalOps,
			"total_database_ops":    totalDbOps,
			"total_cache_hits":      cacheHits,
			"total_cache_misses":    cacheMisses,
			"cache_hit_ratio":       cacheHitRatio,
			"average_response_time_us": atomic.LoadInt64(&dl.averageResponseTime),
			"postgresql_queries":    atomic.LoadInt64(&dl.metrics.PostgreSQLQueries),
			"neo4j_queries":         atomic.LoadInt64(&dl.metrics.Neo4jQueries),
			"vector_operations":     atomic.LoadInt64(&dl.metrics.VectorOperations),
			"graph_traversals":      atomic.LoadInt64(&dl.metrics.GraphTraversals),
			"total_errors":          atomic.LoadInt64(&dl.metrics.ErrorCount),
		},
		"rates": gin.H{
			"requests_per_second":   float64(totalOps) / uptime.Seconds(),
			"requests_per_hour":     float64(totalOps) / uptime.Hours(),
			"db_ops_per_second":     float64(totalDbOps) / uptime.Seconds(),
			"db_ops_per_hour":       float64(totalDbOps) / uptime.Hours(),
		},
		"health": gin.H{
			"status":               "healthy",
			"error_rate":           calculateDatabaseErrorRate(dl),
			"cache_efficiency":     cacheHitRatio,
			"performance_grade":    calculatePerformanceGrade(dl),
		},
		"connections": gin.H{
			"postgresql_active":    dl.pgPool.Stat().AcquiredConns(),
			"postgresql_idle":      dl.pgPool.Stat().IdleConns(),
			"postgresql_total":     dl.pgPool.Stat().TotalConns(),
		},
	}

	// Add Context7 monitoring headers
	c.Header("X-Service-Uptime", fmt.Sprintf("%.0f", uptime.Seconds()))
	c.Header("X-Total-Requests", fmt.Sprintf("%d", totalOps))
	c.Header("X-Cache-Hit-Ratio", fmt.Sprintf("%.2f", cacheHitRatio))
	c.Header("X-Average-Response-Time-Us", fmt.Sprintf("%d", atomic.LoadInt64(&dl.averageResponseTime)))
	c.Header("X-Performance-Grade", calculatePerformanceGrade(dl))

	c.JSON(http.StatusOK, stats)
}

func calculateDatabaseErrorRate(dl *DatabaseLayer) float64 {
	totalOps := atomic.LoadInt64(&dl.totalDatabaseOps)
	if totalOps == 0 {
		return 0.0
	}

	return float64(atomic.LoadInt64(&dl.metrics.ErrorCount)) / float64(totalOps) * 100.0
}

func calculatePerformanceGrade(dl *DatabaseLayer) string {
	avgResponseTime := atomic.LoadInt64(&dl.averageResponseTime)
	errorRate := calculateDatabaseErrorRate(dl)

	if errorRate > 10.0 {
		return "D"
	}
	if avgResponseTime > 100000 { // > 100ms
		return "C"
	}
	if avgResponseTime > 50000 { // > 50ms
		return "B"
	}
	if avgResponseTime > 10000 { // > 10ms
		return "A"
	}
	return "A+"
}

func (dl *DatabaseLayer) Run() error {
	if err := dl.Initialize(); err != nil {
		return err
	}

	router := dl.setupAPI()

	log.Printf("🗄️  Database Integration Layer starting on port %s", dl.config.Port)
	log.Printf("📊 PostgreSQL: Connected with pgvector")
	log.Printf("🕸️  Neo4j: Connected for graph operations")
	log.Printf("🔴 Redis: Connected for caching")

	return router.Run(":" + dl.config.Port)
}

func (dl *DatabaseLayer) Cleanup() {
	log.Println("🧹 Cleaning up Database Integration Layer...")

	if dl.pgPool != nil {
		dl.pgPool.Close()
	}

	if dl.neo4jDriver != nil {
		dl.neo4jDriver.Close(context.Background())
	}

	if dl.redisClient != nil {
		dl.redisClient.Close()
	}

	log.Println("✅ Database Integration Layer cleanup complete")
}

// Utility functions
func vectorToString(vector []float64) string {
	if len(vector) == 0 {
		return "[]"
	}

	strValues := make([]string, len(vector))
	for i, v := range vector {
		strValues[i] = strconv.FormatFloat(v, 'f', -1, 64)
	}
	return "[" + strings.Join(strValues, ",") + "]"
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if i, err := strconv.Atoi(value); err == nil {
			return i
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if b, err := strconv.ParseBool(value); err == nil {
			return b
		}
	}
	return defaultValue
}

func main() {
	dbLayer := NewDatabaseLayer()
	defer dbLayer.Cleanup()

	if err := dbLayer.Run(); err != nil {
		log.Fatalf("💥 Database Integration Layer failed: %v", err)
	}
}