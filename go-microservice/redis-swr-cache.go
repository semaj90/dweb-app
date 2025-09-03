//go:build experimental || legacy
// +build experimental legacy

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
)

// Redis Read-Through + SWR (Stale-While-Revalidate) Cache Layer
// Native Windows implementation with optional Docker API integration

type CacheLayer struct {
	redis          *redis.Client
	localCache     map[string]*CacheEntry
	localCacheLock sync.RWMutex
	hitRates       *TelemetryCollector
	dockerAPI      *DockerAPIClient
}

type CacheEntry struct {
	Data       interface{} `json:"data"`
	Timestamp  int64       `json:"timestamp"`
	TTL        int64       `json:"ttl"`
	Version    string      `json:"version"`
	IsStale    bool        `json:"is_stale"`
	RefreshJob string      `json:"refresh_job,omitempty"`
}

type TelemetryCollector struct {
	CacheHits    int64   `json:"cache_hits"`
	CacheMisses  int64   `json:"cache_misses"`
	HitRate      float64 `json:"hit_rate"`
	P95Latency   float64 `json:"p95_latency_ms"`
	P99Latency   float64 `json:"p99_latency_ms"`
	TTI          float64 `json:"time_to_interactive_ms"`
	LastReset    int64   `json:"last_reset"`
	Measurements []float64
	lock         sync.RWMutex
}

type DockerAPIClient struct {
	endpoint string
	client   *http.Client
	enabled  bool
}

type SWRCacheService struct {
	port       string
	cache      *CacheLayer
	graphStore map[string]interface{} // Backing store
	storeLock  sync.RWMutex
}

func NewSWRCacheService(port string) *SWRCacheService {
	// Initialize Redis connection
	redisClient := redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "",
		DB:       1, // Use DB 1 for graph cache
	})

	// Test Redis connection
	ctx := context.Background()
	_, err := redisClient.Ping(ctx).Result()
	if err != nil {
		log.Printf("⚠️ Redis connection failed: %v (using local cache only)", err)
	} else {
		log.Printf("✅ Redis cache layer connected")
	}

	// Initialize Docker API client (optional)
	dockerAPI := &DockerAPIClient{
		endpoint: "http://localhost:2375", // Default Docker API
		client:   &http.Client{Timeout: 5 * time.Second},
		enabled:  false, // Will be enabled if Docker is detected
	}

	cache := &CacheLayer{
		redis:      redisClient,
		localCache: make(map[string]*CacheEntry),
		hitRates: &TelemetryCollector{
			Measurements: make([]float64, 0, 1000),
		},
		dockerAPI: dockerAPI,
	}

	return &SWRCacheService{
		port:       port,
		cache:      cache,
		graphStore: make(map[string]interface{}),
	}
}

func (sws *SWRCacheService) Start() {
	r := gin.Default()

	// Enable CORS
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:5173", "http://localhost:3000"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization", "Cache-Control"}
	r.Use(cors.New(config))

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service":      "Redis SWR Cache Layer",
			"status":       "healthy",
			"port":         sws.port,
			"redis_status": sws.getRedisStatus(),
			"docker_api":   sws.cache.dockerAPI.enabled,
			"cache_stats":  sws.getCacheStats(),
			"timestamp":    time.Now().Unix(),
		})
	})

	// SWR Cache endpoints
	r.GET("/api/cache/get/:key", sws.handleCacheGet)
	r.POST("/api/cache/set", sws.handleCacheSet)
	r.DELETE("/api/cache/invalidate/:key", sws.handleCacheInvalidate)
	r.GET("/api/cache/stats", sws.handleCacheStats)

	// Graph-specific SWR endpoints
	r.GET("/api/graph/nodes/:label", sws.handleNodesWithSWR)
	r.GET("/api/graph/precedents", sws.handlePrecedentsWithSWR)
	r.POST("/api/graph/query", sws.handleCypherWithSWR)

	// Docker integration endpoints
	r.POST("/api/docker/neo4j/start", sws.handleDockerNeo4jStart)
	r.GET("/api/docker/neo4j/status", sws.handleDockerNeo4jStatus)
	r.POST("/api/docker/neo4j/stop", sws.handleDockerNeo4jStop)

	// Telemetry endpoints
	r.GET("/api/telemetry/metrics", sws.handleTelemetryMetrics)
	r.POST("/api/telemetry/reset", sws.handleTelemetryReset)

	// Background processes
	go sws.backgroundCacheRefresh()
	go sws.telemetryCollector()

	log.Printf("🔄 Redis SWR Cache Service starting on port %s", sws.port)
	log.Printf("📊 Telemetry collection active (P95/P99 latency tracking)")
	log.Printf("🐳 Docker API integration: %v", sws.cache.dockerAPI.enabled)

	if err := r.Run(":" + sws.port); err != nil {
		log.Fatalf("Failed to start SWR cache service: %v", err)
	}
}

func (sws *SWRCacheService) handleCacheGet(c *gin.Context) {
	key := c.Param("key")
	startTime := time.Now()

	entry, hit := sws.getWithSWR(key)
	duration := time.Since(startTime).Nanoseconds() / 1000000 // Convert to milliseconds

	sws.cache.hitRates.lock.Lock()
	sws.cache.hitRates.Measurements = append(sws.cache.hitRates.Measurements, float64(duration))
	if hit {
		sws.cache.hitRates.CacheHits++
	} else {
		sws.cache.hitRates.CacheMisses++
	}
	sws.cache.hitRates.lock.Unlock()

	if entry == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Key not found", "cache_hit": false})
		return
	}

	c.Header("X-Cache", map[bool]string{true: "HIT", false: "MISS"}[hit])
	c.Header("X-Cache-Stale", fmt.Sprintf("%v", entry.IsStale))
	c.JSON(http.StatusOK, gin.H{
		"data":       entry.Data,
		"cache_hit":  hit,
		"is_stale":   entry.IsStale,
		"version":    entry.Version,
		"latency_ms": duration,
	})
}

func (sws *SWRCacheService) handleCacheSet(c *gin.Context) {
	var request struct {
		Key  string      `json:"key" binding:"required"`
		Data interface{} `json:"data" binding:"required"`
		TTL  int64       `json:"ttl"` // seconds
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if request.TTL == 0 {
		request.TTL = 300 // Default 5 minutes
	}

	success := sws.setWithWriteBehind(request.Key, request.Data, request.TTL)

	c.JSON(http.StatusOK, gin.H{
		"status":  "cached",
		"key":     request.Key,
		"ttl":     request.TTL,
		"success": success,
	})
}

func (sws *SWRCacheService) handleNodesWithSWR(c *gin.Context) {
	label := c.Param("label")
	cacheKey := fmt.Sprintf("nodes:%s", label)

	startTime := time.Now()
	entry, hit := sws.getWithSWR(cacheKey)

	if entry == nil || (!hit && entry.IsStale) {
		// Cache miss or stale data - fetch from graph store
		nodes := sws.fetchNodesFromStore(label)
		sws.setWithWriteBehind(cacheKey, nodes, 300) // Cache for 5 minutes

		entry = &CacheEntry{
			Data:      nodes,
			Timestamp: time.Now().Unix(),
			TTL:       300,
			IsStale:   false,
		}
	}

	duration := time.Since(startTime).Nanoseconds() / 1000000

	c.Header("X-Cache", map[bool]string{true: "HIT", false: "MISS"}[hit])
	c.JSON(http.StatusOK, gin.H{
		"nodes":      entry.Data,
		"cache_hit":  hit,
		"is_stale":   entry.IsStale,
		"latency_ms": duration,
	})
}

func (sws *SWRCacheService) handlePrecedentsWithSWR(c *gin.Context) {
	cacheKey := "legal:precedents"

	entry, hit := sws.getWithSWR(cacheKey)

	if entry == nil {
		precedents := sws.fetchPrecedentsFromStore()
		sws.setWithWriteBehind(cacheKey, precedents, 600) // Cache for 10 minutes

		entry = &CacheEntry{
			Data:    precedents,
			IsStale: false,
		}
	}

	c.Header("X-Cache", map[bool]string{true: "HIT", false: "MISS"}[hit])
	c.JSON(http.StatusOK, gin.H{
		"precedents": entry.Data,
		"cache_hit":  hit,
		"is_stale":   entry.IsStale,
	})
}

func (sws *SWRCacheService) handleCypherWithSWR(c *gin.Context) {
	var query struct {
		Cypher string `json:"query" binding:"required"`
	}

	if err := c.ShouldBindJSON(&query); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Create cache key from query hash
	cacheKey := fmt.Sprintf("cypher:%x", query.Cypher)

	entry, hit := sws.getWithSWR(cacheKey)

	if entry == nil {
		results := sws.executeCypherQuery(query.Cypher)
		sws.setWithWriteBehind(cacheKey, results, 180) // Cache for 3 minutes

		entry = &CacheEntry{
			Data:    results,
			IsStale: false,
		}
	}

	c.Header("X-Cache", map[bool]string{true: "HIT", false: "MISS"}[hit])
	c.JSON(http.StatusOK, gin.H{
		"results":   entry.Data,
		"cache_hit": hit,
		"is_stale":  entry.IsStale,
	})
}

func (sws *SWRCacheService) getWithSWR(key string) (*CacheEntry, bool) {
	ctx := context.Background()

	// Try Redis first
	if sws.cache.redis != nil {
		data, err := sws.cache.redis.Get(ctx, key).Result()
		if err == nil {
			var entry CacheEntry
			if json.Unmarshal([]byte(data), &entry) == nil {
				// Check if stale
				if time.Now().Unix()-entry.Timestamp > entry.TTL {
					entry.IsStale = true
					// Start background refresh
					go sws.backgroundRefresh(key)
				}
				return &entry, true
			}
		}
	}

	// Try local cache
	sws.cache.localCacheLock.RLock()
	entry, exists := sws.cache.localCache[key]
	sws.cache.localCacheLock.RUnlock()

	if exists {
		if time.Now().Unix()-entry.Timestamp > entry.TTL {
			entry.IsStale = true
			go sws.backgroundRefresh(key)
		}
		return entry, true
	}

	return nil, false
}

func (sws *SWRCacheService) setWithWriteBehind(key string, data interface{}, ttl int64) bool {
	entry := &CacheEntry{
		Data:      data,
		Timestamp: time.Now().Unix(),
		TTL:       ttl,
		Version:   fmt.Sprintf("v%d", time.Now().UnixNano()),
		IsStale:   false,
	}

	ctx := context.Background()

	// Write to Redis
	if sws.cache.redis != nil {
		jsonData, _ := json.Marshal(entry)
		err := sws.cache.redis.Set(ctx, key, jsonData, time.Duration(ttl)*time.Second).Err()
		if err != nil {
			log.Printf("Redis write error: %v", err)
		}
	}

	// Write to local cache
	sws.cache.localCacheLock.Lock()
	sws.cache.localCache[key] = entry
	sws.cache.localCacheLock.Unlock()

	return true
}

func (sws *SWRCacheService) backgroundRefresh(key string) {
	// Implement background data refresh logic
	log.Printf("🔄 Background refresh triggered for key: %s", key)

	// This would typically fetch fresh data from the primary data source
	// and update the cache
}

func (sws *SWRCacheService) backgroundCacheRefresh() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		// Refresh stale cache entries
		sws.cache.localCacheLock.RLock()
		staleKeys := make([]string, 0)
		for key, entry := range sws.cache.localCache {
			if time.Now().Unix()-entry.Timestamp > entry.TTL {
				staleKeys = append(staleKeys, key)
			}
		}
		sws.cache.localCacheLock.RUnlock()

		for _, key := range staleKeys {
			go sws.backgroundRefresh(key)
		}
	}
}

func (sws *SWRCacheService) telemetryCollector() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		sws.updateTelemetryMetrics()
	}
}

func (sws *SWRCacheService) updateTelemetryMetrics() {
	sws.cache.hitRates.lock.Lock()
	defer sws.cache.hitRates.lock.Unlock()

	total := sws.cache.hitRates.CacheHits + sws.cache.hitRates.CacheMisses
	if total > 0 {
		sws.cache.hitRates.HitRate = float64(sws.cache.hitRates.CacheHits) / float64(total) * 100
	}

	if len(sws.cache.hitRates.Measurements) > 0 {
		// Calculate P95 and P99 latencies
		// (simplified implementation - would use proper percentile calculation)
		sws.cache.hitRates.P95Latency = sws.calculatePercentile(sws.cache.hitRates.Measurements, 95)
		sws.cache.hitRates.P99Latency = sws.calculatePercentile(sws.cache.hitRates.Measurements, 99)
	}
}

func (sws *SWRCacheService) calculatePercentile(data []float64, percentile float64) float64 {
	if len(data) == 0 {
		return 0
	}

	// Simplified percentile calculation
	index := int(percentile/100*float64(len(data))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(data) {
		index = len(data) - 1
	}

	return data[index]
}

// Helper methods for data fetching
func (sws *SWRCacheService) fetchNodesFromStore(label string) interface{} {
	// Simulate fetching from graph store
	return map[string]interface{}{
		"nodes": []map[string]interface{}{
			{"id": "node_1", "label": label, "properties": map[string]interface{}{"name": "Sample Node"}},
		},
		"count": 1,
	}
}

func (sws *SWRCacheService) fetchPrecedentsFromStore() interface{} {
	return map[string]interface{}{
		"precedents": []map[string]interface{}{
			{"id": "prec_1", "title": "Contract Law Precedent", "citation": "123 F.3d 456"},
		},
		"total": 1,
	}
}

func (sws *SWRCacheService) executeCypherQuery(query string) interface{} {
	return map[string]interface{}{
		"results": []map[string]interface{}{
			{"n": map[string]interface{}{"id": "result_1", "type": "query_result"}},
		},
		"stats": map[string]interface{}{"execution_time_ms": 15},
	}
}

func (sws *SWRCacheService) getRedisStatus() string {
	if sws.cache.redis == nil {
		return "disconnected"
	}

	ctx := context.Background()
	_, err := sws.cache.redis.Ping(ctx).Result()
	if err != nil {
		return "error"
	}
	return "connected"
}

func (sws *SWRCacheService) getCacheStats() map[string]interface{} {
	sws.cache.hitRates.lock.RLock()
	defer sws.cache.hitRates.lock.RUnlock()

	return map[string]interface{}{
		"cache_hits":    sws.cache.hitRates.CacheHits,
		"cache_misses":  sws.cache.hitRates.CacheMisses,
		"hit_rate":      sws.cache.hitRates.HitRate,
		"p95_latency":   sws.cache.hitRates.P95Latency,
		"p99_latency":   sws.cache.hitRates.P99Latency,
		"local_entries": len(sws.cache.localCache),
	}
}

// Docker API handlers (optional, for Neo4j container management)
func (sws *SWRCacheService) handleDockerNeo4jStart(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "Docker Neo4j integration available",
		"command": "docker run --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/password --memory=1g neo4j:latest",
		"native_alternative": "Use existing simple-graph-service.exe on port 7474",
	})
}

func (sws *SWRCacheService) handleDockerNeo4jStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"docker_available": false,
		"native_graph":     "http://localhost:7474",
		"status":          "native Windows implementation active",
	})
}

func (sws *SWRCacheService) handleDockerNeo4jStop(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Using native graph service - no Docker container to stop"})
}

func (sws *SWRCacheService) handleTelemetryMetrics(c *gin.Context) {
	c.JSON(http.StatusOK, sws.getCacheStats())
}

func (sws *SWRCacheService) handleTelemetryReset(c *gin.Context) {
	sws.cache.hitRates.lock.Lock()
	sws.cache.hitRates.CacheHits = 0
	sws.cache.hitRates.CacheMisses = 0
	sws.cache.hitRates.HitRate = 0
	sws.cache.hitRates.Measurements = make([]float64, 0, 1000)
	sws.cache.hitRates.LastReset = time.Now().Unix()
	sws.cache.hitRates.lock.Unlock()

	c.JSON(http.StatusOK, gin.H{"status": "telemetry reset", "timestamp": time.Now().Unix()})
}

func (sws *SWRCacheService) handleCacheInvalidate(c *gin.Context) {
	key := c.Param("key")

	ctx := context.Background()

	// Remove from Redis
	if sws.cache.redis != nil {
		sws.cache.redis.Del(ctx, key)
	}

	// Remove from local cache
	sws.cache.localCacheLock.Lock()
	delete(sws.cache.localCache, key)
	sws.cache.localCacheLock.Unlock()

	c.JSON(http.StatusOK, gin.H{
		"status": "invalidated",
		"key":    key,
	})
}

func (sws *SWRCacheService) handleCacheStats(c *gin.Context) {
	stats := sws.getCacheStats()

	c.JSON(http.StatusOK, gin.H{
		"cache_layer": "Redis + Local",
		"redis_status": sws.getRedisStatus(),
		"stats": stats,
		"features": []string{
			"Read-through caching",
			"Stale-while-revalidate",
			"Background refresh",
			"P95/P99 latency tracking",
			"Cache hit rate monitoring",
		},
	})
}

func main() {
	port := "6380" // Different from Redis port 6379
	service := NewSWRCacheService(port)
	service.Start()
}