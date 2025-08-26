// multi_layer_cache.go - Enterprise Multi-Layer Caching System
// Version 2.0 - Memurai (Redis) + PostgreSQL JSONB caching
package cache

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
)

// CacheLevel represents different cache tiers
type CacheLevel int

const (
	L1Cache CacheLevel = iota // In-memory (local)
	L2Cache                   // Redis (distributed)
	L3Cache                   // PostgreSQL JSONB (persistent)
)

// CacheEntry represents a cached item with metadata
type CacheEntry struct {
	Key         string                 `json:"key"`
	Value       interface{}            `json:"value"`
	Namespace   string                 `json:"namespace"`
	TTLSeconds  int                    `json:"ttl_seconds,omitempty"`
	CreatedAt   time.Time             `json:"created_at"`
	ExpiresAt   *time.Time            `json:"expires_at,omitempty"`
	AccessCount int                   `json:"access_count"`
	SizeBytes   int                   `json:"size_bytes,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// CacheStats provides cache performance metrics
type CacheStats struct {
	L1Stats CacheLevelStats `json:"l1_stats"`
	L2Stats CacheLevelStats `json:"l2_stats"`
	L3Stats CacheLevelStats `json:"l3_stats"`
	Total   CacheLevelStats `json:"total"`
}

type CacheLevelStats struct {
	Hits        int64   `json:"hits"`
	Misses      int64   `json:"misses"`
	Sets        int64   `json:"sets"`
	Deletes     int64   `json:"deletes"`
	HitRatio    float64 `json:"hit_ratio"`
	TotalKeys   int64   `json:"total_keys"`
	TotalSize   int64   `json:"total_size_bytes"`
	AvgSetTime  float64 `json:"avg_set_time_ms"`
	AvgGetTime  float64 `json:"avg_get_time_ms"`
}

// MultiLayerCache implements enterprise-grade caching with multiple tiers
type MultiLayerCache struct {
	// L1: In-memory cache (fastest)
	l1Cache map[string]*CacheEntry
	l1Stats CacheLevelStats
	
	// L2: Redis/Memurai cache (distributed)
	redisClient *redis.Client
	l2Stats     CacheLevelStats
	
	// L3: PostgreSQL JSONB cache (persistent)
	pgPool  *pgxpool.Pool
	l3Stats CacheLevelStats
	
	// Configuration
	config MultiLayerCacheConfig
	
	// Performance monitoring
	stats CacheStats
}

// MultiLayerCacheConfig holds cache configuration
type MultiLayerCacheConfig struct {
	// L1 Configuration
	L1MaxSize     int           `json:"l1_max_size"`
	L1DefaultTTL  time.Duration `json:"l1_default_ttl"`
	
	// L2 Configuration (Redis/Memurai)
	L2Address     string        `json:"l2_address"`
	L2Password    string        `json:"l2_password"`
	L2DB          int          `json:"l2_db"`
	L2DefaultTTL  time.Duration `json:"l2_default_ttl"`
	L2MaxRetries  int          `json:"l2_max_retries"`
	
	// L3 Configuration (PostgreSQL)
	L3DefaultTTL  time.Duration `json:"l3_default_ttl"`
	L3CleanupInterval time.Duration `json:"l3_cleanup_interval"`
	
	// General settings
	EnableL1      bool `json:"enable_l1"`
	EnableL2      bool `json:"enable_l2"`
	EnableL3      bool `json:"enable_l3"`
	EnableMetrics bool `json:"enable_metrics"`
	WriteThrough  bool `json:"write_through"` // Write to all layers simultaneously
	ReadThrough   bool `json:"read_through"`  // Populate upper layers on cache miss
}

// NewMultiLayerCache creates a new multi-layer cache instance
func NewMultiLayerCache(config MultiLayerCacheConfig, pgPool *pgxpool.Pool) (*MultiLayerCache, error) {
	cache := &MultiLayerCache{
		l1Cache: make(map[string]*CacheEntry),
		pgPool:  pgPool,
		config:  config,
	}
	
	// Initialize Redis/Memurai client if enabled
	if config.EnableL2 {
		cache.redisClient = redis.NewClient(&redis.Options{
			Addr:         config.L2Address,
			Password:     config.L2Password,
			DB:           config.L2DB,
			MaxRetries:   config.L2MaxRetries,
			DialTimeout:  5 * time.Second,
			ReadTimeout:  3 * time.Second,
			WriteTimeout: 3 * time.Second,
		})
		
		// Test connection
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		if err := cache.redisClient.Ping(ctx).Err(); err != nil {
			log.Printf("Warning: Redis/Memurai connection failed: %v", err)
			config.EnableL2 = false
		} else {
			log.Printf("✅ Connected to Redis/Memurai at %s", config.L2Address)
		}
	}
	
	// Start background cleanup if L3 is enabled
	if config.EnableL3 && config.L3CleanupInterval > 0 {
		go cache.startCleanupRoutine()
	}
	
	log.Printf("MultiLayerCache initialized: L1=%v L2=%v L3=%v", 
		config.EnableL1, config.EnableL2, config.EnableL3)
	
	return cache, nil
}

// Get retrieves a value from the cache, checking all layers
func (c *MultiLayerCache) Get(ctx context.Context, namespace, key string) (interface{}, bool) {
	fullKey := c.buildKey(namespace, key)
	start := time.Now()
	
	// Try L1 cache first (fastest)
	if c.config.EnableL1 {
		if entry, found := c.getFromL1(fullKey); found {
			c.l1Stats.Hits++
			c.updateAccessStats(entry)
			return entry.Value, true
		}
		c.l1Stats.Misses++
	}
	
	// Try L2 cache (Redis/Memurai)
	if c.config.EnableL2 {
		if value, found := c.getFromL2(ctx, fullKey); found {
			c.l2Stats.Hits++
			
			// Write-back to L1 if enabled and read-through is configured
			if c.config.EnableL1 && c.config.ReadThrough {
				c.setToL1(fullKey, value, int(c.config.L1DefaultTTL.Seconds()))
			}
			
			return value, true
		}
		c.l2Stats.Misses++
	}
	
	// Try L3 cache (PostgreSQL JSONB)
	if c.config.EnableL3 {
		if value, found := c.getFromL3(ctx, fullKey); found {
			c.l3Stats.Hits++
			
			// Write-back to upper layers if read-through is enabled
			if c.config.ReadThrough {
				if c.config.EnableL2 {
					c.setToL2(ctx, fullKey, value, int(c.config.L2DefaultTTL.Seconds()))
				}
				if c.config.EnableL1 {
					c.setToL1(fullKey, value, int(c.config.L1DefaultTTL.Seconds()))
				}
			}
			
			return value, true
		}
		c.l3Stats.Misses++
	}
	
	// Update timing statistics
	if c.config.EnableMetrics {
		duration := time.Since(start).Milliseconds()
		c.updateGetTimingStats(float64(duration))
	}
	
	return nil, false
}

// Set stores a value in the cache across configured layers
func (c *MultiLayerCache) Set(ctx context.Context, namespace, key string, value interface{}, ttlSeconds int) error {
	fullKey := c.buildKey(namespace, key)
	start := time.Now()
	
	var errs []error
	
	// Write-through to all enabled layers
	if c.config.EnableL1 {
		if err := c.setToL1(fullKey, value, ttlSeconds); err != nil {
			errs = append(errs, fmt.Errorf("L1 cache error: %w", err))
		} else {
			c.l1Stats.Sets++
		}
	}
	
	if c.config.EnableL2 {
		if err := c.setToL2(ctx, fullKey, value, ttlSeconds); err != nil {
			errs = append(errs, fmt.Errorf("L2 cache error: %w", err))
		} else {
			c.l2Stats.Sets++
		}
	}
	
	if c.config.EnableL3 {
		if err := c.setToL3(ctx, fullKey, value, ttlSeconds); err != nil {
			errs = append(errs, fmt.Errorf("L3 cache error: %w", err))
		} else {
			c.l3Stats.Sets++
		}
	}
	
	// Update timing statistics
	if c.config.EnableMetrics {
		duration := time.Since(start).Milliseconds()
		c.updateSetTimingStats(float64(duration))
	}
	
	// Return combined errors if any
	if len(errs) > 0 {
		return fmt.Errorf("cache set errors: %v", errs)
	}
	
	return nil
}

// Delete removes a value from all cache layers
func (c *MultiLayerCache) Delete(ctx context.Context, namespace, key string) error {
	fullKey := c.buildKey(namespace, key)
	
	var errs []error
	
	// Delete from all layers
	if c.config.EnableL1 {
		c.deleteFromL1(fullKey)
		c.l1Stats.Deletes++
	}
	
	if c.config.EnableL2 {
		if err := c.deleteFromL2(ctx, fullKey); err != nil {
			errs = append(errs, fmt.Errorf("L2 delete error: %w", err))
		} else {
			c.l2Stats.Deletes++
		}
	}
	
	if c.config.EnableL3 {
		if err := c.deleteFromL3(ctx, fullKey); err != nil {
			errs = append(errs, fmt.Errorf("L3 delete error: %w", err))
		} else {
			c.l3Stats.Deletes++
		}
	}
	
	if len(errs) > 0 {
		return fmt.Errorf("cache delete errors: %v", errs)
	}
	
	return nil
}

// L1 Cache Operations (In-Memory)
func (c *MultiLayerCache) getFromL1(key string) (*CacheEntry, bool) {
	entry, found := c.l1Cache[key]
	if !found {
		return nil, false
	}
	
	// Check expiration
	if entry.ExpiresAt != nil && time.Now().After(*entry.ExpiresAt) {
		delete(c.l1Cache, key)
		return nil, false
	}
	
	return entry, true
}

func (c *MultiLayerCache) setToL1(key string, value interface{}, ttlSeconds int) error {
	entry := &CacheEntry{
		Key:         key,
		Value:       value,
		CreatedAt:   time.Now(),
		AccessCount: 0,
	}
	
	if ttlSeconds > 0 {
		expiresAt := time.Now().Add(time.Duration(ttlSeconds) * time.Second)
		entry.ExpiresAt = &expiresAt
		entry.TTLSeconds = ttlSeconds
	}
	
	// Simple LRU eviction if cache is full
	if len(c.l1Cache) >= c.config.L1MaxSize {
		c.evictLRUFromL1()
	}
	
	c.l1Cache[key] = entry
	return nil
}

func (c *MultiLayerCache) deleteFromL1(key string) {
	delete(c.l1Cache, key)
}

func (c *MultiLayerCache) evictLRUFromL1() {
	var oldestKey string
	var oldestTime time.Time
	
	for key, entry := range c.l1Cache {
		if oldestKey == "" || entry.CreatedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.CreatedAt
		}
	}
	
	if oldestKey != "" {
		delete(c.l1Cache, oldestKey)
	}
}

// L2 Cache Operations (Redis/Memurai)
func (c *MultiLayerCache) getFromL2(ctx context.Context, key string) (interface{}, bool) {
	result, err := c.redisClient.Get(ctx, key).Result()
	if err == redis.Nil {
		return nil, false
	} else if err != nil {
		log.Printf("Redis GET error: %v", err)
		return nil, false
	}
	
	var value interface{}
	if err := json.Unmarshal([]byte(result), &value); err != nil {
		log.Printf("Redis JSON unmarshal error: %v", err)
		return nil, false
	}
	
	return value, true
}

func (c *MultiLayerCache) setToL2(ctx context.Context, key string, value interface{}, ttlSeconds int) error {
	jsonValue, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("JSON marshal error: %w", err)
	}
	
	ttl := time.Duration(ttlSeconds) * time.Second
	if ttlSeconds <= 0 {
		ttl = c.config.L2DefaultTTL
	}
	
	return c.redisClient.Set(ctx, key, jsonValue, ttl).Err()
}

func (c *MultiLayerCache) deleteFromL2(ctx context.Context, key string) error {
	return c.redisClient.Del(ctx, key).Err()
}

// L3 Cache Operations (PostgreSQL JSONB)
func (c *MultiLayerCache) getFromL3(ctx context.Context, key string) (interface{}, bool) {
	var cacheValue []byte
	var expiresAt sql.NullTime
	
	query := `
		SELECT cache_value, expires_at
		FROM cache_entries 
		WHERE cache_key = $1 
		AND (expires_at IS NULL OR expires_at > NOW())
	`
	
	err := c.pgPool.QueryRow(ctx, query, key).Scan(&cacheValue, &expiresAt)
	if err != nil {
		if err != sql.ErrNoRows {
			log.Printf("PostgreSQL cache GET error: %v", err)
		}
		return nil, false
	}
	
	// Update access statistics
	c.updateL3AccessStats(ctx, key)
	
	var value interface{}
	if err := json.Unmarshal(cacheValue, &value); err != nil {
		log.Printf("PostgreSQL cache JSON unmarshal error: %v", err)
		return nil, false
	}
	
	return value, true
}

func (c *MultiLayerCache) setToL3(ctx context.Context, key string, value interface{}, ttlSeconds int) error {
	jsonValue, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("JSON marshal error: %w", err)
	}
	
	var expiresAt *time.Time
	if ttlSeconds > 0 {
		expiry := time.Now().Add(time.Duration(ttlSeconds) * time.Second)
		expiresAt = &expiry
	}
	
	sizeBytes := len(jsonValue)
	
	query := `
		INSERT INTO cache_entries (cache_key, cache_namespace, cache_value, ttl_seconds, expires_at, size_bytes)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (cache_key) 
		DO UPDATE SET 
			cache_value = EXCLUDED.cache_value,
			ttl_seconds = EXCLUDED.ttl_seconds,
			expires_at = EXCLUDED.expires_at,
			size_bytes = EXCLUDED.size_bytes,
			last_accessed = NOW()
	`
	
	_, err = c.pgPool.Exec(ctx, query, key, "default", jsonValue, ttlSeconds, expiresAt, sizeBytes)
	return err
}

func (c *MultiLayerCache) deleteFromL3(ctx context.Context, key string) error {
	query := `DELETE FROM cache_entries WHERE cache_key = $1`
	_, err := c.pgPool.Exec(ctx, query, key)
	return err
}

func (c *MultiLayerCache) updateL3AccessStats(ctx context.Context, key string) {
	query := `
		UPDATE cache_entries 
		SET access_count = access_count + 1, last_accessed = NOW()
		WHERE cache_key = $1
	`
	c.pgPool.Exec(ctx, query, key)
}

// Utility methods
func (c *MultiLayerCache) buildKey(namespace, key string) string {
	return fmt.Sprintf("%s:%s", namespace, key)
}

func (c *MultiLayerCache) updateAccessStats(entry *CacheEntry) {
	entry.AccessCount++
}

func (c *MultiLayerCache) updateGetTimingStats(duration float64) {
	// Update average GET timing statistics
	if c.l1Stats.Hits+c.l1Stats.Misses > 0 {
		total := c.l1Stats.Hits + c.l1Stats.Misses
		c.l1Stats.AvgGetTime = (c.l1Stats.AvgGetTime*float64(total-1) + duration) / float64(total)
	}
}

func (c *MultiLayerCache) updateSetTimingStats(duration float64) {
	// Update average SET timing statistics
	if c.l1Stats.Sets > 0 {
		c.l1Stats.AvgSetTime = (c.l1Stats.AvgSetTime*float64(c.l1Stats.Sets-1) + duration) / float64(c.l1Stats.Sets)
	}
}

// GetStats returns comprehensive cache statistics
func (c *MultiLayerCache) GetStats() CacheStats {
	// Calculate hit ratios
	if c.l1Stats.Hits+c.l1Stats.Misses > 0 {
		c.l1Stats.HitRatio = float64(c.l1Stats.Hits) / float64(c.l1Stats.Hits+c.l1Stats.Misses)
	}
	
	if c.l2Stats.Hits+c.l2Stats.Misses > 0 {
		c.l2Stats.HitRatio = float64(c.l2Stats.Hits) / float64(c.l2Stats.Hits+c.l2Stats.Misses)
	}
	
	if c.l3Stats.Hits+c.l3Stats.Misses > 0 {
		c.l3Stats.HitRatio = float64(c.l3Stats.Hits) / float64(c.l3Stats.Hits+c.l3Stats.Misses)
	}
	
	// Calculate totals
	total := CacheLevelStats{
		Hits:    c.l1Stats.Hits + c.l2Stats.Hits + c.l3Stats.Hits,
		Misses:  c.l1Stats.Misses + c.l2Stats.Misses + c.l3Stats.Misses,
		Sets:    c.l1Stats.Sets + c.l2Stats.Sets + c.l3Stats.Sets,
		Deletes: c.l1Stats.Deletes + c.l2Stats.Deletes + c.l3Stats.Deletes,
	}
	
	if total.Hits+total.Misses > 0 {
		total.HitRatio = float64(total.Hits) / float64(total.Hits+total.Misses)
	}
	
	return CacheStats{
		L1Stats: c.l1Stats,
		L2Stats: c.l2Stats,
		L3Stats: c.l3Stats,
		Total:   total,
	}
}

// Cleanup expired entries periodically
func (c *MultiLayerCache) startCleanupRoutine() {
	ticker := time.NewTicker(c.config.L3CleanupInterval)
	defer ticker.Stop()
	
	for range ticker.C {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		
		// Clean L1 cache
		c.cleanupL1()
		
		// Clean L3 cache (PostgreSQL)
		c.cleanupL3(ctx)
		
		cancel()
	}
}

func (c *MultiLayerCache) cleanupL1() {
	now := time.Now()
	for key, entry := range c.l1Cache {
		if entry.ExpiresAt != nil && now.After(*entry.ExpiresAt) {
			delete(c.l1Cache, key)
		}
	}
}

func (c *MultiLayerCache) cleanupL3(ctx context.Context) {
	query := `DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < NOW()`
	_, err := c.pgPool.Exec(ctx, query)
	if err != nil {
		log.Printf("L3 cache cleanup error: %v", err)
	}
}

// Close gracefully shuts down the cache
func (c *MultiLayerCache) Close() error {
	if c.redisClient != nil {
		return c.redisClient.Close()
	}
	return nil
}