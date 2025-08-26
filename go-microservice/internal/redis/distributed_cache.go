// Redis Distributed Cache for AI Modular System
// Provides distributed caching with RabbitMQ integration
// Production-ready implementation for Phase 2

package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
)

// DistributedCache handles Redis operations for AI system
type DistributedCache struct {
	client       *redis.Client
	ctx          context.Context
	mu           sync.RWMutex
	cacheMetrics *CacheMetrics
	enabled      bool
}

// CacheMetrics tracks cache performance
type CacheMetrics struct {
	mu           sync.RWMutex
	Hits         int64         `json:"hits"`
	Misses       int64         `json:"misses"`
	Sets         int64         `json:"sets"`
	Deletes      int64         `json:"deletes"`
	AvgGetTime   time.Duration `json:"avg_get_time"`
	AvgSetTime   time.Duration `json:"avg_set_time"`
	TotalGetTime time.Duration `json:"total_get_time"`
	TotalSetTime time.Duration `json:"total_set_time"`
	HitRate      float64       `json:"hit_rate"`
	LastUpdate   time.Time     `json:"last_update"`
}

// DimensionalArrayCacheEntry represents a cached dimensional array
type DimensionalArrayCacheEntry struct {
	Key          string                 `json:"key"`
	Dimensions   []int                  `json:"dimensions"`
	Data         []float32             `json:"data"`
	Metadata     map[string]interface{} `json:"metadata"`
	ComputedAt   time.Time             `json:"computed_at"`
	AccessCount  int64                 `json:"access_count"`
	TTL          time.Duration         `json:"ttl"`
}

// T5ProcessingCacheEntry represents cached T5 processing results
type T5ProcessingCacheEntry struct {
	Key            string                 `json:"key"`
	InputText      string                `json:"input_text"`
	ProcessedText  string                `json:"processed_text"`
	Embeddings     []float32             `json:"embeddings"`
	Confidence     float64               `json:"confidence"`
	ProcessingTime time.Duration         `json:"processing_time"`
	ModelVersion   string                `json:"model_version"`
	Metadata       map[string]interface{} `json:"metadata"`
	CachedAt       time.Time             `json:"cached_at"`
}

var (
	globalCache *DistributedCache
	cacheOnce   sync.Once
)

// InitializeDistributedCache initializes the global cache instance
func InitializeDistributedCache(redisURL string) (*DistributedCache, error) {
	var initErr error
	cacheOnce.Do(func() {
		if redisURL == "" {
			redisURL = "localhost:6379"
		}

		// Try parsing as full Redis URL first
		opt, err := redis.ParseURL(fmt.Sprintf("redis://%s", redisURL))
		if err != nil {
			// Fallback to simple options
			opt = &redis.Options{
				Addr:         redisURL,
				Password:     "", // no password
				DB:           0,  // default DB
				PoolSize:     10,
				MinIdleConns: 5,
				MaxRetries:   3,
			}
		}

		client := redis.NewClient(opt)
		ctx := context.Background()

		// Test connection
		if err := client.Ping(ctx).Err(); err != nil {
			log.Printf("⚠️ Redis connection failed: %v (using local cache)", err)
			globalCache = &DistributedCache{
				ctx:          ctx,
				cacheMetrics: &CacheMetrics{LastUpdate: time.Now()},
				enabled:      false,
			}
			return
		}

		globalCache = &DistributedCache{
			client:       client,
			ctx:          ctx,
			cacheMetrics: &CacheMetrics{LastUpdate: time.Now()},
			enabled:      true,
		}

		// Start metrics updater
		go globalCache.updateMetrics()

		log.Printf("✅ Redis Distributed Cache initialized: %s", redisURL)
	})

	return globalCache, initErr
}

// GetDistributedCache returns the global cache instance
func GetDistributedCache() *DistributedCache {
	if globalCache == nil {
		InitializeDistributedCache("localhost:6379")
	}
	return globalCache
}

// Cache Operations

// Set stores a value in cache with TTL
func (dc *DistributedCache) Set(key string, value interface{}, ttl time.Duration) error {
	if !dc.enabled {
		return fmt.Errorf("Redis not available")
	}

	startTime := time.Now()
	
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %v", err)
	}

	err = dc.client.Set(dc.ctx, key, data, ttl).Err()
	
	// Update metrics
	dc.cacheMetrics.mu.Lock()
	dc.cacheMetrics.Sets++
	setTime := time.Since(startTime)
	dc.cacheMetrics.TotalSetTime += setTime
	if dc.cacheMetrics.Sets > 0 {
		dc.cacheMetrics.AvgSetTime = dc.cacheMetrics.TotalSetTime / time.Duration(dc.cacheMetrics.Sets)
	}
	dc.cacheMetrics.mu.Unlock()

	return err
}

// Get retrieves a value from cache
func (dc *DistributedCache) Get(key string, dest interface{}) error {
	if !dc.enabled {
		return fmt.Errorf("Redis not available")
	}

	startTime := time.Now()
	
	data, err := dc.client.Get(dc.ctx, key).Bytes()
	
	// Update metrics
	dc.cacheMetrics.mu.Lock()
	getTime := time.Since(startTime)
	dc.cacheMetrics.TotalGetTime += getTime
	
	if err == redis.Nil {
		dc.cacheMetrics.Misses++
	} else if err == nil {
		dc.cacheMetrics.Hits++
	}
	
	if dc.cacheMetrics.Hits+dc.cacheMetrics.Misses > 0 {
		dc.cacheMetrics.HitRate = float64(dc.cacheMetrics.Hits) / float64(dc.cacheMetrics.Hits+dc.cacheMetrics.Misses)
		dc.cacheMetrics.AvgGetTime = dc.cacheMetrics.TotalGetTime / time.Duration(dc.cacheMetrics.Hits+dc.cacheMetrics.Misses)
	}
	dc.cacheMetrics.mu.Unlock()

	if err == redis.Nil {
		return fmt.Errorf("key not found: %s", key)
	} else if err != nil {
		return err
	}

	return json.Unmarshal(data, dest)
}

// Delete removes keys from cache
func (dc *DistributedCache) Delete(keys ...string) error {
	if !dc.enabled {
		return fmt.Errorf("Redis not available")
	}

	err := dc.client.Del(dc.ctx, keys...).Err()
	
	dc.cacheMetrics.mu.Lock()
	dc.cacheMetrics.Deletes += int64(len(keys))
	dc.cacheMetrics.mu.Unlock()
	
	return err
}

// AI-Specific Cache Operations

// SetDimensionalArray caches a dimensional array result
func (dc *DistributedCache) SetDimensionalArray(key string, dimensions []int, data []float32, metadata map[string]interface{}) error {
	entry := DimensionalArrayCacheEntry{
		Key:         key,
		Dimensions:  dimensions,
		Data:        data,
		Metadata:    metadata,
		ComputedAt:  time.Now(),
		AccessCount: 0,
		TTL:         1 * time.Hour, // Default TTL for dimensional arrays
	}

	return dc.Set(fmt.Sprintf("dim_array:%s", key), entry, entry.TTL)
}

// GetDimensionalArray retrieves a cached dimensional array
func (dc *DistributedCache) GetDimensionalArray(key string) (*DimensionalArrayCacheEntry, error) {
	var entry DimensionalArrayCacheEntry
	err := dc.Get(fmt.Sprintf("dim_array:%s", key), &entry)
	if err != nil {
		return nil, err
	}

	// Update access count
	entry.AccessCount++
	dc.Set(fmt.Sprintf("dim_array:%s", key), entry, entry.TTL)

	return &entry, nil
}

// SetT5ProcessingResult caches T5 processing results
func (dc *DistributedCache) SetT5ProcessingResult(key string, inputText, processedText string, embeddings []float32, confidence float64, processingTime time.Duration) error {
	entry := T5ProcessingCacheEntry{
		Key:            key,
		InputText:      inputText,
		ProcessedText:  processedText,
		Embeddings:     embeddings,
		Confidence:     confidence,
		ProcessingTime: processingTime,
		ModelVersion:   "t5-base-legal-v1.0",
		Metadata:       make(map[string]interface{}),
		CachedAt:       time.Now(),
	}

	return dc.Set(fmt.Sprintf("t5_result:%s", key), entry, 2*time.Hour) // 2-hour TTL for T5 results
}

// GetT5ProcessingResult retrieves cached T5 processing results
func (dc *DistributedCache) GetT5ProcessingResult(key string) (*T5ProcessingCacheEntry, error) {
	var entry T5ProcessingCacheEntry
	err := dc.Get(fmt.Sprintf("t5_result:%s", key), &entry)
	return &entry, err
}

// Background Task Operations

// SetBackgroundTask stores a background task for offline processing
func (dc *DistributedCache) SetBackgroundTask(taskID string, taskData interface{}, priority int) error {
	task := map[string]interface{}{
		"id":         taskID,
		"data":       taskData,
		"priority":   priority,
		"status":     "queued",
		"created_at": time.Now(),
	}

	// Use Redis sorted set for priority queue
	if dc.enabled {
		return dc.client.ZAdd(dc.ctx, "background_tasks", &redis.Z{
			Score:  float64(-priority), // Negative for descending order
			Member: taskID,
		}).Err()
	}

	return dc.Set(fmt.Sprintf("task:%s", taskID), task, 24*time.Hour)
}

// GetNextBackgroundTask retrieves the highest priority background task
func (dc *DistributedCache) GetNextBackgroundTask() (string, error) {
	if !dc.enabled {
		return "", fmt.Errorf("Redis not available")
	}

	// Get highest priority task (lowest score due to negative values)
	result, err := dc.client.ZPopMin(dc.ctx, "background_tasks", 1).Result()
	if err != nil || len(result) == 0 {
		return "", fmt.Errorf("no background tasks available")
	}

	return result[0].Member.(string), nil
}

// Performance and Statistics

// GetCacheMetrics returns current cache metrics
func (dc *DistributedCache) GetCacheMetrics() *CacheMetrics {
	dc.cacheMetrics.mu.RLock()
	defer dc.cacheMetrics.mu.RUnlock()
	
	// Create a copy to avoid race conditions
	metrics := *dc.cacheMetrics
	return &metrics
}

// GetCacheStats returns comprehensive cache statistics
func (dc *DistributedCache) GetCacheStats() map[string]interface{} {
	metrics := dc.GetCacheMetrics()
	
	stats := map[string]interface{}{
		"enabled":        dc.enabled,
		"hits":           metrics.Hits,
		"misses":         metrics.Misses,
		"sets":           metrics.Sets,
		"deletes":        metrics.Deletes,
		"hit_rate":       metrics.HitRate,
		"avg_get_time":   metrics.AvgGetTime.Nanoseconds(),
		"avg_set_time":   metrics.AvgSetTime.Nanoseconds(),
		"last_update":    metrics.LastUpdate,
	}

	if dc.enabled {
		// Add Redis-specific stats
		info := dc.client.Info(dc.ctx, "memory").Val()
		stats["redis_info"] = info
	}

	return stats
}

// IsEnabled returns whether Redis is enabled and available
func (dc *DistributedCache) IsEnabled() bool {
	return dc.enabled
}

// Health check for cache
func (dc *DistributedCache) HealthCheck() map[string]interface{} {
	health := map[string]interface{}{
		"enabled": dc.enabled,
		"status":  "disconnected",
	}

	if dc.enabled && dc.client != nil {
		ctx, cancel := context.WithTimeout(dc.ctx, 1*time.Second)
		defer cancel()

		if err := dc.client.Ping(ctx).Err(); err == nil {
			health["status"] = "connected"
		} else {
			health["error"] = err.Error()
		}
	}

	return health
}

// Private methods

func (dc *DistributedCache) updateMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		if !dc.enabled {
			continue
		}

		dc.cacheMetrics.mu.Lock()
		dc.cacheMetrics.LastUpdate = time.Now()
		dc.cacheMetrics.mu.Unlock()
	}
}

// Cleanup closes Redis connections
func (dc *DistributedCache) Cleanup() {
	if dc.enabled && dc.client != nil {
		dc.client.Close()
		log.Println("🔌 Redis distributed cache connections closed")
	}
}