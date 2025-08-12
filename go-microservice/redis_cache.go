//go:build legacy
// +build legacy

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
)

// RedisManager handles all Redis operations including caching and messaging
type RedisManager struct {
	client        *redis.Client
	pubsub        *redis.PubSub
	ctx           context.Context
	mu            sync.RWMutex
	cacheMetrics  *CacheMetrics
	messageQueues map[string]*MessageQueue
	subscribers   map[string][]MessageHandler
}

// CacheMetrics tracks cache performance
type CacheMetrics struct {
	mu              sync.RWMutex
	Hits            int64         `json:"hits"`
	Misses          int64         `json:"misses"`
	Sets            int64         `json:"sets"`
	Deletes         int64         `json:"deletes"`
	AvgGetTime      time.Duration `json:"avg_get_time"`
	AvgSetTime      time.Duration `json:"avg_set_time"`
	TotalGetTime    time.Duration `json:"total_get_time"`
	TotalSetTime    time.Duration `json:"total_set_time"`
	HitRate         float64       `json:"hit_rate"`
	LastUpdate      time.Time     `json:"last_update"`
}

// MessageQueue represents a Redis-backed message queue (similar to BullMQ)
type MessageQueue struct {
	Name        string              `json:"name"`
	Priority    int                 `json:"priority"`
	MaxRetries  int                 `json:"max_retries"`
	RetryDelay  time.Duration       `json:"retry_delay"`
	Processing  map[string]*Job     `json:"-"`
	mu          sync.RWMutex
}

// Job represents a queued job
type Job struct {
	ID          string                 `json:"id"`
	Queue       string                 `json:"queue"`
	Type        string                 `json:"type"`
	Payload     json.RawMessage        `json:"payload"`
	Priority    int                    `json:"priority"`
	Retries     int                    `json:"retries"`
	MaxRetries  int                    `json:"max_retries"`
	Status      string                 `json:"status"` // "pending", "processing", "completed", "failed"
	CreatedAt   time.Time              `json:"created_at"`
	ProcessedAt *time.Time             `json:"processed_at,omitempty"`
	CompletedAt *time.Time             `json:"completed_at,omitempty"`
	Error       string                 `json:"error,omitempty"`
	Result      json.RawMessage        `json:"result,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// BatchInferenceJob represents a batch inference task
type BatchInferenceJob struct {
	JobID       string          `json:"job_id"`
	Documents   []string        `json:"documents"`
	Model       string          `json:"model"`
	BatchSize   int             `json:"batch_size"`
	Priority    int             `json:"priority"`
	CallbackURL string          `json:"callback_url,omitempty"`
}

// CacheEntry represents a cached item with metadata
type CacheEntry struct {
	Key        string          `json:"key"`
	Value      json.RawMessage `json:"value"`
	TTL        time.Duration   `json:"ttl"`
	Tags       []string        `json:"tags"`
	CreatedAt  time.Time       `json:"created_at"`
	AccessedAt time.Time       `json:"accessed_at"`
	HitCount   int64           `json:"hit_count"`
}

// MessageHandler is a function type for handling messages
type MessageHandler func(channel string, payload []byte) error

var (
	redisManager *RedisManager
	redisOnce    sync.Once
)

// InitializeRedisManager initializes the global Redis manager
func InitializeRedisManager(redisURL string) error {
	var initErr error
	redisOnce.Do(func() {
		if redisURL == "" {
			redisURL = "localhost:6379"
		}

		opt, err := redis.ParseURL(fmt.Sprintf("redis://%s", redisURL))
		if err != nil {
			opt = &redis.Options{
				Addr:         redisURL,
				Password:     "", // no password set
				DB:           0,  // use default DB
				PoolSize:     10,
				MinIdleConns: 5,
				MaxRetries:   3,
			}
		}

		client := redis.NewClient(opt)
		ctx := context.Background()

		// Test connection
		if err := client.Ping(ctx).Err(); err != nil {
			initErr = fmt.Errorf("failed to connect to Redis: %v", err)
			return
		}

		redisManager = &RedisManager{
			client:        client,
			ctx:           ctx,
			cacheMetrics:  &CacheMetrics{LastUpdate: time.Now()},
			messageQueues: make(map[string]*MessageQueue),
			subscribers:   make(map[string][]MessageHandler),
		}

		// Initialize default queues
		redisManager.CreateQueue("inference", 1, 3, 5*time.Second)
		redisManager.CreateQueue("embeddings", 2, 3, 5*time.Second)
		redisManager.CreateQueue("processing", 3, 3, 5*time.Second)

		// Start metrics updater
		go redisManager.updateMetrics()

		log.Printf("✅ Redis Manager initialized: %s", redisURL)
	})

	return initErr
}

// GetRedisManager returns the global Redis manager instance
func GetRedisManager() *RedisManager {
	return redisManager
}

// Cache Operations

// Set stores a value in cache with optional TTL
func (rm *RedisManager) Set(key string, value interface{}, ttl time.Duration) error {
	startTime := time.Now()
	
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %v", err)
	}

	err = rm.client.Set(rm.ctx, key, data, ttl).Err()
	
	// Update metrics
	rm.cacheMetrics.mu.Lock()
	rm.cacheMetrics.Sets++
	setTime := time.Since(startTime)
	rm.cacheMetrics.TotalSetTime += setTime
	rm.cacheMetrics.AvgSetTime = rm.cacheMetrics.TotalSetTime / time.Duration(rm.cacheMetrics.Sets)
	rm.cacheMetrics.mu.Unlock()

	return err
}

// Get retrieves a value from cache
func (rm *RedisManager) Get(key string, dest interface{}) error {
	startTime := time.Now()
	
	data, err := rm.client.Get(rm.ctx, key).Bytes()
	
	// Update metrics
	rm.cacheMetrics.mu.Lock()
	getTime := time.Since(startTime)
	rm.cacheMetrics.TotalGetTime += getTime
	
	if err == redis.Nil {
		rm.cacheMetrics.Misses++
	} else if err == nil {
		rm.cacheMetrics.Hits++
	}
	
	if rm.cacheMetrics.Hits+rm.cacheMetrics.Misses > 0 {
		rm.cacheMetrics.HitRate = float64(rm.cacheMetrics.Hits) / float64(rm.cacheMetrics.Hits+rm.cacheMetrics.Misses)
	}
	rm.cacheMetrics.AvgGetTime = rm.cacheMetrics.TotalGetTime / time.Duration(rm.cacheMetrics.Hits+rm.cacheMetrics.Misses)
	rm.cacheMetrics.mu.Unlock()

	if err == redis.Nil {
		return fmt.Errorf("key not found: %s", key)
	} else if err != nil {
		return err
	}

	return json.Unmarshal(data, dest)
}

// Delete removes a key from cache
func (rm *RedisManager) Delete(keys ...string) error {
	err := rm.client.Del(rm.ctx, keys...).Err()
	
	rm.cacheMetrics.mu.Lock()
	rm.cacheMetrics.Deletes += int64(len(keys))
	rm.cacheMetrics.mu.Unlock()
	
	return err
}

// SetWithTags stores a value with tags for grouped invalidation
func (rm *RedisManager) SetWithTags(key string, value interface{}, ttl time.Duration, tags []string) error {
	// Store the main value
	if err := rm.Set(key, value, ttl); err != nil {
		return err
	}

	// Store tags in sets for grouped operations
	for _, tag := range tags {
		tagKey := fmt.Sprintf("tag:%s", tag)
		if err := rm.client.SAdd(rm.ctx, tagKey, key).Err(); err != nil {
			return err
		}
		// Set expiration on tag set
		rm.client.Expire(rm.ctx, tagKey, ttl)
	}

	return nil
}

// InvalidateByTag removes all cache entries with a specific tag
func (rm *RedisManager) InvalidateByTag(tag string) error {
	tagKey := fmt.Sprintf("tag:%s", tag)
	
	// Get all keys with this tag
	keys, err := rm.client.SMembers(rm.ctx, tagKey).Result()
	if err != nil {
		return err
	}

	if len(keys) > 0 {
		// Delete all tagged keys
		if err := rm.Delete(keys...); err != nil {
			return err
		}
	}

	// Delete the tag set itself
	return rm.client.Del(rm.ctx, tagKey).Err()
}

// Message Queue Operations

// CreateQueue creates a new message queue
func (rm *RedisManager) CreateQueue(name string, priority int, maxRetries int, retryDelay time.Duration) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	rm.messageQueues[name] = &MessageQueue{
		Name:       name,
		Priority:   priority,
		MaxRetries: maxRetries,
		RetryDelay: retryDelay,
		Processing: make(map[string]*Job),
	}
}

// EnqueueJob adds a job to a queue
func (rm *RedisManager) EnqueueJob(queueName string, jobType string, payload interface{}) (*Job, error) {
	queue, exists := rm.messageQueues[queueName]
	if !exists {
		return nil, fmt.Errorf("queue not found: %s", queueName)
	}

	payloadData, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	job := &Job{
		ID:         fmt.Sprintf("%s:%d", queueName, time.Now().UnixNano()),
		Queue:      queueName,
		Type:       jobType,
		Payload:    payloadData,
		Priority:   queue.Priority,
		MaxRetries: queue.MaxRetries,
		Status:     "pending",
		CreatedAt:  time.Now(),
	}

	// Store job in Redis
	jobKey := fmt.Sprintf("job:%s", job.ID)
	jobData, _ := json.Marshal(job)
	
	// Add to queue with priority
	score := float64(queue.Priority*1000000 - int(time.Now().Unix()))
	if err := rm.client.ZAdd(rm.ctx, fmt.Sprintf("queue:%s", queueName), &redis.Z{
		Score:  score,
		Member: jobKey,
	}).Err(); err != nil {
		return nil, err
	}

	// Store job data
	if err := rm.client.Set(rm.ctx, jobKey, jobData, 24*time.Hour).Err(); err != nil {
		return nil, err
	}

	// Publish job event
	rm.PublishJobEvent(queueName, "job.created", job)

	return job, nil
}

// DequeueJob gets the next job from a queue
func (rm *RedisManager) DequeueJob(queueName string) (*Job, error) {
	queueKey := fmt.Sprintf("queue:%s", queueName)
	
	// Get highest priority job
	result, err := rm.client.ZPopMax(rm.ctx, queueKey, 1).Result()
	if err != nil || len(result) == 0 {
		return nil, fmt.Errorf("no jobs in queue: %s", queueName)
	}

	jobKey := result[0].Member.(string)
	
	// Get job data
	jobData, err := rm.client.Get(rm.ctx, jobKey).Bytes()
	if err != nil {
		return nil, err
	}

	var job Job
	if err := json.Unmarshal(jobData, &job); err != nil {
		return nil, err
	}

	// Update job status
	now := time.Now()
	job.Status = "processing"
	job.ProcessedAt = &now

	// Save updated job
	updatedData, _ := json.Marshal(job)
	rm.client.Set(rm.ctx, jobKey, updatedData, 24*time.Hour)

	// Add to processing set
	queue := rm.messageQueues[queueName]
	if queue != nil {
		queue.mu.Lock()
		queue.Processing[job.ID] = &job
		queue.mu.Unlock()
	}

	// Publish job event
	rm.PublishJobEvent(queueName, "job.processing", &job)

	return &job, nil
}

// CompleteJob marks a job as completed
func (rm *RedisManager) CompleteJob(job *Job, result interface{}) error {
	resultData, err := json.Marshal(result)
	if err != nil {
		return err
	}

	now := time.Now()
	job.Status = "completed"
	job.CompletedAt = &now
	job.Result = resultData

	// Save completed job
	jobKey := fmt.Sprintf("job:%s", job.ID)
	jobData, _ := json.Marshal(job)
	rm.client.Set(rm.ctx, jobKey, jobData, 7*24*time.Hour) // Keep completed jobs for 7 days

	// Remove from processing
	if queue := rm.messageQueues[job.Queue]; queue != nil {
		queue.mu.Lock()
		delete(queue.Processing, job.ID)
		queue.mu.Unlock()
	}

	// Publish completion event
	rm.PublishJobEvent(job.Queue, "job.completed", job)

	return nil
}

// FailJob marks a job as failed and potentially retries it
func (rm *RedisManager) FailJob(job *Job, err error) error {
	job.Error = err.Error()
	job.Retries++

	if job.Retries < job.MaxRetries {
		// Retry the job
		job.Status = "pending"
		
		// Re-enqueue with delay
		queue := rm.messageQueues[job.Queue]
		if queue != nil {
			time.AfterFunc(queue.RetryDelay, func() {
				rm.EnqueueJob(job.Queue, job.Type, job.Payload)
			})
		}
		
		rm.PublishJobEvent(job.Queue, "job.retrying", job)
	} else {
		// Mark as permanently failed
		job.Status = "failed"
		now := time.Now()
		job.CompletedAt = &now
		
		// Save failed job
		jobKey := fmt.Sprintf("job:%s", job.ID)
		jobData, _ := json.Marshal(job)
		rm.client.Set(rm.ctx, jobKey, jobData, 30*24*time.Hour) // Keep failed jobs for 30 days
		
		rm.PublishJobEvent(job.Queue, "job.failed", job)
	}

	// Remove from processing
	if queue := rm.messageQueues[job.Queue]; queue != nil {
		queue.mu.Lock()
		delete(queue.Processing, job.ID)
		queue.mu.Unlock()
	}

	return nil
}

// Pub/Sub Operations

// Subscribe subscribes to a channel with a handler
func (rm *RedisManager) Subscribe(channel string, handler MessageHandler) error {
	rm.mu.Lock()
	rm.subscribers[channel] = append(rm.subscribers[channel], handler)
	rm.mu.Unlock()

	// Create subscription if not exists
	if rm.pubsub == nil {
		rm.pubsub = rm.client.Subscribe(rm.ctx, channel)
	} else {
		rm.pubsub.Subscribe(rm.ctx, channel)
	}

	// Start listening in a goroutine
	go func() {
		ch := rm.pubsub.Channel()
		for msg := range ch {
			rm.mu.RLock()
			handlers := rm.subscribers[msg.Channel]
			rm.mu.RUnlock()

			for _, h := range handlers {
				go h(msg.Channel, []byte(msg.Payload))
			}
		}
	}()

	return nil
}

// Publish publishes a message to a channel
func (rm *RedisManager) Publish(channel string, message interface{}) error {
	data, err := json.Marshal(message)
	if err != nil {
		return err
	}

	return rm.client.Publish(rm.ctx, channel, data).Err()
}

// PublishJobEvent publishes a job-related event
func (rm *RedisManager) PublishJobEvent(queue string, eventType string, job *Job) {
	event := map[string]interface{}{
		"type":      eventType,
		"queue":     queue,
		"job":       job,
		"timestamp": time.Now(),
	}

	rm.Publish(fmt.Sprintf("jobs:%s", queue), event)
}

// Batch Inference Operations

// ScheduleBatchInference schedules a batch inference job
func (rm *RedisManager) ScheduleBatchInference(documents []string, model string, batchSize int) (*BatchInferenceJob, error) {
	job := &BatchInferenceJob{
		JobID:     fmt.Sprintf("batch_%d", time.Now().UnixNano()),
		Documents: documents,
		Model:     model,
		BatchSize: batchSize,
		Priority:  1,
	}

	// Create job in queue
	queueJob, err := rm.EnqueueJob("inference", "batch_inference", job)
	if err != nil {
		return nil, err
	}

	job.JobID = queueJob.ID

	// Cache job details
	rm.Set(fmt.Sprintf("batch:%s", job.JobID), job, 24*time.Hour)

	return job, nil
}

// GetBatchInferenceStatus gets the status of a batch inference job
func (rm *RedisManager) GetBatchInferenceStatus(jobID string) (*Job, error) {
	var job Job
	err := rm.Get(fmt.Sprintf("job:%s", jobID), &job)
	return &job, err
}

// Stream Operations for Real-time Updates

// StreamAdd adds an event to a stream
func (rm *RedisManager) StreamAdd(stream string, data map[string]interface{}) (string, error) {
	values := make(map[string]interface{})
	for k, v := range data {
		values[k] = v
	}

	result := rm.client.XAdd(rm.ctx, &redis.XAddArgs{
		Stream: stream,
		Values: values,
	})

	return result.Result()
}

// StreamRead reads from a stream
func (rm *RedisManager) StreamRead(stream string, lastID string) ([]redis.XMessage, error) {
	if lastID == "" {
		lastID = "$"
	}

	result, err := rm.client.XRead(rm.ctx, &redis.XReadArgs{
		Streams: []string{stream, lastID},
		Count:   10,
		Block:   1 * time.Second,
	}).Result()

	if err != nil {
		return nil, err
	}

	if len(result) > 0 && len(result[0].Messages) > 0 {
		return result[0].Messages, nil
	}

	return nil, nil
}

// Performance Monitoring

// GetCacheMetrics returns current cache metrics
func (rm *RedisManager) GetCacheMetrics() *CacheMetrics {
	rm.cacheMetrics.mu.RLock()
	defer rm.cacheMetrics.mu.RUnlock()
	return rm.cacheMetrics
}

// GetQueueStats returns statistics for all queues
func (rm *RedisManager) GetQueueStats() map[string]interface{} {
	stats := make(map[string]interface{})
	
	for name, queue := range rm.messageQueues {
		queueKey := fmt.Sprintf("queue:%s", name)
		pending, _ := rm.client.ZCard(rm.ctx, queueKey).Result()
		
		queue.mu.RLock()
		processing := len(queue.Processing)
		queue.mu.RUnlock()
		
		stats[name] = map[string]interface{}{
			"pending":    pending,
			"processing": processing,
			"priority":   queue.Priority,
			"maxRetries": queue.MaxRetries,
		}
	}
	
	return stats
}

// updateMetrics updates cache metrics periodically
func (rm *RedisManager) updateMetrics() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		rm.cacheMetrics.mu.Lock()
		rm.cacheMetrics.LastUpdate = time.Now()
		rm.cacheMetrics.mu.Unlock()
		
		// Publish metrics
		rm.Publish("metrics:cache", rm.cacheMetrics)
	}
}

// Cleanup closes Redis connections
func (rm *RedisManager) Cleanup() {
	if rm.pubsub != nil {
		rm.pubsub.Close()
	}
	if rm.client != nil {
		rm.client.Close()
	}
	log.Println("Redis connections closed")
}
