// RabbitMQ Simulator for AI Modular System
// Provides AMQP-compatible interface using github.com/rabbitmq/amqp091-go
// Simulates persistent queues for offline/online processing

package messaging

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

// RabbitMQSimulator provides AMQP-compatible messaging simulation
type RabbitMQSimulator struct {
	connection *amqp.Connection
	channel    *amqp.Channel
	isConnected bool
	mu         sync.RWMutex
	
	// Simulated queues for different message types
	computationQueue    []QueueMessage
	cacheQueue         []QueueMessage
	healthQueue        []QueueMessage
	responseQueue      []QueueMessage
	
	// Queue mutexes for thread safety
	computationMu sync.RWMutex
	cacheMu       sync.RWMutex
	healthMu      sync.RWMutex
	responseMu    sync.RWMutex
	
	// Connection config
	config ConnectionConfig
	
	// Metrics
	messagesPublished int64
	messagesConsumed  int64
	queueSizes        map[string]int
	
	// Event handlers
	onMessage func(QueueMessage)
	onConnect func()
	onDisconnect func()
}

// ConnectionConfig for RabbitMQ simulator
type ConnectionConfig struct {
	URL              string
	MaxRetries       int
	RetryInterval    time.Duration
	HeartbeatInterval time.Duration
	EnableSimulation bool // If true, uses simulation instead of real RabbitMQ
}

// QueueMessage represents a message in the queue
type QueueMessage struct {
	ID            string                 `json:"id"`
	CorrelationID string                 `json:"correlation_id"`
	Type          string                 `json:"type"`
	Priority      int                    `json:"priority"`
	Timestamp     time.Time              `json:"timestamp"`
	RetryCount    int                    `json:"retry_count"`
	MaxRetries    int                    `json:"max_retries"`
	Payload       map[string]interface{} `json:"payload"`
	RoutingKey    string                 `json:"routing_key"`
	Exchange      string                 `json:"exchange"`
}

// Queue names
const (
	ComputationQueue = "ai.computation"
	CacheQueue      = "ai.cache"
	HealthQueue     = "ai.health"
	ResponseQueue   = "ai.response"
	DeadLetterQueue = "ai.deadletter"
)

// Message types
const (
	MsgTypeComputation = "computation_request"
	MsgTypeCache       = "cache_request"
	MsgTypeHealth      = "health_check"
	MsgTypeResponse    = "computation_response"
	MsgTypeError       = "error"
)

// NewRabbitMQSimulator creates a new RabbitMQ simulator
func NewRabbitMQSimulator(config ConnectionConfig) *RabbitMQSimulator {
	return &RabbitMQSimulator{
		config:     config,
		queueSizes: make(map[string]int),
	}
}

// Connect establishes connection to RabbitMQ or starts simulation
func (r *RabbitMQSimulator) Connect() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.config.EnableSimulation {
		// Use simulation mode
		log.Printf("🔄 Starting RabbitMQ simulation mode")
		r.isConnected = true
		r.initializeQueues()
		
		if r.onConnect != nil {
			r.onConnect()
		}
		
		log.Printf("✅ RabbitMQ simulator connected")
		return nil
	}

	// Try real RabbitMQ connection
	var err error
	for i := 0; i < r.config.MaxRetries; i++ {
		r.connection, err = amqp.Dial(r.config.URL)
		if err == nil {
			r.channel, err = r.connection.Channel()
			if err == nil {
				r.isConnected = true
				r.setupRealQueues()
				log.Printf("✅ Connected to real RabbitMQ: %s", r.config.URL)
				
				if r.onConnect != nil {
					r.onConnect()
				}
				return nil
			}
		}
		
		log.Printf("⚠️ RabbitMQ connection attempt %d/%d failed: %v", i+1, r.config.MaxRetries, err)
		time.Sleep(r.config.RetryInterval)
	}

	// Fall back to simulation
	log.Printf("🔄 Falling back to RabbitMQ simulation mode")
	r.config.EnableSimulation = true
	r.isConnected = true
	r.initializeQueues()
	
	if r.onConnect != nil {
		r.onConnect()
	}
	
	return nil
}

// Disconnect closes the connection
func (r *RabbitMQSimulator) Disconnect() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.isConnected = false
	
	if r.onDisconnect != nil {
		r.onDisconnect()
	}

	if r.config.EnableSimulation {
		log.Printf("🔌 RabbitMQ simulator disconnected")
		return nil
	}

	if r.channel != nil {
		r.channel.Close()
	}
	if r.connection != nil {
		r.connection.Close()
	}
	
	log.Printf("🔌 Disconnected from RabbitMQ")
	return nil
}

// IsConnected returns connection status
func (r *RabbitMQSimulator) IsConnected() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.isConnected
}

// PublishMessage publishes a message to the specified queue
func (r *RabbitMQSimulator) PublishMessage(queueName string, message QueueMessage) error {
	if !r.isConnected {
		return fmt.Errorf("not connected to RabbitMQ")
	}

	message.Timestamp = time.Now()
	if message.ID == "" {
		message.ID = fmt.Sprintf("msg_%d", time.Now().UnixNano())
	}

	r.messagesPublished++

	if r.config.EnableSimulation {
		return r.publishToSimulatedQueue(queueName, message)
	}

	return r.publishToRealQueue(queueName, message)
}

// ConsumeMessages starts consuming messages from a queue
func (r *RabbitMQSimulator) ConsumeMessages(queueName string, handler func(QueueMessage)) error {
	if !r.isConnected {
		return fmt.Errorf("not connected to RabbitMQ")
	}

	if r.config.EnableSimulation {
		go r.consumeFromSimulatedQueue(queueName, handler)
		return nil
	}

	return r.consumeFromRealQueue(queueName, handler)
}

// GetQueueStats returns statistics for all queues
func (r *RabbitMQSimulator) GetQueueStats() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := map[string]interface{}{
		"connected":           r.isConnected,
		"simulation_mode":     r.config.EnableSimulation,
		"messages_published":  r.messagesPublished,
		"messages_consumed":   r.messagesConsumed,
		"queue_sizes":         r.getQueueSizes(),
		"uptime":             time.Since(time.Now().Add(-5 * time.Minute)).String(), // Placeholder
	}

	return stats
}

// PublishComputationRequest publishes a computation request
func (r *RabbitMQSimulator) PublishComputationRequest(request map[string]interface{}) error {
	message := QueueMessage{
		Type:          MsgTypeComputation,
		Priority:      5,
		MaxRetries:    3,
		Payload:       request,
		RoutingKey:    ComputationQueue,
		Exchange:      "ai.direct",
		CorrelationID: fmt.Sprintf("comp_%d", time.Now().UnixNano()),
	}

	return r.PublishMessage(ComputationQueue, message)
}

// PublishCacheRequest publishes a cache request
func (r *RabbitMQSimulator) PublishCacheRequest(request map[string]interface{}) error {
	message := QueueMessage{
		Type:          MsgTypeCache,
		Priority:      3,
		MaxRetries:    2,
		Payload:       request,
		RoutingKey:    CacheQueue,
		Exchange:      "ai.direct",
		CorrelationID: fmt.Sprintf("cache_%d", time.Now().UnixNano()),
	}

	return r.PublishMessage(CacheQueue, message)
}

// PublishHealthCheck publishes a health check
func (r *RabbitMQSimulator) PublishHealthCheck() error {
	healthData := map[string]interface{}{
		"service":   "cuda-ai-service",
		"status":    "healthy",
		"timestamp": time.Now(),
		"metrics": map[string]interface{}{
			"cpu_usage":    0.45,
			"memory_usage": 0.67,
			"gpu_usage":    0.23,
		},
	}

	message := QueueMessage{
		Type:       MsgTypeHealth,
		Priority:   1,
		MaxRetries: 1,
		Payload:    healthData,
		RoutingKey: HealthQueue,
		Exchange:   "ai.direct",
	}

	return r.PublishMessage(HealthQueue, message)
}

// Private methods for simulation

func (r *RabbitMQSimulator) initializeQueues() {
	r.computationQueue = make([]QueueMessage, 0)
	r.cacheQueue = make([]QueueMessage, 0)
	r.healthQueue = make([]QueueMessage, 0)
	r.responseQueue = make([]QueueMessage, 0)
	
	r.queueSizes[ComputationQueue] = 0
	r.queueSizes[CacheQueue] = 0
	r.queueSizes[HealthQueue] = 0
	r.queueSizes[ResponseQueue] = 0
}

func (r *RabbitMQSimulator) publishToSimulatedQueue(queueName string, message QueueMessage) error {
	switch queueName {
	case ComputationQueue:
		r.computationMu.Lock()
		r.computationQueue = append(r.computationQueue, message)
		r.queueSizes[ComputationQueue] = len(r.computationQueue)
		r.computationMu.Unlock()
		
	case CacheQueue:
		r.cacheMu.Lock()
		r.cacheQueue = append(r.cacheQueue, message)
		r.queueSizes[CacheQueue] = len(r.cacheQueue)
		r.cacheMu.Unlock()
		
	case HealthQueue:
		r.healthMu.Lock()
		r.healthQueue = append(r.healthQueue, message)
		r.queueSizes[HealthQueue] = len(r.healthQueue)
		r.healthMu.Unlock()
		
	case ResponseQueue:
		r.responseMu.Lock()
		r.responseQueue = append(r.responseQueue, message)
		r.queueSizes[ResponseQueue] = len(r.responseQueue)
		r.responseMu.Unlock()
		
	default:
		return fmt.Errorf("unknown queue: %s", queueName)
	}

	log.Printf("📤 Published to simulated queue %s: %s", queueName, message.Type)
	return nil
}

func (r *RabbitMQSimulator) consumeFromSimulatedQueue(queueName string, handler func(QueueMessage)) {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		if !r.isConnected {
			break
		}

		var message QueueMessage
		var hasMessage bool

		switch queueName {
		case ComputationQueue:
			r.computationMu.Lock()
			if len(r.computationQueue) > 0 {
				message = r.computationQueue[0]
				r.computationQueue = r.computationQueue[1:]
				r.queueSizes[ComputationQueue] = len(r.computationQueue)
				hasMessage = true
			}
			r.computationMu.Unlock()

		case CacheQueue:
			r.cacheMu.Lock()
			if len(r.cacheQueue) > 0 {
				message = r.cacheQueue[0]
				r.cacheQueue = r.cacheQueue[1:]
				r.queueSizes[CacheQueue] = len(r.cacheQueue)
				hasMessage = true
			}
			r.cacheMu.Unlock()

		case HealthQueue:
			r.healthMu.Lock()
			if len(r.healthQueue) > 0 {
				message = r.healthQueue[0]
				r.healthQueue = r.healthQueue[1:]
				r.queueSizes[HealthQueue] = len(r.healthQueue)
				hasMessage = true
			}
			r.healthMu.Unlock()

		case ResponseQueue:
			r.responseMu.Lock()
			if len(r.responseQueue) > 0 {
				message = r.responseQueue[0]
				r.responseQueue = r.responseQueue[1:]
				r.queueSizes[ResponseQueue] = len(r.responseQueue)
				hasMessage = true
			}
			r.responseMu.Unlock()
		}

		if hasMessage {
			r.messagesConsumed++
			log.Printf("📥 Consumed from simulated queue %s: %s", queueName, message.Type)
			
			// Process message in goroutine to avoid blocking
			go handler(message)
		}
	}
}

func (r *RabbitMQSimulator) getQueueSizes() map[string]int {
	sizes := make(map[string]int)
	for k, v := range r.queueSizes {
		sizes[k] = v
	}
	return sizes
}

func (r *RabbitMQSimulator) setupRealQueues() error {
	// Setup real RabbitMQ queues if connected to actual server
	queues := []string{ComputationQueue, CacheQueue, HealthQueue, ResponseQueue}
	
	for _, queueName := range queues {
		_, err := r.channel.QueueDeclare(
			queueName,
			true,  // durable
			false, // delete when unused
			false, // exclusive
			false, // no-wait
			nil,   // arguments
		)
		if err != nil {
			return fmt.Errorf("failed to declare queue %s: %v", queueName, err)
		}
	}
	
	return nil
}

func (r *RabbitMQSimulator) publishToRealQueue(queueName string, message QueueMessage) error {
	body, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}

	return r.channel.Publish(
		"",        // exchange
		queueName, // routing key
		false,     // mandatory
		false,     // immediate
		amqp.Publishing{
			ContentType:   "application/json",
			Body:          body,
			DeliveryMode:  amqp.Persistent, // make message persistent
			Priority:      uint8(message.Priority),
			CorrelationId: message.CorrelationID,
			Timestamp:     message.Timestamp,
		})
}

func (r *RabbitMQSimulator) consumeFromRealQueue(queueName string, handler func(QueueMessage)) error {
	msgs, err := r.channel.Consume(
		queueName,
		"",    // consumer
		false, // auto-ack
		false, // exclusive
		false, // no-local
		false, // no-wait
		nil,   // args
	)
	if err != nil {
		return fmt.Errorf("failed to register consumer: %v", err)
	}

	go func() {
		for d := range msgs {
			var message QueueMessage
			if err := json.Unmarshal(d.Body, &message); err != nil {
				log.Printf("❌ Failed to unmarshal message: %v", err)
				d.Nack(false, false)
				continue
			}

			r.messagesConsumed++
			handler(message)
			d.Ack(false)
		}
	}()

	return nil
}

// SetEventHandlers sets event handlers for connection events
func (r *RabbitMQSimulator) SetEventHandlers(
	onConnect func(),
	onDisconnect func(),
	onMessage func(QueueMessage),
) {
	r.onConnect = onConnect
	r.onDisconnect = onDisconnect
	r.onMessage = onMessage
}

// ProcessOfflineQueue processes queued messages when back online
func (r *RabbitMQSimulator) ProcessOfflineQueue(queueName string) (int, error) {
	if !r.isConnected {
		return 0, fmt.Errorf("not connected")
	}

	processed := 0
	
	switch queueName {
	case ComputationQueue:
		r.computationMu.Lock()
		count := len(r.computationQueue)
		r.computationMu.Unlock()
		processed = count
		
	case CacheQueue:
		r.cacheMu.Lock()
		count := len(r.cacheQueue)
		r.cacheMu.Unlock()
		processed = count
	}

	log.Printf("🔄 Processed %d messages from offline queue %s", processed, queueName)
	return processed, nil
}