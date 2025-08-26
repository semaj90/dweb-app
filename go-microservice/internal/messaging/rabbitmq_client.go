// Real RabbitMQ Client for AI Modular System
// Uses github.com/rabbitmq/amqp091-go for production-ready messaging

package messaging

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	amqp "github.com/rabbitmq/amqp091-go"
)

// RabbitMQClient handles real RabbitMQ connections and messaging
type RabbitMQClient struct {
	connection *amqp.Connection
	channel    *amqp.Channel
	config     Config
	isConnected bool
	mu         sync.RWMutex
	
	// Metrics
	messagesPublished int64
	messagesConsumed  int64
	
	// Event handlers
	onConnect    func()
	onDisconnect func()
	onError      func(error)
}

// Config for RabbitMQ connection
type Config struct {
	URL           string
	Exchange      string
	MaxRetries    int
	RetryInterval time.Duration
	Queues        map[string]QueueConfig
}

// QueueConfig defines queue properties
type QueueConfig struct {
	Name       string
	Durable    bool
	AutoDelete bool
	Exclusive  bool
	NoWait     bool
	RoutingKey string
	Arguments  amqp.Table
}

// Message represents a RabbitMQ message
type Message struct {
	ID            string                 `json:"id"`
	CorrelationID string                 `json:"correlation_id"`
	Type          string                 `json:"type"`
	Priority      uint8                  `json:"priority"`
	Timestamp     time.Time              `json:"timestamp"`
	RetryCount    int                    `json:"retry_count"`
	MaxRetries    int                    `json:"max_retries"`
	Payload       map[string]interface{} `json:"payload"`
	RoutingKey    string                 `json:"routing_key"`
}

// Queue names for AI Modular System
const (
	QueueComputationRequests = "ai.computation.requests"
	QueueComputationResults  = "ai.computation.results"
	QueueCacheOperations     = "ai.cache.operations"
	QueueHealthChecks        = "ai.health.checks"
	QueueBackgroundTasks     = "ai.background.tasks"
	QueueOfflineProcessing   = "ai.offline.processing"
)

// Message types
const (
	TypeDimensionalArray = "dimensional_array"
	TypeT5Processing     = "t5_processing"
	TypeCacheRequest     = "cache_request"
	TypeHealthCheck      = "health_check"
	TypeBackgroundTask   = "background_task"
	TypeOfflineTask      = "offline_task"
)

// NewRabbitMQClient creates a new RabbitMQ client
func NewRabbitMQClient(config Config) *RabbitMQClient {
	return &RabbitMQClient{
		config: config,
	}
}

// Connect establishes connection to RabbitMQ server
func (r *RabbitMQClient) Connect() error {
	var err error
	
	for attempt := 1; attempt <= r.config.MaxRetries; attempt++ {
		log.Printf("🔄 Connecting to RabbitMQ (attempt %d/%d): %s", attempt, r.config.MaxRetries, r.config.URL)
		
		r.connection, err = amqp.Dial(r.config.URL)
		if err != nil {
			log.Printf("❌ Connection attempt %d failed: %v", attempt, err)
			if attempt < r.config.MaxRetries {
				time.Sleep(r.config.RetryInterval)
				continue
			}
			return fmt.Errorf("failed to connect to RabbitMQ after %d attempts: %v", r.config.MaxRetries, err)
		}
		
		r.channel, err = r.connection.Channel()
		if err != nil {
			log.Printf("❌ Failed to open channel: %v", err)
			r.connection.Close()
			if attempt < r.config.MaxRetries {
				time.Sleep(r.config.RetryInterval)
				continue
			}
			return fmt.Errorf("failed to open channel: %v", err)
		}
		
		break
	}
	
	r.mu.Lock()
	r.isConnected = true
	r.mu.Unlock()
	
	// Declare exchange
	if r.config.Exchange != "" {
		err = r.channel.ExchangeDeclare(
			r.config.Exchange,
			"direct", // type
			true,     // durable
			false,    // auto-deleted
			false,    // internal
			false,    // no-wait
			nil,      // arguments
		)
		if err != nil {
			return fmt.Errorf("failed to declare exchange: %v", err)
		}
	}
	
	// Setup queues
	err = r.setupQueues()
	if err != nil {
		return fmt.Errorf("failed to setup queues: %v", err)
	}
	
	// Setup connection close handler
	go r.handleConnectionClose()
	
	if r.onConnect != nil {
		r.onConnect()
	}
	
	log.Printf("✅ Connected to RabbitMQ: %s", r.config.URL)
	return nil
}

// Disconnect closes the RabbitMQ connection
func (r *RabbitMQClient) Disconnect() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	r.isConnected = false
	
	if r.channel != nil {
		r.channel.Close()
	}
	if r.connection != nil {
		r.connection.Close()
	}
	
	if r.onDisconnect != nil {
		r.onDisconnect()
	}
	
	log.Printf("🔌 Disconnected from RabbitMQ")
	return nil
}

// IsConnected returns the connection status
func (r *RabbitMQClient) IsConnected() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.isConnected
}

// PublishMessage publishes a message to RabbitMQ
func (r *RabbitMQClient) PublishMessage(queueName string, message Message) error {
	if !r.IsConnected() {
		return fmt.Errorf("not connected to RabbitMQ")
	}
	
	if message.Timestamp.IsZero() {
		message.Timestamp = time.Now()
	}
	if message.ID == "" {
		message.ID = fmt.Sprintf("msg_%d", time.Now().UnixNano())
	}
	
	body, err := json.Marshal(message)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}
	
	err = r.channel.Publish(
		r.config.Exchange, // exchange
		queueName,         // routing key
		false,            // mandatory
		false,            // immediate
		amqp.Publishing{
			ContentType:   "application/json",
			Body:          body,
			DeliveryMode:  amqp.Persistent, // persist messages
			Priority:      message.Priority,
			CorrelationId: message.CorrelationID,
			Timestamp:     message.Timestamp,
			MessageId:     message.ID,
		},
	)
	
	if err != nil {
		return fmt.Errorf("failed to publish message: %v", err)
	}
	
	r.messagesPublished++
	log.Printf("📤 Published message to %s: %s (ID: %s)", queueName, message.Type, message.ID)
	return nil
}

// ConsumeMessages starts consuming messages from a queue
func (r *RabbitMQClient) ConsumeMessages(queueName string, handler func(Message) error) error {
	if !r.IsConnected() {
		return fmt.Errorf("not connected to RabbitMQ")
	}
	
	msgs, err := r.channel.Consume(
		queueName,
		"",    // consumer tag
		false, // auto-ack (we'll ack manually)
		false, // exclusive
		false, // no-local
		false, // no-wait
		nil,   // arguments
	)
	if err != nil {
		return fmt.Errorf("failed to register consumer: %v", err)
	}
	
	go func() {
		log.Printf("📥 Started consuming from queue: %s", queueName)
		
		for d := range msgs {
			var message Message
			err := json.Unmarshal(d.Body, &message)
			if err != nil {
				log.Printf("❌ Failed to unmarshal message: %v", err)
				d.Nack(false, false) // negative ack, don't requeue
				continue
			}
			
			// Process message
			err = handler(message)
			if err != nil {
				log.Printf("❌ Message processing failed: %v", err)
				
				// Check retry count
				if message.RetryCount < message.MaxRetries {
					message.RetryCount++
					log.Printf("🔄 Retrying message (attempt %d/%d)", message.RetryCount, message.MaxRetries)
					
					// Republish for retry
					retryErr := r.PublishMessage(queueName, message)
					if retryErr != nil {
						log.Printf("❌ Failed to republish for retry: %v", retryErr)
					}
				} else {
					log.Printf("💀 Message exceeded max retries, sending to dead letter queue")
				}
				
				d.Nack(false, false)
			} else {
				r.messagesConsumed++
				d.Ack(false) // acknowledge successful processing
				log.Printf("✅ Processed message: %s (ID: %s)", message.Type, message.ID)
			}
		}
	}()
	
	return nil
}

// PublishDimensionalArrayRequest publishes a dimensional array computation request
func (r *RabbitMQClient) PublishDimensionalArrayRequest(request map[string]interface{}) error {
	message := Message{
		Type:          TypeDimensionalArray,
		Priority:      5,
		MaxRetries:    3,
		Payload:       request,
		RoutingKey:    QueueComputationRequests,
		CorrelationID: fmt.Sprintf("dim_array_%d", time.Now().UnixNano()),
	}
	
	return r.PublishMessage(QueueComputationRequests, message)
}

// PublishT5ProcessingRequest publishes a T5 processing request
func (r *RabbitMQClient) PublishT5ProcessingRequest(request map[string]interface{}) error {
	message := Message{
		Type:          TypeT5Processing,
		Priority:      4,
		MaxRetries:    3,
		Payload:       request,
		RoutingKey:    QueueComputationRequests,
		CorrelationID: fmt.Sprintf("t5_proc_%d", time.Now().UnixNano()),
	}
	
	return r.PublishMessage(QueueComputationRequests, message)
}

// PublishCacheRequest publishes a cache operation request
func (r *RabbitMQClient) PublishCacheRequest(request map[string]interface{}) error {
	message := Message{
		Type:          TypeCacheRequest,
		Priority:      3,
		MaxRetries:    2,
		Payload:       request,
		RoutingKey:    QueueCacheOperations,
		CorrelationID: fmt.Sprintf("cache_%d", time.Now().UnixNano()),
	}
	
	return r.PublishMessage(QueueCacheOperations, message)
}

// PublishBackgroundTask publishes a background processing task
func (r *RabbitMQClient) PublishBackgroundTask(task map[string]interface{}) error {
	message := Message{
		Type:          TypeBackgroundTask,
		Priority:      2,
		MaxRetries:    1,
		Payload:       task,
		RoutingKey:    QueueBackgroundTasks,
		CorrelationID: fmt.Sprintf("bg_task_%d", time.Now().UnixNano()),
	}
	
	return r.PublishMessage(QueueBackgroundTasks, message)
}

// PublishOfflineTask publishes a task to be processed when back online
func (r *RabbitMQClient) PublishOfflineTask(task map[string]interface{}) error {
	message := Message{
		Type:          TypeOfflineTask,
		Priority:      1,
		MaxRetries:    5,
		Payload:       task,
		RoutingKey:    QueueOfflineProcessing,
		CorrelationID: fmt.Sprintf("offline_%d", time.Now().UnixNano()),
	}
	
	return r.PublishMessage(QueueOfflineProcessing, message)
}

// PublishHealthCheck publishes a health check message
func (r *RabbitMQClient) PublishHealthCheck(status map[string]interface{}) error {
	message := Message{
		Type:          TypeHealthCheck,
		Priority:      1,
		MaxRetries:    1,
		Payload:       status,
		RoutingKey:    QueueHealthChecks,
		CorrelationID: fmt.Sprintf("health_%d", time.Now().UnixNano()),
	}
	
	return r.PublishMessage(QueueHealthChecks, message)
}

// GetStats returns RabbitMQ client statistics
func (r *RabbitMQClient) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"connected":           r.IsConnected(),
		"messages_published":  r.messagesPublished,
		"messages_consumed":   r.messagesConsumed,
		"exchange":           r.config.Exchange,
		"queues":             len(r.config.Queues),
		"connection_url":     r.config.URL,
	}
}

// SetEventHandlers sets event handlers
func (r *RabbitMQClient) SetEventHandlers(onConnect, onDisconnect func(), onError func(error)) {
	r.onConnect = onConnect
	r.onDisconnect = onDisconnect
	r.onError = onError
}

// Private methods

func (r *RabbitMQClient) setupQueues() error {
	// Default queue configurations for AI Modular System
	defaultQueues := map[string]QueueConfig{
		QueueComputationRequests: {
			Name:       QueueComputationRequests,
			Durable:    true,
			AutoDelete: false,
			Exclusive:  false,
			NoWait:     false,
			RoutingKey: QueueComputationRequests,
		},
		QueueComputationResults: {
			Name:       QueueComputationResults,
			Durable:    true,
			AutoDelete: false,
			Exclusive:  false,
			NoWait:     false,
			RoutingKey: QueueComputationResults,
		},
		QueueCacheOperations: {
			Name:       QueueCacheOperations,
			Durable:    true,
			AutoDelete: false,
			Exclusive:  false,
			NoWait:     false,
			RoutingKey: QueueCacheOperations,
		},
		QueueHealthChecks: {
			Name:       QueueHealthChecks,
			Durable:    false, // health checks can be lost
			AutoDelete: true,
			Exclusive:  false,
			NoWait:     false,
			RoutingKey: QueueHealthChecks,
		},
		QueueBackgroundTasks: {
			Name:       QueueBackgroundTasks,
			Durable:    true,
			AutoDelete: false,
			Exclusive:  false,
			NoWait:     false,
			RoutingKey: QueueBackgroundTasks,
		},
		QueueOfflineProcessing: {
			Name:       QueueOfflineProcessing,
			Durable:    true, // important for offline processing
			AutoDelete: false,
			Exclusive:  false,
			NoWait:     false,
			RoutingKey: QueueOfflineProcessing,
		},
	}
	
	// Use default queues if none configured
	if len(r.config.Queues) == 0 {
		r.config.Queues = defaultQueues
	}
	
	// Declare all queues
	for _, queueConfig := range r.config.Queues {
		_, err := r.channel.QueueDeclare(
			queueConfig.Name,
			queueConfig.Durable,
			queueConfig.AutoDelete,
			queueConfig.Exclusive,
			queueConfig.NoWait,
			queueConfig.Arguments,
		)
		if err != nil {
			return fmt.Errorf("failed to declare queue %s: %v", queueConfig.Name, err)
		}
		
		// Bind queue to exchange if exchange is configured
		if r.config.Exchange != "" {
			err = r.channel.QueueBind(
				queueConfig.Name,       // queue name
				queueConfig.RoutingKey, // routing key
				r.config.Exchange,      // exchange
				false,                  // no-wait
				nil,                    // arguments
			)
			if err != nil {
				return fmt.Errorf("failed to bind queue %s: %v", queueConfig.Name, err)
			}
		}
		
		log.Printf("✅ Queue declared and bound: %s", queueConfig.Name)
	}
	
	return nil
}

func (r *RabbitMQClient) handleConnectionClose() {
	closeError := <-r.connection.NotifyClose(make(chan *amqp.Error))
	if closeError != nil {
		log.Printf("🔌 RabbitMQ connection closed: %v", closeError)
		
		r.mu.Lock()
		r.isConnected = false
		r.mu.Unlock()
		
		if r.onError != nil {
			r.onError(closeError)
		}
		
		if r.onDisconnect != nil {
			r.onDisconnect()
		}
		
		// Attempt reconnection
		log.Printf("🔄 Attempting to reconnect to RabbitMQ...")
		go r.reconnect()
	}
}

func (r *RabbitMQClient) reconnect() {
	for {
		time.Sleep(r.config.RetryInterval)
		
		log.Printf("🔄 Reconnecting to RabbitMQ...")
		err := r.Connect()
		if err == nil {
			log.Printf("✅ Reconnected to RabbitMQ")
			break
		}
		
		log.Printf("❌ Reconnection failed: %v", err)
	}
}

// GetDefaultConfig returns a default RabbitMQ configuration
func GetDefaultConfig() Config {
	return Config{
		URL:           "amqp://guest:guest@localhost:5672/",
		Exchange:      "ai.direct",
		MaxRetries:    3,
		RetryInterval: 5 * time.Second,
		Queues:        make(map[string]QueueConfig),
	}
}