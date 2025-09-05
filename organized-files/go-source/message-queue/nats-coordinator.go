package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/stan.go"
)

// Message types for legal AI system coordination
type MessageType string

const (
	DocumentProcessing MessageType = "document.processing"
	VectorIndexing     MessageType = "vector.indexing"
	AIAnalysis         MessageType = "ai.analysis"
	SystemAlert        MessageType = "system.alert"
	MemoryManagement   MessageType = "memory.management"
	ServiceHealth      MessageType = "service.health"
)

// LegalAIMessage represents a coordinated message in the system
type LegalAIMessage struct {
	ID          string                 `json:"id"`
	Type        MessageType            `json:"type"`
	Source      string                 `json:"source"`
	Target      string                 `json:"target"`
	Timestamp   time.Time              `json:"timestamp"`
	Priority    int                    `json:"priority"`
	Payload     map[string]interface{} `json:"payload"`
	RetryCount  int                    `json:"retry_count"`
	MaxRetries  int                    `json:"max_retries"`
}

// NATSCoordinator manages all inter-service communication
type NATSCoordinator struct {
	nc         *nats.Conn
	sc         stan.Conn
	clusterID  string
	clientID   string
	logger     *log.Logger
	
	// Service registry
	services   map[string]*ServiceInfo
	
	// Message handlers
	handlers   map[MessageType]MessageHandler
	
	// Performance metrics
	metrics    *CoordinatorMetrics
}

type ServiceInfo struct {
	Name        string    `json:"name"`
	Host        string    `json:"host"`
	Port        int       `json:"port"`
	Status      string    `json:"status"`
	LastSeen    time.Time `json:"last_seen"`
	LoadFactor  float64   `json:"load_factor"`
	Capabilities []string  `json:"capabilities"`
}

type MessageHandler func(ctx context.Context, msg *LegalAIMessage) error

type CoordinatorMetrics struct {
	MessagesProcessed int64     `json:"messages_processed"`
	ErrorCount        int64     `json:"error_count"`
	AverageLatency    float64   `json:"average_latency"`
	ActiveServices    int       `json:"active_services"`
	LastUpdate        time.Time `json:"last_update"`
}

// NewNATSCoordinator creates a new message coordinator
func NewNATSCoordinator(natsURL, clusterID, clientID string) (*NATSCoordinator, error) {
	// Connect to NATS
	nc, err := nats.Connect(natsURL, 
		nats.ReconnectWait(time.Second),
		nats.MaxReconnects(-1),
		nats.PingInterval(30*time.Second),
		nats.MaxPingsOutstanding(3),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS: %w", err)
	}

	// Connect to NATS Streaming
	sc, err := stan.Connect(clusterID, clientID,
		stan.NatsConn(nc),
		stan.SetConnectionLostHandler(func(_ stan.Conn, reason error) {
			log.Printf("NATS Streaming connection lost: %v", reason)
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS Streaming: %w", err)
	}

	coordinator := &NATSCoordinator{
		nc:        nc,
		sc:        sc,
		clusterID: clusterID,
		clientID:  clientID,
		logger:    log.New(log.Writer(), "[NATS-COORDINATOR] ", log.LstdFlags),
		services:  make(map[string]*ServiceInfo),
		handlers:  make(map[MessageType]MessageHandler),
		metrics: &CoordinatorMetrics{
			LastUpdate: time.Now(),
		},
	}

	// Setup default handlers
	coordinator.setupDefaultHandlers()

	return coordinator, nil
}

// setupDefaultHandlers configures built-in message handlers
func (nc *NATSCoordinator) setupDefaultHandlers() {
	// Document processing handler
	nc.RegisterHandler(DocumentProcessing, func(ctx context.Context, msg *LegalAIMessage) error {
		nc.logger.Printf("Processing document: %s", msg.Payload["document_id"])
		
		// Route to appropriate service based on document type
		docType, ok := msg.Payload["type"].(string)
		if !ok {
			return fmt.Errorf("invalid document type")
		}

		var targetService string
		switch docType {
		case "legal-brief":
			targetService = "legal-bert-service"
		case "contract":
			targetService = "contract-analyzer"
		default:
			targetService = "general-processor"
		}

		// Forward to specific service
		return nc.ForwardMessage(msg, targetService)
	})

	// Vector indexing handler
	nc.RegisterHandler(VectorIndexing, func(ctx context.Context, msg *LegalAIMessage) error {
		nc.logger.Printf("Indexing vectors for: %s", msg.Payload["document_id"])
		
		// Route to vector service
		return nc.ForwardMessage(msg, "qdrant-vector-service")
	})

	// AI analysis handler
	nc.RegisterHandler(AIAnalysis, func(ctx context.Context, msg *LegalAIMessage) error {
		nc.logger.Printf("Running AI analysis: %s", msg.Payload["analysis_type"])
		
		// Distribute load across AI services
		service := nc.selectBestService("ai-analysis")
		if service == nil {
			return fmt.Errorf("no AI analysis service available")
		}

		return nc.ForwardMessage(msg, service.Name)
	})

	// System alert handler
	nc.RegisterHandler(SystemAlert, func(ctx context.Context, msg *LegalAIMessage) error {
		priority := msg.Priority
		nc.logger.Printf("System alert (priority %d): %s", priority, msg.Payload["message"])
		
		// High priority alerts go to monitoring system
		if priority >= 8 {
			return nc.ForwardToMonitoring(msg)
		}
		
		return nil
	})

	// Memory management handler
	nc.RegisterHandler(MemoryManagement, func(ctx context.Context, msg *LegalAIMessage) error {
		action := msg.Payload["action"].(string)
		nc.logger.Printf("Memory management action: %s", action)
		
		switch action {
		case "cleanup":
			return nc.BroadcastMemoryCleanup()
		case "gc":
			return nc.BroadcastGarbageCollection()
		default:
			return fmt.Errorf("unknown memory action: %s", action)
		}
	})
}

// RegisterHandler registers a custom message handler
func (nc *NATSCoordinator) RegisterHandler(msgType MessageType, handler MessageHandler) {
	nc.handlers[msgType] = handler
}

// PublishMessage publishes a message to the coordination system
func (nc *NATSCoordinator) PublishMessage(msg *LegalAIMessage) error {
	msg.Timestamp = time.Now()
	
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	subject := fmt.Sprintf("legal.%s", msg.Type)
	
	// Use NATS Streaming for persistence
	_, err = nc.sc.Publish(subject, data)
	if err != nil {
		return fmt.Errorf("failed to publish message: %w", err)
	}

	nc.metrics.MessagesProcessed++
	return nil
}

// Subscribe sets up subscriptions for all message types
func (nc *NATSCoordinator) Subscribe() error {
	for msgType := range nc.handlers {
		subject := fmt.Sprintf("legal.%s", msgType)
		
		_, err := nc.sc.Subscribe(subject, func(msg *stan.Msg) {
			nc.handleMessage(msg.Data)
		}, stan.DurableName(fmt.Sprintf("%s-durable", nc.clientID)))
		
		if err != nil {
			return fmt.Errorf("failed to subscribe to %s: %w", subject, err)
		}
	}

	// Subscribe to service health updates
	_, err := nc.nc.Subscribe("legal.service.health.*", func(msg *nats.Msg) {
		nc.handleServiceHealth(msg.Subject, msg.Data)
	})

	return err
}

// handleMessage processes incoming messages
func (nc *NATSCoordinator) handleMessage(data []byte) {
	var msg LegalAIMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		nc.logger.Printf("Failed to unmarshal message: %v", err)
		return
	}

	handler, exists := nc.handlers[msg.Type]
	if !exists {
		nc.logger.Printf("No handler for message type: %s", msg.Type)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	start := time.Now()
	err := handler(ctx, &msg)
	duration := time.Since(start)

	if err != nil {
		nc.logger.Printf("Handler error for %s: %v", msg.Type, err)
		nc.retryMessage(&msg)
		nc.metrics.ErrorCount++
	}

	// Update metrics
	nc.updateLatencyMetrics(duration)
}

// retryMessage implements retry logic for failed messages
func (nc *NATSCoordinator) retryMessage(msg *LegalAIMessage) {
	if msg.RetryCount >= msg.MaxRetries {
		nc.logger.Printf("Message %s exceeded max retries", msg.ID)
		return
	}

	msg.RetryCount++
	
	// Exponential backoff
	delay := time.Duration(1<<msg.RetryCount) * time.Second
	time.AfterFunc(delay, func() {
		nc.PublishMessage(msg)
	})
}

// ForwardMessage forwards a message to a specific service
func (nc *NATSCoordinator) ForwardMessage(msg *LegalAIMessage, serviceName string) error {
	subject := fmt.Sprintf("legal.service.%s", serviceName)
	
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	return nc.nc.Publish(subject, data)
}

// selectBestService selects the best available service based on load
func (nc *NATSCoordinator) selectBestService(capability string) *ServiceInfo {
	var best *ServiceInfo
	var lowestLoad float64 = 1.0

	for _, service := range nc.services {
		// Check if service has the required capability
		hasCapability := false
		for _, cap := range service.Capabilities {
			if cap == capability {
				hasCapability = true
				break
			}
		}

		if hasCapability && service.Status == "healthy" && service.LoadFactor < lowestLoad {
			best = service
			lowestLoad = service.LoadFactor
		}
	}

	return best
}

// BroadcastMemoryCleanup sends memory cleanup signal to all services
func (nc *NATSCoordinator) BroadcastMemoryCleanup() error {
	msg := &LegalAIMessage{
		ID:       fmt.Sprintf("cleanup-%d", time.Now().Unix()),
		Type:     MemoryManagement,
		Source:   "coordinator",
		Target:   "all",
		Priority: 7,
		Payload: map[string]interface{}{
			"action": "cleanup",
		},
	}

	return nc.nc.Publish("legal.memory.cleanup", []byte(msg.ID))
}

// BroadcastGarbageCollection triggers GC across all services
func (nc *NATSCoordinator) BroadcastGarbageCollection() error {
	msg := &LegalAIMessage{
		ID:       fmt.Sprintf("gc-%d", time.Now().Unix()),
		Type:     MemoryManagement,
		Source:   "coordinator",
		Target:   "all",
		Priority: 6,
		Payload: map[string]interface{}{
			"action": "gc",
		},
	}

	return nc.nc.Publish("legal.memory.gc", []byte(msg.ID))
}

// handleServiceHealth processes service health updates
func (nc *NATSCoordinator) handleServiceHealth(subject string, data []byte) {
	var serviceInfo ServiceInfo
	if err := json.Unmarshal(data, &serviceInfo); err != nil {
		nc.logger.Printf("Failed to unmarshal service health: %v", err)
		return
	}

	serviceInfo.LastSeen = time.Now()
	nc.services[serviceInfo.Name] = &serviceInfo
	nc.metrics.ActiveServices = len(nc.services)
}

// updateLatencyMetrics updates performance metrics
func (nc *NATSCoordinator) updateLatencyMetrics(duration time.Duration) {
	// Simple moving average
	if nc.metrics.AverageLatency == 0 {
		nc.metrics.AverageLatency = duration.Seconds()
	} else {
		nc.metrics.AverageLatency = (nc.metrics.AverageLatency*0.9) + (duration.Seconds()*0.1)
	}
	nc.metrics.LastUpdate = time.Now()
}

// GetMetrics returns current coordinator metrics
func (nc *NATSCoordinator) GetMetrics() *CoordinatorMetrics {
	return nc.metrics
}

// Close gracefully shuts down the coordinator
func (nc *NATSCoordinator) Close() error {
	if nc.sc != nil {
		nc.sc.Close()
	}
	if nc.nc != nil {
		nc.nc.Close()
	}
	return nil
}

// Main entry point for NATS coordinator service
func main() {
	coordinator, err := NewNATSCoordinator(
		"nats://localhost:4222",
		"legal-ai-cluster",
		"coordinator-001",
	)
	if err != nil {
		log.Fatal("Failed to create coordinator:", err)
	}
	defer coordinator.Close()

	// Start subscription
	if err := coordinator.Subscribe(); err != nil {
		log.Fatal("Failed to start subscriptions:", err)
	}

	log.Println("NATS Legal AI Coordinator started")

	// Keep the service running
	select {}
}