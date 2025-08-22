package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
)

// Real-time Async Communication Layer
// Supports gRPC, WebSocket, and hybrid protocols for legal AI system

type CommunicationLayer struct {
	config        *CommConfig
	grpcServer    *grpc.Server
	httpServer    *http.Server
	wsUpgrader    websocket.Upgrader
	clients       map[string]*WebSocketClient
	clientsMutex  sync.RWMutex
	messageQueue  chan *Message
	grpcClients   map[string]interface{} // Service clients
	metrics       *CommMetrics
}

type CommConfig struct {
	GRPCPort        string
	HTTPPort        string
	EnableTLS       bool
	CertFile        string
	KeyFile         string
	MaxConnections  int
	MessageBuffer   int
	KeepAlive       time.Duration
	EnableGZIP      bool
	AllowedOrigins  []string
}

type WebSocketClient struct {
	ID         string
	Conn       *websocket.Conn
	UserID     string
	CaseID     string
	Send       chan []byte
	Hub        *CommunicationLayer
	LastSeen   time.Time
	IsActive   bool
	Metadata   map[string]interface{}
	mutex      sync.RWMutex
}

type Message struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	FromID      string                 `json:"from_id"`
	ToID        string                 `json:"to_id,omitempty"`
	CaseID      string                 `json:"case_id,omitempty"`
	Payload     map[string]interface{} `json:"payload"`
	Timestamp   time.Time              `json:"timestamp"`
	Priority    int                    `json:"priority"`
	Broadcast   bool                   `json:"broadcast"`
	Protocol    string                 `json:"protocol"`
	Retry       int                    `json:"retry"`
	MaxRetries  int                    `json:"max_retries"`
}

type CommMetrics struct {
	ActiveConnections   int64 `json:"active_connections"`
	TotalMessages       int64 `json:"total_messages"`
	MessagesPerSecond   float64 `json:"messages_per_second"`
	AverageLatency      float64 `json:"average_latency_ms"`
	GRPCCalls           int64 `json:"grpc_calls"`
	WebSocketMessages   int64 `json:"websocket_messages"`
	ErrorCount          int64 `json:"error_count"`
	LastError           string `json:"last_error,omitempty"`
	UptimeSeconds       int64 `json:"uptime_seconds"`
	StartTime           time.Time `json:"-"`
	mutex               sync.RWMutex
}

// Message types for legal AI system
const (
	MSG_CHAT_MESSAGE       = "chat_message"
	MSG_DOCUMENT_UPDATE    = "document_update"
	MSG_AI_RESPONSE        = "ai_response"
	MSG_CASE_UPDATE        = "case_update"
	MSG_EVIDENCE_ALERT     = "evidence_alert"
	MSG_SEARCH_RESULT      = "search_result"
	MSG_SYSTEM_NOTIFICATION = "system_notification"
	MSG_TYPING_INDICATOR   = "typing_indicator"
	MSG_PRESENCE_UPDATE    = "presence_update"
	MSG_GPU_METRICS        = "gpu_metrics"
)

func NewCommunicationLayer() *CommunicationLayer {
	config := &CommConfig{
		GRPCPort:       "8096",
		HTTPPort:       "8097",
		EnableTLS:      false,
		MaxConnections: 10000,
		MessageBuffer:  1000,
		KeepAlive:      30 * time.Second,
		EnableGZIP:     true,
		AllowedOrigins: []string{"http://localhost:5173", "http://localhost:3000"},
	}

	return &CommunicationLayer{
		config: config,
		wsUpgrader: websocket.Upgrader{
			ReadBufferSize:  4096,
			WriteBufferSize: 4096,
			CheckOrigin: func(r *http.Request) bool {
				origin := r.Header.Get("Origin")
				for _, allowed := range config.AllowedOrigins {
					if origin == allowed {
						return true
					}
				}
				return true // Allow all origins in development
			},
		},
		clients:      make(map[string]*WebSocketClient),
		messageQueue: make(chan *Message, config.MessageBuffer),
		grpcClients:  make(map[string]interface{}),
		metrics: &CommMetrics{
			StartTime: time.Now(),
		},
	}
}

func (cl *CommunicationLayer) Start() error {
	log.Println("🌐 Starting Real-time Communication Layer...")

	// Start message processing goroutine
	go cl.processMessages()

	// Start metrics collection
	go cl.collectMetrics()

	// Start gRPC server
	go cl.startGRPCServer()

	// Start HTTP/WebSocket server
	go cl.startHTTPServer()

	log.Printf("✅ Communication Layer started (gRPC: %s, HTTP: %s)", cl.config.GRPCPort, cl.config.HTTPPort)
	return nil
}

func (cl *CommunicationLayer) startGRPCServer() {
	lis, err := net.Listen("tcp", ":"+cl.config.GRPCPort)
	if err != nil {
		log.Fatalf("Failed to listen on gRPC port %s: %v", cl.config.GRPCPort, err)
	}

	// Configure gRPC server with keep-alive
	cl.grpcServer = grpc.NewServer(
		grpc.KeepaliveParams(keepalive.ServerParameters{
			MaxConnectionIdle: 15 * time.Second,
			MaxConnectionAge:  30 * time.Second,
			MaxConnectionAgeGrace: 5 * time.Second,
			Time:              5 * time.Second,
			Timeout:           1 * time.Second,
		}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
			MinTime:             5 * time.Second,
			PermitWithoutStream: true,
		}),
	)

	// Register services (implementations would be added here)
	// pb.RegisterLegalAIServiceServer(cl.grpcServer, &legalAIService{cl})
	
	// Enable reflection for debugging
	reflection.Register(cl.grpcServer)

	log.Printf("🔧 gRPC server listening on :%s", cl.config.GRPCPort)
	if err := cl.grpcServer.Serve(lis); err != nil {
		log.Printf("gRPC server error: %v", err)
	}
}

func (cl *CommunicationLayer) startHTTPServer() {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// CORS middleware
	router.Use(func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		for _, allowed := range cl.config.AllowedOrigins {
			if origin == allowed {
				c.Header("Access-Control-Allow-Origin", origin)
				break
			}
		}
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
		c.Header("Access-Control-Allow-Credentials", "true")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		c.Next()
	})

	// WebSocket endpoint
	router.GET("/ws", cl.handleWebSocket)

	// REST API endpoints
	api := router.Group("/api")
	{
		api.POST("/message", cl.handleMessage)
		api.GET("/metrics", cl.handleMetrics)
		api.GET("/clients", cl.handleClients)
		api.POST("/broadcast", cl.handleBroadcast)
		api.GET("/status", cl.handleStatus)
	}

	// Health check
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":     "healthy",
			"service":    "communication-layer",
			"version":    "2.0.0",
			"timestamp":  time.Now(),
			"grpc_port":  cl.config.GRPCPort,
			"http_port":  cl.config.HTTPPort,
		})
	})

	cl.httpServer = &http.Server{
		Addr:    ":" + cl.config.HTTPPort,
		Handler: router,
	}

	log.Printf("🌐 HTTP/WebSocket server listening on :%s", cl.config.HTTPPort)
	if err := cl.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("HTTP server error: %v", err)
	}
}

func (cl *CommunicationLayer) handleWebSocket(c *gin.Context) {
	conn, err := cl.wsUpgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	// Get client info from query parameters
	userID := c.Query("user_id")
	caseID := c.Query("case_id")
	clientID := fmt.Sprintf("%s_%d", userID, time.Now().UnixNano())

	client := &WebSocketClient{
		ID:       clientID,
		Conn:     conn,
		UserID:   userID,
		CaseID:   caseID,
		Send:     make(chan []byte, 256),
		Hub:      cl,
		LastSeen: time.Now(),
		IsActive: true,
		Metadata: make(map[string]interface{}),
	}

	// Register client
	cl.clientsMutex.Lock()
	cl.clients[clientID] = client
	cl.clientsMutex.Unlock()

	// Update metrics
	cl.metrics.mutex.Lock()
	cl.metrics.ActiveConnections++
	cl.metrics.mutex.Unlock()

	log.Printf("👤 Client connected: %s (User: %s, Case: %s)", clientID, userID, caseID)

	// Send welcome message
	welcome := &Message{
		ID:        fmt.Sprintf("welcome_%d", time.Now().UnixNano()),
		Type:      MSG_SYSTEM_NOTIFICATION,
		ToID:      clientID,
		Payload:   map[string]interface{}{"message": "Connected to Legal AI Communication Layer"},
		Timestamp: time.Now(),
		Protocol:  "websocket",
	}
	cl.sendToClient(client, welcome)

	// Start client handlers
	go client.writePump()
	go client.readPump()
}

func (wsc *WebSocketClient) readPump() {
	defer func() {
		wsc.Hub.unregisterClient(wsc)
		wsc.Conn.Close()
	}()

	wsc.Conn.SetReadLimit(512)
	wsc.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	wsc.Conn.SetPongHandler(func(string) error {
		wsc.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		wsc.LastSeen = time.Now()
		return nil
	})

	for {
		_, messageBytes, err := wsc.Conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		var message Message
		if err := json.Unmarshal(messageBytes, &message); err != nil {
			log.Printf("Message unmarshal error: %v", err)
			continue
		}

		message.FromID = wsc.ID
		message.Timestamp = time.Now()
		message.Protocol = "websocket"

		// Add to message queue
		select {
		case wsc.Hub.messageQueue <- &message:
		default:
			log.Printf("Message queue full, dropping message from %s", wsc.ID)
		}

		// Update metrics
		wsc.Hub.metrics.mutex.Lock()
		wsc.Hub.metrics.WebSocketMessages++
		wsc.Hub.metrics.TotalMessages++
		wsc.Hub.metrics.mutex.Unlock()
	}
}

func (wsc *WebSocketClient) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		wsc.Conn.Close()
	}()

	for {
		select {
		case message, ok := <-wsc.Send:
			wsc.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				wsc.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			if err := wsc.Conn.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}

		case <-ticker.C:
			wsc.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := wsc.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (cl *CommunicationLayer) processMessages() {
	for message := range cl.messageQueue {
		start := time.Now()
		
		switch message.Type {
		case MSG_CHAT_MESSAGE:
			cl.handleChatMessage(message)
		case MSG_DOCUMENT_UPDATE:
			cl.handleDocumentUpdate(message)
		case MSG_AI_RESPONSE:
			cl.handleAIResponse(message)
		case MSG_CASE_UPDATE:
			cl.handleCaseUpdate(message)
		case MSG_EVIDENCE_ALERT:
			cl.handleEvidenceAlert(message)
		case MSG_SEARCH_RESULT:
			cl.handleSearchResult(message)
		case MSG_TYPING_INDICATOR:
			cl.handleTypingIndicator(message)
		case MSG_PRESENCE_UPDATE:
			cl.handlePresenceUpdate(message)
		case MSG_GPU_METRICS:
			cl.handleGPUMetrics(message)
		default:
			log.Printf("Unknown message type: %s", message.Type)
		}

		// Update latency metrics
		latency := float64(time.Since(start).Nanoseconds()) / 1e6
		cl.metrics.mutex.Lock()
		cl.metrics.AverageLatency = (cl.metrics.AverageLatency + latency) / 2.0
		cl.metrics.mutex.Unlock()
	}
}

func (cl *CommunicationLayer) handleChatMessage(message *Message) {
	// Route chat message to appropriate clients
	if message.ToID != "" {
		// Direct message
		if client := cl.getClient(message.ToID); client != nil {
			cl.sendToClient(client, message)
		}
	} else if message.CaseID != "" {
		// Case-based message
		cl.broadcastToCase(message.CaseID, message)
	} else if message.Broadcast {
		// Global broadcast
		cl.broadcastToAll(message)
	}
}

func (cl *CommunicationLayer) handleDocumentUpdate(message *Message) {
	// Notify all clients working on the same case
	if message.CaseID != "" {
		cl.broadcastToCase(message.CaseID, message)
	}
}

func (cl *CommunicationLayer) handleAIResponse(message *Message) {
	// Send AI response to requesting client
	if message.ToID != "" {
		if client := cl.getClient(message.ToID); client != nil {
			cl.sendToClient(client, message)
		}
	}
}

func (cl *CommunicationLayer) handleCaseUpdate(message *Message) {
	// Broadcast case updates to relevant clients
	if message.CaseID != "" {
		cl.broadcastToCase(message.CaseID, message)
	}
}

func (cl *CommunicationLayer) handleEvidenceAlert(message *Message) {
	// High-priority evidence alerts
	message.Priority = 1
	if message.CaseID != "" {
		cl.broadcastToCase(message.CaseID, message)
	}
}

func (cl *CommunicationLayer) handleSearchResult(message *Message) {
	// Send search results to requesting client
	if message.ToID != "" {
		if client := cl.getClient(message.ToID); client != nil {
			cl.sendToClient(client, message)
		}
	}
}

func (cl *CommunicationLayer) handleTypingIndicator(message *Message) {
	// Broadcast typing indicators with short TTL
	if message.CaseID != "" {
		cl.broadcastToCase(message.CaseID, message)
	}
}

func (cl *CommunicationLayer) handlePresenceUpdate(message *Message) {
	// Update client presence status
	if client := cl.getClient(message.FromID); client != nil {
		client.mutex.Lock()
		client.LastSeen = time.Now()
		if status, ok := message.Payload["status"].(string); ok {
			client.Metadata["status"] = status
		}
		client.mutex.Unlock()
	}
}

func (cl *CommunicationLayer) handleGPUMetrics(message *Message) {
	// Broadcast GPU metrics to monitoring clients
	cl.broadcastToAll(message)
}

func (cl *CommunicationLayer) sendToClient(client *WebSocketClient, message *Message) {
	messageBytes, _ := json.Marshal(message)
	
	select {
	case client.Send <- messageBytes:
	default:
		close(client.Send)
		cl.unregisterClient(client)
	}
}

func (cl *CommunicationLayer) broadcastToCase(caseID string, message *Message) {
	cl.clientsMutex.RLock()
	defer cl.clientsMutex.RUnlock()
	
	for _, client := range cl.clients {
		if client.CaseID == caseID && client.IsActive {
			cl.sendToClient(client, message)
		}
	}
}

func (cl *CommunicationLayer) broadcastToAll(message *Message) {
	cl.clientsMutex.RLock()
	defer cl.clientsMutex.RUnlock()
	
	for _, client := range cl.clients {
		if client.IsActive {
			cl.sendToClient(client, message)
		}
	}
}

func (cl *CommunicationLayer) getClient(clientID string) *WebSocketClient {
	cl.clientsMutex.RLock()
	defer cl.clientsMutex.RUnlock()
	return cl.clients[clientID]
}

func (cl *CommunicationLayer) unregisterClient(client *WebSocketClient) {
	cl.clientsMutex.Lock()
	if _, ok := cl.clients[client.ID]; ok {
		delete(cl.clients, client.ID)
		close(client.Send)
		client.IsActive = false
		cl.metrics.mutex.Lock()
		cl.metrics.ActiveConnections--
		cl.metrics.mutex.Unlock()
		log.Printf("👋 Client disconnected: %s", client.ID)
	}
	cl.clientsMutex.Unlock()
}

// REST API Handlers
func (cl *CommunicationLayer) handleMessage(c *gin.Context) {
	var message Message
	if err := c.ShouldBindJSON(&message); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	message.ID = fmt.Sprintf("api_%d", time.Now().UnixNano())
	message.Timestamp = time.Now()
	message.Protocol = "http"

	select {
	case cl.messageQueue <- &message:
		c.JSON(http.StatusOK, gin.H{"status": "queued", "message_id": message.ID})
	default:
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "message queue full"})
	}
}

func (cl *CommunicationLayer) handleMetrics(c *gin.Context) {
	cl.metrics.mutex.RLock()
	metrics := *cl.metrics
	metrics.UptimeSeconds = int64(time.Since(metrics.StartTime).Seconds())
	if metrics.TotalMessages > 0 {
		metrics.MessagesPerSecond = float64(metrics.TotalMessages) / float64(metrics.UptimeSeconds)
	}
	cl.metrics.mutex.RUnlock()

	c.JSON(http.StatusOK, metrics)
}

func (cl *CommunicationLayer) handleClients(c *gin.Context) {
	cl.clientsMutex.RLock()
	clientList := make([]map[string]interface{}, 0, len(cl.clients))
	for _, client := range cl.clients {
		client.mutex.RLock()
		clientInfo := map[string]interface{}{
			"id":        client.ID,
			"user_id":   client.UserID,
			"case_id":   client.CaseID,
			"last_seen": client.LastSeen,
			"is_active": client.IsActive,
			"metadata":  client.Metadata,
		}
		client.mutex.RUnlock()
		clientList = append(clientList, clientInfo)
	}
	cl.clientsMutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{
		"total_clients": len(clientList),
		"clients":       clientList,
	})
}

func (cl *CommunicationLayer) handleBroadcast(c *gin.Context) {
	var message Message
	if err := c.ShouldBindJSON(&message); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	message.ID = fmt.Sprintf("broadcast_%d", time.Now().UnixNano())
	message.Timestamp = time.Now()
	message.Protocol = "http"
	message.Broadcast = true

	cl.broadcastToAll(&message)
	c.JSON(http.StatusOK, gin.H{"status": "broadcasted", "message_id": message.ID})
}

func (cl *CommunicationLayer) handleStatus(c *gin.Context) {
	cl.metrics.mutex.RLock()
	activeConnections := cl.metrics.ActiveConnections
	totalMessages := cl.metrics.TotalMessages
	uptime := time.Since(cl.metrics.StartTime)
	cl.metrics.mutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{
		"service":            "Real-time Communication Layer",
		"version":            "2.0.0",
		"status":             "running",
		"active_connections": activeConnections,
		"total_messages":     totalMessages,
		"uptime_seconds":     int64(uptime.Seconds()),
		"grpc_port":         cl.config.GRPCPort,
		"http_port":         cl.config.HTTPPort,
		"protocols":         []string{"websocket", "grpc", "http"},
		"timestamp":         time.Now(),
	})
}

func (cl *CommunicationLayer) collectMetrics() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		cl.metrics.mutex.Lock()
		uptime := time.Since(cl.metrics.StartTime)
		if cl.metrics.TotalMessages > 0 && uptime.Seconds() > 0 {
			cl.metrics.MessagesPerSecond = float64(cl.metrics.TotalMessages) / uptime.Seconds()
		}
		cl.metrics.UptimeSeconds = int64(uptime.Seconds())
		cl.metrics.mutex.Unlock()
	}
}

func (cl *CommunicationLayer) Shutdown(ctx context.Context) error {
	log.Println("🔄 Shutting down Communication Layer...")

	// Close message queue
	close(cl.messageQueue)

	// Disconnect all WebSocket clients
	cl.clientsMutex.Lock()
	for _, client := range cl.clients {
		client.IsActive = false
		close(client.Send)
		client.Conn.Close()
	}
	cl.clients = make(map[string]*WebSocketClient)
	cl.clientsMutex.Unlock()

	// Shutdown gRPC server
	if cl.grpcServer != nil {
		cl.grpcServer.GracefulStop()
	}

	// Shutdown HTTP server
	if cl.httpServer != nil {
		return cl.httpServer.Shutdown(ctx)
	}

	log.Println("✅ Communication Layer shutdown complete")
	return nil
}

func main() {
	commLayer := NewCommunicationLayer()
	
	if err := commLayer.Start(); err != nil {
		log.Fatalf("💥 Failed to start communication layer: %v", err)
	}

	// Block forever
	select {}
}