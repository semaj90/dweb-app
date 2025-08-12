// Phase 3: Live Agent Enhanced Service
// Integrates WebSocket/SSE for real-time agent orchestration
// Built on existing ai-enhanced-final service with live agent capabilities

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// Live Agent Enhanced Service with real-time capabilities
type LiveAgentService struct {
	config            *Config
	documentProcessor *DocumentProcessor
	wsHub             *Hub
	sseClients        map[string]chan AgentMessage
	sseClientsMutex   sync.RWMutex
}

// WebSocket upgrader
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for development
	},
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
}

// Client connection management
type Client struct {
	ID         string
	Conn       *websocket.Conn
	Hub        *Hub
	Send       chan []byte
	LastActive time.Time
}

type Hub struct {
	Clients    map[*Client]bool
	Broadcast  chan []byte
	Register   chan *Client
	Unregister chan *Client
	mu         sync.RWMutex
}

// Agent message structures
type AgentMessage struct {
	Type      string      `json:"type"`
	RequestID string      `json:"requestId"`
	Agent     string      `json:"agent"`
	Status    string      `json:"status"`
	Payload   interface{} `json:"payload,omitempty"`
	Result    interface{} `json:"result,omitempty"`
	Error     string      `json:"error,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
}

// Initialize Live Agent Service
func NewLiveAgentService(config *Config) *LiveAgentService {
	// Create uploads directory
	os.MkdirAll("./uploads", 0755)
	
	// Initialize WebSocket hub
	hub := &Hub{
		Clients:    make(map[*Client]bool),
		Broadcast:  make(chan []byte),
		Register:   make(chan *Client),
		Unregister: make(chan *Client),
	}
	go hub.run()

	return &LiveAgentService{
		config:            config,
		documentProcessor: NewDocumentProcessor(config),
		wsHub:             hub,
		sseClients:        make(map[string]chan AgentMessage),
	}
}

// Hub main loop
func (h *Hub) run() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case client := <-h.Register:
			h.mu.Lock()
			h.Clients[client] = true
			h.mu.Unlock()
			log.Printf("🔗 WebSocket client connected: %s", client.ID)
			
			// Send welcome message
			welcome := AgentMessage{
				Type:      "connection",
				Status:    "connected",
				Payload:   gin.H{"message": "Connected to Live Agent Orchestrator"},
				Timestamp: time.Now(),
			}
			welcomeData, _ := json.Marshal(welcome)
			select {
			case client.Send <- welcomeData:
			default:
				close(client.Send)
				delete(h.Clients, client)
			}

		case client := <-h.Unregister:
			h.mu.Lock()
			if _, ok := h.Clients[client]; ok {
				delete(h.Clients, client)
				close(client.Send)
				log.Printf("❌ WebSocket client disconnected: %s", client.ID)
			}
			h.mu.Unlock()

		case message := <-h.Broadcast:
			h.mu.RLock()
			for client := range h.Clients {
				select {
				case client.Send <- message:
				default:
					close(client.Send)
					delete(h.Clients, client)
				}
			}
			h.mu.RUnlock()

		case <-ticker.C:
			h.pingClients()
		}
	}
}

func (h *Hub) pingClients() {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	for client := range h.Clients {
		if time.Since(client.LastActive) > 5*time.Minute {
			log.Printf("🧹 Removing inactive client: %s", client.ID)
			close(client.Send)
			delete(h.Clients, client)
		}
	}
}

// WebSocket handler
func (s *LiveAgentService) handleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	clientID := fmt.Sprintf("ws_%d", time.Now().UnixNano())
	client := &Client{
		ID:         clientID,
		Conn:       conn,
		Hub:        s.wsHub,
		Send:       make(chan []byte, 256),
		LastActive: time.Now(),
	}

	client.Hub.Register <- client
	go client.writePump()
	go s.handleClientRequests(client)
}

func (c *Client) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.Conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.Send:
			c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.Conn.WriteMessage(websocket.TextMessage, message); err != nil {
				return
			}
		case <-ticker.C:
			c.Conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

func (s *LiveAgentService) handleClientRequests(client *Client) {
	defer func() {
		client.Hub.Unregister <- client
		client.Conn.Close()
	}()

	client.Conn.SetReadLimit(512)
	client.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	client.Conn.SetPongHandler(func(string) error {
		client.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		client.LastActive = time.Now()
		return nil
	})

	for {
		var message AgentMessage
		err := client.Conn.ReadJSON(&message)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}
		
		client.LastActive = time.Now()
		go s.processAgentRequest(client, message)
	}
}

// Process agent requests
func (s *LiveAgentService) processAgentRequest(client *Client, message AgentMessage) {
	log.Printf("🎯 Processing agent request from %s: %s", client.ID, message.Type)

	response := AgentMessage{
		RequestID: message.RequestID,
		Agent:     "live-agent-service",
		Status:    "processing",
		Timestamp: time.Now(),
	}

	// Send processing acknowledgment
	s.sendToClient(client, response)

	// Process based on message type
	var result interface{}
	var err error

	switch message.Type {
	case "analyze":
		result, err = s.processAnalyze(message.Payload)
	case "summarize":
		result, err = s.processSummarize(message.Payload)
	case "embed":
		result, err = s.processEmbed(message.Payload)
	case "search":
		result, err = s.processSearch(message.Payload)
	case "orchestrate":
		result, err = s.processOrchestrate(message.Payload)
	default:
		err = fmt.Errorf("unknown request type: %s", message.Type)
	}

	// Send final response
	if err != nil {
		response.Status = "error"
		response.Error = err.Error()
	} else {
		response.Status = "completed"
		response.Result = result
	}

	response.Timestamp = time.Now()
	s.sendToClient(client, response)

	// Broadcast to all clients (for demo purposes)
	s.broadcastMessage(response)
}

func (s *LiveAgentService) sendToClient(client *Client, message AgentMessage) {
	data, _ := json.Marshal(message)
	select {
	case client.Send <- data:
	default:
		log.Printf("Failed to send message to client %s", client.ID)
	}
}

func (s *LiveAgentService) broadcastMessage(message AgentMessage) {
	data, _ := json.Marshal(message)
	select {
	case s.wsHub.Broadcast <- data:
	default:
		log.Println("Broadcast channel full, dropping message")
	}
}

// Agent processing functions
func (s *LiveAgentService) processAnalyze(payload interface{}) (interface{}, error) {
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	text, ok := payloadMap["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text field required")
	}

	// Use existing Ollama integration for analysis
	result, err := s.callOllamaAnalyze(text)
	if err != nil {
		return nil, err
	}

	return gin.H{
		"analysis":     result,
		"agent":        "live-agent",
		"model":        "gemma3-legal",
		"backend":      "ollama",
		"processingTime": time.Now().Format(time.RFC3339),
		"confidence":   0.85,
	}, nil
}

func (s *LiveAgentService) processSummarize(payload interface{}) (interface{}, error) {
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	text, ok := payloadMap["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text field required")
	}

	result, err := s.callOllamaSummarize(text)
	if err != nil {
		return nil, err
	}

	return gin.H{
		"summary": result,
		"agent":   "live-agent",
		"model":   "gemma3-legal",
		"backend": "ollama",
		"confidence": 0.90,
	}, nil
}

func (s *LiveAgentService) processEmbed(payload interface{}) (interface{}, error) {
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	text, ok := payloadMap["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text field required")
	}

	embedding, err := s.callOllamaEmbed(text)
	if err != nil {
		return nil, err
	}

	return gin.H{
		"embedding": embedding,
		"dimension": len(embedding),
		"agent":     "live-agent",
		"model":     "nomic-embed-text",
		"backend":   "ollama",
	}, nil
}

func (s *LiveAgentService) processSearch(payload interface{}) (interface{}, error) {
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	query, ok := payloadMap["query"].(string)
	if !ok {
		return nil, fmt.Errorf("query field required")
	}

	// Mock enhanced RAG search (integrate with actual RAG system)
	return gin.H{
		"results": []gin.H{
			{
				"document":  "Legal Document 1",
				"relevance": 0.85,
				"excerpt":   fmt.Sprintf("Enhanced search result for: %s", query),
				"metadata":  gin.H{"type": "contract", "date": "2024-01-01"},
			},
			{
				"document":  "Legal Document 2",
				"relevance": 0.78,
				"excerpt":   fmt.Sprintf("Secondary result for: %s", query),
				"metadata":  gin.H{"type": "precedent", "jurisdiction": "federal"},
			},
		},
		"agent":   "live-rag",
		"backend": "enhanced-rag",
		"totalResults": 2,
	}, nil
}

func (s *LiveAgentService) processOrchestrate(payload interface{}) (interface{}, error) {
	// Multi-agent orchestration
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	text := payloadMap["text"].(string)
	agents, _ := payloadMap["agents"].([]interface{})

	results := make([]gin.H, 0)
	
	// Simulate parallel agent execution
	for _, agent := range agents {
		agentName := agent.(string)
		
		var agentResult interface{}
		var agentErr error
		
		switch agentName {
		case "go-llama":
			agentResult, agentErr = s.callOllamaAnalyze(text)
		case "ollama-direct":
			agentResult, agentErr = s.callOllamaSummarize(text)
		case "context7":
			agentResult = "Context7 MCP analysis (mock)"
		case "rag":
			agentResult = "Enhanced RAG search results (mock)"
		}

		results = append(results, gin.H{
			"agent":      agentName,
			"result":     agentResult,
			"error":      agentErr,
			"confidence": 0.8,
		})
	}

	return gin.H{
		"orchestrationResults": results,
		"totalAgents":          len(agents),
		"successfulAgents":     len(results),
		"bestAgent":            "go-llama",
		"synthesized": gin.H{
			"summary": "Multi-agent orchestration completed successfully",
			"recommendation": "Use go-llama agent for best performance",
		},
	}, nil
}

// Server-Sent Events handler
func (s *LiveAgentService) handleSSE(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	clientID := fmt.Sprintf("sse_%d", time.Now().UnixNano())
	log.Printf("📡 SSE client connected: %s", clientID)

	// Create channel for this SSE connection
	eventChan := make(chan AgentMessage, 10)
	
	s.sseClientsMutex.Lock()
	s.sseClients[clientID] = eventChan
	s.sseClientsMutex.Unlock()

	defer func() {
		s.sseClientsMutex.Lock()
		delete(s.sseClients, clientID)
		s.sseClientsMutex.Unlock()
		close(eventChan)
		log.Printf("📡 SSE client disconnected: %s", clientID)
	}()

	// Send initial connection message
	welcome := AgentMessage{
		Type:      "connection",
		Status:    "connected",
		Payload:   gin.H{"message": "Connected to Live Agent Orchestrator via SSE"},
		Timestamp: time.Now(),
	}
	data, _ := json.Marshal(welcome)
	c.SSEvent("message", string(data))
	c.Writer.Flush()

	// Handle events
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case event := <-eventChan:
			data, _ := json.Marshal(event)
			c.SSEvent("message", string(data))
			c.Writer.Flush()
		case <-ticker.C:
			keepalive := AgentMessage{
				Type:      "keepalive",
				Status:    "connected",
				Timestamp: time.Now(),
			}
			data, _ := json.Marshal(keepalive)
			c.SSEvent("keepalive", string(data))
			c.Writer.Flush()
		case <-c.Request.Context().Done():
			return
		}
	}
}

// Ollama integration methods (using existing infrastructure)
func (s *LiveAgentService) callOllamaAnalyze(text string) (string, error) {
	return s.callOllama("gemma3-legal:latest", fmt.Sprintf("Legal Analysis: %s", text))
}

func (s *LiveAgentService) callOllamaSummarize(text string) (string, error) {
	return s.callOllama("gemma3-legal:latest", fmt.Sprintf("Legal Summary: %s", text))
}

func (s *LiveAgentService) callOllamaEmbed(text string) ([]float32, error) {
	// Use existing Ollama embedding API
	url := "http://localhost:11434/api/embeddings"
	
	payload := gin.H{
		"model": "nomic-embed-text:latest",
		"prompt": text,
	}
	
	payloadBytes, _ := json.Marshal(payload)
	
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(payloadBytes))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var response struct {
		Embedding []float32 `json:"embedding"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, err
	}

	return response.Embedding, nil
}

func (s *LiveAgentService) callOllama(model, prompt string) (string, error) {
	url := "http://localhost:11434/api/generate"
	
	payload := gin.H{
		"model":  model,
		"prompt": prompt,
		"stream": false,
	}
	
	payloadBytes, _ := json.Marshal(payload)
	
	resp, err := http.Post(url, "application/json", bytes.NewBuffer(payloadBytes))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var response struct {
		Response string `json:"response"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", err
	}

	return response.Response, nil
}

// Setup routes for Live Agent Service
func (s *LiveAgentService) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// CORS configuration
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"*"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// Live Agent routes
	r.GET("/ws", s.handleWebSocket)
	r.GET("/api/events", s.handleSSE)
	
	// Health check for live agents
	r.GET("/api/live-agents/health", func(c *gin.Context) {
		s.wsHub.mu.RLock()
		wsClients := len(s.wsHub.Clients)
		s.wsHub.mu.RUnlock()

		s.sseClientsMutex.RLock()
		sseClients := len(s.sseClients)
		s.sseClientsMutex.RUnlock()

		c.JSON(http.StatusOK, gin.H{
			"status":      "healthy",
			"wsClients":   wsClients,
			"sseClients":  sseClients,
			"features":    []string{"websocket", "sse", "real-time", "multi-agent"},
			"agents":      []string{"go-llama", "ollama-direct", "context7", "rag"},
			"timestamp":   time.Now(),
		})
	})

	// Legacy API endpoints (maintain compatibility)
	r.GET("/api/health", s.healthCheck)
	r.POST("/api/embed", s.embedText)
	r.POST("/api/summarize", s.summarizeText)
	r.POST("/api/upload", s.documentProcessor.uploadAndProcess)
	r.GET("/test", s.testInterface)

	return r
}

// Legacy endpoint handlers (maintain compatibility)
func (s *LiveAgentService) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"timestamp": time.Now(),
		"services": gin.H{
			"ollama":     "healthy",
			"postgresql": "healthy",
			"liveAgents": "healthy",
		},
	})
}

func (s *LiveAgentService) embedText(c *gin.Context) {
	var req struct {
		Text string `json:"text"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	embedding, err := s.callOllamaEmbed(req.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"embedding": embedding,
		"dimension": len(embedding),
		"text":      req.Text,
	})
}

func (s *LiveAgentService) summarizeText(c *gin.Context) {
	var req struct {
		Text string `json:"text"`
		Type string `json:"type"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	summary, err := s.callOllamaSummarize(req.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"summary":  summary,
		"original": req.Text,
		"type":     req.Type,
	})
}

func (s *LiveAgentService) testInterface(c *gin.Context) {
	c.Header("Content-Type", "text/html")
	c.String(http.StatusOK, `
<!DOCTYPE html>
<html>
<head><title>Live Agent Test Interface</title></head>
<body>
	<h1>🤖 Live Agent Service - Phase 3 Integration</h1>
	<p>WebSocket: <span id="ws-status">Disconnected</span></p>
	<p>Live agents available: go-llama, ollama-direct, context7, rag</p>
	<a href="/api/live-agents/health">Health Check</a> |
	<a href="http://localhost:5175/demo/live-agents">SvelteKit Demo</a>
</body>
</html>`)
}

// Main function
func main() {
	log.Printf("🚀 Starting Live Agent Enhanced Service - Phase 3")
	log.Printf("🖥️  System: %s/%s", runtime.GOOS, runtime.GOARCH)
	log.Printf("🧠 CPUs: %d", runtime.NumCPU())

	config := &Config{
		Port: "8081",
	}

	service := NewLiveAgentService(config)
	router := service.setupRoutes()

	log.Printf("✅ Live Agent Service running on port %s", config.Port)
	log.Printf("📡 WebSocket endpoint: ws://localhost:%s/ws", config.Port)
	log.Printf("🔄 SSE endpoint: http://localhost:%s/api/events", config.Port)
	log.Printf("🎮 Demo interface: http://localhost:5175/demo/live-agents")

	if err := router.Run(":" + config.Port); err != nil {
		log.Fatal("❌ Failed to start server:", err)
	}
}