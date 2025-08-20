//go:build legacy
// +build legacy

// WebSocket/SSE handler for real-time agent orchestration
// Part of Phase 3: Live Agent Integration

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// WebSocket upgrader
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		// Allow all origins for development (restrict in production)
		return true
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

// Agent request/response structures
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

// Global hub instance
var hub *Hub

// Initialize WebSocket hub
func initWebSocketHub() {
	hub = &Hub{
		Clients:    make(map[*Client]bool),
		Broadcast:  make(chan []byte),
		Register:   make(chan *Client),
		Unregister: make(chan *Client),
	}
	go hub.run()
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
			log.Printf("WebSocket client connected: %s", client.ID)
			
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
				log.Printf("WebSocket client disconnected: %s", client.ID)
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
			// Ping clients to check connection health
			h.pingClients()
		}
	}
}

// Ping clients for health check
func (h *Hub) pingClients() {
	h.mu.Lock()
	defer h.mu.Unlock()
	
	for client := range h.Clients {
		if time.Since(client.LastActive) > 5*time.Minute {
			log.Printf("Removing inactive client: %s", client.ID)
			close(client.Send)
			delete(h.Clients, client)
		}
	}
}

// Broadcast message to all clients
func broadcastToClients(message AgentMessage) {
	if hub == nil {
		return
	}
	
	data, err := json.Marshal(message)
	if err != nil {
		log.Printf("Error marshaling broadcast message: %v", err)
		return
	}
	
	select {
	case hub.Broadcast <- data:
	default:
		log.Println("Broadcast channel full, dropping message")
	}
}

// WebSocket handler
func handleWebSocket(c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	clientID := fmt.Sprintf("client_%d", time.Now().UnixNano())
	client := &Client{
		ID:         clientID,
		Conn:       conn,
		Hub:        hub,
		Send:       make(chan []byte, 256),
		LastActive: time.Now(),
	}

	// Register client
	client.Hub.Register <- client

	// Start goroutines
	go client.writePump()
	go client.readPump()
}

// Client read pump
func (c *Client) readPump() {
	defer func() {
		c.Hub.Unregister <- c
		c.Conn.Close()
	}()

	c.Conn.SetReadLimit(512)
	c.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.Conn.SetPongHandler(func(string) error {
		c.Conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		c.LastActive = time.Now()
		return nil
	})

	for {
		var message AgentMessage
		err := c.Conn.ReadJSON(&message)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}
		
		c.LastActive = time.Now()
		// Handle incoming messages (agent requests, etc.)
		go handleAgentRequest(c, message)
	}
}

// Client write pump
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
				log.Printf("WebSocket write error: %v", err)
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

// Handle agent requests from WebSocket clients
func handleAgentRequest(client *Client, message AgentMessage) {
	log.Printf("Processing agent request from %s: %s", client.ID, message.Type)

	response := AgentMessage{
		RequestID: message.RequestID,
		Agent:     "go-backend",
		Status:    "processing",
		Timestamp: time.Now(),
	}

	// Send processing acknowledgment
	responseData, _ := json.Marshal(response)
	select {
	case client.Send <- responseData:
	default:
		log.Printf("Failed to send processing ack to client %s", client.ID)
	}

	// Process request based on type
	var result interface{}
	var err error

	switch message.Type {
	case "analyze":
		result, err = processAnalyzeRequest(message.Payload)
	case "summarize":
		result, err = processSummarizeRequest(message.Payload)
	case "embed":
		result, err = processEmbedRequest(message.Payload)
	case "search":
		result, err = processSearchRequest(message.Payload)
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
	responseData, _ = json.Marshal(response)
	select {
	case client.Send <- responseData:
	default:
		log.Printf("Failed to send result to client %s", client.ID)
	}
}

// Server-Sent Events handler
func handleSSE(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	clientID := fmt.Sprintf("sse_%d", time.Now().UnixNano())
	log.Printf("SSE client connected: %s", clientID)

	// Create a channel for this SSE connection
	eventChan := make(chan AgentMessage, 10)
	
	// Register SSE client (simplified - could use a separate SSE hub)
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		
		for {
			select {
			case event := <-eventChan:
				data, _ := json.Marshal(event)
				c.SSEvent("message", string(data))
				c.Writer.Flush()
			case <-ticker.C:
				// Send keepalive
				keepalive := AgentMessage{
					Type:      "keepalive",
					Status:    "connected",
					Timestamp: time.Now(),
				}
				data, _ := json.Marshal(keepalive)
				c.SSEvent("keepalive", string(data))
				c.Writer.Flush()
			case <-c.Request.Context().Done():
				log.Printf("SSE client disconnected: %s", clientID)
				return
			}
		}
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

	// Keep connection alive
	<-c.Request.Context().Done()
}

// Process request functions (integrate with existing Ollama handlers)
func processAnalyzeRequest(payload interface{}) (interface{}, error) {
	// Convert payload to expected format
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	text, ok := payloadMap["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text field required")
	}

	// Use existing Ollama integration
	result, err := callOllamaGenerate("gemma3-legal:latest", fmt.Sprintf("Analyze: %s", text))
	if err != nil {
		return nil, err
	}

	return gin.H{
		"analysis": result,
		"agent":    "go-llama",
		"model":    "gemma3-legal",
		"backend":  "ollama",
	}, nil
}

func processSummarizeRequest(payload interface{}) (interface{}, error) {
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	text, ok := payloadMap["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text field required")
	}

	result, err := callOllamaGenerate("gemma3-legal:latest", fmt.Sprintf("Summarize: %s", text))
	if err != nil {
		return nil, err
	}

	return gin.H{
		"summary": result,
		"agent":   "go-llama",
		"model":   "gemma3-legal",
		"backend": "ollama",
	}, nil
}

func processEmbedRequest(payload interface{}) (interface{}, error) {
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	text, ok := payloadMap["text"].(string)
	if !ok {
		return nil, fmt.Errorf("text field required")
	}

	embedding, err := callOllamaEmbed(text)
	if err != nil {
		return nil, err
	}

	return gin.H{
		"embedding": embedding,
		"dimension": len(embedding),
		"agent":     "go-llama",
		"model":     "nomic-embed-text",
		"backend":   "ollama",
	}, nil
}

func processSearchRequest(payload interface{}) (interface{}, error) {
	payloadMap, ok := payload.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid payload format")
	}

	query, ok := payloadMap["query"].(string)
	if !ok {
		return nil, fmt.Errorf("query field required")
	}

	// Placeholder for enhanced RAG search
	return gin.H{
		"results": []gin.H{
			{
				"document":  "Legal Document 1",
				"relevance": 0.85,
				"excerpt":   fmt.Sprintf("Search result for: %s", query),
			},
		},
		"agent":   "go-rag",
		"backend": "enhanced-rag",
	}, nil
}

// Ollama integration functions
func callOllamaGenerate(model, prompt string) (string, error) {
	ollamaURL := "http://localhost:11434" // Default Ollama URL
	if url := os.Getenv("OLLAMA_URL"); url != "" {
		ollamaURL = url
	}

	requestBody := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"temperature": 0.7,
			"top_p":       0.9,
		},
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("error marshaling request: %v", err)
	}

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post(ollamaURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("error calling Ollama: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Ollama returned status %d", resp.StatusCode)
	}

	var ollamaResponse struct {
		Response string `json:"response"`
		Done     bool   `json:"done"`
		Error    string `json:"error,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&ollamaResponse); err != nil {
		return "", fmt.Errorf("error decoding response: %v", err)
	}

	if ollamaResponse.Error != "" {
		return "", fmt.Errorf("Ollama error: %s", ollamaResponse.Error)
	}

	return ollamaResponse.Response, nil
}

func callOllamaEmbed(text string) ([]float64, error) {
	ollamaURL := "http://localhost:11434" // Default Ollama URL
	if url := os.Getenv("OLLAMA_URL"); url != "" {
		ollamaURL = url
	}

	requestBody := map[string]interface{}{
		"model":  "nomic-embed-text", // Default embedding model
		"prompt": text,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %v", err)
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(ollamaURL+"/api/embeddings", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error calling Ollama embeddings: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Ollama embeddings returned status %d", resp.StatusCode)
	}

	var embedResponse struct {
		Embedding []float64 `json:"embedding"`
		Error     string    `json:"error,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&embedResponse); err != nil {
		return nil, fmt.Errorf("error decoding embedding response: %v", err)
	}

	if embedResponse.Error != "" {
		return nil, fmt.Errorf("Ollama embedding error: %s", embedResponse.Error)
	}

	return embedResponse.Embedding, nil
}

// Add WebSocket and SSE routes to main router
func addLiveAgentRoutes(r *gin.Engine) {
	// Initialize WebSocket hub
	initWebSocketHub()

	// WebSocket endpoint
	r.GET("/ws", handleWebSocket)
	
	// Server-Sent Events endpoint
	r.GET("/api/events", handleSSE)
	
	// Health check for live agent system
	r.GET("/api/live-agents/health", func(c *gin.Context) {
		hub.mu.RLock()
		clientCount := len(hub.Clients)
		hub.mu.RUnlock()

		c.JSON(http.StatusOK, gin.H{
			"status":      "healthy",
			"wsClients":   clientCount,
			"features":    []string{"websocket", "sse", "real-time"},
			"agents":      []string{"go-llama", "ollama-direct", "context7", "rag"},
			"timestamp":   time.Now(),
		})
	})

	// Trigger broadcast message (for testing)
	r.POST("/api/live-agents/broadcast", func(c *gin.Context) {
		var message AgentMessage
		if err := c.ShouldBindJSON(&message); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		message.Timestamp = time.Now()
		broadcastToClients(message)
		
		c.JSON(http.StatusOK, gin.H{
			"status": "broadcasted",
			"message": message,
		})
	})
}