package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// StreamingManager handles real-time streaming and chunking operations
type StreamingManager struct {
	wsUpgrader     websocket.Upgrader
	activeStreams  map[string]*StreamSession
	sseClients     map[string]*SSEClient
	mu             sync.RWMutex
	chunkProcessor *ChunkStreamProcessor
	metrics        *StreamMetrics
}

// StreamSession represents an active streaming session
type StreamSession struct {
	ID            string                 `json:"id"`
	Type          string                 `json:"type"` // "websocket", "sse", "http-chunk"
	ClientID      string                 `json:"client_id"`
	Created       time.Time              `json:"created"`
	LastActivity  time.Time              `json:"last_activity"`
	BytesSent     int64                  `json:"bytes_sent"`
	BytesReceived int64                  `json:"bytes_received"`
	MessageCount  int64                  `json:"message_count"`
	IsActive      bool                   `json:"is_active"`
	Metadata      map[string]interface{} `json:"metadata"`
	conn          *websocket.Conn
	sseWriter     io.Writer
	context       context.Context
	cancel        context.CancelFunc
}

// SSEClient represents a Server-Sent Events client
type SSEClient struct {
	ID           string        `json:"id"`
	Writer       io.Writer     `json:"-"`
	Flusher      http.Flusher `json:"-"`
	MessageChan  chan []byte   `json:"-"`
	CloseChan    chan bool     `json:"-"`
	LastPing     time.Time     `json:"last_ping"`
	MessagesSent int64         `json:"messages_sent"`
}

// ChunkStreamProcessor processes streaming chunks
type ChunkStreamProcessor struct {
	bufferSize   int
	maxChunkSize int
	workers      int
	inputQueue   chan *StreamChunk
	outputQueue  chan *ProcessedChunk
	errorQueue   chan error
	wg           sync.WaitGroup
}

// StreamChunk represents a chunk of streaming data
type StreamChunk struct {
	ID        string          `json:"id"`
	SessionID string          `json:"session_id"`
	Index     int             `json:"index"`
	Data      []byte          `json:"data"`
	Size      int             `json:"size"`
	Timestamp time.Time       `json:"timestamp"`
	IsLast    bool            `json:"is_last"`
	Metadata  json.RawMessage `json:"metadata,omitempty"`
}

// ProcessedChunk represents a processed chunk ready for output
type ProcessedChunk struct {
	StreamChunk
	ProcessedData interface{}   `json:"processed_data"`
	ProcessTime   time.Duration `json:"process_time"`
	Error         string        `json:"error,omitempty"`
}

// StreamMetrics tracks streaming performance
type StreamMetrics struct {
	mu                sync.RWMutex
	ActiveConnections int64         `json:"active_connections"`
	TotalConnections  int64         `json:"total_connections"`
	BytesStreamed     int64         `json:"bytes_streamed"`
	ChunksProcessed   int64         `json:"chunks_processed"`
	AverageLatency    time.Duration `json:"average_latency"`
	ErrorCount        int64         `json:"error_count"`
	LastUpdate        time.Time     `json:"last_update"`
}

// StreamRequest represents a streaming request
type StreamRequest struct {
	Type        string                 `json:"type"` // "process", "inference", "transform"
	Data        json.RawMessage        `json:"data"`
	ChunkSize   int                    `json:"chunk_size"`
	Compression bool                   `json:"compression"`
	Options     map[string]interface{} `json:"options"`
}

// StreamResponse represents a streaming response
type StreamResponse struct {
	SessionID string          `json:"session_id"`
	ChunkID   string          `json:"chunk_id"`
	Index     int             `json:"index"`
	Data      json.RawMessage `json:"data"`
	IsLast    bool            `json:"is_last"`
	Timestamp time.Time       `json:"timestamp"`
	Metadata  interface{}     `json:"metadata,omitempty"`
}

var (
	streamManager     *StreamingManager
	streamManagerOnce sync.Once
)

// InitializeStreamingManager initializes the global streaming manager
func InitializeStreamingManager() error {
	var initErr error
	streamManagerOnce.Do(func() {
		streamManager = &StreamingManager{
			wsUpgrader: websocket.Upgrader{
				ReadBufferSize:  1024,
				WriteBufferSize: 1024,
				CheckOrigin: func(r *http.Request) bool {
					// Allow connections from any origin
					return true
				},
			},
			activeStreams: make(map[string]*StreamSession),
			sseClients:    make(map[string]*SSEClient),
			metrics:       &StreamMetrics{LastUpdate: time.Now()},
			chunkProcessor: &ChunkStreamProcessor{
				bufferSize:   4096,
				maxChunkSize: 1024 * 1024, // 1MB
				workers:      4,
				inputQueue:   make(chan *StreamChunk, 100),
				outputQueue:  make(chan *ProcessedChunk, 100),
				errorQueue:   make(chan error, 10),
			},
		}

		// Start chunk processor workers
		streamManager.chunkProcessor.start()

		// Start metrics updater
		go streamManager.updateMetrics()

		log.Println("✅ Streaming Manager initialized")
	})

	return initErr
}

// GetStreamingManager returns the global streaming manager instance
func GetStreamingManager() *StreamingManager {
	return streamManager
}

// WebSocket Streaming

// HandleWebSocketStream handles WebSocket streaming connections
func (sm *StreamingManager) HandleWebSocketStream(c *gin.Context) {
	conn, err := sm.wsUpgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	// Create session
	ctx, cancel := context.WithCancel(context.Background())
	session := &StreamSession{
		ID:           fmt.Sprintf("ws_%d", time.Now().UnixNano()),
		Type:         "websocket",
		ClientID:     c.ClientIP(),
		Created:      time.Now(),
		LastActivity: time.Now(),
		IsActive:     true,
		conn:         conn,
		context:      ctx,
		cancel:       cancel,
		Metadata:     make(map[string]interface{}),
	}

	// Register session
	sm.mu.Lock()
	sm.activeStreams[session.ID] = session
	sm.metrics.ActiveConnections++
	sm.metrics.TotalConnections++
	sm.mu.Unlock()

	// Start ping/pong handler
	go sm.handlePingPong(session)

	// Handle messages
	sm.handleWebSocketMessages(session)

	// Cleanup on disconnect
	sm.removeSession(session.ID)
}

// handleWebSocketMessages handles incoming WebSocket messages
func (sm *StreamingManager) handleWebSocketMessages(session *StreamSession) {
	for {
		messageType, data, err := session.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		session.LastActivity = time.Now()
		session.BytesReceived += int64(len(data))
		session.MessageCount++

		// Process message based on type
		if messageType == websocket.TextMessage {
			sm.processStreamMessage(session, data)
		} else if messageType == websocket.BinaryMessage {
			sm.processStreamBinary(session, data)
		}
	}
}

// processStreamMessage processes text stream messages
func (sm *StreamingManager) processStreamMessage(session *StreamSession, data []byte) {
	var req StreamRequest
	if err := json.Unmarshal(data, &req); err != nil {
		sm.sendError(session, fmt.Errorf("invalid request: %v", err))
		return
	}

	// Create chunks from data
	chunks := sm.createChunks(req.Data, req.ChunkSize, session.ID)
	
	// Process chunks
	for _, chunk := range chunks {
		sm.chunkProcessor.inputQueue <- chunk
	}

	// Send processed chunks back
	go sm.sendProcessedChunks(session)
}

// processStreamBinary processes binary stream data
func (sm *StreamingManager) processStreamBinary(session *StreamSession, data []byte) {
	// Process binary data (e.g., for file uploads, media streaming)
	chunk := &StreamChunk{
		ID:        fmt.Sprintf("chunk_%d", time.Now().UnixNano()),
		SessionID: session.ID,
		Data:      data,
		Size:      len(data),
		Timestamp: time.Now(),
	}

	sm.chunkProcessor.inputQueue <- chunk
}

// sendProcessedChunks sends processed chunks back to client
func (sm *StreamingManager) sendProcessedChunks(session *StreamSession) {
	for {
		select {
		case chunk := <-sm.chunkProcessor.outputQueue:
			if chunk.SessionID != session.ID {
				continue
			}

			response := StreamResponse{
				SessionID: session.ID,
				ChunkID:   chunk.ID,
				Index:     chunk.Index,
				Data:      chunk.Data,
				IsLast:    chunk.IsLast,
				Timestamp: chunk.Timestamp,
				Metadata:  chunk.ProcessedData,
			}

			data, _ := json.Marshal(response)
			if err := session.conn.WriteMessage(websocket.TextMessage, data); err != nil {
				log.Printf("Error sending chunk: %v", err)
				return
			}

			session.BytesSent += int64(len(data))

			if chunk.IsLast {
				return
			}

		case <-session.context.Done():
			return
		}
	}
}

// Server-Sent Events (SSE) Streaming

// HandleSSEStream handles Server-Sent Events streaming
func (sm *StreamingManager) HandleSSEStream(c *gin.Context) {
	// Set headers for SSE
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	// Get writer and flusher
	writer := c.Writer
	flusher, ok := writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Streaming not supported"})
		return
	}

	// Create SSE client
	client := &SSEClient{
		ID:          fmt.Sprintf("sse_%d", time.Now().UnixNano()),
		Writer:      writer,
		Flusher:     flusher,
		MessageChan: make(chan []byte, 10),
		CloseChan:   make(chan bool),
		LastPing:    time.Now(),
	}

	// Register client
	sm.mu.Lock()
	sm.sseClients[client.ID] = client
	sm.metrics.ActiveConnections++
	sm.mu.Unlock()

	// Send initial connection event
	fmt.Fprintf(writer, "event: connected\ndata: {\"client_id\": \"%s\"}\n\n", client.ID)
	flusher.Flush()

	// Handle SSE streaming
	sm.handleSSEClient(client)

	// Cleanup
	sm.removeSSEClient(client.ID)
}

// handleSSEClient handles SSE client messages
func (sm *StreamingManager) handleSSEClient(client *SSEClient) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case message := <-client.MessageChan:
			fmt.Fprintf(client.Writer, "data: %s\n\n", message)
			client.Flusher.Flush()
			client.MessagesSent++
			sm.metrics.BytesStreamed += int64(len(message))

		case <-ticker.C:
			// Send ping to keep connection alive
			fmt.Fprintf(client.Writer, "event: ping\ndata: {\"timestamp\": \"%s\"}\n\n", 
				time.Now().Format(time.RFC3339))
			client.Flusher.Flush()
			client.LastPing = time.Now()

		case <-client.CloseChan:
			return
		}
	}
}

// SendSSEMessage sends a message to an SSE client
func (sm *StreamingManager) SendSSEMessage(clientID string, data interface{}) error {
	sm.mu.RLock()
	client, exists := sm.sseClients[clientID]
	sm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("client not found: %s", clientID)
	}

	message, err := json.Marshal(data)
	if err != nil {
		return err
	}

	select {
	case client.MessageChan <- message:
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("timeout sending message to client")
	}
}

// HTTP Chunked Transfer Encoding

// HandleChunkedStream handles HTTP chunked transfer encoding
func (sm *StreamingManager) HandleChunkedStream(c *gin.Context) {
	c.Header("Transfer-Encoding", "chunked")
	c.Header("Content-Type", "application/json")

	writer := c.Writer
	flusher, ok := writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Chunked transfer not supported"})
		return
	}

	// Read request body
	var req StreamRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Create session
	session := &StreamSession{
		ID:           fmt.Sprintf("http_chunk_%d", time.Now().UnixNano()),
		Type:         "http-chunk",
		ClientID:     c.ClientIP(),
		Created:      time.Now(),
		LastActivity: time.Now(),
		IsActive:     true,
		sseWriter:    writer,
		Metadata:     make(map[string]interface{}),
	}

	// Process and stream chunks
	chunks := sm.createChunks(req.Data, req.ChunkSize, session.ID)
	
	for _, chunk := range chunks {
		// Process chunk
		processed := sm.processChunk(chunk)
		
		// Send chunk
		response := StreamResponse{
			SessionID: session.ID,
			ChunkID:   chunk.ID,
			Index:     chunk.Index,
			Data:      processed.Data,
			IsLast:    chunk.IsLast,
			Timestamp: chunk.Timestamp,
		}

		data, _ := json.Marshal(response)
		writer.Write(data)
		writer.Write([]byte("\n"))
		flusher.Flush()

		session.BytesSent += int64(len(data))
	}
}

// Chunk Processing

// createChunks creates chunks from data
func (sm *StreamingManager) createChunks(data json.RawMessage, chunkSize int, sessionID string) []*StreamChunk {
	if chunkSize == 0 {
		chunkSize = 4096 // Default 4KB chunks
	}

	dataBytes := []byte(data)
	totalSize := len(dataBytes)
	chunks := make([]*StreamChunk, 0)
	
	for i := 0; i < totalSize; i += chunkSize {
		end := i + chunkSize
		if end > totalSize {
			end = totalSize
		}

		chunk := &StreamChunk{
			ID:        fmt.Sprintf("chunk_%d_%d", time.Now().UnixNano(), i/chunkSize),
			SessionID: sessionID,
			Index:     i / chunkSize,
			Data:      dataBytes[i:end],
			Size:      end - i,
			Timestamp: time.Now(),
			IsLast:    end >= totalSize,
		}

		chunks = append(chunks, chunk)
	}

	return chunks
}

// processChunk processes a single chunk
func (sm *StreamingManager) processChunk(chunk *StreamChunk) *ProcessedChunk {
	startTime := time.Now()
	
	// Simulate processing (in production, apply actual transformations)
	processed := &ProcessedChunk{
		StreamChunk: *chunk,
		ProcessTime: time.Since(startTime),
	}

	// Parse JSON if possible
	if simdParser != nil {
		result, err := simdParser.ParseJSON(chunk.Data)
		if err == nil {
			processed.ProcessedData = result.Data
		} else {
			processed.Error = err.Error()
		}
	}

	sm.metrics.mu.Lock()
	sm.metrics.ChunksProcessed++
	sm.metrics.mu.Unlock()

	return processed
}

// ChunkStreamProcessor implementation

// start starts the chunk processor workers
func (csp *ChunkStreamProcessor) start() {
	for i := 0; i < csp.workers; i++ {
		csp.wg.Add(1)
		go csp.worker()
	}
}

// worker processes chunks from the input queue
func (csp *ChunkStreamProcessor) worker() {
	defer csp.wg.Done()

	for chunk := range csp.inputQueue {
		processed := streamManager.processChunk(chunk)
		csp.outputQueue <- processed
	}
}

// stop stops the chunk processor
func (csp *ChunkStreamProcessor) stop() {
	close(csp.inputQueue)
	csp.wg.Wait()
	close(csp.outputQueue)
	close(csp.errorQueue)
}

// Utility Functions

// handlePingPong handles WebSocket ping/pong for connection health
func (sm *StreamingManager) handlePingPong(session *StreamSession) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	session.conn.SetPongHandler(func(string) error {
		session.LastActivity = time.Now()
		return nil
	})

	for {
		select {
		case <-ticker.C:
			if err := session.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		case <-session.context.Done():
			return
		}
	}
}

// sendError sends an error message to the client
func (sm *StreamingManager) sendError(session *StreamSession, err error) {
	errorMsg := map[string]interface{}{
		"error":     err.Error(),
		"timestamp": time.Now(),
	}

	data, _ := json.Marshal(errorMsg)
	
	if session.Type == "websocket" && session.conn != nil {
		session.conn.WriteMessage(websocket.TextMessage, data)
	}
}

// removeSession removes and cleans up a session
func (sm *StreamingManager) removeSession(sessionID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if session, exists := sm.activeStreams[sessionID]; exists {
		session.IsActive = false
		if session.cancel != nil {
			session.cancel()
		}
		delete(sm.activeStreams, sessionID)
		sm.metrics.ActiveConnections--
	}
}

// removeSSEClient removes an SSE client
func (sm *StreamingManager) removeSSEClient(clientID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if client, exists := sm.sseClients[clientID]; exists {
		close(client.CloseChan)
		close(client.MessageChan)
		delete(sm.sseClients, clientID)
		sm.metrics.ActiveConnections--
	}
}

// BroadcastToAll broadcasts a message to all active connections
func (sm *StreamingManager) BroadcastToAll(data interface{}) {
	message, err := json.Marshal(data)
	if err != nil {
		log.Printf("Error marshaling broadcast message: %v", err)
		return
	}

	sm.mu.RLock()
	defer sm.mu.RUnlock()

	// Broadcast to WebSocket clients
	for _, session := range sm.activeStreams {
		if session.Type == "websocket" && session.conn != nil {
			session.conn.WriteMessage(websocket.TextMessage, message)
		}
	}

	// Broadcast to SSE clients
	for _, client := range sm.sseClients {
		select {
		case client.MessageChan <- message:
		default:
			// Client buffer full, skip
		}
	}
}

// GetStreamMetrics returns current streaming metrics
func (sm *StreamingManager) GetStreamMetrics() *StreamMetrics {
	sm.metrics.mu.RLock()
	defer sm.metrics.mu.RUnlock()
	return sm.metrics
}

// updateMetrics updates streaming metrics periodically
func (sm *StreamingManager) updateMetrics() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		sm.metrics.mu.Lock()
		sm.metrics.LastUpdate = time.Now()
		sm.metrics.mu.Unlock()
	}
}

// StreamFile streams a file in chunks
func (sm *StreamingManager) StreamFile(c *gin.Context, filePath string, chunkSize int) error {
	file, err := c.File(filePath)
	if err != nil {
		return err
	}

	// Set headers for streaming
	c.Header("Content-Type", "application/octet-stream")
	c.Header("Transfer-Encoding", "chunked")

	reader := bufio.NewReader(file)
	buffer := make([]byte, chunkSize)
	
	for {
		n, err := reader.Read(buffer)
		if n > 0 {
			c.Writer.Write(buffer[:n])
			c.Writer.Flush()
		}
		
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}
	}

	return nil
}
