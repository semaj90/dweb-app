package server

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/quic-go/quic-go"
	"github.com/go-kratos/kratos/v2/log"
)

// QUICCoordinator manages ultra-low latency communication
type QUICCoordinator struct {
	listener     *quic.Listener
	connections  sync.Map // map[string]*QUICConnection
	handlers     sync.Map // map[string]StreamHandler
	logger       log.Logger
	config       *QUICConfig
	shutdownChan chan struct{}
}

// QUICConfig holds QUIC server configuration
type QUICConfig struct {
	Address         string
	Port            int
	MaxStreams      int
	IdleTimeout     time.Duration
	HandshakeTimeout time.Duration
	CertFile        string
	KeyFile         string
	EnableMetrics   bool
}

// QUICConnection represents a client connection
type QUICConnection struct {
	conn       quic.Connection
	clientID   string
	lastSeen   time.Time
	streams    sync.Map
	metrics    *ConnectionMetrics
}

// ConnectionMetrics tracks connection performance
type ConnectionMetrics struct {
	StreamsOpened   int64
	StreamsClosed   int64
	BytesSent      int64
	BytesReceived  int64
	LastLatency    time.Duration
	AvgLatency     time.Duration
	ErrorCount     int64
}

// StreamHandler defines the interface for stream processing
type StreamHandler func(ctx context.Context, stream quic.Stream, conn *QUICConnection) error

// LegalAIMessage represents messages in the legal AI system
type LegalAIMessage struct {
	Type      string                 `json:"type"`
	ID        string                 `json:"id"`
	Source    string                 `json:"source"`
	Target    string                 `json:"target"`
	Timestamp time.Time              `json:"timestamp"`
	Priority  int                    `json:"priority"`
	Payload   map[string]interface{} `json:"payload"`
}

// NewQUICCoordinator creates a new QUIC coordinator
func NewQUICCoordinator(config *QUICConfig, logger log.Logger) (*QUICCoordinator, error) {
	if config == nil {
		config = &QUICConfig{
			Address:          "0.0.0.0",
			Port:            9443,
			MaxStreams:      1000,
			IdleTimeout:     30 * time.Second,
			HandshakeTimeout: 10 * time.Second,
			EnableMetrics:   true,
		}
	}

	coordinator := &QUICCoordinator{
		logger:       logger,
		config:       config,
		shutdownChan: make(chan struct{}),
	}

	// Setup default handlers
	coordinator.setupDefaultHandlers()

	return coordinator, nil
}

// Start initializes and starts the QUIC server
func (qc *QUICCoordinator) Start() error {
	// Generate TLS config
	tlsConfig, err := qc.generateTLSConfig()
	if err != nil {
		return fmt.Errorf("failed to generate TLS config: %w", err)
	}

	// Configure QUIC
	quicConfig := &quic.Config{
		MaxIdleTimeout:        qc.config.IdleTimeout,
		MaxIncomingStreams:    int64(qc.config.MaxStreams),
		MaxIncomingUniStreams: int64(qc.config.MaxStreams / 2),
		HandshakeIdleTimeout:  qc.config.HandshakeTimeout,
		KeepAlivePeriod:      15 * time.Second,
		EnableDatagrams:      true,
	}

	// Create listener
	addr := fmt.Sprintf("%s:%d", qc.config.Address, qc.config.Port)
	listener, err := quic.ListenAddr(addr, tlsConfig, quicConfig)
	if err != nil {
		return fmt.Errorf("failed to start QUIC listener: %w", err)
	}

	qc.listener = listener
	qc.logger.Log(log.LevelInfo, "msg", "QUIC coordinator started", "addr", addr)

	// Start accepting connections
	go qc.acceptConnections()

	// Start metrics collection if enabled
	if qc.config.EnableMetrics {
		go qc.startMetricsCollection()
	}

	return nil
}

// acceptConnections handles incoming QUIC connections
func (qc *QUICCoordinator) acceptConnections() {
	for {
		select {
		case <-qc.shutdownChan:
			return
		default:
			conn, err := qc.listener.Accept(context.Background())
			if err != nil {
				qc.logger.Log(log.LevelError, "msg", "Failed to accept connection", "error", err)
				continue
			}

			// Handle connection in goroutine
			go qc.handleConnection(conn)
		}
	}
}

// handleConnection manages a single QUIC connection
func (qc *QUICCoordinator) handleConnection(conn quic.Connection) {
	clientID := qc.generateClientID(conn)
	
	qconn := &QUICConnection{
		conn:     conn,
		clientID: clientID,
		lastSeen: time.Now(),
		metrics:  &ConnectionMetrics{},
	}

	qc.connections.Store(clientID, qconn)
	qc.logger.Log(log.LevelInfo, "msg", "New QUIC connection", "client_id", clientID, "remote_addr", conn.RemoteAddr())

	// Handle streams for this connection
	go qc.handleStreams(qconn)

	// Wait for connection to close
	<-conn.Context().Done()
	
	qc.connections.Delete(clientID)
	qc.logger.Log(log.LevelInfo, "msg", "QUIC connection closed", "client_id", clientID)
}

// handleStreams processes streams for a connection
func (qc *QUICCoordinator) handleStreams(qconn *QUICConnection) {
	for {
		select {
		case <-qconn.conn.Context().Done():
			return
		default:
			stream, err := qconn.conn.AcceptStream(context.Background())
			if err != nil {
				qc.logger.Log(log.LevelError, "msg", "Failed to accept stream", "error", err)
				continue
			}

			// Handle stream in goroutine
			go qc.handleStream(stream, qconn)
		}
	}
}

// handleStream processes a single stream
func (qc *QUICCoordinator) handleStream(stream quic.Stream, qconn *QUICConnection) {
	defer stream.Close()

	startTime := time.Now()
	qconn.metrics.StreamsOpened++

	// Read message type first
	msgType, err := qc.readMessageType(stream)
	if err != nil {
		qc.logger.Log(log.LevelError, "msg", "Failed to read message type", "error", err)
		qconn.metrics.ErrorCount++
		return
	}

	// Get handler for message type
	handlerFunc, exists := qc.handlers.Load(msgType)
	if !exists {
		qc.logger.Log(log.LevelWarn, "msg", "No handler for message type", "message_type", msgType)
		qc.sendError(stream, fmt.Sprintf("Unknown message type: %s", msgType))
		return
	}

	handler := handlerFunc.(StreamHandler)

	// Execute handler with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err = handler(ctx, stream, qconn)
	if err != nil {
		qc.logger.Log(log.LevelError, "msg", "Stream handler error", "error", err)
		qconn.metrics.ErrorCount++
		qc.sendError(stream, err.Error())
		return
	}

	// Update metrics
	duration := time.Since(startTime)
	qconn.metrics.LastLatency = duration
	qconn.metrics.AvgLatency = (qconn.metrics.AvgLatency + duration) / 2
	qconn.metrics.StreamsClosed++
	qconn.lastSeen = time.Now()
}

// setupDefaultHandlers configures built-in stream handlers
func (qc *QUICCoordinator) setupDefaultHandlers() {
	// Document processing handler
	qc.RegisterHandler("document-process", qc.handleDocumentProcessing)
	
	// Vector search handler
	qc.RegisterHandler("vector-search", qc.handleVectorSearch)
	
	// Real-time analysis handler
	qc.RegisterHandler("realtime-analysis", qc.handleRealtimeAnalysis)
	
	// Bulk operations handler
	qc.RegisterHandler("bulk-operation", qc.handleBulkOperation)
	
	// Health check handler
	qc.RegisterHandler("health-check", qc.handleHealthCheck)
	
	// Metrics handler
	qc.RegisterHandler("metrics", qc.handleMetricsRequest)
}

// RegisterHandler registers a stream handler for a message type
func (qc *QUICCoordinator) RegisterHandler(msgType string, handler StreamHandler) {
	qc.handlers.Store(msgType, handler)
	qc.logger.Log(log.LevelInfo, "msg", "Registered QUIC handler", "message_type", msgType)
}

// handleDocumentProcessing processes legal documents via QUIC
func (qc *QUICCoordinator) handleDocumentProcessing(ctx context.Context, stream quic.Stream, conn *QUICConnection) error {
	// Read document data
	data, err := qc.readStreamData(stream)
	if err != nil {
		return fmt.Errorf("failed to read document data: %w", err)
	}

	var docRequest struct {
		DocumentID string `json:"document_id"`
		Content    string `json:"content"`
		Metadata   map[string]interface{} `json:"metadata"`
		Priority   int    `json:"priority"`
	}

	if err := json.Unmarshal(data, &docRequest); err != nil {
		return fmt.Errorf("failed to parse document request: %w", err)
	}

	qc.logger.Log(log.LevelInfo, "msg", "Processing document via QUIC", "document_id", docRequest.DocumentID, "client_id", conn.clientID)

	// Simulate document processing (replace with actual legal AI processing)
	result := map[string]interface{}{
		"document_id": docRequest.DocumentID,
		"status":      "processed",
		"analysis": map[string]interface{}{
			"document_type": "contract",
			"risk_level":   "medium",
			"confidence":   0.85,
			"processing_time": time.Since(time.Now()).Milliseconds(),
		},
		"timestamp": time.Now(),
		"processed_by": "quic-coordinator",
	}

	// Send response
	return qc.sendStreamResponse(stream, result)
}

// handleVectorSearch performs vector similarity search via QUIC
func (qc *QUICCoordinator) handleVectorSearch(ctx context.Context, stream quic.Stream, conn *QUICConnection) error {
	data, err := qc.readStreamData(stream)
	if err != nil {
		return fmt.Errorf("failed to read search data: %w", err)
	}

	var searchRequest struct {
		Query     string  `json:"query"`
		Limit     int     `json:"limit"`
		Threshold float64 `json:"threshold"`
		Filters   map[string]interface{} `json:"filters"`
	}

	if err := json.Unmarshal(data, &searchRequest); err != nil {
		return fmt.Errorf("failed to parse search request: %w", err)
	}

	qc.logger.Log(log.LevelInfo, "msg", "Vector search via QUIC", "query", searchRequest.Query)

	// Simulate vector search (replace with actual Qdrant/PGVector search)
	results := map[string]interface{}{
		"query": searchRequest.Query,
		"results": []map[string]interface{}{
			{
				"document_id": "doc_123",
				"score":      0.92,
				"title":      "Legal Contract Analysis",
				"snippet":    "Relevant contract clause...",
			},
			{
				"document_id": "doc_456",
				"score":      0.87,
				"title":      "Compliance Guidelines",
				"snippet":    "Regulatory requirements...",
			},
		},
		"total_results": 2,
		"search_time": time.Since(time.Now()).Milliseconds(),
		"timestamp": time.Now(),
	}

	return qc.sendStreamResponse(stream, results)
}

// handleRealtimeAnalysis provides real-time legal analysis via QUIC
func (qc *QUICCoordinator) handleRealtimeAnalysis(ctx context.Context, stream quic.Stream, conn *QUICConnection) error {
	data, err := qc.readStreamData(stream)
	if err != nil {
		return fmt.Errorf("failed to read analysis data: %w", err)
	}

	var analysisRequest struct {
		Text        string `json:"text"`
		AnalysisType string `json:"analysis_type"`
		Streaming   bool   `json:"streaming"`
	}

	if err := json.Unmarshal(data, &analysisRequest); err != nil {
		return fmt.Errorf("failed to parse analysis request: %w", err)
	}

	qc.logger.Log(log.LevelInfo, "msg", "Real-time analysis via QUIC", "analysis_type", analysisRequest.AnalysisType)

	if analysisRequest.Streaming {
		// Stream analysis results as they become available
		return qc.streamAnalysisResults(stream, analysisRequest.Text)
	}

	// Single response analysis
	result := map[string]interface{}{
		"analysis_type": analysisRequest.AnalysisType,
		"results": map[string]interface{}{
			"sentiment":    "neutral",
			"risk_factors": []string{"clause_ambiguity", "jurisdiction_complexity"},
			"confidence":   0.78,
			"recommendations": []string{
				"Review section 4.2 for clarity",
				"Clarify termination conditions",
			},
		},
		"processing_time": time.Since(time.Now()).Milliseconds(),
		"timestamp": time.Now(),
	}

	return qc.sendStreamResponse(stream, result)
}

// streamAnalysisResults streams real-time analysis updates
func (qc *QUICCoordinator) streamAnalysisResults(stream quic.Stream, text string) error {
	// Simulate streaming analysis (replace with actual AI processing)
	updates := []map[string]interface{}{
		{"step": "tokenization", "progress": 0.2, "status": "processing"},
		{"step": "classification", "progress": 0.4, "status": "processing"},
		{"step": "risk_analysis", "progress": 0.6, "status": "processing"},
		{"step": "recommendations", "progress": 0.8, "status": "processing"},
		{"step": "complete", "progress": 1.0, "status": "completed", "results": map[string]interface{}{
			"document_type": "contract",
			"risk_level":   "medium",
			"confidence":   0.85,
		}},
	}

	for _, update := range updates {
		update["timestamp"] = time.Now()
		
		data, err := json.Marshal(update)
		if err != nil {
			return err
		}

		if err := qc.writeStreamData(stream, data); err != nil {
			return err
		}

		// Simulate processing time
		time.Sleep(100 * time.Millisecond)
	}

	return nil
}

// handleBulkOperation processes bulk operations via QUIC
func (qc *QUICCoordinator) handleBulkOperation(ctx context.Context, stream quic.Stream, conn *QUICConnection) error {
	data, err := qc.readStreamData(stream)
	if err != nil {
		return fmt.Errorf("failed to read bulk data: %w", err)
	}

	var bulkRequest struct {
		Operation string                   `json:"operation"`
		Items     []map[string]interface{} `json:"items"`
		BatchSize int                      `json:"batch_size"`
	}

	if err := json.Unmarshal(data, &bulkRequest); err != nil {
		return fmt.Errorf("failed to parse bulk request: %w", err)
	}

	qc.logger.Log(log.LevelInfo, "msg", "Bulk operation via QUIC", "operation", bulkRequest.Operation, "item_count", len(bulkRequest.Items))

	// Process items in batches
	batchSize := bulkRequest.BatchSize
	if batchSize <= 0 {
		batchSize = 10
	}

	results := make([]map[string]interface{}, 0, len(bulkRequest.Items))

	for i := 0; i < len(bulkRequest.Items); i += batchSize {
		end := i + batchSize
		if end > len(bulkRequest.Items) {
			end = len(bulkRequest.Items)
		}

		batch := bulkRequest.Items[i:end]
		batchResults := qc.processBatch(bulkRequest.Operation, batch)
		results = append(results, batchResults...)

		// Send progress update
		progress := map[string]interface{}{
			"type":     "progress",
			"progress": float64(end) / float64(len(bulkRequest.Items)),
			"processed": end,
			"total":    len(bulkRequest.Items),
		}

		progressData, _ := json.Marshal(progress)
		qc.writeStreamData(stream, progressData)
	}

	// Send final results
	finalResult := map[string]interface{}{
		"type":        "complete",
		"operation":   bulkRequest.Operation,
		"total_items": len(bulkRequest.Items),
		"results":     results,
		"timestamp":   time.Now(),
	}

	return qc.sendStreamResponse(stream, finalResult)
}

// processBatch simulates batch processing
func (qc *QUICCoordinator) processBatch(operation string, items []map[string]interface{}) []map[string]interface{} {
	results := make([]map[string]interface{}, len(items))
	
	for i, item := range items {
		results[i] = map[string]interface{}{
			"id":        item["id"],
			"operation": operation,
			"status":    "completed",
			"result":    fmt.Sprintf("Processed %s", operation),
		}
	}
	
	return results
}

// handleHealthCheck responds to health check requests
func (qc *QUICCoordinator) handleHealthCheck(ctx context.Context, stream quic.Stream, conn *QUICConnection) error {
	health := map[string]interface{}{
		"status":       "healthy",
		"connections":  qc.getConnectionCount(),
		"uptime":      time.Since(time.Now()).Seconds(),
		"version":     "1.0.0",
		"timestamp":   time.Now(),
	}

	return qc.sendStreamResponse(stream, health)
}

// handleMetricsRequest returns performance metrics
func (qc *QUICCoordinator) handleMetricsRequest(ctx context.Context, stream quic.Stream, conn *QUICConnection) error {
	metrics := qc.collectMetrics()
	return qc.sendStreamResponse(stream, metrics)
}

// Utility methods

func (qc *QUICCoordinator) readMessageType(stream quic.Stream) (string, error) {
	// Read first 4 bytes for length
	lengthBuf := make([]byte, 4)
	if _, err := io.ReadFull(stream, lengthBuf); err != nil {
		return "", err
	}

	// Read message type
	msgType := make([]byte, int(lengthBuf[0]))
	if _, err := io.ReadFull(stream, msgType); err != nil {
		return "", err
	}

	return string(msgType), nil
}

func (qc *QUICCoordinator) readStreamData(stream quic.Stream) ([]byte, error) {
	// Read length prefix (4 bytes)
	lengthBuf := make([]byte, 4)
	if _, err := io.ReadFull(stream, lengthBuf); err != nil {
		return nil, err
	}

	length := int(lengthBuf[0])<<24 | int(lengthBuf[1])<<16 | int(lengthBuf[2])<<8 | int(lengthBuf[3])

	// Read data
	data := make([]byte, length)
	if _, err := io.ReadFull(stream, data); err != nil {
		return nil, err
	}

	return data, nil
}

func (qc *QUICCoordinator) writeStreamData(stream quic.Stream, data []byte) error {
	// Write length prefix
	length := len(data)
	lengthBuf := []byte{
		byte(length >> 24),
		byte(length >> 16),
		byte(length >> 8),
		byte(length),
	}

	if _, err := stream.Write(lengthBuf); err != nil {
		return err
	}

	// Write data
	_, err := stream.Write(data)
	return err
}

func (qc *QUICCoordinator) sendStreamResponse(stream quic.Stream, response interface{}) error {
	data, err := json.Marshal(response)
	if err != nil {
		return err
	}

	return qc.writeStreamData(stream, data)
}

func (qc *QUICCoordinator) sendError(stream quic.Stream, errorMsg string) error {
	errorResponse := map[string]interface{}{
		"error":     errorMsg,
		"timestamp": time.Now(),
	}

	return qc.sendStreamResponse(stream, errorResponse)
}

func (qc *QUICCoordinator) generateClientID(conn quic.Connection) string {
	return fmt.Sprintf("client-%d-%s", time.Now().Unix(), conn.RemoteAddr().String())
}

func (qc *QUICCoordinator) generateTLSConfig() (*tls.Config, error) {
	if qc.config.CertFile != "" && qc.config.KeyFile != "" {
		cert, err := tls.LoadX509KeyPair(qc.config.CertFile, qc.config.KeyFile)
		if err != nil {
			return nil, err
		}
		
		return &tls.Config{
			Certificates: []tls.Certificate{cert},
			NextProtos:   []string{"legal-ai-quic"},
		}, nil
	}

	// Generate self-signed certificate for development
	return qc.generateSelfSignedTLS()
}

func (qc *QUICCoordinator) generateSelfSignedTLS() (*tls.Config, error) {
	// Simplified self-signed cert generation for development
	// In production, use proper certificates
	return &tls.Config{
		InsecureSkipVerify: true,
		NextProtos:        []string{"legal-ai-quic"},
	}, nil
}

func (qc *QUICCoordinator) getConnectionCount() int {
	count := 0
	qc.connections.Range(func(key, value interface{}) bool {
		count++
		return true
	})
	return count
}

func (qc *QUICCoordinator) collectMetrics() map[string]interface{} {
	totalStreams := int64(0)
	totalErrors := int64(0)
	avgLatency := time.Duration(0)
	connectionCount := 0

	qc.connections.Range(func(key, value interface{}) bool {
		conn := value.(*QUICConnection)
		totalStreams += conn.metrics.StreamsOpened
		totalErrors += conn.metrics.ErrorCount
		avgLatency += conn.metrics.AvgLatency
		connectionCount++
		return true
	})

	if connectionCount > 0 {
		avgLatency = avgLatency / time.Duration(connectionCount)
	}

	return map[string]interface{}{
		"total_connections": connectionCount,
		"total_streams":    totalStreams,
		"total_errors":     totalErrors,
		"avg_latency_ms":   avgLatency.Milliseconds(),
		"timestamp":        time.Now(),
	}
}

func (qc *QUICCoordinator) startMetricsCollection() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			metrics := qc.collectMetrics()
			qc.logger.Log(log.LevelInfo, "msg", "QUIC Metrics", "metrics", metrics)
		case <-qc.shutdownChan:
			return
		}
	}
}

// Shutdown gracefully shuts down the QUIC coordinator
func (qc *QUICCoordinator) Shutdown() error {
	close(qc.shutdownChan)
	
	if qc.listener != nil {
		return qc.listener.Close()
	}
	
	return nil
}