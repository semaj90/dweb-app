package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"

	"github.com/lucas-clemente/quic-go"
	"github.com/lucas-clemente/quic-go/http3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"google.golang.org/protobuf/proto"
)

// QUIC/gRPC/IPC Tensor Transport Layer
// Handles high-performance tensor data streaming between services

type TensorTransport struct {
	// QUIC server for high-performance streaming
	quicServer   *quic.Listener
	quicClients  map[string]*quic.Connection
	
	// gRPC server for typed service calls
	grpcServer   *grpc.Server
	grpcClients  map[string]*grpc.ClientConn
	
	// IPC for local inter-process communication
	ipcConnections map[string]*IPCConnection
	
	// Tensor processing
	tensorQueue   chan TensorOperation
	resultCache   map[string]*TensorResult
	
	// Configuration
	config        TransportConfig
	
	// Synchronization
	mu            sync.RWMutex
	activeStreams map[string]*TensorStream
	
	// Performance metrics
	metrics       TransportMetrics
}

type TransportConfig struct {
	QUICPort        string
	GRPCPort        string
	IPCSocketPath   string
	MaxStreams      int
	BufferSize      int
	CompressionLevel int
	EnableTLS       bool
	CertFile        string
	KeyFile         string
	
	// Tensor-specific settings
	MaxTensorSize   int64
	ChunkSize       int
	TimeoutSeconds  int
}

type TensorOperation struct {
	ID            string                 `json:"id"`
	Type          string                 `json:"type"` // "multiply", "add", "conv", "attention", "som"
	InputTensors  []Tensor               `json:"input_tensors"`
	Parameters    map[string]interface{} `json:"parameters"`
	Priority      int                    `json:"priority"`
	RequiresGPU   bool                   `json:"requires_gpu"`
	Timestamp     time.Time              `json:"timestamp"`
	StreamID      string                 `json:"stream_id"`
}

type Tensor struct {
	ID          string    `json:"id"`
	Shape       []int     `json:"shape"`
	DataType    string    `json:"data_type"` // "float32", "float64", "int32", "int64"
	Data        []byte    `json:"data"`
	Compressed  bool      `json:"compressed"`
	Metadata    map[string]interface{} `json:"metadata"`
	Checksum    string    `json:"checksum"`
}

type TensorResult struct {
	OperationID   string                 `json:"operation_id"`
	OutputTensors []Tensor               `json:"output_tensors"`
	Success       bool                   `json:"success"`
	Error         string                 `json:"error,omitempty"`
	ProcessingTime time.Duration         `json:"processing_time"`
	WorkerID      string                 `json:"worker_id"`
	GPUUsed       bool                   `json:"gpu_used"`
	Metadata      map[string]interface{} `json:"metadata"`
}

type TensorStream struct {
	ID            string
	Connection    interface{} // quic.Stream or grpc stream
	Type          string      // "quic", "grpc", "ipc"
	Active        bool
	LastActivity  time.Time
	BytesTransferred int64
	OperationsCount  int
}

type IPCConnection struct {
	Path     string
	Conn     net.Conn
	Active   bool
	LastUsed time.Time
}

type TransportMetrics struct {
	TotalOperations   int64     `json:"total_operations"`
	CompletedOps      int64     `json:"completed_operations"`
	FailedOps         int64     `json:"failed_operations"`
	BytesTransferred  int64     `json:"bytes_transferred"`
	AverageLatency    float64   `json:"average_latency_ms"`
	ThroughputMBps    float64   `json:"throughput_mbps"`
	ActiveStreams     int       `json:"active_streams"`
	QUICConnections   int       `json:"quic_connections"`
	GRPCConnections   int       `json:"grpc_connections"`
	IPCConnections    int       `json:"ipc_connections"`
	LastUpdated       time.Time `json:"last_updated"`
}

// Protobuf-style message definitions (simplified)
type TensorServiceRequest struct {
	Operation *TensorOperation `json:"operation"`
	StreamId  string           `json:"stream_id"`
}

type TensorServiceResponse struct {
	Result   *TensorResult `json:"result"`
	StreamId string        `json:"stream_id"`
	Done     bool          `json:"done"`
}

func NewTensorTransport() *TensorTransport {
	config := loadTransportConfig()
	
	return &TensorTransport{
		config:         config,
		quicClients:    make(map[string]*quic.Connection),
		grpcClients:    make(map[string]*grpc.ClientConn),
		ipcConnections: make(map[string]*IPCConnection),
		tensorQueue:    make(chan TensorOperation, 1000),
		resultCache:    make(map[string]*TensorResult),
		activeStreams:  make(map[string]*TensorStream),
	}
}

func loadTransportConfig() TransportConfig {
	return TransportConfig{
		QUICPort:         getEnv("QUIC_PORT", "8100"),
		GRPCPort:         getEnv("GRPC_PORT", "8101"),
		IPCSocketPath:    getEnv("IPC_SOCKET", "/tmp/tensor.sock"),
		MaxStreams:       getEnvInt("MAX_STREAMS", 100),
		BufferSize:       getEnvInt("BUFFER_SIZE", 1024*1024), // 1MB
		CompressionLevel: getEnvInt("COMPRESSION", 6),
		EnableTLS:        getEnvBool("ENABLE_TLS", true),
		CertFile:         getEnv("CERT_FILE", "cert.pem"),
		KeyFile:          getEnv("KEY_FILE", "key.pem"),
		MaxTensorSize:    getEnvInt64("MAX_TENSOR_SIZE", 100*1024*1024), // 100MB
		ChunkSize:        getEnvInt("CHUNK_SIZE", 64*1024), // 64KB
		TimeoutSeconds:   getEnvInt("TIMEOUT_SECONDS", 30),
	}
}

func (t *TensorTransport) Initialize() error {
	log.Println("🚀 Initializing QUIC/gRPC/IPC Tensor Transport...")
	
	// Start QUIC server
	if err := t.startQUICServer(); err != nil {
		return fmt.Errorf("QUIC server failed: %w", err)
	}
	
	// Start gRPC server
	if err := t.startGRPCServer(); err != nil {
		return fmt.Errorf("gRPC server failed: %w", err)
	}
	
	// Start IPC server
	if err := t.startIPCServer(); err != nil {
		return fmt.Errorf("IPC server failed: %w", err)
	}
	
	// Start tensor processor
	go t.processTensorQueue()
	
	// Start metrics collector
	go t.collectMetrics()
	
	// Start connection monitor
	go t.monitorConnections()
	
	log.Println("✅ Tensor Transport initialized")
	return nil
}

func (t *TensorTransport) startQUICServer() error {
	// Generate TLS config for QUIC
	tlsConfig := t.generateTLSConfig()
	
	// Create QUIC config
	quicConfig := &quic.Config{
		MaxIncomingStreams: int64(t.config.MaxStreams),
		KeepAlivePeriod:   30 * time.Second,
	}
	
	// Start QUIC listener
	listener, err := quic.ListenAddr(fmt.Sprintf(":%s", t.config.QUICPort), tlsConfig, quicConfig)
	if err != nil {
		return err
	}
	
	t.quicServer = listener
	
	// Accept connections
	go func() {
		for {
			conn, err := listener.Accept(context.Background())
			if err != nil {
				log.Printf("QUIC accept error: %v", err)
				continue
			}
			
			go t.handleQUICConnection(conn)
		}
	}()
	
	log.Printf("✅ QUIC server listening on port %s", t.config.QUICPort)
	return nil
}

func (t *TensorTransport) startGRPCServer() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", t.config.GRPCPort))
	if err != nil {
		return err
	}
	
	// Create gRPC server with options
	var opts []grpc.ServerOption
	
	if t.config.EnableTLS {
		// Add TLS credentials if enabled
		creds, err := credentials.NewServerTLSFromFile(t.config.CertFile, t.config.KeyFile)
		if err != nil {
			log.Printf("TLS credentials failed, using insecure: %v", err)
		} else {
			opts = append(opts, grpc.Creds(creds))
		}
	}
	
	// Configure buffer sizes and compression
	opts = append(opts,
		grpc.MaxRecvMsgSize(int(t.config.MaxTensorSize)),
		grpc.MaxSendMsgSize(int(t.config.MaxTensorSize)),
	)
	
	t.grpcServer = grpc.NewServer(opts...)
	
	// Register tensor service
	// RegisterTensorServiceServer(t.grpcServer, &TensorServiceImpl{transport: t})
	
	// Enable reflection for debugging
	reflection.Register(t.grpcServer)
	
	go func() {
		if err := t.grpcServer.Serve(lis); err != nil {
			log.Printf("gRPC server error: %v", err)
		}
	}()
	
	log.Printf("✅ gRPC server listening on port %s", t.config.GRPCPort)
	return nil
}

func (t *TensorTransport) startIPCServer() error {
	// Remove existing socket if it exists
	os.Remove(t.config.IPCSocketPath)
	
	listener, err := net.Listen("unix", t.config.IPCSocketPath)
	if err != nil {
		return err
	}
	
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				log.Printf("IPC accept error: %v", err)
				continue
			}
			
			go t.handleIPCConnection(conn)
		}
	}()
	
	log.Printf("✅ IPC server listening on %s", t.config.IPCSocketPath)
	return nil
}

func (t *TensorTransport) handleQUICConnection(conn quic.Connection) {
	connID := fmt.Sprintf("quic_%s", conn.RemoteAddr().String())
	t.mu.Lock()
	t.quicClients[connID] = &conn
	t.mu.Unlock()
	
	log.Printf("🔗 New QUIC connection: %s", connID)
	
	defer func() {
		t.mu.Lock()
		delete(t.quicClients, connID)
		t.mu.Unlock()
		conn.CloseWithError(0, "server shutdown")
	}()
	
	// Handle streams
	for {
		stream, err := conn.AcceptStream(context.Background())
		if err != nil {
			break
		}
		
		streamID := fmt.Sprintf("%s_stream_%d", connID, stream.StreamID())
		t.activeStreams[streamID] = &TensorStream{
			ID:           streamID,
			Connection:   stream,
			Type:         "quic",
			Active:       true,
			LastActivity: time.Now(),
		}
		
		go t.handleTensorStream(stream, streamID)
	}
}

func (t *TensorTransport) handleIPCConnection(conn net.Conn) {
	connID := fmt.Sprintf("ipc_%d", time.Now().UnixNano())
	
	t.mu.Lock()
	t.ipcConnections[connID] = &IPCConnection{
		Path:     t.config.IPCSocketPath,
		Conn:     conn,
		Active:   true,
		LastUsed: time.Now(),
	}
	t.mu.Unlock()
	
	defer func() {
		t.mu.Lock()
		delete(t.ipcConnections, connID)
		t.mu.Unlock()
		conn.Close()
	}()
	
	log.Printf("🔗 New IPC connection: %s", connID)
	
	// Handle IPC messages
	decoder := json.NewDecoder(conn)
	encoder := json.NewEncoder(conn)
	
	for {
		var req TensorServiceRequest
		if err := decoder.Decode(&req); err != nil {
			if err != io.EOF {
				log.Printf("IPC decode error: %v", err)
			}
			break
		}
		
		// Process tensor operation
		result := t.processTensorOperation(*req.Operation)
		
		response := TensorServiceResponse{
			Result:   &result,
			StreamId: req.StreamId,
			Done:     true,
		}
		
		if err := encoder.Encode(response); err != nil {
			log.Printf("IPC encode error: %v", err)
			break
		}
		
		t.mu.Lock()
		if ipc, exists := t.ipcConnections[connID]; exists {
			ipc.LastUsed = time.Now()
		}
		t.mu.Unlock()
	}
}

func (t *TensorTransport) handleTensorStream(stream interface{}, streamID string) {
	defer func() {
		t.mu.Lock()
		delete(t.activeStreams, streamID)
		t.mu.Unlock()
	}()
	
	switch s := stream.(type) {
	case quic.Stream:
		t.handleQUICTensorStream(s, streamID)
	case grpc.ServerStream:
		t.handleGRPCTensorStream(s, streamID)
	}
}

func (t *TensorTransport) handleQUICTensorStream(stream quic.Stream, streamID string) {
	defer stream.Close()
	
	log.Printf("📡 Handling QUIC tensor stream: %s", streamID)
	
	// Read tensor operations from stream
	decoder := json.NewDecoder(stream)
	encoder := json.NewEncoder(stream)
	
	for {
		var operation TensorOperation
		if err := decoder.Decode(&operation); err != nil {
			if err != io.EOF {
				log.Printf("QUIC stream decode error: %v", err)
			}
			break
		}
		
		// Update stream activity
		t.mu.Lock()
		if streamInfo, exists := t.activeStreams[streamID]; exists {
			streamInfo.LastActivity = time.Now()
			streamInfo.OperationsCount++
		}
		t.mu.Unlock()
		
		// Process operation
		operation.StreamID = streamID
		result := t.processTensorOperation(operation)
		
		// Send result back
		response := TensorServiceResponse{
			Result:   &result,
			StreamId: streamID,
			Done:     true,
		}
		
		if err := encoder.Encode(response); err != nil {
			log.Printf("QUIC stream encode error: %v", err)
			break
		}
		
		// Update metrics
		t.metrics.BytesTransferred += int64(len(operation.InputTensors))
		t.metrics.CompletedOps++
	}
}

func (t *TensorTransport) handleGRPCTensorStream(stream grpc.ServerStream, streamID string) {
	log.Printf("📡 Handling gRPC tensor stream: %s", streamID)
	// Implementation for gRPC streaming would go here
}

func (t *TensorTransport) processTensorOperation(operation TensorOperation) TensorResult {
	start := time.Now()
	
	result := TensorResult{
		OperationID:    operation.ID,
		Success:        false,
		ProcessingTime: 0,
		WorkerID:       "main_worker",
		GPUUsed:        operation.RequiresGPU,
		Metadata:       make(map[string]interface{}),
	}
	
	defer func() {
		result.ProcessingTime = time.Since(start)
		t.metrics.TotalOperations++
		
		if result.Success {
			t.metrics.CompletedOps++
		} else {
			t.metrics.FailedOps++
		}
	}()
	
	// Cache check
	cacheKey := t.generateCacheKey(operation)
	t.mu.RLock()
	if cached, exists := t.resultCache[cacheKey]; exists {
		t.mu.RUnlock()
		cached.Metadata["cache_hit"] = true
		return *cached
	}
	t.mu.RUnlock()
	
	// Process based on operation type
	switch operation.Type {
	case "multiply":
		result = t.tensorMultiply(operation)
	case "add":
		result = t.tensorAdd(operation)
	case "conv":
		result = t.tensorConvolution(operation)
	case "attention":
		result = t.computeAttention(operation)
	case "som":
		result = t.computeSOM(operation)
	case "embedding":
		result = t.generateEmbedding(operation)
	default:
		result.Error = fmt.Sprintf("unknown operation type: %s", operation.Type)
		return result
	}
	
	// Cache successful results
	if result.Success && len(result.OutputTensors) > 0 {
		t.mu.Lock()
		t.resultCache[cacheKey] = &result
		t.mu.Unlock()
		
		// Cleanup old cache entries
		go t.cleanupCache()
	}
	
	return result
}

func (t *TensorTransport) processTensorQueue() {
	for operation := range t.tensorQueue {
		result := t.processTensorOperation(operation)
		
		// Send result to appropriate stream if needed
		if streamInfo, exists := t.activeStreams[operation.StreamID]; exists {
			t.sendResultToStream(streamInfo, result)
		}
	}
}

func (t *TensorTransport) sendResultToStream(stream *TensorStream, result TensorResult) {
	switch stream.Type {
	case "quic":
		if quicStream, ok := stream.Connection.(quic.Stream); ok {
			encoder := json.NewEncoder(quicStream)
			response := TensorServiceResponse{
				Result:   &result,
				StreamId: stream.ID,
				Done:     true,
			}
			encoder.Encode(response)
		}
	case "grpc":
		// Handle gRPC stream response
	case "ipc":
		// Handle IPC response
	}
}

// Tensor operation implementations (simplified mock implementations)
func (t *TensorTransport) tensorMultiply(operation TensorOperation) TensorResult {
	if len(operation.InputTensors) < 2 {
		return TensorResult{
			OperationID: operation.ID,
			Success:     false,
			Error:       "tensor multiply requires at least 2 input tensors",
		}
	}
	
	// Mock tensor multiplication
	tensor1 := operation.InputTensors[0]
	tensor2 := operation.InputTensors[1]
	
	// Create result tensor (simplified)
	resultTensor := Tensor{
		ID:       fmt.Sprintf("result_%s", operation.ID),
		Shape:    tensor1.Shape, // Simplified - actual would depend on operation
		DataType: tensor1.DataType,
		Data:     make([]byte, len(tensor1.Data)), // Mock result data
		Metadata: map[string]interface{}{
			"operation": "multiply",
			"input_shapes": [][]int{tensor1.Shape, tensor2.Shape},
		},
	}
	
	// Mock computation delay
	time.Sleep(10 * time.Millisecond)
	
	return TensorResult{
		OperationID:   operation.ID,
		OutputTensors: []Tensor{resultTensor},
		Success:       true,
		GPUUsed:       operation.RequiresGPU,
		Metadata: map[string]interface{}{
			"computation_type": "tensor_multiply",
			"flops": calculateFLOPs(tensor1.Shape, tensor2.Shape),
		},
	}
}

func (t *TensorTransport) tensorAdd(operation TensorOperation) TensorResult {
	if len(operation.InputTensors) < 2 {
		return TensorResult{
			OperationID: operation.ID,
			Success:     false,
			Error:       "tensor add requires at least 2 input tensors",
		}
	}
	
	tensor1 := operation.InputTensors[0]
	tensor2 := operation.InputTensors[1]
	
	resultTensor := Tensor{
		ID:       fmt.Sprintf("add_result_%s", operation.ID),
		Shape:    tensor1.Shape,
		DataType: tensor1.DataType,
		Data:     make([]byte, len(tensor1.Data)),
		Metadata: map[string]interface{}{
			"operation": "add",
			"broadcast": areBroadcastCompatible(tensor1.Shape, tensor2.Shape),
		},
	}
	
	time.Sleep(5 * time.Millisecond)
	
	return TensorResult{
		OperationID:   operation.ID,
		OutputTensors: []Tensor{resultTensor},
		Success:       true,
		GPUUsed:       operation.RequiresGPU,
	}
}

func (t *TensorTransport) tensorConvolution(operation TensorOperation) TensorResult {
	if len(operation.InputTensors) < 2 {
		return TensorResult{
			OperationID: operation.ID,
			Success:     false,
			Error:       "convolution requires input tensor and kernel",
		}
	}
	
	input := operation.InputTensors[0]
	kernel := operation.InputTensors[1]
	
	// Calculate output shape for convolution
	outputShape := calculateConvOutputShape(input.Shape, kernel.Shape, operation.Parameters)
	
	resultTensor := Tensor{
		ID:       fmt.Sprintf("conv_result_%s", operation.ID),
		Shape:    outputShape,
		DataType: input.DataType,
		Data:     make([]byte, calculateTensorSize(outputShape, input.DataType)),
		Metadata: map[string]interface{}{
			"operation": "convolution",
			"kernel_size": kernel.Shape,
			"stride": operation.Parameters["stride"],
			"padding": operation.Parameters["padding"],
		},
	}
	
	// Simulate GPU-accelerated convolution
	if operation.RequiresGPU {
		time.Sleep(20 * time.Millisecond) // GPU convolution
	} else {
		time.Sleep(100 * time.Millisecond) // CPU convolution
	}
	
	return TensorResult{
		OperationID:   operation.ID,
		OutputTensors: []Tensor{resultTensor},
		Success:       true,
		GPUUsed:       operation.RequiresGPU,
		Metadata: map[string]interface{}{
			"computation_type": "convolution",
			"output_shape": outputShape,
		},
	}
}

func (t *TensorTransport) computeAttention(operation TensorOperation) TensorResult {
	if len(operation.InputTensors) < 3 {
		return TensorResult{
			OperationID: operation.ID,
			Success:     false,
			Error:       "attention requires query, key, value tensors",
		}
	}
	
	query := operation.InputTensors[0]
	key := operation.InputTensors[1]
	value := operation.InputTensors[2]
	
	// Calculate attention output shape
	attentionShape := []int{query.Shape[0], query.Shape[1], value.Shape[2]}
	
	resultTensor := Tensor{
		ID:       fmt.Sprintf("attention_result_%s", operation.ID),
		Shape:    attentionShape,
		DataType: query.DataType,
		Data:     make([]byte, calculateTensorSize(attentionShape, query.DataType)),
		Metadata: map[string]interface{}{
			"operation": "attention",
			"head_dim": operation.Parameters["head_dim"],
			"num_heads": operation.Parameters["num_heads"],
		},
	}
	
	// Simulate attention computation
	time.Sleep(30 * time.Millisecond)
	
	return TensorResult{
		OperationID:   operation.ID,
		OutputTensors: []Tensor{resultTensor},
		Success:       true,
		GPUUsed:       true, // Attention typically uses GPU
		Metadata: map[string]interface{}{
			"computation_type": "multi_head_attention",
			"attention_weights": "computed",
		},
	}
}

func (t *TensorTransport) computeSOM(operation TensorOperation) TensorResult {
	if len(operation.InputTensors) < 1 {
		return TensorResult{
			OperationID: operation.ID,
			Success:     false,
			Error:       "SOM requires input data tensor",
		}
	}
	
	input := operation.InputTensors[0]
	
	// SOM parameters
	mapSize := getIntParam(operation.Parameters, "map_size", 10)
	dimensions := getIntParam(operation.Parameters, "dimensions", 2)
	
	// Create SOM weight matrix
	somShape := []int{mapSize, mapSize, input.Shape[len(input.Shape)-1]}
	
	resultTensor := Tensor{
		ID:       fmt.Sprintf("som_result_%s", operation.ID),
		Shape:    somShape,
		DataType: input.DataType,
		Data:     make([]byte, calculateTensorSize(somShape, input.DataType)),
		Metadata: map[string]interface{}{
			"operation": "som",
			"map_size": mapSize,
			"dimensions": dimensions,
			"training_iterations": operation.Parameters["iterations"],
		},
	}
	
	// Simulate SOM training
	time.Sleep(50 * time.Millisecond)
	
	return TensorResult{
		OperationID:   operation.ID,
		OutputTensors: []Tensor{resultTensor},
		Success:       true,
		GPUUsed:       operation.RequiresGPU,
		Metadata: map[string]interface{}{
			"computation_type": "self_organizing_map",
			"convergence": "achieved",
		},
	}
}

func (t *TensorTransport) generateEmbedding(operation TensorOperation) TensorResult {
	if len(operation.InputTensors) < 1 {
		return TensorResult{
			OperationID: operation.ID,
			Success:     false,
			Error:       "embedding requires input tensor",
		}
	}
	
	input := operation.InputTensors[0]
	embeddingDim := getIntParam(operation.Parameters, "embedding_dim", 384)
	
	embeddingShape := []int{input.Shape[0], embeddingDim}
	
	resultTensor := Tensor{
		ID:       fmt.Sprintf("embedding_result_%s", operation.ID),
		Shape:    embeddingShape,
		DataType: "float32",
		Data:     make([]byte, calculateTensorSize(embeddingShape, "float32")),
		Metadata: map[string]interface{}{
			"operation": "embedding",
			"model": operation.Parameters["model"],
			"embedding_dim": embeddingDim,
		},
	}
	
	// Simulate embedding generation
	time.Sleep(25 * time.Millisecond)
	
	return TensorResult{
		OperationID:   operation.ID,
		OutputTensors: []Tensor{resultTensor},
		Success:       true,
		GPUUsed:       operation.RequiresGPU,
		Metadata: map[string]interface{}{
			"computation_type": "embedding_generation",
			"vector_norm": "normalized",
		},
	}
}

// Client connection methods
func (t *TensorTransport) ConnectQUIC(address string) error {
	tlsConfig := &tls.Config{
		InsecureSkipVerify: true, // For development only
	}
	
	conn, err := quic.DialAddr(context.Background(), address, tlsConfig, nil)
	if err != nil {
		return err
	}
	
	t.mu.Lock()
	t.quicClients[address] = &conn
	t.mu.Unlock()
	
	log.Printf("🔗 Connected to QUIC server: %s", address)
	return nil
}

func (t *TensorTransport) ConnectGRPC(address string) error {
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithInsecure()) // For development
	
	conn, err := grpc.Dial(address, opts...)
	if err != nil {
		return err
	}
	
	t.mu.Lock()
	t.grpcClients[address] = conn
	t.mu.Unlock()
	
	log.Printf("🔗 Connected to gRPC server: %s", address)
	return nil
}

func (t *TensorTransport) SendTensorOperation(address string, operation TensorOperation, protocol string) (*TensorResult, error) {
	switch protocol {
	case "quic":
		return t.sendViaQUIC(address, operation)
	case "grpc":
		return t.sendViaGRPC(address, operation)
	case "ipc":
		return t.sendViaIPC(address, operation)
	default:
		return nil, fmt.Errorf("unsupported protocol: %s", protocol)
	}
}

func (t *TensorTransport) sendViaQUIC(address string, operation TensorOperation) (*TensorResult, error) {
	t.mu.RLock()
	conn, exists := t.quicClients[address]
	t.mu.RUnlock()
	
	if !exists {
		if err := t.ConnectQUIC(address); err != nil {
			return nil, err
		}
		t.mu.RLock()
		conn = t.quicClients[address]
		t.mu.RUnlock()
	}
	
	stream, err := (*conn).OpenStreamSync(context.Background())
	if err != nil {
		return nil, err
	}
	defer stream.Close()
	
	// Send operation
	encoder := json.NewEncoder(stream)
	if err := encoder.Encode(operation); err != nil {
		return nil, err
	}
	
	// Read response
	decoder := json.NewDecoder(stream)
	var response TensorServiceResponse
	if err := decoder.Decode(&response); err != nil {
		return nil, err
	}
	
	return response.Result, nil
}

func (t *TensorTransport) sendViaGRPC(address string, operation TensorOperation) (*TensorResult, error) {
	// gRPC client implementation would go here
	return nil, fmt.Errorf("gRPC client not implemented")
}

func (t *TensorTransport) sendViaIPC(socketPath string, operation TensorOperation) (*TensorResult, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	
	// Send request
	encoder := json.NewEncoder(conn)
	request := TensorServiceRequest{
		Operation: &operation,
		StreamId:  fmt.Sprintf("ipc_%d", time.Now().UnixNano()),
	}
	
	if err := encoder.Encode(request); err != nil {
		return nil, err
	}
	
	// Read response
	decoder := json.NewDecoder(conn)
	var response TensorServiceResponse
	if err := decoder.Decode(&response); err != nil {
		return nil, err
	}
	
	return response.Result, nil
}

// Monitoring and maintenance
func (t *TensorTransport) collectMetrics() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		t.mu.Lock()
		
		t.metrics.ActiveStreams = len(t.activeStreams)
		t.metrics.QUICConnections = len(t.quicClients)
		t.metrics.GRPCConnections = len(t.grpcClients)
		t.metrics.IPCConnections = len(t.ipcConnections)
		t.metrics.LastUpdated = time.Now()
		
		// Calculate throughput
		if t.metrics.CompletedOps > 0 {
			elapsed := time.Since(time.Now().Add(-5 * time.Second))
			t.metrics.ThroughputMBps = float64(t.metrics.BytesTransferred) / elapsed.Seconds() / (1024 * 1024)
		}
		
		t.mu.Unlock()
	}
}

func (t *TensorTransport) monitorConnections() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		t.mu.Lock()
		
		// Clean up idle streams
		for id, stream := range t.activeStreams {
			if time.Since(stream.LastActivity) > 5*time.Minute {
				stream.Active = false
				delete(t.activeStreams, id)
				log.Printf("🧹 Cleaned up idle stream: %s", id)
			}
		}
		
		// Clean up idle IPC connections
		for id, ipc := range t.ipcConnections {
			if time.Since(ipc.LastUsed) > 10*time.Minute {
				ipc.Conn.Close()
				delete(t.ipcConnections, id)
				log.Printf("🧹 Cleaned up idle IPC connection: %s", id)
			}
		}
		
		t.mu.Unlock()
	}
}

func (t *TensorTransport) cleanupCache() {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	// Remove old cache entries if cache is too large
	if len(t.resultCache) > 1000 {
		// Simple cleanup - remove random entries (in production, use LRU)
		for key := range t.resultCache {
			delete(t.resultCache, key)
			if len(t.resultCache) <= 800 {
				break
			}
		}
	}
}

func (t *TensorTransport) GetMetrics() TransportMetrics {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.metrics
}

func (t *TensorTransport) generateTLSConfig() *tls.Config {
	if !t.config.EnableTLS {
		// Generate self-signed cert for development
		cert, err := tls.LoadX509KeyPair(t.config.CertFile, t.config.KeyFile)
		if err != nil {
			// Generate in-memory cert for development
			cert = generateSelfSignedCert()
		}
		
		return &tls.Config{
			Certificates: []tls.Certificate{cert},
			NextProtos:   []string{"quic-tensor-transport"},
		}
	}
	
	return &tls.Config{
		NextProtos: []string{"quic-tensor-transport"},
	}
}

func (t *TensorTransport) generateCacheKey(operation TensorOperation) string {
	// Simple cache key generation
	return fmt.Sprintf("%s_%s_%d", operation.Type, operation.ID[:8], len(operation.InputTensors))
}

// Utility functions
func calculateFLOPs(shape1, shape2 []int) int64 {
	// Simplified FLOP calculation
	return int64(shape1[0] * shape1[1] * shape2[1] * 2)
}

func areBroadcastCompatible(shape1, shape2 []int) bool {
	// Simplified broadcast compatibility check
	return len(shape1) == len(shape2)
}

func calculateConvOutputShape(inputShape, kernelShape []int, params map[string]interface{}) []int {
	// Simplified convolution output shape calculation
	stride := getIntParam(params, "stride", 1)
	padding := getIntParam(params, "padding", 0)
	
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	
	// Calculate spatial dimensions (simplified)
	if len(inputShape) >= 2 && len(kernelShape) >= 2 {
		outputShape[len(outputShape)-2] = (inputShape[len(inputShape)-2] + 2*padding - kernelShape[0]) / stride + 1
		outputShape[len(outputShape)-1] = (inputShape[len(inputShape)-1] + 2*padding - kernelShape[1]) / stride + 1
	}
	
	return outputShape
}

func calculateTensorSize(shape []int, dataType string) int {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	switch dataType {
	case "float32":
		return size * 4
	case "float64":
		return size * 8
	case "int32":
		return size * 4
	case "int64":
		return size * 8
	default:
		return size * 4 // Default to float32
	}
}

func getIntParam(params map[string]interface{}, key string, defaultValue int) int {
	if val, exists := params[key]; exists {
		if intVal, ok := val.(int); ok {
			return intVal
		}
		if floatVal, ok := val.(float64); ok {
			return int(floatVal)
		}
	}
	return defaultValue
}

func generateSelfSignedCert() tls.Certificate {
	// In production, use proper certificates
	// This is a placeholder for development
	return tls.Certificate{}
}

func (t *TensorTransport) Shutdown() error {
	log.Println("🛑 Shutting down Tensor Transport...")
	
	// Close QUIC server
	if t.quicServer != nil {
		t.quicServer.Close()
	}
	
	// Stop gRPC server
	if t.grpcServer != nil {
		t.grpcServer.GracefulStop()
	}
	
	// Close all connections
	t.mu.Lock()
	for _, conn := range t.quicClients {
		(*conn).CloseWithError(0, "shutdown")
	}
	for _, conn := range t.grpcClients {
		conn.Close()
	}
	for _, ipc := range t.ipcConnections {
		ipc.Conn.Close()
	}
	t.mu.Unlock()
	
	// Close channels
	close(t.tensorQueue)
	
	log.Println("✅ Tensor Transport shutdown complete")
	return nil
}

func main() {
	transport := NewTensorTransport()
	defer transport.Shutdown()
	
	if err := transport.Initialize(); err != nil {
		log.Fatalf("💥 Transport initialization failed: %v", err)
	}
	
	// Keep running
	select {}
}