package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health/grpc_health_v1"
)

// healthServer implements the gRPC health check service
type healthServer struct{}

// Check implements the Check method for gRPC health checking
func (h *healthServer) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	return &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	}, nil
}

// Watch implements the Watch method for gRPC health checking
func (h *healthServer) Watch(req *grpc_health_v1.HealthCheckRequest, stream grpc_health_v1.Health_WatchServer) error {
	return stream.Send(&grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	})
}

// List implements the List method for gRPC health checking (added to fix interface compliance)
func (h *healthServer) List(ctx context.Context, req *grpc_health_v1.HealthListRequest) (*grpc_health_v1.HealthListResponse, error) {
	return &grpc_health_v1.HealthListResponse{}, nil
}

// handleGPUComputeRequest handles GPU computation requests
func (s *EnhancedLegalAIService) handleGPUComputeRequest(c *gin.Context) {
	var request struct {
		Data      interface{} `json:"data"`
		Operation string      `json:"operation"`
		Options   map[string]interface{} `json:"options"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Process with GPU if available
	requestData := map[string]interface{}{
		"type": request.Operation,
		"data": request.Data,
		"options": request.Options,
	}
	result, err := s.gpuManager.ProcessRequest(requestData)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"result":  result,
		"gpu_used": s.gpuManager.IsAvailable(),
	})
}

// handleSOMTrainingRequest handles Self-Organizing Map training
func (s *EnhancedLegalAIService) handleSOMTrainingRequest(c *gin.Context) {
	var request struct {
		Data       [][]float64 `json:"data"`
		Width      int         `json:"width"`
		Height     int         `json:"height"`
		Iterations int         `json:"iterations"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Train SOM using TrainWithRequest
	requestData := map[string]interface{}{
		"input_vectors": request.Data,
		"width": request.Width,
		"height": request.Height,
		"iterations": request.Iterations,
	}
	result := s.som.TrainWithRequest(requestData)
	
	// Check for errors in result
	if errorMsg, exists := result["error"]; exists {
		c.JSON(http.StatusInternalServerError, gin.H{"error": errorMsg})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "SOM training completed",
		"clusters": s.som.GetClusters(),
	})
}

// handleXStateEvent handles XState machine events
func (s *EnhancedLegalAIService) handleXStateEvent(c *gin.Context) {
	var event struct {
		Type    string      `json:"type"`
		Payload interface{} `json:"payload"`
		MachineID string    `json:"machine_id"`
	}

	if err := c.ShouldBindJSON(&event); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Process state transition
	stateEvent := &StateEvent{
		MachineID: event.MachineID,
		Type:      event.Type,
		Data:      event.Payload.(map[string]interface{}),
		Timestamp: time.Now(),
	}
	err := s.stateManager.ProcessEvent(stateEvent)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Event processed",
		"timestamp": time.Now().UTC(),
	})
}

// startGRPCServer starts the gRPC server
func (s *EnhancedLegalAIService) startGRPCServer() error {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		return fmt.Errorf("failed to listen on gRPC port: %w", err)
	}

	s.grpcServer = grpc.NewServer()
	
	// Register health service
	grpc_health_v1.RegisterHealthServer(s.grpcServer, &healthServer{})

	go func() {
		if err := s.grpcServer.Serve(lis); err != nil {
			log.Printf("gRPC server error: %v", err)
		}
	}()

	return nil
}

// setupMessageConsumers sets up RabbitMQ message consumers
func (s *EnhancedLegalAIService) setupMessageConsumers() error {
	if s.rabbitmq == nil {
		return fmt.Errorf("RabbitMQ not initialized")
	}

	// Set up consumers for different message types
	consumers := map[string]func([]byte) error{
		"legal.document.process": s.processDocumentMessage,
		"legal.analysis.request": s.processAnalysisMessage,
		"legal.gpu.compute":      s.processGPUMessage,
	}

	for queue, handler := range consumers {
		if err := s.rabbitmq.SetupConsumer(queue, handler); err != nil {
			return fmt.Errorf("failed to setup consumer for %s: %w", queue, err)
		}
	}

	return nil
}

// Duplicate methods removed - using implementations from main.go

// WebSocket-specific handlers with different signatures
func (s *EnhancedLegalAIService) handleGPUComputeRequestWS(conn *WSConnection, message map[string]interface{}) {
	// Extract data from WebSocket message
	request := struct {
		Data      interface{} `json:"data"`
		Operation string      `json:"operation"`
		Options   map[string]interface{} `json:"options"`
	}{
		Data:      message["data"],
		Operation: getStringFromMap(message, "operation"),
		Options:   getMapFromMap(message, "options"),
	}

	// Process with GPU if available
	requestData := map[string]interface{}{
		"type": request.Operation,
		"data": request.Data,
		"options": request.Options,
	}
	result, err := s.gpuManager.ProcessRequest(requestData)
	
	response := map[string]interface{}{
		"type": "gpu_compute_response",
		"success": err == nil,
	}
	
	if err != nil {
		response["error"] = err.Error()
	} else {
		response["result"] = result
		response["gpu_used"] = s.gpuManager.IsAvailable()
	}

	s.sendWebSocketMessage(conn, response)
}

func (s *EnhancedLegalAIService) handleSOMTrainingRequestWS(conn *WSConnection, message map[string]interface{}) {
	// Extract training parameters from message
	data := getFloatArrayFromMap(message, "data")
	width := getIntFromMap(message, "width")
	height := getIntFromMap(message, "height")
	iterations := getIntFromMap(message, "iterations")

	// Train SOM
	err := s.som.Train(data, width, height, iterations)
	
	response := map[string]interface{}{
		"type": "som_training_response",
		"success": err == nil,
	}
	
	if err != nil {
		response["error"] = err.Error()
	} else {
		response["message"] = "SOM training completed"
		response["clusters"] = s.som.GetClusters()
	}

	s.sendWebSocketMessage(conn, response)
}

func (s *EnhancedLegalAIService) handleXStateEventWS(conn *WSConnection, message map[string]interface{}) {
	eventType := getStringFromMap(message, "type")
	payload := message["payload"]
	machineID := getStringFromMap(message, "machine_id")

	// Process state transition
	newState, err := s.stateManager.ProcessEvent(machineID, eventType, payload)
	
	response := map[string]interface{}{
		"type": "xstate_event_response",
		"success": err == nil,
	}
	
	if err != nil {
		response["error"] = err.Error()
	} else {
		response["new_state"] = newState
		response["timestamp"] = time.Now().UTC()
	}

	s.sendWebSocketMessage(conn, response)
}

// Helper functions for extracting data from maps
func getStringFromMap(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func getIntFromMap(m map[string]interface{}, key string) int {
	if val, ok := m[key].(float64); ok {
		return int(val)
	}
	if val, ok := m[key].(int); ok {
		return val
	}
	return 0
}

func getMapFromMap(m map[string]interface{}, key string) map[string]interface{} {
	if val, ok := m[key].(map[string]interface{}); ok {
		return val
	}
	return make(map[string]interface{})
}

func getFloatArrayFromMap(m map[string]interface{}, key string) [][]float64 {
	if val, ok := m[key].([]interface{}); ok {
		result := make([][]float64, len(val))
		for i, row := range val {
			if rowSlice, ok := row.([]interface{}); ok {
				result[i] = make([]float64, len(rowSlice))
				for j, cell := range rowSlice {
					if f, ok := cell.(float64); ok {
						result[i][j] = f
					}
				}
			}
		}
		return result
	}
	return nil
}

// Message processing handlers
func (s *EnhancedLegalAIService) processDocumentMessage(data []byte) error {
	var msg map[string]interface{}
	if err := json.Unmarshal(data, &msg); err != nil {
		return err
	}

	// Process document with AI
	log.Printf("Processing document message: %v", msg)
	return nil
}

func (s *EnhancedLegalAIService) processAnalysisMessage(data []byte) error {
	var request LegalAnalysisRequest
	if err := json.Unmarshal(data, &request); err != nil {
		return err
	}

	// Process with AI processor
	_, err := s.aiProcessor.ProcessLegalDocument(context.Background(), &request)
	return err
}

func (s *EnhancedLegalAIService) processGPUMessage(data []byte) error {
	var msg map[string]interface{}
	if err := json.Unmarshal(data, &msg); err != nil {
		return err
	}

	// Process with GPU manager
	log.Printf("Processing GPU message: %v", msg)
	return nil
}

// Health check helpers
func (s *EnhancedLegalAIService) checkDatabase() bool {
	if s.db == nil {
		return false
	}
	
	sqlDB, err := s.db.DB()
	if err != nil {
		return false
	}
	
	return sqlDB.Ping() == nil
}

func (s *EnhancedLegalAIService) checkAI() bool {
	if s.aiProcessor == nil {
		return false
	}
	
	return s.aiProcessor.HealthCheck() == nil
}

func (s *EnhancedLegalAIService) checkRabbitMQ() bool {
	if s.rabbitmq == nil {
		return false
	}
	
	return s.rabbitmq.IsConnected()
}

// Duplicate healthServer removed