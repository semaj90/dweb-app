// Deprecated: moved to pkg/enhancedrag and cmd/enhanced-rag. Keeping file for reference but not compiled.
//go:build ignore
// +build ignore

package ignored

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// Enhanced RAG Service with Real-time LLM Training
// Integrates with Context7 for advanced legal AI assistance

type EnhancedRAGService struct {
	port           string
	context7Client *Context7Client
	userPatterns   *UserPatternStore
	llmTrainer     *RealTimeLLMTrainer
	wsConnections  map[string]*websocket.Conn
	wsLock         sync.RWMutex
}

type Context7Client struct {
	baseURL    string
	httpClient *http.Client
}

type UserPatternStore struct {
	patterns map[string]*UserBehaviorPattern
	lock     sync.RWMutex
}

type UserBehaviorPattern struct {
	UserID           string                 `json:"userId"`
	SearchQueries    []string               `json:"searchQueries"`
	DocumentAccess   []DocumentInteraction  `json:"documentAccess"`
	LegalConcepts    []string               `json:"legalConcepts"`
	XStateEvents     []XStateEvent          `json:"xstateEvents"`
	PersonalizedData map[string]interface{} `json:"personalizedData"`
	LastUpdate       time.Time              `json:"lastUpdate"`
}

type DocumentInteraction struct {
	DocumentID  string    `json:"documentId"`
	AccessTime  time.Time `json:"accessTime"`
	Duration    int       `json:"duration"`
	ScrollDepth float64   `json:"scrollDepth"`
	Annotations []string  `json:"annotations"`
	EntityFocus []string  `json:"entityFocus"`
}

type XStateEvent struct {
	Event     string                 `json:"event"`
	State     string                 `json:"state"`
	Context   map[string]interface{} `json:"context"`
	Timestamp time.Time              `json:"timestamp"`
}

type RealTimeLLMTrainer struct {
	trainingSessions map[string]*TrainingSession
	lock             sync.RWMutex
}

type TrainingSession struct {
	UserID       string                 `json:"userId"`
	Interactions []LLMInteraction       `json:"interactions"`
	Feedback     []UserFeedback         `json:"feedback"`
	Adaptations  map[string]interface{} `json:"adaptations"`
	StartTime    time.Time              `json:"startTime"`
}

type LLMInteraction struct {
	Query      string                 `json:"query"`
	Response   string                 `json:"response"`
	Context    map[string]interface{} `json:"context"`
	Timestamp  time.Time              `json:"timestamp"`
	Relevance  float64                `json:"relevance"`
	UserRating int                    `json:"userRating"`
}

type UserFeedback struct {
	InteractionID string    `json:"interactionId"`
	Rating        int       `json:"rating"`
	Comments      string    `json:"comments"`
	Timestamp     time.Time `json:"timestamp"`
}

// WebSocket upgrader for enhanced RAG
var enhancedRAGUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for development
	},
}

func NewEnhancedRAGService() *EnhancedRAGService {
	port := os.Getenv("ENHANCED_RAG_PORT")
	if port == "" {
		port = "8095"
	}

	context7URL := os.Getenv("CONTEXT7_URL")
	if context7URL == "" {
		context7URL = "http://localhost:4100"
	}

	return &EnhancedRAGService{
		port: port,
		context7Client: &Context7Client{
			baseURL:    context7URL,
			httpClient: &http.Client{Timeout: 30 * time.Second},
		},
		userPatterns: &UserPatternStore{
			patterns: make(map[string]*UserBehaviorPattern),
		},
		llmTrainer: &RealTimeLLMTrainer{
			trainingSessions: make(map[string]*TrainingSession),
		},
		wsConnections: make(map[string]*websocket.Conn),
	}
}

// Context7 Integration Methods
func (c *Context7Client) AnalyzeStack(component string, context string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"component": component,
		"context":   context,
		"timestamp": time.Now().Unix(),
	}

	resp, err := c.makeRequest("POST", "/analyze-stack", payload)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func (c *Context7Client) GetLibraryDocs(libraryId string, topic string) (map[string]interface{}, error) {
	payload := map[string]interface{}{
		"libraryId": libraryId,
		"topic":     topic,
		"tokens":    10000,
	}

	resp, err := c.makeRequest("POST", "/get-library-docs", payload)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func (c *Context7Client) makeRequest(method, endpoint string, payload interface{}) ([]byte, error) {
	// Implementation for HTTP requests to Context7
	// This would make actual HTTP calls to the Context7 multicore server
	log.Printf("Context7 %s %s: %+v", method, endpoint, payload)

	// Placeholder response for now
	return json.Marshal(map[string]interface{}{
		"status": "success",
		"data":   payload,
	})
}

// User Pattern Management
func (ups *UserPatternStore) UpdatePattern(userID string, pattern *UserBehaviorPattern) {
	ups.lock.Lock()
	defer ups.lock.Unlock()

	pattern.LastUpdate = time.Now()
	ups.patterns[userID] = pattern

	log.Printf("Updated user pattern for %s with %d search queries",
		userID, len(pattern.SearchQueries))
}

func (ups *UserPatternStore) GetPattern(userID string) *UserBehaviorPattern {
	ups.lock.RLock()
	defer ups.lock.RUnlock()

	return ups.patterns[userID]
}

// Real-time LLM Training
func (rlt *RealTimeLLMTrainer) StartSession(userID string) *TrainingSession {
	rlt.lock.Lock()
	defer rlt.lock.Unlock()

	session := &TrainingSession{
		UserID:       userID,
		Interactions: make([]LLMInteraction, 0),
		Feedback:     make([]UserFeedback, 0),
		Adaptations:  make(map[string]interface{}),
		StartTime:    time.Now(),
	}

	rlt.trainingSessions[userID] = session
	log.Printf("Started LLM training session for user %s", userID)

	return session
}

func (rlt *RealTimeLLMTrainer) AddInteraction(userID string, interaction LLMInteraction) {
	rlt.lock.Lock()
	defer rlt.lock.Unlock()

	if session, exists := rlt.trainingSessions[userID]; exists {
		interaction.Timestamp = time.Now()
		session.Interactions = append(session.Interactions, interaction)

		// Analyze interaction for real-time adaptations
		rlt.analyzeInteraction(session, interaction)
	}
}

func (rlt *RealTimeLLMTrainer) analyzeInteraction(session *TrainingSession, interaction LLMInteraction) {
	// Real-time analysis of user interactions for LLM adaptation
	if interaction.UserRating >= 4 {
		session.Adaptations["positive_patterns"] = append(
			session.Adaptations["positive_patterns"].([]string),
			interaction.Query,
		)
	}

	// Extract legal concepts from successful interactions
	if interaction.Relevance > 0.8 {
		session.Adaptations["high_relevance_queries"] = append(
			session.Adaptations["high_relevance_queries"].([]string),
			interaction.Query,
		)
	}
}

// HTTP Handlers
func (s *EnhancedRAGService) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger())
	r.Use(gin.Recovery())

	// CORS configuration for frontend integration
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"http://localhost:5173", "http://localhost:3000"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	config.AllowCredentials = true
	r.Use(cors.New(config))

	// Health check
	r.GET("/health", s.healthHandler)

	// WebSocket for real-time tracking
	r.GET("/ws/:userId", s.websocketHandler)

	// User pattern tracking
	r.POST("/patterns/:userId", s.updatePatternHandler)
	r.GET("/patterns/:userId", s.getPatternHandler)

	// LLM training endpoints
	r.POST("/llm/session/:userId", s.startTrainingSessionHandler)
	r.POST("/llm/interaction/:userId", s.addInteractionHandler)
	r.POST("/llm/feedback/:userId", s.addFeedbackHandler)

	// Context7 integration
	r.POST("/context7/analyze", s.context7AnalyzeHandler)
	r.POST("/context7/docs", s.context7DocsHandler)

	// Enhanced search with personalization
	r.POST("/search/personalized", s.personalizedSearchHandler)

	// Document analysis with legal entity extraction
	r.POST("/documents/analyze", s.analyzeDocumentHandler)

	return r
}

func (s *EnhancedRAGService) healthHandler(c *gin.Context) {
	// Check Context7 connectivity
	context7Health := "false"
	if resp, err := http.Get(s.context7Client.baseURL + "/health"); err == nil && resp.StatusCode == 200 {
		context7Health = "true"
		resp.Body.Close()
	}

	c.JSON(http.StatusOK, gin.H{
		"status":                "healthy",
		"context7_connected":    context7Health,
		"active_patterns":       len(s.userPatterns.patterns),
		"training_sessions":     len(s.llmTrainer.trainingSessions),
		"websocket_connections": len(s.wsConnections),
		"timestamp":             time.Now().Format(time.RFC3339),
	})
}

func (s *EnhancedRAGService) websocketHandler(c *gin.Context) {
	userID := c.Param("userId")

	conn, err := enhancedRAGUpgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	s.wsLock.Lock()
	s.wsConnections[userID] = conn
	s.wsLock.Unlock()

	defer func() {
		s.wsLock.Lock()
		delete(s.wsConnections, userID)
		s.wsLock.Unlock()
	}()

	log.Printf("WebSocket connected for user %s", userID)

	// Handle incoming messages
	for {
		var msg map[string]interface{}
		if err := conn.ReadJSON(&msg); err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		// Process real-time user behavior
		s.processRealtimeEvent(userID, msg)
	}
}

func (s *EnhancedRAGService) processRealtimeEvent(userID string, event map[string]interface{}) {
	// Process real-time user behavior events
	eventType := event["type"].(string)

	switch eventType {
	case "search_query":
		s.handleSearchEvent(userID, event)
	case "document_view":
		s.handleDocumentEvent(userID, event)
	case "xstate_transition":
		s.handleXStateEvent(userID, event)
	}
}

func (s *EnhancedRAGService) handleSearchEvent(userID string, event map[string]interface{}) {
	pattern := s.userPatterns.GetPattern(userID)
	if pattern == nil {
		pattern = &UserBehaviorPattern{
			UserID:        userID,
			SearchQueries: make([]string, 0),
		}
	}

	query := event["query"].(string)
	pattern.SearchQueries = append(pattern.SearchQueries, query)

	s.userPatterns.UpdatePattern(userID, pattern)

	// Send real-time recommendations via WebSocket
	s.sendRealtimeRecommendations(userID, query)
}

func (s *EnhancedRAGService) handleDocumentEvent(userID string, event map[string]interface{}) {
	// Handle document interaction events
	log.Printf("Document event for user %s: %+v", userID, event)
}

func (s *EnhancedRAGService) handleXStateEvent(userID string, event map[string]interface{}) {
	// Handle XState machine transitions
	log.Printf("XState event for user %s: %+v", userID, event)
}

func (s *EnhancedRAGService) sendRealtimeRecommendations(userID string, query string) {
	s.wsLock.RLock()
	conn, exists := s.wsConnections[userID]
	s.wsLock.RUnlock()

	if !exists {
		return
	}

	recommendations := map[string]interface{}{
		"type":           "recommendations",
		"query":          query,
		"suggestions":    []string{"related concept 1", "related concept 2"},
		"legal_entities": []string{"entity 1", "entity 2"},
		"timestamp":      time.Now().Unix(),
	}

	if err := conn.WriteJSON(recommendations); err != nil {
		log.Printf("WebSocket write error: %v", err)
	}
}

func (s *EnhancedRAGService) updatePatternHandler(c *gin.Context) {
	userID := c.Param("userId")

	var pattern UserBehaviorPattern
	if err := c.ShouldBindJSON(&pattern); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	pattern.UserID = userID
	s.userPatterns.UpdatePattern(userID, &pattern)

	c.JSON(http.StatusOK, gin.H{"status": "updated"})
}

func (s *EnhancedRAGService) getPatternHandler(c *gin.Context) {
	userID := c.Param("userId")
	pattern := s.userPatterns.GetPattern(userID)

	if pattern == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "pattern not found"})
		return
	}

	c.JSON(http.StatusOK, pattern)
}

func (s *EnhancedRAGService) startTrainingSessionHandler(c *gin.Context) {
	userID := c.Param("userId")
	session := s.llmTrainer.StartSession(userID)

	c.JSON(http.StatusOK, session)
}

func (s *EnhancedRAGService) addInteractionHandler(c *gin.Context) {
	userID := c.Param("userId")

	var interaction LLMInteraction
	if err := c.ShouldBindJSON(&interaction); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	s.llmTrainer.AddInteraction(userID, interaction)

	c.JSON(http.StatusOK, gin.H{"status": "added"})
}

func (s *EnhancedRAGService) addFeedbackHandler(c *gin.Context) {
	userID := c.Param("userId")

	var feedback UserFeedback
	if err := c.ShouldBindJSON(&feedback); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	feedback.Timestamp = time.Now()

	// Add feedback to training session
	s.llmTrainer.lock.Lock()
	if session, exists := s.llmTrainer.trainingSessions[userID]; exists {
		session.Feedback = append(session.Feedback, feedback)
	}
	s.llmTrainer.lock.Unlock()

	c.JSON(http.StatusOK, gin.H{"status": "added"})
}

func (s *EnhancedRAGService) context7AnalyzeHandler(c *gin.Context) {
	var req struct {
		Component string `json:"component"`
		Context   string `json:"context"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := s.context7Client.AnalyzeStack(req.Component, req.Context)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (s *EnhancedRAGService) context7DocsHandler(c *gin.Context) {
	var req struct {
		LibraryId string `json:"libraryId"`
		Topic     string `json:"topic"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := s.context7Client.GetLibraryDocs(req.LibraryId, req.Topic)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

func (s *EnhancedRAGService) personalizedSearchHandler(c *gin.Context) {
	var req struct {
		UserID string `json:"userId"`
		Query  string `json:"query"`
		Limit  int    `json:"limit"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Get user patterns for personalization
	pattern := s.userPatterns.GetPattern(req.UserID)

	// Mock personalized search results
	results := map[string]interface{}{
		"query":            req.Query,
		"personalized":     pattern != nil,
		"results":          []string{"result 1", "result 2", "result 3"},
		"legal_entities":   []string{"contract", "liability", "damages"},
		"related_concepts": []string{"tort law", "contract breach"},
		"confidence_score": 0.92,
	}

	c.JSON(http.StatusOK, results)
}

func (s *EnhancedRAGService) analyzeDocumentHandler(c *gin.Context) {
	var req struct {
		DocumentID string `json:"documentId"`
		Content    string `json:"content"`
		UserID     string `json:"userId"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Mock document analysis with legal entity extraction
	analysis := map[string]interface{}{
		"document_id":      req.DocumentID,
		"legal_entities":   []string{"plaintiff", "defendant", "contract", "damages"},
		"key_concepts":     []string{"breach of contract", "monetary damages", "specific performance"},
		"sentiment":        "neutral",
		"complexity_score": 7.5,
		"embeddings":       []float64{0.1, 0.2, 0.3}, // Mock embeddings
		"summary":          "Legal document analysis complete",
	}

	c.JSON(http.StatusOK, analysis)
}

func (s *EnhancedRAGService) Start() error {
	r := s.setupRoutes()

	log.Printf("🚀 Enhanced RAG Service starting on port %s", s.port)
	log.Printf("📡 Context7 integration: %s", s.context7Client.baseURL)
	log.Printf("🔗 WebSocket endpoint: ws://localhost:%s/ws/{userId}", s.port)
	log.Printf("🧠 Real-time LLM training enabled")
	log.Printf("👤 Personalized search and recommendations ready")

	return r.Run(":" + s.port)
}

// This file is excluded from builds.
