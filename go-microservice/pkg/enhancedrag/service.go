//go:build legacy
// +build legacy

package enhancedrag

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

type Service struct {
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

var upgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool { return true }}

func New() *Service {
	port := os.Getenv("ENHANCED_RAG_PORT")
	if port == "" {
		port = "8094"
	}
	ctxURL := os.Getenv("CONTEXT7_URL")
	if ctxURL == "" {
		ctxURL = "http://localhost:4100"
	}
	return &Service{port: port, context7Client: &Context7Client{baseURL: ctxURL, httpClient: &http.Client{Timeout: 30 * time.Second}}, userPatterns: &UserPatternStore{patterns: make(map[string]*UserBehaviorPattern)}, llmTrainer: &RealTimeLLMTrainer{trainingSessions: make(map[string]*TrainingSession)}, wsConnections: make(map[string]*websocket.Conn)}
}

func (c *Context7Client) AnalyzeStack(component, context string) (map[string]interface{}, error) {
	payload := map[string]interface{}{"component": component, "context": context, "timestamp": time.Now().Unix()}
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

func (c *Context7Client) GetLibraryDocs(libraryId, topic string) (map[string]interface{}, error) {
	payload := map[string]interface{}{"libraryId": libraryId, "topic": topic, "tokens": 10000}
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
	log.Printf("Context7 %s %s: %+v", method, endpoint, payload)
	return json.Marshal(map[string]interface{}{"status": "success", "data": payload})
}

func (s *Service) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())
	cfg := cors.DefaultConfig()
	cfg.AllowOrigins = []string{"http://localhost:5173", "http://localhost:3000"}
	cfg.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	cfg.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	cfg.AllowCredentials = true
	r.Use(cors.New(cfg))
	r.GET("/health", s.healthHandler)
	r.GET("/ws/:userId", s.websocketHandler)
	r.POST("/patterns/:userId", s.updatePatternHandler)
	r.GET("/patterns/:userId", s.getPatternHandler)
	r.POST("/llm/session/:userId", s.startTrainingSessionHandler)
	r.POST("/llm/interaction/:userId", s.addInteractionHandler)
	r.POST("/llm/feedback/:userId", s.addFeedbackHandler)
	r.POST("/context7/analyze", s.context7AnalyzeHandler)
	r.POST("/context7/docs", s.context7DocsHandler)
	r.POST("/search/personalized", s.personalizedSearchHandler)
	r.POST("/documents/analyze", s.analyzeDocumentHandler)
	return r
}

func (s *Service) healthHandler(c *gin.Context) {
	ctxt := "false"
	if resp, err := http.Get(s.context7Client.baseURL + "/health"); err == nil && resp.StatusCode == 200 {
		ctxt = "true"
		resp.Body.Close()
	}
	c.JSON(http.StatusOK, gin.H{"status": "healthy", "context7_connected": ctxt, "active_patterns": len(s.userPatterns.patterns), "training_sessions": len(s.llmTrainer.trainingSessions), "websocket_connections": len(s.wsConnections), "timestamp": time.Now().Format(time.RFC3339)})
}

func (s *Service) websocketHandler(c *gin.Context) {
	userID := c.Param("userId")
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()
	s.wsLock.Lock()
	s.wsConnections[userID] = conn
	s.wsLock.Unlock()
	defer func() { s.wsLock.Lock(); delete(s.wsConnections, userID); s.wsLock.Unlock() }()
	for {
		var msg map[string]interface{}
		if err := conn.ReadJSON(&msg); err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}
		s.processRealtimeEvent(userID, msg)
	}
}

func (s *Service) processRealtimeEvent(userID string, event map[string]interface{}) {
	t, _ := event["type"].(string)
	switch t {
	case "search_query":
		s.handleSearchEvent(userID, event)
	case "document_view":
		s.handleDocumentEvent(userID, event)
	case "xstate_transition":
		s.handleXStateEvent(userID, event)
	}
}

func (s *Service) handleSearchEvent(userID string, event map[string]interface{}) {
	pattern := s.userPatterns.GetPattern(userID)
	if pattern == nil {
		pattern = &UserBehaviorPattern{UserID: userID, SearchQueries: []string{}}
	}
	if q, ok := event["query"].(string); ok {
		pattern.SearchQueries = append(pattern.SearchQueries, q)
		s.userPatterns.UpdatePattern(userID, pattern)
		s.sendRealtimeRecommendations(userID, q)
	}
}

func (s *Service) handleDocumentEvent(userID string, event map[string]interface{}) {
	log.Printf("Document event for %s: %+v", userID, event)
}
func (s *Service) handleXStateEvent(userID string, event map[string]interface{}) {
	log.Printf("XState event for %s: %+v", userID, event)
}

func (s *Service) sendRealtimeRecommendations(userID, query string) {
	s.wsLock.RLock()
	conn, ok := s.wsConnections[userID]
	s.wsLock.RUnlock()
	if !ok {
		return
	}
	rec := map[string]interface{}{"type": "recommendations", "query": query, "suggestions": []string{"related concept 1", "related concept 2"}, "legal_entities": []string{"entity 1", "entity 2"}, "timestamp": time.Now().Unix()}
	if err := conn.WriteJSON(rec); err != nil {
		log.Printf("WebSocket write error: %v", err)
	}
}

func (ups *UserPatternStore) UpdatePattern(userID string, pattern *UserBehaviorPattern) {
	ups.lock.Lock()
	defer ups.lock.Unlock()
	pattern.LastUpdate = time.Now()
	ups.patterns[userID] = pattern
}
func (ups *UserPatternStore) GetPattern(userID string) *UserBehaviorPattern {
	ups.lock.RLock()
	defer ups.lock.RUnlock()
	return ups.patterns[userID]
}
func (rlt *RealTimeLLMTrainer) StartSession(userID string) *TrainingSession {
	rlt.lock.Lock()
	defer rlt.lock.Unlock()
	s := &TrainingSession{UserID: userID, Interactions: []LLMInteraction{}, Feedback: []UserFeedback{}, Adaptations: map[string]interface{}{}, StartTime: time.Now()}
	rlt.trainingSessions[userID] = s
	return s
}
func (rlt *RealTimeLLMTrainer) AddInteraction(userID string, in LLMInteraction) {
	rlt.lock.Lock()
	defer rlt.lock.Unlock()
	if s, ok := rlt.trainingSessions[userID]; ok {
		in.Timestamp = time.Now()
		s.Interactions = append(s.Interactions, in)
	}
}

func (s *Service) startTrainingSessionHandler(c *gin.Context) {
	userID := c.Param("userId")
	c.JSON(http.StatusOK, s.llmTrainer.StartSession(userID))
}
func (s *Service) addInteractionHandler(c *gin.Context) {
	userID := c.Param("userId")
	var in LLMInteraction
	if err := c.ShouldBindJSON(&in); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	s.llmTrainer.AddInteraction(userID, in)
	c.JSON(http.StatusOK, gin.H{"status": "added"})
}
func (s *Service) addFeedbackHandler(c *gin.Context) {
	userID := c.Param("userId")
	var fb UserFeedback
	if err := c.ShouldBindJSON(&fb); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	fb.Timestamp = time.Now()
	s.llmTrainer.lock.Lock()
	if sess, ok := s.llmTrainer.trainingSessions[userID]; ok {
		sess.Feedback = append(sess.Feedback, fb)
	}
	s.llmTrainer.lock.Unlock()
	c.JSON(http.StatusOK, gin.H{"status": "added"})
}
func (s *Service) updatePatternHandler(c *gin.Context) {
	userID := c.Param("userId")
	var p UserBehaviorPattern
	if err := c.ShouldBindJSON(&p); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	p.UserID = userID
	s.userPatterns.UpdatePattern(userID, &p)
	c.JSON(http.StatusOK, gin.H{"status": "updated"})
}
func (s *Service) getPatternHandler(c *gin.Context) {
	userID := c.Param("userId")
	if p := s.userPatterns.GetPattern(userID); p != nil {
		c.JSON(http.StatusOK, p)
	} else {
		c.JSON(http.StatusNotFound, gin.H{"error": "pattern not found"})
	}
}
func (s *Service) context7AnalyzeHandler(c *gin.Context) {
	var req struct{ Component, Context string }
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if res, err := s.context7Client.AnalyzeStack(req.Component, req.Context); err == nil {
		c.JSON(http.StatusOK, res)
	} else {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}
func (s *Service) context7DocsHandler(c *gin.Context) {
	var req struct{ LibraryId, Topic string }
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if res, err := s.context7Client.GetLibraryDocs(req.LibraryId, req.Topic); err == nil {
		c.JSON(http.StatusOK, res)
	} else {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}
func (s *Service) personalizedSearchHandler(c *gin.Context) {
	var req struct {
		UserID, Query string
		Limit         int
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	p := s.userPatterns.GetPattern(req.UserID)
	c.JSON(http.StatusOK, map[string]interface{}{"query": req.Query, "personalized": p != nil, "results": []string{"result 1", "result 2", "result 3"}, "legal_entities": []string{"contract", "liability", "damages"}, "related_concepts": []string{"tort law", "contract breach"}, "confidence_score": 0.92})
}
func (s *Service) analyzeDocumentHandler(c *gin.Context) {
	var req struct{ DocumentID, Content, UserID string }
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, map[string]interface{}{"document_id": req.DocumentID, "legal_entities": []string{"plaintiff", "defendant", "contract", "damages"}, "key_concepts": []string{"breach of contract", "monetary damages", "specific performance"}, "sentiment": "neutral", "complexity_score": 7.5, "embeddings": []float64{0.1, 0.2, 0.3}, "summary": "Legal document analysis complete"})
}

func (s *Service) Start() error {
	r := s.setupRoutes()
	log.Printf("🚀 Enhanced RAG Service starting on port %s", s.port)
	log.Printf("📡 Context7 integration: %s", s.context7Client.baseURL)
	log.Printf("🔗 WebSocket endpoint: ws://localhost:%s/ws/{userId}", s.port)
	return r.Run(":" + s.port)
}
