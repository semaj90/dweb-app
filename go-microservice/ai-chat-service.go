package main

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/lib/pq"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

// Chat message structures
type ChatMessage struct {
	ID          string                 `json:"id" db:"id"`
	SessionID   string                 `json:"session_id" db:"session_id"`
	UserID      string                 `json:"user_id" db:"user_id"`
	CaseID      string                 `json:"case_id" db:"case_id"`
	Role        string                 `json:"role" db:"role"` // user, assistant, system
	Content     string                 `json:"content" db:"content"`
	Timestamp   time.Time              `json:"timestamp" db:"timestamp"`
	TokenCount  int                    `json:"token_count" db:"token_count"`
	Metadata    map[string]interface{} `json:"metadata" db:"metadata"`
	Analysis    *MessageAnalysis       `json:"analysis,omitempty"`
	Context     *RAGContext           `json:"context,omitempty"`
}

type MessageAnalysis struct {
	Intent       []string               `json:"intent"`
	Entities     []Entity               `json:"entities"`
	Sentiment    float64                `json:"sentiment"`
	Confidence   float64                `json:"confidence"`
	Topics       []string               `json:"topics"`
	LegalConcepts []string              `json:"legal_concepts"`
	SOMCluster   int                    `json:"som_cluster"`
	Embeddings   []float32              `json:"embeddings"`
	Metadata     map[string]interface{} `json:"metadata"`
}

type Entity struct {
	Text       string  `json:"text"`
	Type       string  `json:"type"`
	Confidence float64 `json:"confidence"`
	StartPos   int     `json:"start_pos"`
	EndPos     int     `json:"end_pos"`
}

type RAGContext struct {
	RelevantDocs    []RelevantDocument     `json:"relevant_docs"`
	UserHistory     []ChatMessage          `json:"user_history"`
	CaseContext     map[string]interface{} `json:"case_context"`
	Recommendations []Recommendation       `json:"recommendations"`
	DidYouMean      []string               `json:"did_you_mean"`
}

type RelevantDocument struct {
	ID           string                 `json:"id"`
	Title        string                 `json:"title"`
	Content      string                 `json:"content"`
	Relevance    float64                `json:"relevance"`
	Source       string                 `json:"source"`
	Metadata     map[string]interface{} `json:"metadata"`
	Highlights   []string               `json:"highlights"`
}

type Recommendation struct {
	Type        string                 `json:"type"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Confidence  float64                `json:"confidence"`
	Action      string                 `json:"action"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type ChatSession struct {
	ID            string                 `json:"id"`
	UserID        string                 `json:"user_id"`
	CaseID        string                 `json:"case_id"`
	StartTime     time.Time              `json:"start_time"`
	LastActivity  time.Time              `json:"last_activity"`
	MessageCount  int                    `json:"message_count"`
	TokensUsed    int                    `json:"tokens_used"`
	Context       map[string]interface{} `json:"context"`
	IsActive      bool                   `json:"is_active"`
}

type StreamingResponse struct {
	SessionID string `json:"session_id"`
	MessageID string `json:"message_id"`
	Token     string `json:"token"`
	Complete  bool   `json:"complete"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// AI Chat Service
type AIChatService struct {
	db          *sql.DB
	redis       *redis.Client
	upgrader    websocket.Upgrader
	llama       *LlamaService
	ragEngine   *EnhancedRAGEngine
	som         *SOM
	sessions    map[string]*ChatSession
	sessionMux  sync.RWMutex
	grpcServer  *grpc.Server
}

// Llama.cpp integration service
type LlamaService struct {
	modelPath    string
	contextSize  int
	threads      int
	gpuLayers    int
	temperature  float64
	topK         int
	topP         float64
	repeatPenalty float64
	mu           sync.Mutex
	isLoaded     bool
	process      *exec.Cmd
}

// Enhanced RAG Engine with SOM integration
type EnhancedRAGEngine struct {
	db            *sql.DB
	redis         *redis.Client
	som           *SOM
	vectorDim     int
	maxResults    int
	similarityThreshold float64
}

func NewAIChatService() *AIChatService {
	// Initialize database connection
	db, err := sql.Open("postgres", os.Getenv("DATABASE_URL"))
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	// Initialize Redis
	rdb := redis.NewClient(&redis.Options{
		Addr:     os.Getenv("REDIS_HOST") + ":" + os.Getenv("REDIS_PORT"),
		Password: os.Getenv("REDIS_PASSWORD"),
		DB:       0,
	})

	// Initialize Llama service
	llama := &LlamaService{
		modelPath:     os.Getenv("LLAMA_MODEL_PATH"),
		contextSize:   4096,
		threads:       8,
		gpuLayers:     32,
		temperature:   0.7,
		topK:          40,
		topP:          0.95,
		repeatPenalty: 1.1,
	}

	// Initialize SOM for intent clustering
	som := &SOM{
		Width:        20,
		Height:       20,
		LearningRate: 0.1,
		Radius:       5.0,
		Iterations:   1000,
	}
	som.initializeNodes(384) // 384-dimensional embeddings

	// Initialize Enhanced RAG
	ragEngine := &EnhancedRAGEngine{
		db:                  db,
		redis:               rdb,
		som:                 som,
		vectorDim:           384,
		maxResults:          10,
		similarityThreshold: 0.7,
	}

	service := &AIChatService{
		db:        db,
		redis:     rdb,
		llama:     llama,
		ragEngine: ragEngine,
		som:       som,
		sessions:  make(map[string]*ChatSession),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}

	// Load Llama model
	go llama.LoadModel()

	return service
}

func (l *LlamaService) LoadModel() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.isLoaded {
		return nil
	}

	log.Printf("🦙 Loading Llama model: %s", l.modelPath)

	// Start llama.cpp server process
	args := []string{
		"-m", l.modelPath,
		"-c", fmt.Sprintf("%d", l.contextSize),
		"-t", fmt.Sprintf("%d", l.threads),
		"-ngl", fmt.Sprintf("%d", l.gpuLayers),
		"--host", "127.0.0.1",
		"--port", "8085",
		"--api-key", "your-api-key",
	}

	l.process = exec.Command("./llama-server", args...)
	l.process.Stdout = os.Stdout
	l.process.Stderr = os.Stderr

	if err := l.process.Start(); err != nil {
		return fmt.Errorf("failed to start llama server: %v", err)
	}

	// Wait for server to be ready
	time.Sleep(5 * time.Second)
	l.isLoaded = true

	log.Println("✅ Llama model loaded successfully")
	return nil
}

func (l *LlamaService) GenerateResponse(prompt string, maxTokens int) (string, error) {
	if !l.isLoaded {
		return "", fmt.Errorf("llama model not loaded")
	}

	// Call llama.cpp API
	requestBody := map[string]interface{}{
		"prompt":      prompt,
		"max_tokens":  maxTokens,
		"temperature": l.temperature,
		"top_k":       l.topK,
		"top_p":       l.topP,
		"repeat_penalty": l.repeatPenalty,
		"stop":        []string{"</s>", "[INST]", "[/INST]"},
	}

	jsonBody, _ := json.Marshal(requestBody)
	
	resp, err := http.Post("http://127.0.0.1:8085/completion", 
		"application/json", 
		strings.NewReader(string(jsonBody)))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if content, ok := result["content"].(string); ok {
		return content, nil
	}

	return "", fmt.Errorf("invalid response format")
}

func (l *LlamaService) StreamResponse(prompt string, maxTokens int, callback func(string)) error {
	if !l.isLoaded {
		return fmt.Errorf("llama model not loaded")
	}

	// Streaming implementation
	requestBody := map[string]interface{}{
		"prompt":      prompt,
		"max_tokens":  maxTokens,
		"temperature": l.temperature,
		"stream":      true,
	}

	jsonBody, _ := json.Marshal(requestBody)
	
	resp, err := http.Post("http://127.0.0.1:8085/completion", 
		"application/json", 
		strings.NewReader(string(jsonBody)))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			if choices, ok := chunk["choices"].([]interface{}); ok && len(choices) > 0 {
				if choice, ok := choices[0].(map[string]interface{}); ok {
					if delta, ok := choice["delta"].(map[string]interface{}); ok {
						if content, ok := delta["content"].(string); ok {
							callback(content)
						}
					}
				}
			}
		}
	}

	return scanner.Err()
}

// Enhanced RAG Engine Implementation
func (r *EnhancedRAGEngine) SearchRelevantDocuments(query string, caseID string, userID string) ([]RelevantDocument, error) {
	// Generate query embedding
	queryEmbedding, err := r.generateEmbedding(query)
	if err != nil {
		return nil, err
	}

	// Vector similarity search using pgvector
	searchQuery := `
		SELECT 
			id, title, content, metadata,
			1 - (embedding <=> $1) as similarity
		FROM legal_documents 
		WHERE case_id = $2 
			AND (1 - (embedding <=> $1)) > $3
		ORDER BY similarity DESC
		LIMIT $4
	`

	rows, err := r.db.Query(searchQuery, pq.Array(queryEmbedding), caseID, r.similarityThreshold, r.maxResults)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var documents []RelevantDocument
	for rows.Next() {
		var doc RelevantDocument
		var metadataJSON string
		
		err := rows.Scan(&doc.ID, &doc.Title, &doc.Content, &metadataJSON, &doc.Relevance)
		if err != nil {
			continue
		}

		json.Unmarshal([]byte(metadataJSON), &doc.Metadata)
		
		// Generate highlights
		doc.Highlights = r.generateHighlights(doc.Content, query)
		
		documents = append(documents, doc)
	}

	return documents, nil
}

func (r *EnhancedRAGEngine) generateEmbedding(text string) ([]float64, error) {
	// Call sentence transformer service
	requestBody := map[string]interface{}{
		"texts": []string{text},
		"model": "sentence-transformers/all-MiniLM-L6-v2",
	}

	jsonBody, _ := json.Marshal(requestBody)
	
	resp, err := http.Post("http://localhost:8083/embed", 
		"application/json", 
		strings.NewReader(string(jsonBody)))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	if embeddings, ok := result["embeddings"].([]interface{}); ok && len(embeddings) > 0 {
		if embedding, ok := embeddings[0].([]interface{}); ok {
			result := make([]float64, len(embedding))
			for i, val := range embedding {
				if f, ok := val.(float64); ok {
					result[i] = f
				}
			}
			return result, nil
		}
	}

	return nil, fmt.Errorf("invalid embedding response")
}

func (r *EnhancedRAGEngine) generateHighlights(content, query string) []string {
	// Simple keyword highlighting
	words := strings.Fields(strings.ToLower(query))
	contentLower := strings.ToLower(content)
	
	var highlights []string
	sentences := strings.Split(content, ".")
	
	for _, sentence := range sentences {
		sentenceLower := strings.ToLower(sentence)
		for _, word := range words {
			if strings.Contains(sentenceLower, word) {
				highlights = append(highlights, strings.TrimSpace(sentence))
				break
			}
		}
	}
	
	if len(highlights) > 3 {
		highlights = highlights[:3]
	}
	
	return highlights
}

// HTTP Handlers
func (s *AIChatService) setupRoutes() *gin.Engine {
	r := gin.Default()

	// Enable CORS
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	})

	// Chat endpoints
	r.POST("/api/chat/session", s.createChatSession)
	r.GET("/api/chat/session/:sessionId", s.getChatSession)
	r.POST("/api/chat/message", s.sendMessage)
	r.GET("/api/chat/history/:sessionId", s.getChatHistory)
	r.POST("/api/chat/analyze", s.analyzeMessage)
	r.GET("/api/chat/recommendations/:sessionId", s.getRecommendations)
	r.POST("/api/chat/report", s.generateReport)

	// WebSocket endpoint for real-time chat
	r.GET("/api/chat/ws/:sessionId", s.handleWebSocket)

	// Health check
	r.GET("/api/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status": "healthy",
			"llama_loaded": s.llama.isLoaded,
			"timestamp": time.Now(),
		})
	})

	return r
}

func (s *AIChatService) createChatSession(c *gin.Context) {
	var req struct {
		UserID string `json:"user_id"`
		CaseID string `json:"case_id"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	session := &ChatSession{
		ID:           generateUUID(),
		UserID:       req.UserID,
		CaseID:       req.CaseID,
		StartTime:    time.Now(),
		LastActivity: time.Now(),
		IsActive:     true,
		Context:      make(map[string]interface{}),
	}

	// Store session in memory and Redis
	s.sessionMux.Lock()
	s.sessions[session.ID] = session
	s.sessionMux.Unlock()

	sessionJSON, _ := json.Marshal(session)
	s.redis.Set(context.Background(), fmt.Sprintf("session:%s", session.ID), sessionJSON, time.Hour*24)

	c.JSON(200, session)
}

func (s *AIChatService) sendMessage(c *gin.Context) {
	var req struct {
		SessionID string `json:"session_id"`
		Content   string `json:"content"`
		UserID    string `json:"user_id"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Create user message
	userMessage := &ChatMessage{
		ID:        generateUUID(),
		SessionID: req.SessionID,
		UserID:    req.UserID,
		Role:      "user",
		Content:   req.Content,
		Timestamp: time.Now(),
	}

	// Analyze user message
	analysis, err := s.analyzeUserMessage(userMessage)
	if err != nil {
		log.Printf("Failed to analyze message: %v", err)
	} else {
		userMessage.Analysis = analysis
	}

	// Get RAG context
	ragContext, err := s.buildRAGContext(userMessage)
	if err != nil {
		log.Printf("Failed to build RAG context: %v", err)
	} else {
		userMessage.Context = ragContext
	}

	// Store user message
	s.storeMessage(userMessage)

	// Generate AI response
	aiResponse, err := s.generateAIResponse(userMessage)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to generate response"})
		return
	}

	// Store AI response
	s.storeMessage(aiResponse)

	c.JSON(200, gin.H{
		"user_message": userMessage,
		"ai_response":  aiResponse,
	})
}

func (s *AIChatService) analyzeUserMessage(message *ChatMessage) (*MessageAnalysis, error) {
	// Generate embeddings
	embedding, err := s.ragEngine.generateEmbedding(message.Content)
	if err != nil {
		return nil, err
	}

	// Convert to float32 for SOM
	embedding32 := make([]float32, len(embedding))
	for i, v := range embedding {
		embedding32[i] = float32(v)
	}

	// Get SOM cluster
	clusterID, distance := s.som.findBestMatchingUnit(embedding32)

	// Extract intent and entities (simplified)
	intent := extractIntent(message.Content)
	entities := extractEntities(message.Content)
	sentiment := analyzeSentiment(message.Content)
	
	analysis := &MessageAnalysis{
		Intent:        intent,
		Entities:      entities,
		Sentiment:     sentiment,
		Confidence:    1.0 - distance/100.0, // Convert distance to confidence
		Topics:        extractTopics(message.Content),
		LegalConcepts: extractLegalConcepts(message.Content),
		SOMCluster:    clusterID,
		Embeddings:    embedding32,
		Metadata: map[string]interface{}{
			"analysis_timestamp": time.Now(),
			"cluster_distance":   distance,
		},
	}

	return analysis, nil
}

func (s *AIChatService) buildRAGContext(message *ChatMessage) (*RAGContext, error) {
	// Get session
	session := s.sessions[message.SessionID]
	if session == nil {
		return nil, fmt.Errorf("session not found")
	}

	// Search relevant documents
	relevantDocs, err := s.ragEngine.SearchRelevantDocuments(message.Content, session.CaseID, message.UserID)
	if err != nil {
		return nil, err
	}

	// Get user history
	userHistory, err := s.getUserHistory(message.UserID, 10)
	if err != nil {
		log.Printf("Failed to get user history: %v", err)
	}

	// Generate recommendations
	recommendations := s.generateRecommendations(message, relevantDocs, userHistory)

	// Generate "did you mean" suggestions
	didYouMean := s.generateDidYouMean(message.Content)

	context := &RAGContext{
		RelevantDocs:    relevantDocs,
		UserHistory:     userHistory,
		CaseContext:     session.Context,
		Recommendations: recommendations,
		DidYouMean:      didYouMean,
	}

	return context, nil
}

func (s *AIChatService) generateAIResponse(userMessage *ChatMessage) (*ChatMessage, error) {
	// Build enhanced prompt with RAG context
	prompt := s.buildEnhancedPrompt(userMessage)

	// Generate response using Llama
	responseContent, err := s.llama.GenerateResponse(prompt, 512)
	if err != nil {
		return nil, err
	}

	aiMessage := &ChatMessage{
		ID:         generateUUID(),
		SessionID:  userMessage.SessionID,
		UserID:     "assistant",
		Role:       "assistant",
		Content:    responseContent,
		Timestamp:  time.Now(),
		TokenCount: estimateTokenCount(responseContent),
	}

	return aiMessage, nil
}

func (s *AIChatService) buildEnhancedPrompt(userMessage *ChatMessage) string {
	var promptBuilder strings.Builder

	promptBuilder.WriteString("[INST] You are a specialized legal AI assistant. ")
	
	// Add relevant documents context
	if userMessage.Context != nil && len(userMessage.Context.RelevantDocs) > 0 {
		promptBuilder.WriteString("\n\nRelevant documents:\n")
		for i, doc := range userMessage.Context.RelevantDocs {
			if i >= 3 { // Limit to top 3 documents
				break
			}
			promptBuilder.WriteString(fmt.Sprintf("- %s: %s\n", doc.Title, doc.Content[:200]))
		}
	}

	// Add user intent analysis
	if userMessage.Analysis != nil {
		promptBuilder.WriteString(fmt.Sprintf("\n\nUser intent analysis:\n"))
		promptBuilder.WriteString(fmt.Sprintf("- Intent: %v\n", userMessage.Analysis.Intent))
		promptBuilder.WriteString(fmt.Sprintf("- Legal concepts: %v\n", userMessage.Analysis.LegalConcepts))
		promptBuilder.WriteString(fmt.Sprintf("- Sentiment: %.2f\n", userMessage.Analysis.Sentiment))
	}

	promptBuilder.WriteString(fmt.Sprintf("\n\nUser question: %s\n\n", userMessage.Content))
	promptBuilder.WriteString("Please provide a comprehensive, legally accurate response. [/INST]")

	return promptBuilder.String()
}

func (s *AIChatService) handleWebSocket(c *gin.Context) {
	sessionID := c.Param("sessionId")
	
	conn, err := s.upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	log.Printf("WebSocket connected for session: %s", sessionID)

	for {
		var message map[string]interface{}
		if err := conn.ReadJSON(&message); err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		// Handle different message types
		switch message["type"] {
		case "chat_message":
			s.handleStreamingChat(conn, sessionID, message)
		case "typing":
			s.handleTypingIndicator(sessionID, message)
		case "attention":
			s.handleAttentionTracking(sessionID, message)
		}
	}
}

func (s *AIChatService) handleStreamingChat(conn *websocket.Conn, sessionID string, message map[string]interface{}) {
	content, ok := message["content"].(string)
	if !ok {
		return
	}

	userID, _ := message["user_id"].(string)

	// Create user message
	userMessage := &ChatMessage{
		ID:        generateUUID(),
		SessionID: sessionID,
		UserID:    userID,
		Role:      "user",
		Content:   content,
		Timestamp: time.Now(),
	}

	// Analyze and build context
	analysis, _ := s.analyzeUserMessage(userMessage)
	userMessage.Analysis = analysis

	ragContext, _ := s.buildRAGContext(userMessage)
	userMessage.Context = ragContext

	// Store user message
	s.storeMessage(userMessage)

	// Send user message confirmation
	conn.WriteJSON(map[string]interface{}{
		"type":    "message_received",
		"message": userMessage,
	})

	// Generate streaming response
	prompt := s.buildEnhancedPrompt(userMessage)
	responseID := generateUUID()

	s.llama.StreamResponse(prompt, 512, func(token string) {
		response := StreamingResponse{
			SessionID: sessionID,
			MessageID: responseID,
			Token:     token,
			Complete:  false,
		}
		conn.WriteJSON(map[string]interface{}{
			"type":     "streaming_token",
			"response": response,
		})
	})

	// Send completion message
	conn.WriteJSON(map[string]interface{}{
		"type": "streaming_complete",
		"response": StreamingResponse{
			SessionID: sessionID,
			MessageID: responseID,
			Complete:  true,
		},
	})
}

// Helper functions
func generateUUID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

func estimateTokenCount(text string) int {
	return len(strings.Fields(text))
}

func extractIntent(content string) []string {
	// Simplified intent extraction
	intents := []string{}
	content = strings.ToLower(content)
	
	if strings.Contains(content, "search") || strings.Contains(content, "find") {
		intents = append(intents, "search")
	}
	if strings.Contains(content, "analyze") || strings.Contains(content, "review") {
		intents = append(intents, "analysis")
	}
	if strings.Contains(content, "draft") || strings.Contains(content, "write") {
		intents = append(intents, "generation")
	}
	if strings.Contains(content, "explain") || strings.Contains(content, "what") {
		intents = append(intents, "explanation")
	}
	
	return intents
}

func extractEntities(content string) []Entity {
	// Simplified entity extraction
	entities := []Entity{}
	
	// Look for common legal entities
	words := strings.Fields(content)
	for i, word := range words {
		if isLegalEntity(word) {
			entities = append(entities, Entity{
				Text:       word,
				Type:       "legal_term",
				Confidence: 0.8,
				StartPos:   i,
				EndPos:     i + 1,
			})
		}
	}
	
	return entities
}

func isLegalEntity(word string) bool {
	legalTerms := []string{"contract", "agreement", "liability", "damages", "breach", "clause", "statute", "regulation"}
	word = strings.ToLower(word)
	for _, term := range legalTerms {
		if word == term {
			return true
		}
	}
	return false
}

func analyzeSentiment(content string) float64 {
	// Simplified sentiment analysis
	positive := []string{"good", "great", "excellent", "positive", "helpful"}
	negative := []string{"bad", "terrible", "negative", "problem", "issue"}
	
	content = strings.ToLower(content)
	score := 0.0
	
	for _, word := range positive {
		if strings.Contains(content, word) {
			score += 0.1
		}
	}
	
	for _, word := range negative {
		if strings.Contains(content, word) {
			score -= 0.1
		}
	}
	
	// Normalize to [-1, 1]
	if score > 1 {
		score = 1
	}
	if score < -1 {
		score = -1
	}
	
	return score
}

func extractTopics(content string) []string {
	// Simplified topic extraction
	topics := []string{}
	content = strings.ToLower(content)
	
	topicMap := map[string]string{
		"contract": "Contract Law",
		"property": "Property Law",
		"criminal": "Criminal Law",
		"family":   "Family Law",
		"business": "Business Law",
		"tax":      "Tax Law",
	}
	
	for keyword, topic := range topicMap {
		if strings.Contains(content, keyword) {
			topics = append(topics, topic)
		}
	}
	
	return topics
}

func extractLegalConcepts(content string) []string {
	// Simplified legal concept extraction
	concepts := []string{}
	content = strings.ToLower(content)
	
	conceptMap := map[string]string{
		"liability":     "Legal Liability",
		"negligence":    "Negligence",
		"breach":        "Breach of Contract",
		"damages":       "Damages",
		"jurisdiction":  "Jurisdiction",
		"precedent":     "Legal Precedent",
	}
	
	for keyword, concept := range conceptMap {
		if strings.Contains(content, keyword) {
			concepts = append(concepts, concept)
		}
	}
	
	return concepts
}

func (s *AIChatService) storeMessage(message *ChatMessage) error {
	query := `
		INSERT INTO chat_messages 
		(id, session_id, user_id, case_id, role, content, timestamp, token_count, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`
	
	metadataJSON, _ := json.Marshal(message.Metadata)
	
	_, err := s.db.Exec(query, message.ID, message.SessionID, message.UserID, 
		message.CaseID, message.Role, message.Content, message.Timestamp, 
		message.TokenCount, metadataJSON)
	
	return err
}

func (s *AIChatService) getUserHistory(userID string, limit int) ([]ChatMessage, error) {
	query := `
		SELECT id, session_id, user_id, case_id, role, content, timestamp, token_count
		FROM chat_messages 
		WHERE user_id = $1 
		ORDER BY timestamp DESC 
		LIMIT $2
	`
	
	rows, err := s.db.Query(query, userID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	
	var messages []ChatMessage
	for rows.Next() {
		var msg ChatMessage
		err := rows.Scan(&msg.ID, &msg.SessionID, &msg.UserID, &msg.CaseID, 
			&msg.Role, &msg.Content, &msg.Timestamp, &msg.TokenCount)
		if err != nil {
			continue
		}
		messages = append(messages, msg)
	}
	
	return messages, nil
}

func (s *AIChatService) generateRecommendations(message *ChatMessage, docs []RelevantDocument, history []ChatMessage) []Recommendation {
	recommendations := []Recommendation{}
	
	// Analyze user intent and suggest actions
	if message.Analysis != nil {
		for _, intent := range message.Analysis.Intent {
			switch intent {
			case "search":
				recommendations = append(recommendations, Recommendation{
					Type:        "action",
					Title:       "Advanced Search",
					Description: "Try using more specific legal terms for better results",
					Confidence:  0.8,
					Action:      "refine_search",
				})
			case "analysis":
				recommendations = append(recommendations, Recommendation{
					Type:        "tool",
					Title:       "Document Analysis",
					Description: "Run comprehensive analysis on relevant documents",
					Confidence:  0.9,
					Action:      "analyze_documents",
				})
			case "generation":
				recommendations = append(recommendations, Recommendation{
					Type:        "template",
					Title:       "Legal Templates",
					Description: "Use pre-approved templates for faster drafting",
					Confidence:  0.7,
					Action:      "use_template",
				})
			}
		}
	}
	
	// Recommend based on similar cases
	if len(docs) > 0 {
		recommendations = append(recommendations, Recommendation{
			Type:        "reference",
			Title:       "Similar Cases",
			Description: fmt.Sprintf("Found %d similar cases that might be relevant", len(docs)),
			Confidence:  0.85,
			Action:      "view_similar_cases",
		})
	}
	
	return recommendations
}

func (s *AIChatService) generateDidYouMean(query string) []string {
	// Simple "did you mean" implementation
	suggestions := []string{}
	
	// Common legal term corrections
	corrections := map[string]string{
		"agrement":    "agreement",
		"liabilty":    "liability",
		"breech":      "breach",
		"negligance":  "negligence",
		"guarentee":   "guarantee",
		"recieve":     "receive",
	}
	
	words := strings.Fields(strings.ToLower(query))
	for i, word := range words {
		if correction, exists := corrections[word]; exists {
			newWords := make([]string, len(words))
			copy(newWords, words)
			newWords[i] = correction
			suggestions = append(suggestions, strings.Join(newWords, " "))
		}
	}
	
	return suggestions
}

func (s *AIChatService) handleTypingIndicator(sessionID string, message map[string]interface{}) {
	// Broadcast typing indicator to other session participants
	// Implementation depends on your WebSocket broadcast mechanism
}

func (s *AIChatService) handleAttentionTracking(sessionID string, message map[string]interface{}) {
	// Track user attention for context switching
	attentionData := map[string]interface{}{
		"session_id": sessionID,
		"timestamp":  time.Now(),
		"type":       message["attention_type"],
		"data":       message["data"],
	}
	
	// Store in Redis for analysis
	key := fmt.Sprintf("attention:%s:%d", sessionID, time.Now().Unix())
	data, _ := json.Marshal(attentionData)
	s.redis.Set(context.Background(), key, data, time.Hour)
}

func main() {
	service := NewAIChatService()
	
	// Setup HTTP routes
	router := service.setupRoutes()
	
	// Start gRPC server (for internal service communication)
	go func() {
		lis, err := net.Listen("tcp", ":50052")
		if err != nil {
			log.Fatalf("Failed to listen: %v", err)
		}
		
		s := grpc.NewServer()
		reflection.Register(s)
		
		log.Println("🚀 gRPC server starting on :50052")
		if err := s.Serve(lis); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()
	
	// Start HTTP server
	log.Println("🚀 AI Chat Service starting on :8086")
	log.Fatal(http.ListenAndServe(":8086", router))
}