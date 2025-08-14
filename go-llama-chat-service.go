package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
)

// Go-Llama Chat Service with llama.cpp integration
// Supports streaming inference, chat history, user context awareness

type ChatService struct {
	db          *pgxpool.Pool
	redis       *redis.Client
	config      ChatConfig
	llamaCmd    *exec.Cmd
	llamaStdin  *bufio.Writer
	llamaStdout *bufio.Scanner
	mu          sync.RWMutex
	wsUpgrader  websocket.Upgrader
	activeChats map[string]*ChatSession
	
	// User activity tracking
	userActivity map[string]UserActivity
	contextStore map[string]ChatContext
}

type ChatConfig struct {
	Port         string
	DatabaseURL  string
	RedisURL     string
	LlamaCppPath string
	ModelPath    string
	ContextSize  int
	Threads      int
	GPULayers    int
	Temperature  float64
	TopP         float64
	TopK         int
}

type ChatRequest struct {
	Message     string                 `json:"message"`
	UserID      string                 `json:"user_id"`
	SessionID   string                 `json:"session_id"`
	CaseID      string                 `json:"case_id,omitempty"`
	Context     map[string]interface{} `json:"context,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
}

type ChatResponse struct {
	Response     string                 `json:"response"`
	SessionID    string                 `json:"session_id"`
	UserID       string                 `json:"user_id"`
	Timestamp    time.Time              `json:"timestamp"`
	TokenCount   int                    `json:"token_count"`
	ProcessingTime float64              `json:"processing_time_ms"`
	UserIntent   string                 `json:"user_intent"`
	Suggestions  []string               `json:"suggestions"`
	Context      ChatContext            `json:"context"`
	Streaming    bool                   `json:"streaming"`
}

type ChatSession struct {
	ID           string                 `json:"id"`
	UserID       string                 `json:"user_id"`
	Messages     []ChatMessage          `json:"messages"`
	Context      ChatContext            `json:"context"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	IsActive     bool                   `json:"is_active"`
}

type ChatMessage struct {
	ID        string    `json:"id"`
	Role      string    `json:"role"` // "user", "assistant", "system"
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata"`
}

type ChatContext struct {
	CaseID         string                 `json:"case_id"`
	DocumentRefs   []string               `json:"document_refs"`
	LegalEntities  []string               `json:"legal_entities"`
	Keywords       []string               `json:"keywords"`
	UserIntent     string                 `json:"user_intent"`
	Confidence     float64                `json:"confidence"`
	RecentActions  []UserAction           `json:"recent_actions"`
	Preferences    map[string]interface{} `json:"preferences"`
}

type UserActivity struct {
	UserID        string    `json:"user_id"`
	IsTyping      bool      `json:"is_typing"`
	LastActivity  time.Time `json:"last_activity"`
	CurrentPage   string    `json:"current_page"`
	WebcamActive  bool      `json:"webcam_active"`
	AttentionLevel string   `json:"attention_level"` // "high", "medium", "low"
	InteractionPattern string `json:"interaction_pattern"`
}

type UserAction struct {
	Type      string                 `json:"type"`
	Details   map[string]interface{} `json:"details"`
	Timestamp time.Time              `json:"timestamp"`
}

type StreamResponse struct {
	Token     string  `json:"token"`
	Done      bool    `json:"done"`
	SessionID string  `json:"session_id"`
	Progress  float64 `json:"progress"`
}

func NewChatService() *ChatService {
	config := loadChatConfig()
	
	return &ChatService{
		config:      config,
		activeChats: make(map[string]*ChatSession),
		userActivity: make(map[string]UserActivity),
		contextStore: make(map[string]ChatContext),
		wsUpgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}
}

func loadChatConfig() ChatConfig {
	return ChatConfig{
		Port:         getEnv("CHAT_PORT", "8099"),
		DatabaseURL:  getEnv("DATABASE_URL", "postgresql://legal_admin:123456@localhost:5432/legal_ai_db"),
		RedisURL:     getEnv("REDIS_URL", "redis://localhost:6379"),
		LlamaCppPath: getEnv("LLAMA_CPP_PATH", "C:\\Users\\james\\llama.cpp\\main.exe"),
		ModelPath:    getEnv("LLAMA_MODEL_PATH", "C:\\Users\\james\\models\\gemma-2b-it-q4_k_m.gguf"),
		ContextSize:  getEnvInt("CONTEXT_SIZE", 4096),
		Threads:      getEnvInt("THREADS", 8),
		GPULayers:    getEnvInt("GPU_LAYERS", 35),
		Temperature:  getEnvFloat("TEMPERATURE", 0.7),
		TopP:         getEnvFloat("TOP_P", 0.9),
		TopK:         getEnvInt("TOP_K", 40),
	}
}

func (c *ChatService) Initialize() error {
	log.Println("🚀 Initializing Go-Llama Chat Service...")
	
	// Initialize database
	var err error
	c.db, err = pgxpool.New(context.Background(), c.config.DatabaseURL)
	if err != nil {
		return fmt.Errorf("database connection failed: %w", err)
	}
	
	// Initialize Redis
	opt, err := redis.ParseURL(c.config.RedisURL)
	if err != nil {
		return fmt.Errorf("redis URL parsing failed: %w", err)
	}
	c.redis = redis.NewClient(opt)
	
	// Initialize llama.cpp process
	if err := c.initializeLlama(); err != nil {
		return fmt.Errorf("llama.cpp initialization failed: %w", err)
	}
	
	// Start user activity monitoring
	go c.monitorUserActivity()
	
	log.Println("✅ Go-Llama Chat Service initialized")
	return nil
}

func (c *ChatService) initializeLlama() error {
	log.Printf("🦙 Starting llama.cpp with model: %s", c.config.ModelPath)
	
	args := []string{
		"-m", c.config.ModelPath,
		"-c", fmt.Sprintf("%d", c.config.ContextSize),
		"-t", fmt.Sprintf("%d", c.config.Threads),
		"-ngl", fmt.Sprintf("%d", c.config.GPULayers),
		"--temp", fmt.Sprintf("%.2f", c.config.Temperature),
		"--top-p", fmt.Sprintf("%.2f", c.config.TopP),
		"--top-k", fmt.Sprintf("%d", c.config.TopK),
		"-i", // Interactive mode
		"--color",
		"--verbose-prompt",
	}
	
	c.llamaCmd = exec.Command(c.config.LlamaCppPath, args...)
	
	// Setup stdin/stdout pipes
	stdin, err := c.llamaCmd.StdinPipe()
	if err != nil {
		return err
	}
	c.llamaStdin = bufio.NewWriter(stdin)
	
	stdout, err := c.llamaCmd.StdoutPipe()
	if err != nil {
		return err
	}
	c.llamaStdout = bufio.NewScanner(stdout)
	
	// Start the process
	if err := c.llamaCmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama.cpp: %w", err)
	}
	
	log.Println("✅ llama.cpp process started successfully")
	return nil
}

func (c *ChatService) HandleChat(ctx *gin.Context) {
	var req ChatRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	
	start := time.Now()
	
	// Get or create chat session
	session := c.getOrCreateSession(req.UserID, req.SessionID)
	
	// Update user activity
	c.updateUserActivity(req.UserID, map[string]interface{}{
		"action": "chat",
		"message_length": len(req.Message),
	})
	
	// Analyze user intent
	userIntent := c.analyzeUserIntent(req.Message, session.Context)
	
	// Build enhanced context
	enhancedContext := c.buildEnhancedContext(req, session)
	
	if req.Stream {
		c.handleStreamingChat(ctx, req, session, enhancedContext)
	} else {
		c.handleStandardChat(ctx, req, session, enhancedContext, start, userIntent)
	}
}

func (c *ChatService) handleStandardChat(ctx *gin.Context, req ChatRequest, session *ChatSession, enhancedContext string, start time.Time, userIntent string) {
	// Generate response using llama.cpp
	response, tokenCount, err := c.generateResponse(req.Message, enhancedContext)
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	
	// Add messages to session
	userMsg := ChatMessage{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   req.Message,
		Timestamp: time.Now(),
		Metadata:  map[string]interface{}{"user_intent": userIntent},
	}
	
	assistantMsg := ChatMessage{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()+1),
		Role:      "assistant",
		Content:   response,
		Timestamp: time.Now(),
		Metadata:  map[string]interface{}{"token_count": tokenCount},
	}
	
	session.Messages = append(session.Messages, userMsg, assistantMsg)
	session.UpdatedAt = time.Now()
	
	// Update context
	session.Context = c.updateChatContext(session.Context, req.Message, response, userIntent)
	
	// Store session
	c.storeSession(session)
	
	// Generate suggestions
	suggestions := c.generateSuggestions(response, session.Context)
	
	chatResponse := ChatResponse{
		Response:       response,
		SessionID:      session.ID,
		UserID:         req.UserID,
		Timestamp:      time.Now(),
		TokenCount:     tokenCount,
		ProcessingTime: float64(time.Since(start).Nanoseconds()) / 1e6,
		UserIntent:     userIntent,
		Suggestions:    suggestions,
		Context:        session.Context,
		Streaming:      false,
	}
	
	ctx.JSON(http.StatusOK, chatResponse)
}

func (c *ChatService) handleStreamingChat(ctx *gin.Context, req ChatRequest, session *ChatSession, enhancedContext string) {
	// Upgrade to WebSocket for streaming
	conn, err := c.wsUpgrader.Upgrade(ctx.Writer, ctx.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()
	
	// Start streaming response
	responseChan := make(chan string, 100)
	doneChan := make(chan bool, 1)
	
	go c.generateStreamingResponse(req.Message, enhancedContext, responseChan, doneChan)
	
	fullResponse := ""
	tokenCount := 0
	
	for {
		select {
		case token := <-responseChan:
			fullResponse += token
			tokenCount++
			
			streamResp := StreamResponse{
				Token:     token,
				Done:      false,
				SessionID: session.ID,
				Progress:  float64(tokenCount) / float64(req.MaxTokens),
			}
			
			if err := conn.WriteJSON(streamResp); err != nil {
				return
			}
			
		case <-doneChan:
			// Send final message
			finalResp := StreamResponse{
				Token:     "",
				Done:      true,
				SessionID: session.ID,
				Progress:  1.0,
			}
			
			conn.WriteJSON(finalResp)
			
			// Update session with complete response
			c.updateSessionWithResponse(session, req.Message, fullResponse, tokenCount)
			return
		}
	}
}

func (c *ChatService) generateResponse(message, context string) (string, int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Build prompt with context
	prompt := c.buildPrompt(message, context)
	
	// Send to llama.cpp
	if _, err := c.llamaStdin.WriteString(prompt + "\n"); err != nil {
		return "", 0, err
	}
	c.llamaStdin.Flush()
	
	// Read response
	var response strings.Builder
	tokenCount := 0
	
	for c.llamaStdout.Scan() {
		line := c.llamaStdout.Text()
		
		// Check for end of response
		if strings.Contains(line, "<|end|>") || strings.Contains(line, "###") {
			break
		}
		
		response.WriteString(line)
		tokenCount++
		
		// Prevent infinite loops
		if tokenCount > 1000 {
			break
		}
	}
	
	return response.String(), tokenCount, nil
}

func (c *ChatService) generateStreamingResponse(message, context string, responseChan chan<- string, doneChan chan<- bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	defer func() { doneChan <- true }()
	
	prompt := c.buildPrompt(message, context)
	
	if _, err := c.llamaStdin.WriteString(prompt + "\n"); err != nil {
		return
	}
	c.llamaStdin.Flush()
	
	for c.llamaStdout.Scan() {
		line := c.llamaStdout.Text()
		
		if strings.Contains(line, "<|end|>") || strings.Contains(line, "###") {
			break
		}
		
		// Split line into tokens and stream them
		tokens := strings.Fields(line)
		for _, token := range tokens {
			responseChan <- token + " "
			time.Sleep(10 * time.Millisecond) // Simulate streaming delay
		}
	}
}

func (c *ChatService) buildPrompt(message, context string) string {
	systemPrompt := `You are a helpful AI legal assistant. You have access to legal documents and case information. Provide accurate, helpful responses based on the available context.`
	
	if context != "" {
		systemPrompt += "\n\nContext: " + context
	}
	
	return fmt.Sprintf(`<|system|>%s<|end|>
<|user|>%s<|end|>
<|assistant|>`, systemPrompt, message)
}

func (c *ChatService) buildEnhancedContext(req ChatRequest, session *ChatSession) string {
	context := fmt.Sprintf("User: %s, Session: %s", req.UserID, req.SessionID)
	
	if req.CaseID != "" {
		context += fmt.Sprintf(", Case: %s", req.CaseID)
	}
	
	// Add recent conversation history
	if len(session.Messages) > 0 {
		context += "\n\nRecent conversation:"
		for i := len(session.Messages) - 5; i < len(session.Messages) && i >= 0; i++ {
			if i < len(session.Messages) {
				msg := session.Messages[i]
				context += fmt.Sprintf("\n%s: %s", msg.Role, msg.Content)
			}
		}
	}
	
	// Add legal context if available
	if len(session.Context.DocumentRefs) > 0 {
		context += "\n\nRelevant documents: " + strings.Join(session.Context.DocumentRefs, ", ")
	}
	
	return context
}

func (c *ChatService) analyzeUserIntent(message string, context ChatContext) string {
	message = strings.ToLower(message)
	
	if strings.Contains(message, "search") || strings.Contains(message, "find") {
		return "search"
	} else if strings.Contains(message, "summarize") || strings.Contains(message, "summary") {
		return "summarize"
	} else if strings.Contains(message, "analyze") || strings.Contains(message, "analysis") {
		return "analyze"
	} else if strings.Contains(message, "draft") || strings.Contains(message, "write") {
		return "draft"
	} else if strings.Contains(message, "explain") || strings.Contains(message, "what is") {
		return "explain"
	}
	
	return "general"
}

func (c *ChatService) getOrCreateSession(userID, sessionID string) *ChatSession {
	if sessionID == "" {
		sessionID = fmt.Sprintf("session_%d", time.Now().UnixNano())
	}
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if session, exists := c.activeChats[sessionID]; exists {
		return session
	}
	
	session := &ChatSession{
		ID:        sessionID,
		UserID:    userID,
		Messages:  []ChatMessage{},
		Context:   ChatContext{UserIntent: "general", Confidence: 0.5},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		IsActive:  true,
	}
	
	c.activeChats[sessionID] = session
	return session
}

func (c *ChatService) updateChatContext(context ChatContext, userMessage, response, intent string) ChatContext {
	context.UserIntent = intent
	context.Confidence = 0.8
	
	// Extract keywords from messages
	words := strings.Fields(strings.ToLower(userMessage + " " + response))
	for _, word := range words {
		if len(word) > 3 && !contains(context.Keywords, word) {
			context.Keywords = append(context.Keywords, word)
		}
	}
	
	// Add user action
	context.RecentActions = append(context.RecentActions, UserAction{
		Type:      "chat_message",
		Details:   map[string]interface{}{"intent": intent, "length": len(userMessage)},
		Timestamp: time.Now(),
	})
	
	// Keep only recent actions
	if len(context.RecentActions) > 10 {
		context.RecentActions = context.RecentActions[len(context.RecentActions)-10:]
	}
	
	return context
}

func (c *ChatService) generateSuggestions(response string, context ChatContext) []string {
	suggestions := []string{}
	
	switch context.UserIntent {
	case "search":
		suggestions = append(suggestions, "Refine search criteria", "Search in specific documents")
	case "summarize":
		suggestions = append(suggestions, "Generate detailed summary", "Create bullet points")
	case "analyze":
		suggestions = append(suggestions, "Deep dive analysis", "Compare with similar cases")
	case "draft":
		suggestions = append(suggestions, "Review draft", "Add legal citations")
	default:
		suggestions = append(suggestions, "Get more details", "Search related topics")
	}
	
	return suggestions
}

func (c *ChatService) updateUserActivity(userID string, activity map[string]interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	current := c.userActivity[userID]
	current.UserID = userID
	current.LastActivity = time.Now()
	
	if action, ok := activity["action"].(string); ok {
		current.RecentActions = append(current.RecentActions, UserAction{
			Type:      action,
			Details:   activity,
			Timestamp: time.Now(),
		})
	}
	
	// Determine attention level based on activity
	if time.Since(current.LastActivity) < 30*time.Second {
		current.AttentionLevel = "high"
	} else if time.Since(current.LastActivity) < 2*time.Minute {
		current.AttentionLevel = "medium"
	} else {
		current.AttentionLevel = "low"
	}
	
	c.userActivity[userID] = current
}

func (c *ChatService) monitorUserActivity() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		c.mu.Lock()
		for userID, activity := range c.userActivity {
			// Update attention level based on time since last activity
			if time.Since(activity.LastActivity) > 5*time.Minute {
				activity.AttentionLevel = "low"
				c.userActivity[userID] = activity
			}
			
			// Clean up old sessions
			for sessionID, session := range c.activeChats {
				if time.Since(session.UpdatedAt) > 30*time.Minute {
					c.storeSession(session)
					delete(c.activeChats, sessionID)
				}
			}
		}
		c.mu.Unlock()
	}
}

func (c *ChatService) storeSession(session *ChatSession) {
	// Store session in database (simplified)
	sessionData, _ := json.Marshal(session)
	c.redis.Set(context.Background(), "chat_session:"+session.ID, sessionData, time.Hour)
}

func (c *ChatService) updateSessionWithResponse(session *ChatSession, userMessage, response string, tokenCount int) {
	userMsg := ChatMessage{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		Role:      "user",
		Content:   userMessage,
		Timestamp: time.Now(),
	}
	
	assistantMsg := ChatMessage{
		ID:        fmt.Sprintf("msg_%d", time.Now().UnixNano()+1),
		Role:      "assistant",
		Content:   response,
		Timestamp: time.Now(),
		Metadata:  map[string]interface{}{"token_count": tokenCount},
	}
	
	session.Messages = append(session.Messages, userMsg, assistantMsg)
	session.UpdatedAt = time.Now()
	
	c.storeSession(session)
}

// WebSocket handler for real-time user activity
func (c *ChatService) HandleActivityWebSocket(ctx *gin.Context) {
	conn, err := c.wsUpgrader.Upgrade(ctx.Writer, ctx.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()
	
	userID := ctx.Query("user_id")
	if userID == "" {
		return
	}
	
	for {
		var activity map[string]interface{}
		if err := conn.ReadJSON(&activity); err != nil {
			break
		}
		
		c.updateUserActivity(userID, activity)
		
		// Send back activity status
		current := c.userActivity[userID]
		conn.WriteJSON(map[string]interface{}{
			"user_id":         userID,
			"attention_level": current.AttentionLevel,
			"suggestions":     c.getActivityBasedSuggestions(current),
			"timestamp":       time.Now(),
		})
	}
}

func (c *ChatService) getActivityBasedSuggestions(activity UserActivity) []string {
	suggestions := []string{}
	
	switch activity.AttentionLevel {
	case "low":
		suggestions = append(suggestions, "Take a break", "Review recent work")
	case "medium":
		suggestions = append(suggestions, "Continue current task", "Check for updates")
	case "high":
		suggestions = append(suggestions, "Deep focus session", "Tackle complex tasks")
	}
	
	return suggestions
}

func (c *ChatService) HandleStatus(ctx *gin.Context) {
	status := map[string]interface{}{
		"service":       "Go-Llama Chat Service",
		"status":        "running",
		"active_chats":  len(c.activeChats),
		"active_users":  len(c.userActivity),
		"llama_running": c.llamaCmd.Process != nil,
		"model_path":    c.config.ModelPath,
		"context_size":  c.config.ContextSize,
		"gpu_layers":    c.config.GPULayers,
		"timestamp":     time.Now(),
	}
	
	ctx.JSON(http.StatusOK, status)
}

func (c *ChatService) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())
	
	// CORS middleware
	router.Use(func(ctx *gin.Context) {
		ctx.Header("Access-Control-Allow-Origin", "*")
		ctx.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		ctx.Header("Access-Control-Allow-Headers", "Content-Type")
		
		if ctx.Request.Method == "OPTIONS" {
			ctx.AbortWithStatus(204)
			return
		}
		
		ctx.Next()
	})
	
	// API routes
	api := router.Group("/api")
	{
		api.POST("/chat", c.HandleChat)
		api.GET("/status", c.HandleStatus)
		api.GET("/sessions/:user_id", func(ctx *gin.Context) {
			userID := ctx.Param("user_id")
			sessions := []ChatSession{}
			
			for _, session := range c.activeChats {
				if session.UserID == userID {
					sessions = append(sessions, *session)
				}
			}
			
			ctx.JSON(http.StatusOK, sessions)
		})
	}
	
	// WebSocket endpoints
	router.GET("/ws/chat", c.HandleChat)
	router.GET("/ws/activity", c.HandleActivityWebSocket)
	
	// Root endpoint
	router.GET("/", func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"service": "Go-Llama Chat Service",
			"version": "1.0.0",
			"status":  "running",
			"model":   c.config.ModelPath,
			"endpoints": []string{
				"/api/chat", "/api/status", "/api/sessions/:user_id",
				"/ws/chat", "/ws/activity",
			},
		})
	})
	
	return router
}

func (c *ChatService) Run() error {
	if err := c.Initialize(); err != nil {
		return err
	}
	
	router := c.setupRoutes()
	
	log.Printf("🚀 Go-Llama Chat Service starting on port %s", c.config.Port)
	log.Printf("🦙 Model: %s", c.config.ModelPath)
	log.Printf("🔧 Context: %d, GPU Layers: %d", c.config.ContextSize, c.config.GPULayers)
	log.Printf("💬 WebSocket: ws://localhost:%s/ws/chat", c.config.Port)
	
	return router.Run(":" + c.config.Port)
}

func (c *ChatService) Cleanup() {
	if c.llamaCmd != nil && c.llamaCmd.Process != nil {
		c.llamaCmd.Process.Kill()
	}
	
	if c.db != nil {
		c.db.Close()
	}
	
	if c.redis != nil {
		c.redis.Close()
	}
}

// Utility functions
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if f, err := strconv.ParseFloat(value, 64); err == nil {
			return f
		}
	}
	return defaultValue
}

func main() {
	service := NewChatService()
	defer service.Cleanup()
	
	if err := service.Run(); err != nil {
		log.Fatalf("💥 Chat service failed: %v", err)
	}
}