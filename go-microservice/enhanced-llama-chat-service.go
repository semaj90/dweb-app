/*
Enhanced Go Llama Chat Service with PostgreSQL Integration
Stateful chat service with long-term memory using pgvector

Prerequisites:
1. PostgreSQL with pgvector extension
2. Ollama running with nomic-embed-text model
3. Environment variables: DATABASE_URL, OLLAMA_URL, MODEL_PATH

Database Setup:
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    embedding VECTOR(384),  -- nomic-embed-text dimensions
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_chat_conversation_id (conversation_id),
    INDEX idx_chat_created_at (created_at),
    INDEX idx_chat_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
);

CREATE TABLE IF NOT EXISTS conversation_metadata (
    conversation_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    title TEXT,
    summary TEXT,
    total_messages INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
*/

package main

import (
    "bytes"
    "database/sql"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "os"
    "strconv"
    "strings"
    "time"

    _ "github.com/lib/pq"
    "github.com/google/uuid"
)

// Request/Response structures
type ChatRequest struct {
    ConversationID string `json:"conversation_id"`
    UserID         string `json:"user_id,omitempty"`
    Message        string `json:"message"`
    Temperature    float32 `json:"temperature,omitempty"`
    MaxTokens      int     `json:"max_tokens,omitempty"`
    SystemPrompt   string  `json:"system_prompt,omitempty"`
}

type ChatResponse struct {
    ConversationID string            `json:"conversation_id"`
    Response       string            `json:"response"`
    MessageID      string            `json:"message_id"`
    Metadata       map[string]interface{} `json:"metadata"`
    Error          string            `json:"error,omitempty"`
}

type ChatHistory struct {
    ID             string                 `json:"id"`
    ConversationID string                 `json:"conversation_id"`
    UserID         string                 `json:"user_id"`
    Role           string                 `json:"role"`
    Content        string                 `json:"content"`
    Embedding      []float64              `json:"embedding,omitempty"`
    Metadata       map[string]interface{} `json:"metadata"`
    CreatedAt      time.Time             `json:"created_at"`
}

type ConversationSummary struct {
    ConversationID string                 `json:"conversation_id"`
    UserID         string                 `json:"user_id"`
    Title          string                 `json:"title"`
    Summary        string                 `json:"summary"`
    TotalMessages  int                    `json:"total_messages"`
    LastActivity   time.Time             `json:"last_activity"`
    Metadata       map[string]interface{} `json:"metadata"`
}

type OllamaEmbeddingRequest struct {
    Model  string `json:"model"`
    Prompt string `json:"prompt"`
}

type OllamaGenerateRequest struct {
    Model  string `json:"model"`
    Prompt string `json:"prompt"`
    Temperature float32 `json:"options,omitempty"`
    Stream bool `json:"stream"`
}

type OllamaGenerateResponse struct {
    Model     string `json:"model"`
    CreatedAt string `json:"created_at"`
    Response  string `json:"response"`
    Done      bool   `json:"done"`
}

type OllamaEmbeddingResponse struct {
    Embedding []float64 `json:"embedding"`
}

// Enhanced App with database connections
type App struct {
    db        *sql.DB
    ollamaURL string
    modelName string
}

// Initialize the enhanced chat service
func NewApp() (*App, error) {
    // Get configuration from environment
    databaseURL := os.Getenv("DATABASE_URL")
    if databaseURL == "" {
        databaseURL = "postgresql://postgres:123456@localhost:5432/legal_ai_db"
    }

    ollamaURL := os.Getenv("OLLAMA_URL")
    if ollamaURL == "" {
        ollamaURL = "http://localhost:11434"
    }

    modelName := os.Getenv("MODEL_NAME")
    if modelName == "" {
        modelName = "gemma3-legal"
    }

    log.Printf("Initializing Enhanced Llama Chat Service...")
    log.Printf("Database: %s", databaseURL)
    log.Printf("Ollama: %s", ollamaURL)
    log.Printf("Model: %s", modelName)

    // Connect to PostgreSQL
    db, err := sql.Open("postgres", databaseURL)
    if err != nil {
        return nil, fmt.Errorf("database connection failed: %v", err)
    }

    // Test database connection
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("database ping failed: %v", err)
    }

    log.Println("✅ Database connected successfully")

    // Test Ollama connection
    resp, err := http.Get(ollamaURL + "/api/tags")
    if err != nil {
        log.Printf("Warning: Cannot connect to Ollama at %s: %v", ollamaURL, err)
    } else {
        resp.Body.Close()
        log.Println("✅ Ollama connection verified")
    }

    return &App{
        db:        db,
        ollamaURL: ollamaURL,
        modelName: modelName,
    }, nil
}

// Generate embeddings using Ollama
func (app *App) generateEmbedding(text string) ([]float64, error) {
    reqBody := OllamaEmbeddingRequest{
        Model:  "nomic-embed-text",
        Prompt: text,
    }

    jsonData, err := json.Marshal(reqBody)
    if err != nil {
        return nil, err
    }

    resp, err := http.Post(
        app.ollamaURL+"/api/embeddings",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return nil, fmt.Errorf("ollama request failed: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("ollama returned status %d", resp.StatusCode)
    }

    var embeddingResp OllamaEmbeddingResponse
    if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
        return nil, err
    }

    return embeddingResp.Embedding, nil
}

// Generate text completion using Ollama
func (app *App) generateCompletion(prompt string, temperature float32, maxTokens int) (string, error) {
    reqBody := OllamaGenerateRequest{
        Model:       app.modelName,
        Prompt:      prompt,
        Temperature: temperature,
        Stream:      false,
    }

    jsonData, err := json.Marshal(reqBody)
    if err != nil {
        return "", err
    }

    resp, err := http.Post(
        app.ollamaURL+"/api/generate",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    if err != nil {
        return "", fmt.Errorf("ollama request failed: %v", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return "", fmt.Errorf("ollama returned status %d", resp.StatusCode)
    }

    var generateResp OllamaGenerateResponse
    if err := json.NewDecoder(resp.Body).Decode(&generateResp); err != nil {
        return "", err
    }

    return generateResp.Response, nil
}

// Store message in chat history with embedding
func (app *App) storeMessage(conversationID, userID, role, content string, metadata map[string]interface{}) (string, error) {
    messageID := uuid.New().String()

    // Generate embedding for semantic search
    embedding, err := app.generateEmbedding(content)
    if err != nil {
        log.Printf("Warning: Failed to generate embedding: %v", err)
        // Continue without embedding rather than failing
        embedding = nil
    }

    // Convert embedding to PostgreSQL array format
    var embeddingStr string
    if embedding != nil {
        floatStrings := make([]string, len(embedding))
        for i, v := range embedding {
            floatStrings[i] = fmt.Sprintf("%f", v)
        }
        embeddingStr = "[" + strings.Join(floatStrings, ",") + "]"
    }

    // Convert metadata to JSON
    metadataJSON, err := json.Marshal(metadata)
    if err != nil {
        metadataJSON = []byte("{}")
    }

    // Insert into chat_history
    query := `
        INSERT INTO chat_history (id, conversation_id, user_id, role, content, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    `

    var embeddingValue interface{}
    if embeddingStr != "" {
        embeddingValue = embeddingStr
    } else {
        embeddingValue = nil
    }

    _, err = app.db.Exec(query, messageID, conversationID, userID, role, content, embeddingValue, string(metadataJSON))
    if err != nil {
        return "", fmt.Errorf("failed to store message: %v", err)
    }

    // Update conversation metadata
    app.updateConversationMetadata(conversationID, userID)

    return messageID, nil
}

// Update conversation metadata and statistics
func (app *App) updateConversationMetadata(conversationID, userID string) error {
    query := `
        INSERT INTO conversation_metadata (conversation_id, user_id, total_messages, last_activity)
        VALUES ($1, $2, 1, NOW())
        ON CONFLICT (conversation_id) 
        DO UPDATE SET 
            total_messages = conversation_metadata.total_messages + 1,
            last_activity = NOW()
    `

    _, err := app.db.Exec(query, conversationID, userID)
    return err
}

// Retrieve conversation history with context window
func (app *App) getConversationHistory(conversationID string, limit int) ([]ChatHistory, error) {
    if limit <= 0 {
        limit = 10 // Default context window
    }

    query := `
        SELECT id, conversation_id, user_id, role, content, metadata, created_at
        FROM chat_history 
        WHERE conversation_id = $1 
        ORDER BY created_at DESC 
        LIMIT $2
    `

    rows, err := app.db.Query(query, conversationID, limit)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var history []ChatHistory
    for rows.Next() {
        var msg ChatHistory
        var metadataJSON string

        err := rows.Scan(
            &msg.ID,
            &msg.ConversationID,
            &msg.UserID,
            &msg.Role,
            &msg.Content,
            &metadataJSON,
            &msg.CreatedAt,
        )
        if err != nil {
            return nil, err
        }

        // Parse metadata JSON
        if err := json.Unmarshal([]byte(metadataJSON), &msg.Metadata); err != nil {
            msg.Metadata = make(map[string]interface{})
        }

        history = append(history, msg)
    }

    // Reverse to get chronological order
    for i := len(history)/2 - 1; i >= 0; i-- {
        opp := len(history) - 1 - i
        history[i], history[opp] = history[opp], history[i]
    }

    return history, nil
}

// Semantic search in conversation history
func (app *App) semanticSearch(query string, conversationID string, limit int) ([]ChatHistory, error) {
    if limit <= 0 {
        limit = 5
    }

    // Generate embedding for search query
    queryEmbedding, err := app.generateEmbedding(query)
    if err != nil {
        return nil, fmt.Errorf("failed to generate query embedding: %v", err)
    }

    // Convert to PostgreSQL array format
    floatStrings := make([]string, len(queryEmbedding))
    for i, v := range queryEmbedding {
        floatStrings[i] = fmt.Sprintf("%f", v)
    }
    queryEmbeddingStr := "[" + strings.Join(floatStrings, ",") + "]"

    sqlQuery := `
        SELECT id, conversation_id, user_id, role, content, metadata, created_at,
               1 - (embedding <=> $1::vector) AS similarity
        FROM chat_history 
        WHERE conversation_id = $2 
          AND embedding IS NOT NULL
          AND 1 - (embedding <=> $1::vector) > 0.7
        ORDER BY similarity DESC
        LIMIT $3
    `

    rows, err := app.db.Query(sqlQuery, queryEmbeddingStr, conversationID, limit)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var results []ChatHistory
    for rows.Next() {
        var msg ChatHistory
        var metadataJSON string
        var similarity float64

        err := rows.Scan(
            &msg.ID,
            &msg.ConversationID,
            &msg.UserID,
            &msg.Role,
            &msg.Content,
            &metadataJSON,
            &msg.CreatedAt,
            &similarity,
        )
        if err != nil {
            return nil, err
        }

        if err := json.Unmarshal([]byte(metadataJSON), &msg.Metadata); err != nil {
            msg.Metadata = make(map[string]interface{})
        }

        msg.Metadata["similarity"] = similarity
        results = append(results, msg)
    }

    return results, nil
}

// Build context prompt from conversation history
func (app *App) buildContextPrompt(history []ChatHistory, systemPrompt string) string {
    var prompt strings.Builder

    if systemPrompt != "" {
        prompt.WriteString(systemPrompt + "\n\n")
    } else {
        prompt.WriteString("You are a helpful legal AI assistant. Provide accurate, professional legal information.\n\n")
    }

    prompt.WriteString("Conversation History:\n")
    
    for _, msg := range history {
        if msg.Role == "user" {
            prompt.WriteString(fmt.Sprintf("Human: %s\n", msg.Content))
        } else {
            prompt.WriteString(fmt.Sprintf("Assistant: %s\n", msg.Content))
        }
    }

    return prompt.String()
}

// Enhanced chat handler with memory and context
func (app *App) handleEnhancedChat(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Only POST method allowed", http.StatusMethodNotAllowed)
        return
    }

    var req ChatRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    // Set defaults
    if req.ConversationID == "" {
        req.ConversationID = uuid.New().String()
    }
    if req.Temperature == 0 {
        req.Temperature = 0.7
    }
    if req.MaxTokens == 0 {
        req.MaxTokens = 1024
    }

    log.Printf("Processing chat request for conversation: %s", req.ConversationID)

    // Store user message
    userMetadata := map[string]interface{}{
        "temperature": req.Temperature,
        "max_tokens":  req.MaxTokens,
        "timestamp":   time.Now().Unix(),
    }

    userMessageID, err := app.storeMessage(req.ConversationID, req.UserID, "user", req.Message, userMetadata)
    if err != nil {
        log.Printf("Error storing user message: %v", err)
        // Continue despite storage error
    }

    // Get conversation history for context
    history, err := app.getConversationHistory(req.ConversationID, 20)
    if err != nil {
        log.Printf("Error retrieving history: %v", err)
        history = []ChatHistory{} // Continue with empty history
    }

    // Build context-aware prompt
    contextPrompt := app.buildContextPrompt(history, req.SystemPrompt)
    contextPrompt += fmt.Sprintf("\nHuman: %s\nAssistant:", req.Message)

    log.Printf("Generated context prompt length: %d characters", len(contextPrompt))

    // Generate AI response using Ollama
    result, err := app.generateCompletion(contextPrompt, req.Temperature, req.MaxTokens)
    if err != nil {
        log.Printf("Inference error: %v", err)
        response := ChatResponse{
            ConversationID: req.ConversationID,
            Error:          err.Error(),
        }
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(http.StatusInternalServerError)
        json.NewEncoder(w).Encode(response)
        return
    }

    // Clean up the response
    result = strings.TrimSpace(result)
    if strings.HasPrefix(result, "Assistant:") {
        result = strings.TrimSpace(result[10:])
    }

    // Store AI response
    aiMetadata := map[string]interface{}{
        "model":           app.modelName,
        "provider":        "ollama",
        "temperature":     req.Temperature,
        "max_tokens":      req.MaxTokens,
        "user_message_id": userMessageID,
        "response_length": len(result),
        "timestamp":       time.Now().Unix(),
    }

    aiMessageID, err := app.storeMessage(req.ConversationID, req.UserID, "assistant", result, aiMetadata)
    if err != nil {
        log.Printf("Error storing AI message: %v", err)
        // Continue despite storage error
    }

    log.Printf("Chat response generated successfully. Length: %d", len(result))

    // Send response
    response := ChatResponse{
        ConversationID: req.ConversationID,
        Response:       result,
        MessageID:      aiMessageID,
        Metadata: map[string]interface{}{
            "user_message_id": userMessageID,
            "history_length":  len(history),
            "model":           app.modelName,
            "provider":        "ollama",
        },
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// Get conversation summaries
func (app *App) handleGetConversations(w http.ResponseWriter, r *http.Request) {
    userID := r.URL.Query().Get("user_id")
    limitStr := r.URL.Query().Get("limit")
    
    limit := 10
    if limitStr != "" {
        if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
            limit = l
        }
    }

    query := `
        SELECT conversation_id, user_id, title, summary, total_messages, last_activity, metadata
        FROM conversation_metadata
    `
    args := []interface{}{}

    if userID != "" {
        query += " WHERE user_id = $1"
        args = append(args, userID)
        query += " ORDER BY last_activity DESC LIMIT $2"
        args = append(args, limit)
    } else {
        query += " ORDER BY last_activity DESC LIMIT $1"
        args = append(args, limit)
    }

    rows, err := app.db.Query(query, args...)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer rows.Close()

    var conversations []ConversationSummary
    for rows.Next() {
        var conv ConversationSummary
        var metadataJSON sql.NullString
        
        err := rows.Scan(
            &conv.ConversationID,
            &conv.UserID,
            &conv.Title,
            &conv.Summary,
            &conv.TotalMessages,
            &conv.LastActivity,
            &metadataJSON,
        )
        if err != nil {
            continue
        }

        if metadataJSON.Valid {
            json.Unmarshal([]byte(metadataJSON.String), &conv.Metadata)
        } else {
            conv.Metadata = make(map[string]interface{})
        }

        conversations = append(conversations, conv)
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(conversations)
}

// Health check endpoint
func (app *App) handleHealth(w http.ResponseWriter, r *http.Request) {
    status := map[string]interface{}{
        "status":    "healthy",
        "timestamp": time.Now().Unix(),
        "database":  "connected",
        "model":     "loaded",
        "ollama":    "available",
    }

    // Test database connection
    if err := app.db.Ping(); err != nil {
        status["database"] = "error: " + err.Error()
        status["status"] = "degraded"
    }

    // Test Ollama connection
    resp, err := http.Get(app.ollamaURL + "/api/tags")
    if err != nil || resp.StatusCode != http.StatusOK {
        status["ollama"] = "unavailable"
        status["status"] = "degraded"
    }
    if resp != nil {
        resp.Body.Close()
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(status)
}

// Cleanup resources
func (app *App) cleanup() {
    if app.db != nil {
        app.db.Close()
    }
}

func main() {
    app, err := NewApp()
    if err != nil {
        log.Fatalf("Failed to initialize app: %v", err)
    }
    defer app.cleanup()

    // Set up HTTP routes
    mux := http.NewServeMux()
    mux.HandleFunc("/api/chat", app.handleEnhancedChat)
    mux.HandleFunc("/api/conversations", app.handleGetConversations)
    mux.HandleFunc("/api/health", app.handleHealth)

    // Get port from environment
    port := os.Getenv("PORT")
    if port == "" {
        port = "8081"
    }

    serverAddr := fmt.Sprintf(":%s", port)
    log.Printf("🚀 Enhanced Go Llama Chat Service starting on %s", serverAddr)
    log.Printf("💾 Database: PostgreSQL with pgvector")
    log.Printf("🧠 Embedding: Ollama nomic-embed-text")
    log.Printf("🤖 Model: Ollama %s with context memory", app.modelName)
    
    log.Println("API Endpoints:")
    log.Println("  POST /api/chat - Enhanced chat with memory")
    log.Println("  GET  /api/conversations - List conversations")
    log.Println("  GET  /api/health - Service health check")

    if err := http.ListenAndServe(serverAddr, mux); err != nil {
        log.Fatalf("Server failed: %v", err)
    }
}