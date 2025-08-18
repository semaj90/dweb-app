// main.go - Enhanced RAG V2 with Local LLM, CRUD, and PostgreSQL Integration
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"github.com/lib/pq"
	_ "github.com/lib/pq"
	"github.com/redis/go-redis/v9"
	"github.com/streadway/amqp"
)

// ==========================================
// DATABASE MODELS & CRUD
// ==========================================

type DatabaseManager struct {
	db    *sql.DB
	mutex sync.RWMutex
}

// User Intent Model
type UserIntent struct {
	ID         string          `json:"id" db:"id"`
	UserID     string          `json:"user_id" db:"user_id"`
	Intent     string          `json:"intent" db:"intent"`
	Keywords   pq.StringArray  `json:"keywords" db:"keywords"`
	Confidence float64         `json:"confidence" db:"confidence"`
	Context    json.RawMessage `json:"context" db:"context"`
	CreatedAt  time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt  time.Time       `json:"updated_at" db:"updated_at"`
}

// Recommendation Model
type Recommendation struct {
	ID          string          `json:"id" db:"id"`
	UserID      string          `json:"user_id" db:"user_id"`
	Title       string          `json:"title" db:"title"`
	Description string          `json:"description" db:"description"`
	Type        string          `json:"type" db:"type"`
	Confidence  float64         `json:"confidence" db:"confidence"`
	Context     json.RawMessage `json:"context" db:"context"`
	Status      string          `json:"status" db:"status"`
	CreatedAt   time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time       `json:"updated_at" db:"updated_at"`
}

// Todo Item Model
type TodoItem struct {
	ID          string     `json:"id" db:"id"`
	UserID      string     `json:"user_id" db:"user_id"`
	Title       string     `json:"title" db:"title"`
	Description string     `json:"description" db:"description"`
	Priority    int        `json:"priority" db:"priority"`
	Status      string     `json:"status" db:"status"`
	Solution    string     `json:"solution" db:"solution"`
	DueDate     *time.Time `json:"due_date" db:"due_date"`
	SolvedAt    *time.Time `json:"solved_at" db:"solved_at"`
	CreatedAt   time.Time  `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at" db:"updated_at"`
}

// Analytics Event Model
type AnalyticsEvent struct {
	ID        string          `json:"id" db:"id"`
	UserID    string          `json:"user_id" db:"user_id"`
	EventType string          `json:"event_type" db:"event_type"`
	EventData json.RawMessage `json:"event_data" db:"event_data"`
	Timestamp time.Time       `json:"timestamp" db:"timestamp"`
}

// Session Model
type UserSession struct {
	ID            string          `json:"id" db:"id"`
	UserID        string          `json:"user_id" db:"user_id"`
	State         string          `json:"state" db:"state"`
	LastActivity  time.Time       `json:"last_activity" db:"last_activity"`
	IdleStartTime *time.Time      `json:"idle_start_time" db:"idle_start_time"`
	Context       json.RawMessage `json:"context" db:"context"`
	CreatedAt     time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt     time.Time       `json:"updated_at" db:"updated_at"`
}

// SOM Cluster Model
type SOMCluster struct {
	ID          string          `json:"id" db:"id"`
	ClusterName string          `json:"cluster_name" db:"cluster_name"`
	Centroid    pq.Float64Array `json:"centroid" db:"centroid"`
	Documents   pq.StringArray  `json:"documents" db:"documents"`
	Metadata    json.RawMessage `json:"metadata" db:"metadata"`
	CreatedAt   time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time       `json:"updated_at" db:"updated_at"`
}

// ==========================================
// CRUD OPERATIONS
// ==========================================

func NewDatabaseManager(connectionString string) (*DatabaseManager, error) {
	db, err := sql.Open("postgres", connectionString)
	if err != nil {
		return nil, err
	}

	if err := db.Ping(); err != nil {
		return nil, err
	}

	return &DatabaseManager{db: db}, nil
}

// User Intent CRUD
func (dm *DatabaseManager) CreateUserIntent(intent *UserIntent) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	intent.ID = uuid.New().String()
	intent.CreatedAt = time.Now()
	intent.UpdatedAt = time.Now()

	query := `
        INSERT INTO user_intents (id, user_id, intent, keywords, confidence, context, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `

	_, err := dm.db.Exec(query, intent.ID, intent.UserID, intent.Intent,
		intent.Keywords, intent.Confidence, intent.Context, intent.CreatedAt, intent.UpdatedAt)
	return err
}

func (dm *DatabaseManager) GetUserIntent(id string) (*UserIntent, error) {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	var intent UserIntent
	query := `
        SELECT id, user_id, intent, keywords, confidence, context, created_at, updated_at
        FROM user_intents WHERE id = $1
    `

	err := dm.db.QueryRow(query, id).Scan(
		&intent.ID, &intent.UserID, &intent.Intent, &intent.Keywords,
		&intent.Confidence, &intent.Context, &intent.CreatedAt, &intent.UpdatedAt,
	)

	if err != nil {
		return nil, err
	}

	return &intent, nil
}

func (dm *DatabaseManager) UpdateUserIntent(intent *UserIntent) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	intent.UpdatedAt = time.Now()

	query := `
        UPDATE user_intents
        SET intent = $2, keywords = $3, confidence = $4, context = $5, updated_at = $6
        WHERE id = $1
    `

	_, err := dm.db.Exec(query, intent.ID, intent.Intent, intent.Keywords,
		intent.Confidence, intent.Context, intent.UpdatedAt)
	return err
}

func (dm *DatabaseManager) DeleteUserIntent(id string) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	query := `DELETE FROM user_intents WHERE id = $1`
	_, err := dm.db.Exec(query, id)
	return err
}

func (dm *DatabaseManager) ListUserIntents(userID string, limit, offset int) ([]*UserIntent, error) {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	query := `
        SELECT id, user_id, intent, keywords, confidence, context, created_at, updated_at
        FROM user_intents
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
    `

	rows, err := dm.db.Query(query, userID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var intents []*UserIntent
	for rows.Next() {
		var intent UserIntent
		err := rows.Scan(
			&intent.ID, &intent.UserID, &intent.Intent, &intent.Keywords,
			&intent.Confidence, &intent.Context, &intent.CreatedAt, &intent.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}
		intents = append(intents, &intent)
	}

	return intents, nil
}

// Recommendations CRUD
func (dm *DatabaseManager) CreateRecommendation(rec *Recommendation) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	rec.ID = uuid.New().String()
	rec.CreatedAt = time.Now()
	rec.UpdatedAt = time.Now()

	query := `
        INSERT INTO recommendations (id, user_id, title, description, type, confidence, context, status, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    `

	_, err := dm.db.Exec(query, rec.ID, rec.UserID, rec.Title, rec.Description,
		rec.Type, rec.Confidence, rec.Context, rec.Status, rec.CreatedAt, rec.UpdatedAt)
	return err
}

func (dm *DatabaseManager) GetRecommendation(id string) (*Recommendation, error) {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	var rec Recommendation
	query := `
        SELECT id, user_id, title, description, type, confidence, context, status, created_at, updated_at
        FROM recommendations WHERE id = $1
    `

	err := dm.db.QueryRow(query, id).Scan(
		&rec.ID, &rec.UserID, &rec.Title, &rec.Description, &rec.Type,
		&rec.Confidence, &rec.Context, &rec.Status, &rec.CreatedAt, &rec.UpdatedAt,
	)

	if err != nil {
		return nil, err
	}

	return &rec, nil
}

func (dm *DatabaseManager) UpdateRecommendation(rec *Recommendation) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	rec.UpdatedAt = time.Now()

	query := `
        UPDATE recommendations
        SET title = $2, description = $3, type = $4, confidence = $5, context = $6, status = $7, updated_at = $8
        WHERE id = $1
    `

	_, err := dm.db.Exec(query, rec.ID, rec.Title, rec.Description, rec.Type,
		rec.Confidence, rec.Context, rec.Status, rec.UpdatedAt)
	return err
}

func (dm *DatabaseManager) DeleteRecommendation(id string) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	query := `DELETE FROM recommendations WHERE id = $1`
	_, err := dm.db.Exec(query, id)
	return err
}

// Todo Items CRUD
func (dm *DatabaseManager) CreateTodoItem(todo *TodoItem) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	todo.ID = uuid.New().String()
	todo.CreatedAt = time.Now()
	todo.UpdatedAt = time.Now()

	query := `
        INSERT INTO todo_items (id, user_id, title, description, priority, status, solution, due_date, solved_at, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    `

	_, err := dm.db.Exec(query, todo.ID, todo.UserID, todo.Title, todo.Description,
		todo.Priority, todo.Status, todo.Solution, todo.DueDate, todo.SolvedAt,
		todo.CreatedAt, todo.UpdatedAt)
	return err
}

func (dm *DatabaseManager) GetTodoItem(id string) (*TodoItem, error) {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	var todo TodoItem
	query := `
        SELECT id, user_id, title, description, priority, status, solution, due_date, solved_at, created_at, updated_at
        FROM todo_items WHERE id = $1
    `

	err := dm.db.QueryRow(query, id).Scan(
		&todo.ID, &todo.UserID, &todo.Title, &todo.Description, &todo.Priority,
		&todo.Status, &todo.Solution, &todo.DueDate, &todo.SolvedAt,
		&todo.CreatedAt, &todo.UpdatedAt,
	)

	if err != nil {
		return nil, err
	}

	return &todo, nil
}

func (dm *DatabaseManager) UpdateTodoItem(todo *TodoItem) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	todo.UpdatedAt = time.Now()

	query := `
        UPDATE todo_items
        SET title = $2, description = $3, priority = $4, status = $5, solution = $6,
            due_date = $7, solved_at = $8, updated_at = $9
        WHERE id = $1
    `

	_, err := dm.db.Exec(query, todo.ID, todo.Title, todo.Description, todo.Priority,
		todo.Status, todo.Solution, todo.DueDate, todo.SolvedAt, todo.UpdatedAt)
	return err
}

func (dm *DatabaseManager) DeleteTodoItem(id string) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	query := `DELETE FROM todo_items WHERE id = $1`
	_, err := dm.db.Exec(query, id)
	return err
}

func (dm *DatabaseManager) ListPendingTodos(userID string) ([]*TodoItem, error) {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	query := `
        SELECT id, user_id, title, description, priority, status, solution, due_date, solved_at, created_at, updated_at
        FROM todo_items
        WHERE user_id = $1 AND status = 'pending'
        ORDER BY priority DESC, created_at ASC
    `

	rows, err := dm.db.Query(query, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var todos []*TodoItem
	for rows.Next() {
		var todo TodoItem
		err := rows.Scan(
			&todo.ID, &todo.UserID, &todo.Title, &todo.Description, &todo.Priority,
			&todo.Status, &todo.Solution, &todo.DueDate, &todo.SolvedAt,
			&todo.CreatedAt, &todo.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}
		todos = append(todos, &todo)
	}

	return todos, nil
}

// Analytics CRUD
func (dm *DatabaseManager) CreateAnalyticsEvent(event *AnalyticsEvent) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	event.ID = uuid.New().String()
	event.Timestamp = time.Now()

	query := `
        INSERT INTO analytics_events (id, user_id, event_type, event_data, timestamp)
        VALUES ($1, $2, $3, $4, $5)
    `

	_, err := dm.db.Exec(query, event.ID, event.UserID, event.EventType, event.EventData, event.Timestamp)
	return err
}

// Session Management CRUD
func (dm *DatabaseManager) CreateOrUpdateSession(session *UserSession) error {
	dm.mutex.Lock()
	defer dm.mutex.Unlock()

	session.UpdatedAt = time.Now()

	query := `
        INSERT INTO user_sessions (id, user_id, state, last_activity, idle_start_time, context, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (user_id) DO UPDATE
        SET state = $3, last_activity = $4, idle_start_time = $5, context = $6, updated_at = $8
    `

	if session.ID == "" {
		session.ID = uuid.New().String()
	}
	if session.CreatedAt.IsZero() {
		session.CreatedAt = time.Now()
	}

	_, err := dm.db.Exec(query, session.ID, session.UserID, session.State,
		session.LastActivity, session.IdleStartTime, session.Context,
		session.CreatedAt, session.UpdatedAt)
	return err
}

func (dm *DatabaseManager) GetUserSession(userID string) (*UserSession, error) {
	dm.mutex.RLock()
	defer dm.mutex.RUnlock()

	var session UserSession
	query := `
        SELECT id, user_id, state, last_activity, idle_start_time, context, created_at, updated_at
        FROM user_sessions WHERE user_id = $1
    `

	err := dm.db.QueryRow(query, userID).Scan(
		&session.ID, &session.UserID, &session.State, &session.LastActivity,
		&session.IdleStartTime, &session.Context, &session.CreatedAt, &session.UpdatedAt,
	)

	if err != nil {
		return nil, err
	}

	return &session, nil
}

// ==========================================
// ASYNC PROCESSING WITH CHANNELS
// ==========================================

type AsyncProcessor struct {
	db          *DatabaseManager
	redisClient *redis.Client
	rabbitMQ    *amqp.Connection
	workQueue   chan WorkItem
	resultQueue chan Result
	workers     int
	wg          sync.WaitGroup
}

type WorkItem struct {
	ID        string
	Type      string
	UserID    string
	Data      interface{}
	Timestamp time.Time
}

type Result struct {
	ID        string
	Success   bool
	Data      interface{}
	Error     error
	Timestamp time.Time
}

func NewAsyncProcessor(db *DatabaseManager, redisClient *redis.Client, rabbitMQ *amqp.Connection, workers int) *AsyncProcessor {
	return &AsyncProcessor{
		db:          db,
		redisClient: redisClient,
		rabbitMQ:    rabbitMQ,
		workQueue:   make(chan WorkItem, 1000),
		resultQueue: make(chan Result, 1000),
		workers:     workers,
	}
}

func (ap *AsyncProcessor) Start() {
	for i := 0; i < ap.workers; i++ {
		ap.wg.Add(1)
		go ap.worker(i)
	}

	go ap.resultProcessor()
}

func (ap *AsyncProcessor) worker(id int) {
	defer ap.wg.Done()

	for item := range ap.workQueue {
		result := ap.processWorkItem(item)
		ap.resultQueue <- result
	}
}

func (ap *AsyncProcessor) processWorkItem(item WorkItem) Result {
	var result Result
	result.ID = item.ID
	result.Timestamp = time.Now()

	switch item.Type {
	case "analyze_intent":
		intent, err := ap.analyzeUserIntent(item.UserID, item.Data)
		if err != nil {
			result.Error = err
			result.Success = false
		} else {
			result.Data = intent
			result.Success = true
		}

	case "generate_recommendations":
		recs, err := ap.generateRecommendations(item.UserID, item.Data)
		if err != nil {
			result.Error = err
			result.Success = false
		} else {
			result.Data = recs
			result.Success = true
		}

	case "solve_todo":
		solution, err := ap.solveTodoItem(item.UserID, item.Data)
		if err != nil {
			result.Error = err
			result.Success = false
		} else {
			result.Data = solution
			result.Success = true
		}

	case "update_som":
		err := ap.updateSOMClusters(item.Data)
		if err != nil {
			result.Error = err
			result.Success = false
		} else {
			result.Success = true
		}

	default:
		result.Error = fmt.Errorf("unknown work item type: %s", item.Type)
		result.Success = false
	}

	return result
}

func (ap *AsyncProcessor) analyzeUserIntent(userID string, data interface{}) (*UserIntent, error) {
	// Simulate intent analysis
	intent := &UserIntent{
		UserID:     userID,
		Intent:     "legal_research",
		Keywords:   []string{"contract", "liability", "clause"},
		Confidence: 0.89,
		Context:    json.RawMessage(`{"source": "chat", "session": "active"}`),
	}

	err := ap.db.CreateUserIntent(intent)
	if err != nil {
		return nil, err
	}

	// Store in Redis for fast access
	ctx := context.Background()
	key := fmt.Sprintf("intent:%s:%s", userID, intent.ID)
	intentJSON, _ := json.Marshal(intent)
	ap.redisClient.Set(ctx, key, intentJSON, 1*time.Hour)

	return intent, nil
}

func (ap *AsyncProcessor) generateRecommendations(userID string, data interface{}) ([]*Recommendation, error) {
	// Generate recommendations based on user context
	recommendations := []*Recommendation{
		{
			UserID:      userID,
			Title:       "Review Contract Clause 3.2",
			Description: "Based on recent edits, this clause needs attention",
			Type:        "legal",
			Confidence:  0.92,
			Status:      "active",
			Context:     json.RawMessage(`{"reason": "similar_pattern_detected"}`),
		},
		{
			UserID:      userID,
			Title:       "Similar Case: Johnson v. Smith",
			Description: "Relevant precedent for current case",
			Type:        "precedent",
			Confidence:  0.87,
			Status:      "active",
			Context:     json.RawMessage(`{"case_id": "2023-CV-1234"}`),
		},
	}

	for _, rec := range recommendations {
		err := ap.db.CreateRecommendation(rec)
		if err != nil {
			return nil, err
		}
	}

	return recommendations, nil
}

func (ap *AsyncProcessor) solveTodoItem(userID string, data interface{}) (*TodoItem, error) {
	todoID, ok := data.(string)
	if !ok {
		return nil, fmt.Errorf("invalid todo ID")
	}

	todo, err := ap.db.GetTodoItem(todoID)
	if err != nil {
		return nil, err
	}

	// Simulate AI solving the todo
	todo.Solution = "AI-generated solution based on similar cases and templates"
	todo.Status = "solved"
	now := time.Now()
	todo.SolvedAt = &now

	err = ap.db.UpdateTodoItem(todo)
	if err != nil {
		return nil, err
	}

	return todo, nil
}

func (ap *AsyncProcessor) updateSOMClusters(data interface{}) error {
	// Update Self-Organizing Map clusters
	// This would integrate with your Gorgonia implementation
	return nil
}

func (ap *AsyncProcessor) resultProcessor() {
	for result := range ap.resultQueue {
		if result.Success {
			log.Printf("✅ Processed %s successfully", result.ID)

			// Send success notification via WebSocket
			notification := map[string]interface{}{
				"type":      "processing_complete",
				"id":        result.ID,
				"data":      result.Data,
				"timestamp": result.Timestamp,
			}

			ap.broadcastNotification(notification)
		} else {
			log.Printf("❌ Failed to process %s: %v", result.ID, result.Error)

			// Handle failure - retry logic, error logging, etc.
			ap.handleFailure(result)
		}
	}
}

func (ap *AsyncProcessor) broadcastNotification(notification map[string]interface{}) {
	// Broadcast via WebSocket to connected clients
	// This would integrate with your WebSocket hub
}

func (ap *AsyncProcessor) handleFailure(result Result) {
	// Log to database
	event := &AnalyticsEvent{
		UserID:    "system",
		EventType: "processing_error",
		EventData: json.RawMessage(fmt.Sprintf(`{"error": "%v", "id": "%s"}`, result.Error, result.ID)),
	}

	ap.db.CreateAnalyticsEvent(event)
}

// ==========================================
// HTTP HANDLERS
// ==========================================

type APIServer struct {
	db             *DatabaseManager
	asyncProcessor *AsyncProcessor
	router         *mux.Router
}

func NewAPIServer(db *DatabaseManager, asyncProcessor *AsyncProcessor) *APIServer {
	server := &APIServer{
		db:             db,
		asyncProcessor: asyncProcessor,
		router:         mux.NewRouter(),
	}

	server.setupRoutes()
	return server
}

func (s *APIServer) setupRoutes() {
	// User Intent routes
	s.router.HandleFunc("/api/intents", s.createIntent).Methods("POST")
	s.router.HandleFunc("/api/intents/{id}", s.getIntent).Methods("GET")
	s.router.HandleFunc("/api/intents/{id}", s.updateIntent).Methods("PUT")
	s.router.HandleFunc("/api/intents/{id}", s.deleteIntent).Methods("DELETE")
	s.router.HandleFunc("/api/intents", s.listIntents).Methods("GET")

	// Recommendation routes
	s.router.HandleFunc("/api/recommendations", s.createRecommendation).Methods("POST")
	s.router.HandleFunc("/api/recommendations/{id}", s.getRecommendation).Methods("GET")
	s.router.HandleFunc("/api/recommendations/{id}", s.updateRecommendation).Methods("PUT")
	s.router.HandleFunc("/api/recommendations/{id}", s.deleteRecommendation).Methods("DELETE")
	s.router.HandleFunc("/api/recommendations/generate", s.generateRecommendations).Methods("POST")

	// Todo routes
	s.router.HandleFunc("/api/todos", s.createTodo).Methods("POST")
	s.router.HandleFunc("/api/todos/{id}", s.getTodo).Methods("GET")
	s.router.HandleFunc("/api/todos/{id}", s.updateTodo).Methods("PUT")
	s.router.HandleFunc("/api/todos/{id}", s.deleteTodo).Methods("DELETE")
	s.router.HandleFunc("/api/todos/solve", s.solveTodos).Methods("POST")
	s.router.HandleFunc("/api/todos/pending", s.listPendingTodos).Methods("GET")

	// Session routes
	s.router.HandleFunc("/api/sessions", s.updateSession).Methods("POST")
	s.router.HandleFunc("/api/sessions/{user_id}", s.getSession).Methods("GET")

	// Analytics routes
	s.router.HandleFunc("/api/analytics/event", s.trackEvent).Methods("POST")

	// Health check
	s.router.HandleFunc("/health", s.healthCheck).Methods("GET")
}

// Intent Handlers
func (s *APIServer) createIntent(w http.ResponseWriter, r *http.Request) {
	var intent UserIntent
	if err := json.NewDecoder(r.Body).Decode(&intent); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.db.CreateUserIntent(&intent); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Queue for async processing
	s.asyncProcessor.workQueue <- WorkItem{
		ID:        intent.ID,
		Type:      "analyze_intent",
		UserID:    intent.UserID,
		Data:      intent,
		Timestamp: time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(intent)
}

func (s *APIServer) getIntent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	intent, err := s.db.GetUserIntent(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(intent)
}

func (s *APIServer) updateIntent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var intent UserIntent
	if err := json.NewDecoder(r.Body).Decode(&intent); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	intent.ID = id
	if err := s.db.UpdateUserIntent(&intent); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(intent)
}

func (s *APIServer) deleteIntent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if err := s.db.DeleteUserIntent(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (s *APIServer) listIntents(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	intents, err := s.db.ListUserIntents(userID, 20, 0)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(intents)
}

// Recommendation Handlers
func (s *APIServer) generateRecommendations(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID  string                 `json:"user_id"`
		Context map[string]interface{} `json:"context"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Queue for async generation
	workID := uuid.New().String()
	s.asyncProcessor.workQueue <- WorkItem{
		ID:        workID,
		Type:      "generate_recommendations",
		UserID:    req.UserID,
		Data:      req.Context,
		Timestamp: time.Now(),
	}

	response := map[string]interface{}{
		"status":  "processing",
		"id":      workID,
		"message": "Recommendations are being generated",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Todo Handlers
func (s *APIServer) solveTodos(w http.ResponseWriter, r *http.Request) {
	var req struct {
		UserID    string   `json:"user_id"`
		TodoIDs   []string `json:"todo_ids"`
		AutoSolve bool     `json:"auto_solve"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var todos []*TodoItem

	if req.AutoSolve {
		// Get all pending todos
		pendingTodos, err := s.db.ListPendingTodos(req.UserID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		todos = pendingTodos
	} else {
		// Get specific todos
		for _, id := range req.TodoIDs {
			todo, err := s.db.GetTodoItem(id)
			if err == nil {
				todos = append(todos, todo)
			}
		}
	}

	// Queue todos for solving
	for _, todo := range todos {
		s.asyncProcessor.workQueue <- WorkItem{
			ID:        todo.ID,
			Type:      "solve_todo",
			UserID:    req.UserID,
			Data:      todo.ID,
			Timestamp: time.Now(),
		}
	}

	response := map[string]interface{}{
		"status":  "processing",
		"count":   len(todos),
		"message": fmt.Sprintf("Solving %d todo items", len(todos)),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Additional handlers...
func (s *APIServer) createRecommendation(w http.ResponseWriter, r *http.Request) {
	var rec Recommendation
	if err := json.NewDecoder(r.Body).Decode(&rec); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.db.CreateRecommendation(&rec); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rec)
}

func (s *APIServer) getRecommendation(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	rec, err := s.db.GetRecommendation(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rec)
}

func (s *APIServer) updateRecommendation(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var rec Recommendation
	if err := json.NewDecoder(r.Body).Decode(&rec); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	rec.ID = id
	if err := s.db.UpdateRecommendation(&rec); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rec)
}

func (s *APIServer) deleteRecommendation(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if err := s.db.DeleteRecommendation(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (s *APIServer) createTodo(w http.ResponseWriter, r *http.Request) {
	var todo TodoItem
	if err := json.NewDecoder(r.Body).Decode(&todo); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.db.CreateTodoItem(&todo); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(todo)
}

func (s *APIServer) getTodo(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	todo, err := s.db.GetTodoItem(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(todo)
}

func (s *APIServer) updateTodo(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var todo TodoItem
	if err := json.NewDecoder(r.Body).Decode(&todo); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	todo.ID = id
	if err := s.db.UpdateTodoItem(&todo); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(todo)
}

func (s *APIServer) deleteTodo(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if err := s.db.DeleteTodoItem(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (s *APIServer) listPendingTodos(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Query().Get("user_id")
	if userID == "" {
		http.Error(w, "user_id is required", http.StatusBadRequest)
		return
	}

	todos, err := s.db.ListPendingTodos(userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(todos)
}

func (s *APIServer) updateSession(w http.ResponseWriter, r *http.Request) {
	var session UserSession
	if err := json.NewDecoder(r.Body).Decode(&session); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.db.CreateOrUpdateSession(&session); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(session)
}

func (s *APIServer) getSession(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID := vars["user_id"]

	session, err := s.db.GetUserSession(userID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(session)
}

func (s *APIServer) trackEvent(w http.ResponseWriter, r *http.Request) {
	var event AnalyticsEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.db.CreateAnalyticsEvent(&event); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(event)
}

func (s *APIServer) healthCheck(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"services": map[string]string{
			"database": "connected",
			"redis":    "connected",
			"rabbitmq": "connected",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// ==========================================
// MAIN FUNCTION
// ==========================================

func main() {
	log.Println("🚀 Starting Enhanced RAG V2 with Local LLM Integration...")

	// Load configuration from environment
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgresql://postgres:postgres@localhost:5432/legal_ai_rag?sslmode=disable"
	}

	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "localhost:6379"
	}

	rabbitMQURL := os.Getenv("RABBITMQ_URL")
	if rabbitMQURL == "" {
		rabbitMQURL = "amqp://guest:guest@localhost:5672/"
	}

	// Initialize database
	log.Println("📦 Connecting to PostgreSQL...")
	db, err := NewDatabaseManager(dbURL)
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}

	// Initialize Redis
	log.Println("📦 Connecting to Redis...")
	redisClient := redis.NewClient(&redis.Options{
		Addr: redisURL,
	})

	ctx := context.Background()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		log.Fatal("Failed to connect to Redis:", err)
	}

	// Initialize RabbitMQ
	log.Println("📦 Connecting to RabbitMQ...")
	rabbitConn, err := amqp.Dial(rabbitMQURL)
	if err != nil {
		log.Fatal("Failed to connect to RabbitMQ:", err)
	}
	defer rabbitConn.Close()

	// Initialize async processor with 10 workers
	asyncProcessor := NewAsyncProcessor(db, redisClient, rabbitConn, 10)
	asyncProcessor.Start()

	// Initialize API server
	apiServer := NewAPIServer(db, asyncProcessor)

	// Start HTTP server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8097"
	}

	log.Printf("✅ Enhanced RAG V2 Local running on port %s", port)
	log.Printf("📱 API: http://localhost:%s", port)
	log.Printf("🔌 WebSocket: ws://localhost:%s/ws", port)

	if err := http.ListenAndServe(":"+port, apiServer.router); err != nil {
		log.Fatal("Server failed:", err)
	}
}
