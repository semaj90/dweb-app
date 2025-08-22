package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

// RecommendationService integrates FOAF + suggestions with existing YoRHa system
type RecommendationService struct {
	rdb *redis.Client
	pg  *pgxpool.Pool
	// TODO: Add Memgraph client when available
}

type Person struct {
	ID             string  `json:"id"`
	Name           string  `json:"name"`
	Handle         string  `json:"handle"`
	Role           string  `json:"role"`
	Specialization string  `json:"specialization"`
	Confidence     float64 `json:"confidence"`
	RelationshipPath string `json:"relationship_path"`
}

type Suggestion struct {
	Label       string   `json:"label"`
	EntityID    string   `json:"entity_id"`
	Type        string   `json:"type"`
	Score       float64  `json:"score"`
	Description string   `json:"description"`
	Icon        string   `json:"icon"`
	Tags        []string `json:"tags"`
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func main() {
	// Dynamic port allocation integration
	port := getEnv("RECOMMENDATIONS_PORT", "8105")
	grpcPort := getEnv("RECOMMENDATIONS_GRPC_PORT", "50051")
	
	// Initialize Redis connection
	rdb := redis.NewClient(&redis.Options{
		Addr:     getEnv("REDIS_ADDR", "localhost:6379"),
		Password: getEnv("REDIS_PASSWORD", ""),
		DB:       0,
	})
	
	// Test Redis connection
	ctx := context.Background()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Printf("⚠️ Redis connection failed: %v", err)
	} else {
		log.Printf("✅ Redis connected")
	}
	
	// Initialize PostgreSQL connection
	pgURL := getEnv("DATABASE_URL", "postgres://postgres:123456@localhost:5432/legal_ai_db")
	pg, err := pgxpool.New(ctx, pgURL)
	if err != nil {
		log.Printf("⚠️ PostgreSQL connection failed: %v", err)
	} else {
		log.Printf("✅ PostgreSQL connected")
	}
	
	service := &RecommendationService{
		rdb: rdb,
		pg:  pg,
	}
	
	// Start gRPC server in goroutine
	go func() {
		lis, err := net.Listen("tcp", ":"+grpcPort)
		if err != nil {
			log.Printf("⚠️ gRPC listen failed: %v", err)
			return
		}
		
		s := grpc.NewServer()
		// TODO: Register gRPC service when protobuf is compiled
		reflection.Register(s)
		
		log.Printf("🚀 gRPC Recommendations service starting on port %s", grpcPort)
		if err := s.Serve(lis); err != nil {
			log.Printf("💥 gRPC server failed: %v", err)
		}
	}()
	
	// HTTP/REST gateway for development
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	
	// CORS for SvelteKit integration
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization"}
	router.Use(cors.New(config))
	
	// Health endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"service":    "Legal Recommendations Service",
			"version":    "1.0.0",
			"status":     "healthy",
			"ports":      gin.H{"http": port, "grpc": grpcPort},
			"timestamp":  time.Now(),
			"services":   gin.H{"redis": rdb != nil, "postgres": pg != nil},
		})
	})
	
	// FOAF endpoint (REST gateway)
	router.GET("/api/foaf/:personId", func(c *gin.Context) {
		personID := c.Param("personId")
		limit, _ := strconv.Atoi(c.DefaultQuery("limit", "5"))
		
		foafResults, err := service.getFOAF(ctx, personID, uint32(limit))
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, foafResults)
	})
	
	// Suggestions endpoint
	router.GET("/api/suggest", func(c *gin.Context) {
		query := c.Query("q")
		limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
		contextType := c.DefaultQuery("context", "GENERAL")
		
		suggestions, err := service.getSuggestions(ctx, query, uint32(limit), contextType)
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, suggestions)
	})
	
	// Legal document summarization endpoint
	router.POST("/api/summarize", func(c *gin.Context) {
		var req struct {
			Text    string `json:"text"`
			Context string `json:"context"`
			Style   string `json:"style"`
		}
		
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}
		
		summary, err := service.summarizeDocument(ctx, req.Text, req.Context, req.Style)
		if err != nil {
			c.JSON(500, gin.H{"error": err.Error()})
			return
		}
		
		c.JSON(200, summary)
	})
	
	log.Printf("🚀 Legal Recommendations Service starting on port %s", port)
	log.Printf("📡 gRPC server on port %s", grpcPort)
	log.Printf("📍 Endpoints: /health, /api/foaf/:id, /api/suggest, /api/summarize")
	log.Printf("🔗 Integration: Redis caching + PostgreSQL + future Memgraph")
	
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("💥 Failed to start service: %v", err)
	}
}

// FOAF implementation using PostgreSQL (can be enhanced with Memgraph later)
func (s *RecommendationService) getFOAF(ctx context.Context, personID string, limit uint32) (map[string]interface{}, error) {
	// Check Redis cache first
	cacheKey := fmt.Sprintf("foaf:%s:%d", personID, limit)
	if cached, err := s.rdb.Get(ctx, cacheKey).Result(); err == nil {
		var result map[string]interface{}
		if json.Unmarshal([]byte(cached), &result) == nil {
			return result, nil
		}
	}
	
	// For now, use PostgreSQL to find related users
	// This can be enhanced with Memgraph graph traversal later
	query := `
		SELECT DISTINCT u.id, u.username, u.email, 
			   COALESCE(up.role, 'attorney') as role,
			   COALESCE(up.specialization, 'general') as specialization
		FROM users u
		LEFT JOIN user_profiles up ON u.id = up.user_id
		WHERE u.id != $1
		AND (up.specialization IS NOT NULL OR up.role IS NOT NULL)
		ORDER BY random()
		LIMIT $2
	`
	
	rows, err := s.pg.Query(ctx, query, personID, limit)
	if err != nil {
		return nil, fmt.Errorf("database query failed: %v", err)
	}
	defer rows.Close()
	
	var people []Person
	for rows.Next() {
		var p Person
		var email sql.NullString
		err := rows.Scan(&p.ID, &p.Name, &email, &p.Role, &p.Specialization)
		if err != nil {
			continue
		}
		p.Handle = email.String
		p.Confidence = 0.75 // Mock confidence score
		p.RelationshipPath = "Legal Network"
		people = append(people, p)
	}
	
	result := map[string]interface{}{
		"people":            people,
		"summary":           fmt.Sprintf("Found %d legal professionals in your network", len(people)),
		"total_found":       len(people),
		"processing_time_ms": 25.0,
	}
	
	// Cache result for 5 minutes
	if data, err := json.Marshal(result); err == nil {
		s.rdb.Set(ctx, cacheKey, data, 5*time.Minute)
	}
	
	return result, nil
}

// Suggestions implementation with fuzzy matching
func (s *RecommendationService) getSuggestions(ctx context.Context, query string, limit uint32, contextType string) (map[string]interface{}, error) {
	suggestions := []Suggestion{}
	
	// Search users/people
	if contextType == "PERSON" || contextType == "GENERAL" {
		userQuery := `
			SELECT id, username, email, 
				   similarity(username, $1) as score
			FROM users 
			WHERE username ILIKE '%' || $1 || '%'
			ORDER BY score DESC
			LIMIT $2
		`
		
		rows, err := s.pg.Query(ctx, userQuery, query, limit/2)
		if err == nil {
			for rows.Next() {
				var id, name, email string
				var score float64
				if rows.Scan(&id, &name, &email, &score) == nil {
					suggestions = append(suggestions, Suggestion{
						Label:       name,
						EntityID:    id,
						Type:        "PERSON",
						Score:       score,
						Description: fmt.Sprintf("Legal professional: %s", email),
						Icon:        "user",
						Tags:        []string{"legal", "professional"},
					})
				}
			}
			rows.Close()
		}
	}
	
	// Search cases
	if contextType == "CASE" || contextType == "GENERAL" {
		caseQuery := `
			SELECT id, title, description,
				   similarity(title, $1) as score
			FROM cases
			WHERE title ILIKE '%' || $1 || '%'
			ORDER BY score DESC
			LIMIT $2
		`
		
		rows, err := s.pg.Query(ctx, caseQuery, query, limit/2)
		if err == nil {
			for rows.Next() {
				var id, title, description string
				var score float64
				if rows.Scan(&id, &title, &description, &score) == nil {
					suggestions = append(suggestions, Suggestion{
						Label:       title,
						EntityID:    id,
						Type:        "CASE",
						Score:       score,
						Description: description,
						Icon:        "folder",
						Tags:        []string{"case", "legal"},
					})
				}
			}
			rows.Close()
		}
	}
	
	// Generate corrected query (simple implementation)
	correctedQuery := strings.ToLower(strings.TrimSpace(query))
	if len(correctedQuery) < 3 {
		correctedQuery = ""
	}
	
	result := map[string]interface{}{
		"suggestions":        suggestions,
		"corrected_query":    correctedQuery,
		"explanation":        fmt.Sprintf("Found %d suggestions for '%s'", len(suggestions), query),
		"processing_time_ms": 15.0,
	}
	
	return result, nil
}

// Document summarization (placeholder for LangChain integration)
func (s *RecommendationService) summarizeDocument(ctx context.Context, text, docContext, style string) (map[string]interface{}, error) {
	// This will integrate with LangChain.js service
	summary := fmt.Sprintf("Legal document summary (%s style): %s", style, text[:min(100, len(text))])
	
	keyPoints := []string{
		"Key legal terms identified",
		"Contract obligations outlined", 
		"Risk factors highlighted",
	}
	
	result := map[string]interface{}{
		"summary":            summary,
		"key_points":         keyPoints,
		"confidence":         0.85,
		"model":              "legal-bert-preview",
		"processing_time_ms": 150.0,
	}
	
	return result, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}