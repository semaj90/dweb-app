//go:build experimental
// +build experimental

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"runtime"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
)

// Simple Vector Service - Native Windows, No Docker
type SimpleVectorService struct {
	pgPool      *pgxpool.Pool
	redisClient *redis.Client
	upgrader    websocket.Upgrader
}

// Vector operation request/response types
type VectorRequest struct {
	RequestID string    `json:"request_id"`
	Vector    []float64 `json:"vector"`
	Operation string    `json:"operation"`
}

type VectorResponse struct {
	RequestID    string    `json:"request_id"`
	Success      bool      `json:"success"`
	Result       []float64 `json:"result,omitempty"`
	Score        float64   `json:"score,omitempty"`
	Message      string    `json:"message,omitempty"`
	ProcessingMs int64     `json:"processing_ms"`
}

// System status
type SystemStatus struct {
	Service     string `json:"service"`
	Version     string `json:"version"`
	Status      string `json:"status"`
	Uptime      string `json:"uptime"`
	GoVersion   string `json:"go_version"`
	Platform    string `json:"platform"`
	DatabaseOK  bool   `json:"database_ok"`
	RedisOK     bool   `json:"redis_ok"`
	CudaEnabled bool   `json:"cuda_enabled"`
}

var startTime = time.Now()

func NewSimpleVectorService() *SimpleVectorService {
	service := &SimpleVectorService{
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}

	// Initialize database connection (PostgreSQL)
	dbURL := "postgres://legal_admin:123456@localhost:5432/legal_ai_db?sslmode=disable"
	pgPool, err := pgxpool.New(context.Background(), dbURL)
	if err != nil {
		log.Printf("Warning: Database connection failed: %v", err)
	} else {
		service.pgPool = pgPool
		log.Println("✅ PostgreSQL connected successfully")
	}

	// Initialize Redis connection
	service.redisClient = redis.NewClient(&redis.Options{
		Addr: "localhost:6379",
		DB:   0,
	})

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := service.redisClient.Ping(ctx).Err(); err != nil {
		log.Printf("Warning: Redis connection failed: %v", err)
	} else {
		log.Println("✅ Redis connected successfully")
	}

	return service
}

// Vector operations
func (s *SimpleVectorService) ProcessVector(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	w.Header().Set("Content-Type", "application/json")

	var req VectorRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	var result []float64
	var score float64
	var message string

	// Process based on operation type
	switch req.Operation {
	case "normalize":
		result = s.normalizeVector(req.Vector)
		message = "Vector normalized successfully"

	case "magnitude":
		score = s.calculateMagnitude(req.Vector)
		message = "Vector magnitude calculated"

	case "cosine_similarity":
		// For demo, calculate cosine similarity with itself (should be 1.0)
		score = s.cosineSimilarity(req.Vector, req.Vector)
		message = "Cosine similarity calculated"

	case "rotate":
		result = s.rotateVector(req.Vector, 0.1) // 0.1 radian rotation
		message = "Vector rotated successfully"

	default:
		http.Error(w, "Unsupported operation", http.StatusBadRequest)
		return
	}

	response := VectorResponse{
		RequestID:    req.RequestID,
		Success:      true,
		Result:       result,
		Score:        score,
		Message:      message,
		ProcessingMs: time.Since(startTime).Milliseconds(),
	}

	// Log to database if available
	if s.pgPool != nil {
		go s.logOperation(req, response)
	}

	json.NewEncoder(w).Encode(response)
}

// System health check
func (s *SimpleVectorService) HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Check database
	dbOK := false
	if s.pgPool != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		err := s.pgPool.Ping(ctx)
		cancel()
		dbOK = (err == nil)
	}

	// Check Redis
	redisOK := false
	if s.redisClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		err := s.redisClient.Ping(ctx).Err()
		cancel()
		redisOK = (err == nil)
	}

	status := SystemStatus{
		Service:     "Simple Vector Service",
		Version:     "2.0-native-windows",
		Status:      "healthy",
		Uptime:      time.Since(startTime).String(),
		GoVersion:   runtime.Version(),
		Platform:    fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
		DatabaseOK:  dbOK,
		RedisOK:     redisOK,
		CudaEnabled: s.isCudaAvailable(),
	}

	json.NewEncoder(w).Encode(status)
}

// WebSocket handler for real-time vector operations
func (s *SimpleVectorService) WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := s.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	log.Println("WebSocket client connected")

	for {
		var req VectorRequest
		if err := conn.ReadJSON(&req); err != nil {
			log.Printf("WebSocket read error: %v", err)
			break
		}

		// Process the vector operation
		startTime := time.Now()
		var result []float64
		var score float64

		switch req.Operation {
		case "normalize":
			result = s.normalizeVector(req.Vector)
		case "magnitude":
			score = s.calculateMagnitude(req.Vector)
		case "cosine_similarity":
			score = s.cosineSimilarity(req.Vector, req.Vector)
		case "rotate":
			result = s.rotateVector(req.Vector, 0.1)
		}

		response := VectorResponse{
			RequestID:    req.RequestID,
			Success:      true,
			Result:       result,
			Score:        score,
			Message:      fmt.Sprintf("Operation '%s' completed", req.Operation),
			ProcessingMs: time.Since(startTime).Milliseconds(),
		}

		if err := conn.WriteJSON(response); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}

	log.Println("WebSocket client disconnected")
}

// Vector math operations
func (s *SimpleVectorService) normalizeVector(vector []float64) []float64 {
	magnitude := s.calculateMagnitude(vector)
	if magnitude == 0 {
		return vector
	}

	result := make([]float64, len(vector))
	for i, v := range vector {
		result[i] = v / magnitude
	}
	return result
}

func (s *SimpleVectorService) calculateMagnitude(vector []float64) float64 {
	sum := 0.0
	for _, v := range vector {
		sum += v * v
	}
	return math.Sqrt(sum)
}

func (s *SimpleVectorService) cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := 0.0
	magnitudeA := 0.0
	magnitudeB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}

	magnitudeA = math.Sqrt(magnitudeA)
	magnitudeB = math.Sqrt(magnitudeB)

	if magnitudeA == 0 || magnitudeB == 0 {
		return 0.0
	}

	return dotProduct / (magnitudeA * magnitudeB)
}

func (s *SimpleVectorService) rotateVector(vector []float64, angle float64) []float64 {
	if len(vector) < 2 {
		return vector
	}

	// Simple 2D rotation for first two components
	result := make([]float64, len(vector))
	copy(result, vector)

	cos := math.Cos(angle)
	sin := math.Sin(angle)

	x := vector[0]
	y := vector[1]

	result[0] = x*cos - y*sin
	result[1] = x*sin + y*cos

	return result
}

// Database logging
func (s *SimpleVectorService) logOperation(req VectorRequest, resp VectorResponse) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	query := `
		INSERT INTO vector_operations (request_id, operation, input_size, processing_time_ms, success)
		VALUES ($1, $2, $3, $4, $5)
		ON CONFLICT DO NOTHING
	`

	_, err := s.pgPool.Exec(ctx, query,
		req.RequestID,
		req.Operation,
		len(req.Vector),
		resp.ProcessingMs,
		resp.Success,
	)

	if err != nil {
		log.Printf("Database logging error: %v", err)
	}
}

// Check CUDA availability
func (s *SimpleVectorService) isCudaAvailable() bool {
	// Simple check for NVIDIA drivers/CUDA
	return runtime.GOOS == "windows" // Placeholder - could check nvidia-smi
}

func main() {
	fmt.Println("====================================")
	fmt.Println("Simple Enterprise Vector Service v2.0")
	fmt.Println("Native Windows - No Docker")
	fmt.Println("====================================")

	service := NewSimpleVectorService()

	// Setup HTTP routes
	router := mux.NewRouter()

	// API endpoints
	router.HandleFunc("/api/vector", service.ProcessVector).Methods("POST")
	router.HandleFunc("/api/health", service.HealthCheck).Methods("GET")
	router.HandleFunc("/ws", service.WebSocketHandler)

	// Simple web interface
	router.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		html := `<!DOCTYPE html>
<html>
<head>
    <title>Simple Vector Service</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; }
        .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        button { padding: 10px 20px; margin: 5px; background: #007cba; color: white; border: none; border-radius: 3px; cursor: pointer; }
        textarea { width: 100%; height: 100px; }
        #output { background: #000; color: #0f0; padding: 20px; font-family: monospace; min-height: 200px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Simple Vector Service v2.0</h1>
        <p><strong>Native Windows Deployment</strong> - No Docker, No Complex Dependencies</p>

        <div class="endpoint">
            <h3>API Endpoints:</h3>
            <ul>
                <li><code>POST /api/vector</code> - Vector operations</li>
                <li><code>GET /api/health</code> - System health</li>
                <li><code>WS /ws</code> - WebSocket real-time</li>
            </ul>
        </div>

        <div>
            <h3>Test Vector Operations:</h3>
            <textarea id="vectorInput" placeholder='{"request_id": "test-1", "vector": [1, 2, 3, 4], "operation": "normalize"}'></textarea><br>
            <button onclick="testVector()">Test Vector API</button>
            <button onclick="testHealth()">Check Health</button>
            <button onclick="clearOutput()">Clear</button>
        </div>

        <div id="output">Service ready - Click buttons to test...</div>
    </div>

    <script>
        function log(msg) {
            document.getElementById('output').innerHTML += new Date().toLocaleTimeString() + ': ' + msg + '\n';
        }

        function testVector() {
            const data = document.getElementById('vectorInput').value || '{"request_id": "test-1", "vector": [1, 2, 3, 4], "operation": "normalize"}';

            fetch('/api/vector', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: data
            })
            .then(response => response.json())
            .then(data => log('Vector result: ' + JSON.stringify(data, null, 2)))
            .catch(error => log('Error: ' + error));
        }

        function testHealth() {
            fetch('/api/health')
            .then(response => response.json())
            .then(data => log('Health: ' + JSON.stringify(data, null, 2)))
            .catch(error => log('Error: ' + error));
        }

        function clearOutput() {
            document.getElementById('output').innerHTML = '';
        }

        log('Simple Vector Service Web Interface Ready');
    </script>
</body>
</html>`
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(html))
	}).Methods("GET")

	// Start server
	port := "8095"
	fmt.Printf("🌐 Server starting on http://localhost:%s\n", port)
	fmt.Printf("🔧 Health check: http://localhost:%s/api/health\n", port)
	fmt.Printf("📊 Web interface: http://localhost:%s\n", port)
	fmt.Println("====================================")

	log.Fatal(http.ListenAndServe(":"+port, router))
}