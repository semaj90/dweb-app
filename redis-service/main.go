// redis-service/main.go
// Lightweight HTTP wrapper for Redis operations
// Provides REST API for Redis operations and pub/sub

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/redis/go-redis/v9"
)

var (
	rdb      *redis.Client
	ctx      = context.Background()
	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins for development
		},
	}
)

// Request/Response types
type SetRequest struct {
	Key   string `json:"key"`
	Value string `json:"value"`
	TTL   int    `json:"ttl,omitempty"` // seconds
}

type GetResponse struct {
	Key   string `json:"key"`
	Value string `json:"value"`
	Found bool   `json:"found"`
}

type PubRequest struct {
	Channel string `json:"channel"`
	Message string `json:"message"`
}

type HealthResponse struct {
	Status    string            `json:"status"`
	Connected bool              `json:"connected"`
	Info      map[string]string `json:"info"`
	Uptime    int64             `json:"uptime"`
}

type StatsResponse struct {
	ConnectedClients string `json:"connected_clients"`
	UsedMemory       string `json:"used_memory"`
	UsedMemoryHuman  string `json:"used_memory_human"`
	KeyspaceHits     string `json:"keyspace_hits"`
	KeyspaceMisses   string `json:"keyspace_misses"`
}

var startTime = time.Now()

func main() {
	// Redis configuration
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}

	redisPassword := os.Getenv("REDIS_PASSWORD")
	redisDB := 0

	if dbStr := os.Getenv("REDIS_DB"); dbStr != "" {
		if db, err := strconv.Atoi(dbStr); err == nil {
			redisDB = db
		}
	}

	// Initialize Redis client
	rdb = redis.NewClient(&redis.Options{
		Addr:     redisAddr,
		Password: redisPassword,
		DB:       redisDB,
	})

	// Test connection
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		log.Printf("⚠️ Redis connection failed: %v", err)
		log.Printf("Continuing anyway - some endpoints may fail")
	} else {
		log.Printf("✅ Connected to Redis at %s", redisAddr)
	}

	// Setup HTTP router
	r := mux.NewRouter()

	// CORS middleware
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	})

	// Health endpoint
	r.HandleFunc("/health", healthHandler).Methods("GET")
	r.HandleFunc("/stats", statsHandler).Methods("GET")

	// Basic Redis operations
	r.HandleFunc("/set", setHandler).Methods("POST")
	r.HandleFunc("/get/{key}", getHandler).Methods("GET")
	r.HandleFunc("/del/{key}", delHandler).Methods("DELETE")
	r.HandleFunc("/exists/{key}", existsHandler).Methods("GET")

	// List operations
	r.HandleFunc("/lpush", lpushHandler).Methods("POST")
	r.HandleFunc("/rpush", rpushHandler).Methods("POST")
	r.HandleFunc("/lpop/{key}", lpopHandler).Methods("POST")
	r.HandleFunc("/rpop/{key}", rpopHandler).Methods("POST")
	r.HandleFunc("/lrange/{key}/{start}/{stop}", lrangeHandler).Methods("GET")

	// Hash operations
	r.HandleFunc("/hset", hsetHandler).Methods("POST")
	r.HandleFunc("/hget/{key}/{field}", hgetHandler).Methods("GET")
	r.HandleFunc("/hgetall/{key}", hgetallHandler).Methods("GET")

	// Pub/Sub operations
	r.HandleFunc("/publish", publishHandler).Methods("POST")
	r.HandleFunc("/subscribe/{channel}", subscribeWebSocketHandler).Methods("GET")

	// Job queue helpers (for integration with orchestrator)
	r.HandleFunc("/jobs/status/{jobId}", jobStatusHandler).Methods("GET")
	r.HandleFunc("/jobs/recent", recentJobsHandler).Methods("GET")

	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}

	log.Printf("🚀 Redis HTTP service starting on port %s", port)
	log.Printf("📊 Health endpoint: http://localhost:%s/health", port)
	log.Printf("📈 Stats endpoint: http://localhost:%s/stats", port)

	log.Fatal(http.ListenAndServe(":"+port, r))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	connected := true
	info := map[string]string{}

	// Try to ping Redis
	_, err := rdb.Ping(ctx).Result()
	if err != nil {
		connected = false
		info["error"] = err.Error()
	} else {
		// Get Redis info
		if _, err := rdb.Info(ctx).Result(); err == nil {
			info["redis_version"] = "available"
		} else {
			info["info_error"] = err.Error()
		}
	}

	response := HealthResponse{
		Status:    func() string { if connected { return "healthy" } else { return "unhealthy" } }(),
		Connected: connected,
		Info:      info,
		Uptime:    int64(time.Since(startTime).Seconds()),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func statsHandler(w http.ResponseWriter, r *http.Request) {
	rawInfo, err := rdb.Info(ctx, "stats", "memory", "clients").Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get Redis stats: %v", err), http.StatusInternalServerError)
		return
	}

	stats := StatsResponse{
		ConnectedClients: "N/A",
		UsedMemory:       "N/A",
		UsedMemoryHuman:  "N/A",
		KeyspaceHits:     "N/A",
		KeyspaceMisses:   "N/A",
	}

	// Very lightweight parsing – lines are in format key:value
	for _, line := range strings.Split(rawInfo, "\n") {
		if len(line) == 0 || strings.HasPrefix(line, "#") { // skip comments
			continue
		}
		if parts := strings.SplitN(line, ":", 2); len(parts) == 2 {
			key := parts[0]
			val := strings.TrimSpace(parts[1])
			switch key {
			case "connected_clients":
				stats.ConnectedClients = val
			case "used_memory":
				stats.UsedMemory = val
			case "used_memory_human":
				stats.UsedMemoryHuman = val
			case "keyspace_hits":
				stats.KeyspaceHits = val
			case "keyspace_misses":
				stats.KeyspaceMisses = val
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func setHandler(w http.ResponseWriter, r *http.Request) {
	var req SetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	if req.Key == "" {
		http.Error(w, "Key is required", http.StatusBadRequest)
		return
	}

	var err error
	if req.TTL > 0 {
		err = rdb.Set(ctx, req.Key, req.Value, time.Duration(req.TTL)*time.Second).Err()
	} else {
		err = rdb.Set(ctx, req.Key, req.Value, 0).Err()
	}

	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func getHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]

	val, err := rdb.Get(ctx, key).Result()
	found := true

	if err == redis.Nil {
		found = false
		val = ""
	} else if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	response := GetResponse{
		Key:   key,
		Value: val,
		Found: found,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func delHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]

	deleted, err := rdb.Del(ctx, key).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"deleted": deleted > 0,
		"count":   deleted,
	})
}

func existsHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]

	exists, err := rdb.Exists(ctx, key).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"exists": exists > 0})
}

func lpushHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Key   string   `json:"key"`
		Values []string `json:"values"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	interfaces := make([]interface{}, len(req.Values))
	for i, v := range req.Values {
		interfaces[i] = v
	}

	length, err := rdb.LPush(ctx, req.Key, interfaces...).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int64{"length": length})
}

func rpushHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Key   string   `json:"key"`
		Values []string `json:"values"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	interfaces := make([]interface{}, len(req.Values))
	for i, v := range req.Values {
		interfaces[i] = v
	}

	length, err := rdb.RPush(ctx, req.Key, interfaces...).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int64{"length": length})
}

func lpopHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]

	val, err := rdb.LPop(ctx, key).Result()
	if err == redis.Nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"value": nil, "found": false})
		return
	} else if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"value": val, "found": true})
}

func rpopHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]

	val, err := rdb.RPop(ctx, key).Result()
	if err == redis.Nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"value": nil, "found": false})
		return
	} else if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"value": val, "found": true})
}

func lrangeHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]
	start, _ := strconv.ParseInt(vars["start"], 10, 64)
	stop, _ := strconv.ParseInt(vars["stop"], 10, 64)

	vals, err := rdb.LRange(ctx, key, start, stop).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string][]string{"values": vals})
}

func hsetHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Key    string            `json:"key"`
		Fields map[string]string `json:"fields"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	err := rdb.HMSet(ctx, req.Key, req.Fields).Err()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func hgetHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]
	field := vars["field"]

	val, err := rdb.HGet(ctx, key, field).Result()
	found := true

	if err == redis.Nil {
		found = false
		val = ""
	} else if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"key":   key,
		"field": field,
		"value": val,
		"found": found,
	})
}

func hgetallHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	key := vars["key"]

	vals, err := rdb.HGetAll(ctx, key).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"key":    key,
		"fields": vals,
	})
}

func publishHandler(w http.ResponseWriter, r *http.Request) {
	var req PubRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	subscribers, err := rdb.Publish(ctx, req.Channel, req.Message).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]int64{"subscribers": subscribers})
}

func subscribeWebSocketHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	channel := vars["channel"]

	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade failed: %v", err)
		return
	}
	defer conn.Close()

	pubsub := rdb.Subscribe(ctx, channel)
	defer pubsub.Close()

	log.Printf("WebSocket client subscribed to channel: %s", channel)

	for {
		msg, err := pubsub.ReceiveMessage(ctx)
		if err != nil {
			log.Printf("Redis subscription error: %v", err)
			break
		}

		response := map[string]string{
			"channel": msg.Channel,
			"message": msg.Payload,
			"timestamp": time.Now().Format(time.RFC3339),
		}

		if err := conn.WriteJSON(response); err != nil {
			log.Printf("WebSocket write error: %v", err)
			break
		}
	}
}

// Job queue integration helpers
func jobStatusHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	jobId := vars["jobId"]

	vals, err := rdb.HGetAll(ctx, "job:"+jobId).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	if len(vals) == 0 {
		http.Error(w, "Job not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(vals)
}

func recentJobsHandler(w http.ResponseWriter, r *http.Request) {
	limit := 10
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if parsedLimit, err := strconv.Atoi(limitStr); err == nil && parsedLimit > 0 && parsedLimit <= 100 {
			limit = parsedLimit
		}
	}

	jobs, err := rdb.LRange(ctx, "job_history", 0, int64(limit-1)).Result()
	if err != nil {
		http.Error(w, fmt.Sprintf("Redis error: %v", err), http.StatusInternalServerError)
		return
	}

	var jobList []map[string]interface{}
	for _, job := range jobs {
		var jobData map[string]interface{}
		if err := json.Unmarshal([]byte(job), &jobData); err == nil {
			jobList = append(jobList, jobData)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"jobs":  jobList,
		"total": len(jobList),
	})
}