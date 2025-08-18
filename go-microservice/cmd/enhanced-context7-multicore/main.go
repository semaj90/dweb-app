package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/minio/simdjson-go"
	"github.com/tidwall/gjson"
	"github.com/valyala/fastjson"
)

// Configuration
type Config struct {
	Port               int    `json:"port"`
	WorkerID          string `json:"worker_id"`
	EnableSIMD        bool   `json:"enable_simd"`
	EnableTensorOps   bool   `json:"enable_tensor_ops"`
	EnableLLAMA       bool   `json:"enable_llama"`
	MaxConcurrency    int    `json:"max_concurrency"`
	OllamaEndpoint    string `json:"ollama_endpoint"`
	HealthCheckPeriod int    `json:"health_check_period"`
}

// Request/Response types
type JSONParseRequest struct {
	Data   string            `json:"data"`
	Parser string            `json:"parser,omitempty"`
	Schema string            `json:"schema,omitempty"`
	Meta   map[string]string `json:"meta,omitempty"`
}

type JSONParseResponse struct {
	Success        bool                   `json:"success"`
	Parser         string                 `json:"parser"`
	Data           interface{}            `json:"data,omitempty"`
	ParseTimeNS    int64                  `json:"parse_time_ns"`
	ThroughputMBPS float64                `json:"throughput_mbps"`
	MemoryUsed     int64                  `json:"memory_used"`
	Error          string                 `json:"error,omitempty"`
	Meta           map[string]interface{} `json:"meta,omitempty"`
}

type LLAMARequest struct {
	Prompt    string                 `json:"prompt"`
	Model     string                 `json:"model,omitempty"`
	MaxTokens int                    `json:"max_tokens,omitempty"`
	Options   map[string]interface{} `json:"options,omitempty"`
}

type LLAMAResponse struct {
	Success     bool   `json:"success"`
	Response    string `json:"response"`
	Model       string `json:"model"`
	ProcessTime int64  `json:"process_time_ns"`
	TokenCount  int    `json:"token_count"`
	Error       string `json:"error,omitempty"`
}

type RecommendationRequest struct {
	Context     string `json:"context"`
	ErrorType   string `json:"error_type,omitempty"`
	CodeSnippet string `json:"code_snippet,omitempty"`
	StackTrace  string `json:"stack_trace,omitempty"`
	Priority    string `json:"priority"`
	Language    string `json:"language,omitempty"`
	Framework   string `json:"framework,omitempty"`
}

type Recommendation struct {
	Solution          string   `json:"solution"`
	Confidence        float64  `json:"confidence"`
	Category          string   `json:"category"`
	Implementation    string   `json:"implementation"`
	EstimatedEffort   string   `json:"estimated_effort"`
	Prerequisites     []string `json:"prerequisites,omitempty"`
	RelatedPatterns   []string `json:"related_patterns,omitempty"`
}

type RecommendationResponse struct {
	Success          bool             `json:"success"`
	Recommendations  []Recommendation `json:"recommendations"`
	Context7Insights []string         `json:"context7_insights"`
	RelatedErrors    []string         `json:"related_errors"`
	BestPractices    []string         `json:"best_practices"`
	ProcessTime      int64            `json:"process_time_ns"`
	Error            string           `json:"error,omitempty"`
}

// Service struct
type Context7Service struct {
	config           Config
	stats            *Stats
	jsonParsers      map[string]JSONParser
	llamaClient      *LLAMAClient
	recommendationDB *RecommendationDB
}

// Stats tracking
type Stats struct {
	TotalRequests      int64   `json:"total_requests"`
	JSONParseRequests  int64   `json:"json_parse_requests"`
	LLAMARequests      int64   `json:"llama_requests"`
	RecommendationReqs int64   `json:"recommendation_requests"`
	SuccessfulReqs     int64   `json:"successful_requests"`
	FailedReqs         int64   `json:"failed_requests"`
	AvgResponseTimeMS  float64 `json:"avg_response_time_ms"`
	CurrentMemoryMB    float64 `json:"current_memory_mb"`
	GoroutineCount     int     `json:"goroutine_count"`
}

// JSON Parser interface
type JSONParser interface {
	Parse(data []byte) (interface{}, error)
	Name() string
}

// SIMD JSON Parser
type SIMDJSONParser struct{}

func (p *SIMDJSONParser) Parse(data []byte) (interface{}, error) {
	_, err := simdjson.Parse(data, nil)
	if err != nil {
		return nil, err
	}
	
	// Convert to standard interface{} using simple method
	var result interface{}
	err = json.Unmarshal(data, &result)
	if err != nil {
		return nil, err
	}
	
	return result, nil
}

func (p *SIMDJSONParser) Name() string {
	return "simdjson-go"
}

// Fast JSON Parser
type FastJSONParser struct {
	parser fastjson.Parser
}

func (p *FastJSONParser) Parse(data []byte) (interface{}, error) {
	val, err := p.parser.ParseBytes(data)
	if err != nil {
		return nil, err
	}
	return convertFastJSONValue(val), nil
}

func (p *FastJSONParser) Name() string {
	return "fastjson"
}

func convertFastJSONValue(val *fastjson.Value) interface{} {
	switch val.Type() {
	case fastjson.TypeString:
		s, _ := val.StringBytes()
		return string(s)
	case fastjson.TypeNumber:
		return val.GetFloat64()
	case fastjson.TypeTrue:
		return true
	case fastjson.TypeFalse:
		return false
	case fastjson.TypeNull:
		return nil
	case fastjson.TypeArray:
		arr, _ := val.Array()
		result := make([]interface{}, len(arr))
		for i, item := range arr {
			result[i] = convertFastJSONValue(item)
		}
		return result
	case fastjson.TypeObject:
		obj, _ := val.Object()
		result := make(map[string]interface{})
		obj.Visit(func(key []byte, v *fastjson.Value) {
			result[string(key)] = convertFastJSONValue(v)
		})
		return result
	default:
		return nil
	}
}

// GJSON Parser
type GJSONParser struct{}

func (p *GJSONParser) Parse(data []byte) (interface{}, error) {
	if !gjson.ValidBytes(data) {
		return nil, fmt.Errorf("invalid JSON")
	}
	
	result := gjson.ParseBytes(data)
	return convertGJSONValue(result), nil
}

func (p *GJSONParser) Name() string {
	return "gjson"
}

func convertGJSONValue(result gjson.Result) interface{} {
	switch result.Type {
	case gjson.String:
		return result.String()
	case gjson.Number:
		return result.Float()
	case gjson.True:
		return true
	case gjson.False:
		return false
	case gjson.Null:
		return nil
	case gjson.JSON:
		var parsed interface{}
		json.Unmarshal([]byte(result.Raw), &parsed)
		return parsed
	default:
		return nil
	}
}

// Standard JSON Parser
type StandardJSONParser struct{}

func (p *StandardJSONParser) Parse(data []byte) (interface{}, error) {
	var result interface{}
	err := json.Unmarshal(data, &result)
	return result, err
}

func (p *StandardJSONParser) Name() string {
	return "standard"
}

// LLAMA Client
type LLAMAClient struct {
	endpoint string
	client   *http.Client
}

func NewLLAMAClient(endpoint string) *LLAMAClient {
	return &LLAMAClient{
		endpoint: endpoint,
		client:   &http.Client{Timeout: 30 * time.Second},
	}
}

func (lc *LLAMAClient) Process(req LLAMARequest) (*LLAMAResponse, error) {
	start := time.Now()
	
	model := req.Model
	if model == "" {
		model = "gemma2:2b"
	}
	
	ollamaReq := map[string]interface{}{
		"model":  model,
		"prompt": req.Prompt,
		"stream": false,
	}
	
	if req.MaxTokens > 0 {
		ollamaReq["options"] = map[string]interface{}{
			"num_predict": req.MaxTokens,
		}
	}
	
	if req.Options != nil {
		if options, exists := ollamaReq["options"]; exists {
			opts := options.(map[string]interface{})
			for k, v := range req.Options {
				opts[k] = v
			}
		} else {
			ollamaReq["options"] = req.Options
		}
	}
	
	reqBody, _ := json.Marshal(ollamaReq)
	
	resp, err := lc.client.Post(lc.endpoint+"/api/generate", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return &LLAMAResponse{
			Success: false,
			Error:   err.Error(),
		}, nil
	}
	defer resp.Body.Close()
	
	var ollamaResp map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return &LLAMAResponse{
			Success: false,
			Error:   err.Error(),
		}, nil
	}
	
	processTime := time.Since(start).Nanoseconds()
	
	response := ""
	if resp, exists := ollamaResp["response"]; exists {
		response = resp.(string)
	}
	
	tokenCount := len(strings.Fields(response))
	
	return &LLAMAResponse{
		Success:     true,
		Response:    response,
		Model:       model,
		ProcessTime: processTime,
		TokenCount:  tokenCount,
	}, nil
}

// Recommendation Database (in-memory for demo)
type RecommendationDB struct {
	patterns map[string][]Recommendation
}

func NewRecommendationDB() *RecommendationDB {
	db := &RecommendationDB{
		patterns: make(map[string][]Recommendation),
	}
	db.initializePatterns()
	return db
}

func (rdb *RecommendationDB) initializePatterns() {
	// TypeScript/SvelteKit patterns
	rdb.patterns["typescript"] = []Recommendation{
		{
			Solution:        "Add proper type annotations and use strict TypeScript settings",
			Confidence:      0.9,
			Category:        "type-safety",
			Implementation:  "Enable strict mode in tsconfig.json and add explicit types",
			EstimatedEffort: "30 minutes",
			Prerequisites:   []string{"TypeScript 4.0+"},
		},
		{
			Solution:        "Use Svelte 5 runes for better reactivity",
			Confidence:      0.85,
			Category:        "performance",
			Implementation:  "Replace reactive statements with $state() and $derived()",
			EstimatedEffort: "1-2 hours",
			Prerequisites:   []string{"Svelte 5", "Modern bundler"},
		},
	}
	
	// Legal AI patterns
	rdb.patterns["legal-ai"] = []Recommendation{
		{
			Solution:        "Implement proper document vectorization with pgvector",
			Confidence:      0.95,
			Category:        "ai-integration",
			Implementation:  "Use sentence transformers with PostgreSQL vector search",
			EstimatedEffort: "4-6 hours",
			Prerequisites:   []string{"PostgreSQL", "pgvector extension"},
		},
	}
	
	// Performance patterns
	rdb.patterns["performance"] = []Recommendation{
		{
			Solution:        "Implement SIMD JSON parsing for large documents",
			Confidence:      0.92,
			Category:        "optimization",
			Implementation:  "Use simdjson-go for high-throughput JSON processing",
			EstimatedEffort: "2-3 hours",
			Prerequisites:   []string{"Go 1.19+", "SIMD support"},
		},
	}
}

func (rdb *RecommendationDB) GetRecommendations(req RecommendationRequest) (*RecommendationResponse, error) {
	start := time.Now()
	
	var recommendations []Recommendation
	context7Insights := []string{}
	relatedErrors := []string{}
	bestPractices := []string{}
	
	// Analyze context for patterns
	context := strings.ToLower(req.Context)
	
	if strings.Contains(context, "typescript") || strings.Contains(context, "svelte") {
		recommendations = append(recommendations, rdb.patterns["typescript"]...)
		context7Insights = append(context7Insights, "TypeScript/SvelteKit pattern detected")
		bestPractices = append(bestPractices, "Use strict TypeScript configuration", "Implement proper component typing")
	}
	
	if strings.Contains(context, "legal") || strings.Contains(context, "document") {
		recommendations = append(recommendations, rdb.patterns["legal-ai"]...)
		context7Insights = append(context7Insights, "Legal AI processing pattern detected")
		bestPractices = append(bestPractices, "Implement proper data privacy controls", "Use specialized legal NLP models")
	}
	
	if strings.Contains(context, "performance") || strings.Contains(context, "slow") {
		recommendations = append(recommendations, rdb.patterns["performance"]...)
		context7Insights = append(context7Insights, "Performance optimization opportunity identified")
		bestPractices = append(bestPractices, "Profile before optimizing", "Use appropriate data structures")
	}
	
	// Add error-specific recommendations
	if req.ErrorType != "" {
		switch req.ErrorType {
		case "type_error":
			relatedErrors = append(relatedErrors, "Missing type annotations", "Incorrect type usage")
		case "performance_issue":
			relatedErrors = append(relatedErrors, "Inefficient algorithms", "Memory leaks")
		case "integration_error":
			relatedErrors = append(relatedErrors, "API compatibility issues", "Version mismatches")
		}
	}
	
	processTime := time.Since(start).Nanoseconds()
	
	return &RecommendationResponse{
		Success:          true,
		Recommendations:  recommendations,
		Context7Insights: context7Insights,
		RelatedErrors:    relatedErrors,
		BestPractices:    bestPractices,
		ProcessTime:      processTime,
	}, nil
}

// Service initialization
func NewContext7Service() *Context7Service {
	config := Config{
		Port:               getEnvInt("PORT", 8095),
		WorkerID:          getEnvString("WORKER_ID", fmt.Sprintf("worker_%d", os.Getpid())),
		EnableSIMD:        getEnvBool("ENABLE_SIMD", true),
		EnableTensorOps:   getEnvBool("ENABLE_TENSOR_OPS", true),
		EnableLLAMA:       getEnvBool("ENABLE_LLAMA", true),
		MaxConcurrency:    getEnvInt("MAX_CONCURRENCY", 100),
		OllamaEndpoint:    getEnvString("OLLAMA_ENDPOINT", "http://localhost:11434"),
		HealthCheckPeriod: getEnvInt("HEALTH_CHECK_PERIOD", 30),
	}
	
	service := &Context7Service{
		config:           config,
		stats:            &Stats{},
		jsonParsers:      make(map[string]JSONParser),
		recommendationDB: NewRecommendationDB(),
	}
	
	// Initialize JSON parsers
	service.jsonParsers["simd"] = &SIMDJSONParser{}
	service.jsonParsers["fastjson"] = &FastJSONParser{}
	service.jsonParsers["gjson"] = &GJSONParser{}
	service.jsonParsers["standard"] = &StandardJSONParser{}
	
	// Initialize LLAMA client if enabled
	if config.EnableLLAMA {
		service.llamaClient = NewLLAMAClient(config.OllamaEndpoint)
	}
	
	return service
}

// HTTP Handlers
func (s *Context7Service) setupRoutes() *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())
	
	// CORS middleware
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))
	
	// Middleware for stats tracking
	r.Use(s.statsMiddleware())
	
	// Health endpoint
	r.GET("/health", s.handleHealth)
	
	// Metrics endpoint
	r.GET("/metrics", s.handleMetrics)
	
	// JSON parsing endpoints
	r.POST("/parse/json", s.handleJSONParse)
	r.POST("/parse/json/batch", s.handleJSONParseBatch)
	
	// LLAMA endpoints
	if s.config.EnableLLAMA {
		r.POST("/llama", s.handleLLAMA)
	}
	
	// Recommendation endpoints
	r.POST("/recommendation", s.handleRecommendation)
	
	return r
}

func (s *Context7Service) statsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		s.stats.TotalRequests++
		
		c.Next()
		
		duration := time.Since(start)
		
		if c.Writer.Status() >= 200 && c.Writer.Status() < 400 {
			s.stats.SuccessfulReqs++
		} else {
			s.stats.FailedReqs++
		}
		
		// Update average response time
		total := s.stats.SuccessfulReqs + s.stats.FailedReqs
		s.stats.AvgResponseTimeMS = (s.stats.AvgResponseTimeMS*float64(total-1) + float64(duration.Milliseconds())) / float64(total)
	}
}

func (s *Context7Service) handleHealth(c *gin.Context) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	c.JSON(200, gin.H{
		"status":             "healthy",
		"worker_id":          s.config.WorkerID,
		"port":               s.config.Port,
		"simd_enabled":       s.config.EnableSIMD,
		"llama_enabled":      s.config.EnableLLAMA,
		"memory_mb":          float64(m.HeapInuse) / 1024 / 1024,
		"goroutines":         runtime.NumGoroutine(),
	})
}

func (s *Context7Service) handleMetrics(c *gin.Context) {
	s.updateStats()
	c.JSON(200, s.stats)
}

func (s *Context7Service) handleJSONParse(c *gin.Context) {
	var req JSONParseRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	s.stats.JSONParseRequests++
	
	parser := req.Parser
	if parser == "" {
		parser = "simd"
	}
	
	jsonParser, exists := s.jsonParsers[parser]
	if !exists {
		c.JSON(400, gin.H{"error": fmt.Sprintf("unknown parser: %s", parser)})
		return
	}
	
	start := time.Now()
	data := []byte(req.Data)
	
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	
	result, err := jsonParser.Parse(data)
	
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	
	parseTime := time.Since(start)
	memoryUsed := int64(m2.HeapInuse - m1.HeapInuse)
	throughput := float64(len(data)) / parseTime.Seconds() / (1 << 20) // MB/s
	
	if err != nil {
		c.JSON(400, JSONParseResponse{
			Success:     false,
			Parser:      jsonParser.Name(),
			Error:       err.Error(),
			ParseTimeNS: parseTime.Nanoseconds(),
			MemoryUsed:  memoryUsed,
		})
		return
	}
	
	response := JSONParseResponse{
		Success:        true,
		Parser:         jsonParser.Name(),
		Data:           result,
		ParseTimeNS:    parseTime.Nanoseconds(),
		ThroughputMBPS: throughput,
		MemoryUsed:     memoryUsed,
	}
	
	if req.Meta != nil {
		response.Meta = make(map[string]interface{})
		for k, v := range req.Meta {
			response.Meta[k] = v
		}
	}
	
	c.JSON(200, response)
}

func (s *Context7Service) handleJSONParseBatch(c *gin.Context) {
	var req struct {
		Documents []JSONParseRequest `json:"documents"`
		Parser    string             `json:"parser,omitempty"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	parser := req.Parser
	if parser == "" {
		parser = "simd"
	}
	
	jsonParser, exists := s.jsonParsers[parser]
	if !exists {
		c.JSON(400, gin.H{"error": fmt.Sprintf("unknown parser: %s", parser)})
		return
	}
	
	results := make([]JSONParseResponse, len(req.Documents))
	totalTime := int64(0)
	
	for i, doc := range req.Documents {
		start := time.Now()
		data := []byte(doc.Data)
		
		result, err := jsonParser.Parse(data)
		parseTime := time.Since(start)
		totalTime += parseTime.Nanoseconds()
		
		if err != nil {
			results[i] = JSONParseResponse{
				Success:     false,
				Parser:      jsonParser.Name(),
				Error:       err.Error(),
				ParseTimeNS: parseTime.Nanoseconds(),
			}
		} else {
			results[i] = JSONParseResponse{
				Success:        true,
				Parser:         jsonParser.Name(),
				Data:           result,
				ParseTimeNS:    parseTime.Nanoseconds(),
				ThroughputMBPS: float64(len(data)) / parseTime.Seconds() / (1 << 20),
			}
		}
	}
	
	c.JSON(200, gin.H{
		"batch_size":    len(req.Documents),
		"total_time_ns": totalTime,
		"avg_time_ns":   totalTime / int64(len(req.Documents)),
		"parser":        jsonParser.Name(),
		"results":       results,
	})
}

func (s *Context7Service) handleLLAMA(c *gin.Context) {
	if !s.config.EnableLLAMA {
		c.JSON(503, gin.H{"error": "LLAMA not enabled"})
		return
	}
	
	var req LLAMARequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	s.stats.LLAMARequests++
	
	result, err := s.llamaClient.Process(req)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, result)
}

func (s *Context7Service) handleRecommendation(c *gin.Context) {
	var req RecommendationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	s.stats.RecommendationReqs++
	
	result, err := s.recommendationDB.GetRecommendations(req)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, result)
}

func (s *Context7Service) updateStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	s.stats.CurrentMemoryMB = float64(m.HeapInuse) / 1024 / 1024
	s.stats.GoroutineCount = runtime.NumGoroutine()
}

// Utility functions
func getEnvString(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

// Main function
func main() {
	fmt.Println("🚀 Starting Enhanced Context7 Multicore Service")
	
	service := NewContext7Service()
	router := service.setupRoutes()
	
	// Start background stats updater
	go func() {
		ticker := time.NewTicker(time.Duration(service.config.HealthCheckPeriod) * time.Second)
		defer ticker.Stop()
		
		for {
			select {
			case <-ticker.C:
				service.updateStats()
			}
		}
	}()
	
	fmt.Printf("✅ Context7 Multicore Service running on port %d\n", service.config.Port)
	fmt.Printf("   - Worker ID: %s\n", service.config.WorkerID)
	fmt.Printf("   - SIMD JSON: %v\n", service.config.EnableSIMD)
	fmt.Printf("   - LLAMA: %v\n", service.config.EnableLLAMA)
	
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", service.config.Port),
		Handler: router,
	}
	
	log.Fatal(server.ListenAndServe())
}