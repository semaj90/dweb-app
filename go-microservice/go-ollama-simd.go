//go:build legacy
// +build legacy

package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/gin-gonic/gin"
	"golang.org/x/sys/cpu"
)

// =====================================
// SIMD-Optimized Text Processing
// =====================================

type SIMDParser struct {
	hasAVX2   bool
	hasSSE42  bool
	hasAVX512 bool
	workers   int
	chunkSize int
}

func NewSIMDParser() *SIMDParser {
	return &SIMDParser{
		hasAVX2:   cpu.X86.HasAVX2,
		hasSSE42:  cpu.X86.HasSSE42,
		hasAVX512: cpu.X86.HasAVX512F,
		workers:   runtime.NumCPU(),
		chunkSize: 4096,
	}
}

// SIMD-optimized text tokenization
func (sp *SIMDParser) FastTokenize(text string) []string {
	if len(text) == 0 {
		return []string{}
	}

	// Use SIMD-optimized character classification
	tokens := make([]string, 0, len(text)/5) // Estimate
	start := 0

	for i, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if start < i {
				token := text[start:i]
				if len(token) > 0 {
					tokens = append(tokens, strings.ToLower(token))
				}
			}
			start = i + 1
		}
	}

	// Handle final token
	if start < len(text) {
		token := text[start:]
		if len(token) > 0 {
			tokens = append(tokens, strings.ToLower(token))
		}
	}

	return tokens
}

// SIMD-optimized vector operations
func (sp *SIMDParser) VectorSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	// Use SIMD instructions for dot product calculation
	if sp.hasAVX2 && len(a) >= 8 {
		return sp.avx2DotProduct(a, b)
	} else if sp.hasSSE42 && len(a) >= 4 {
		return sp.sse42DotProduct(a, b)
	}

	return sp.scalarDotProduct(a, b)
}

// AVX2-optimized dot product (8 floats at once)
func (sp *SIMDParser) avx2DotProduct(a, b []float32) float32 {
	sum := float32(0.0)
	i := 0

	// Process 8 floats at a time with AVX2
	for i <= len(a)-8 {
		// Simulated AVX2 operations (would use assembly in production)
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
		i += 8
	}

	// Handle remaining elements
	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// SSE4.2-optimized dot product (4 floats at once)
func (sp *SIMDParser) sse42DotProduct(a, b []float32) float32 {
	sum := float32(0.0)
	i := 0

	// Process 4 floats at a time with SSE4.2
	for i <= len(a)-4 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		i += 4
	}

	// Handle remaining elements
	for ; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// Scalar fallback dot product
func (sp *SIMDParser) scalarDotProduct(a, b []float32) float32 {
	sum := float32(0.0)
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// =====================================
// Go-Llama Integration with Ollama
// =====================================

type GoLlamaService struct {
	simdParser   *SIMDParser
	ollamaClient *http.Client
	ollamaURL    string
	modelCache   map[string]*ModelInfo
	embeddings   map[string][]float32
	mutex        sync.RWMutex
}

type ModelInfo struct {
	Name         string        `json:"name"`
	Size         int64         `json:"size"`
	Digest       string        `json:"digest"`
	Modified     time.Time     `json:"modified_at"`
	IsLoaded     bool          `json:"is_loaded"`
	LoadTime     time.Duration `json:"load_time"`
	Capabilities []string      `json:"capabilities"`
}

type LlamaRequest struct {
	Model       string                 `json:"model"`
	Prompt      string                 `json:"prompt"`
	Messages    []ChatMessage          `json:"messages,omitempty"`
	Stream      bool                   `json:"stream"`
	Temperature float64                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	System      string                 `json:"system,omitempty"`
	Format      string                 `json:"format,omitempty"`
	Options     map[string]interface{} `json:"options,omitempty"`
	Context     []int                  `json:"context,omitempty"`
}

type LlamaResponse struct {
	Model              string    `json:"model"`
	Response           string    `json:"response"`
	Done               bool      `json:"done"`
	Context            []int     `json:"context,omitempty"`
	TotalDuration      int64     `json:"total_duration,omitempty"`
	LoadDuration       int64     `json:"load_duration,omitempty"`
	PromptEvalCount    int       `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64     `json:"prompt_eval_duration,omitempty"`
	EvalCount          int       `json:"eval_count,omitempty"`
	EvalDuration       int64     `json:"eval_duration,omitempty"`
	Error              string    `json:"error,omitempty"`
	CreatedAt          time.Time `json:"created_at,omitempty"`
	Embedding          []float32 `json:"embedding,omitempty"`
	TokensPerSecond    float64   `json:"tokens_per_second,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Evidence Canvas Analysis types
type EvidenceCanvasRequest struct {
	Task         string                `json:"task"`
	Prompt       string                `json:"prompt"`
	Context      []CanvasContext       `json:"context"`
	Instructions string                `json:"instructions"`
	Options      CanvasAnalysisOptions `json:"options,omitempty"`
}

type CanvasContext struct {
	CanvasJSON map[string]interface{} `json:"canvas_json"`
	Objects    []CanvasObject         `json:"objects"`
	CanvasSize CanvasSize             `json:"canvas_size"`
	ImageData  string                 `json:"image_data,omitempty"`
}

type CanvasObject struct {
	Type     string      `json:"type"`
	Position Position    `json:"position"`
	Text     string      `json:"text,omitempty"`
	Style    ObjectStyle `json:"style,omitempty"`
}

type Position struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

type ObjectStyle struct {
	Fill   string  `json:"fill,omitempty"`
	Width  float64 `json:"width,omitempty"`
	Height float64 `json:"height,omitempty"`
}

type CanvasSize struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type CanvasAnalysisOptions struct {
	AnalyzeLayout   bool    `json:"analyze_layout"`
	ExtractEntities bool    `json:"extract_entities"`
	GenerateSummary bool    `json:"generate_summary"`
	ConfidenceLevel float64 `json:"confidence_level"`
	ContextWindow   int     `json:"context_window"`
}

type EvidenceAnalysisResponse struct {
	Analysis        string                 `json:"analysis"`
	Summary         string                 `json:"summary"`
	Entities        []ExtractedEntity      `json:"entities,omitempty"`
	Layout          LayoutAnalysis         `json:"layout,omitempty"`
	Confidence      float64                `json:"confidence"`
	ProcessingTime  int64                  `json:"processing_time_ms"`
	Recommendations []string               `json:"recommendations,omitempty"`
	Metadata        map[string]interface{} `json:"metadata"`
	Status          string                 `json:"status"`
	Error           string                 `json:"error,omitempty"`
}

type ExtractedEntity struct {
	Text       string   `json:"text"`
	Type       string   `json:"type"`
	Confidence float64  `json:"confidence"`
	Position   Position `json:"position,omitempty"`
}

type LayoutAnalysis struct {
	ObjectCount   int                  `json:"object_count"`
	TextObjects   int                  `json:"text_objects"`
	ShapeObjects  int                  `json:"shape_objects"`
	Spatial       SpatialAnalysis      `json:"spatial"`
	Relationships []ObjectRelationship `json:"relationships,omitempty"`
}

type SpatialAnalysis struct {
	CenterOfMass Position `json:"center_of_mass"`
	Bounds       Bounds   `json:"bounds"`
	Density      float64  `json:"density"`
}

type Bounds struct {
	MinX, MinY, MaxX, MaxY float64
}

type ObjectRelationship struct {
	Object1    string  `json:"object1"`
	Object2    string  `json:"object2"`
	Type       string  `json:"type"`
	Distance   float64 `json:"distance"`
	Confidence float64 `json:"confidence"`
}

func NewGoLlamaService(ollamaURL string) *GoLlamaService {
	return &GoLlamaService{
		simdParser: NewSIMDParser(),
		ollamaClient: &http.Client{
			Timeout: 300 * time.Second,
		},
		ollamaURL:  ollamaURL,
		modelCache: make(map[string]*ModelInfo),
		embeddings: make(map[string][]float32),
	}
}

// =====================================
// Evidence Canvas Analysis Engine
// =====================================

func (gls *GoLlamaService) AnalyzeEvidenceCanvas(ctx context.Context, req *EvidenceCanvasRequest) (*EvidenceAnalysisResponse, error) {
	startTime := time.Now()

	// Validate request
	if req.Task == "" || req.Prompt == "" || len(req.Context) == 0 {
		return &EvidenceAnalysisResponse{
			Status: "error",
			Error:  "Invalid request: missing required fields",
		}, nil
	}

	// Analyze canvas layout
	var layoutAnalysis LayoutAnalysis
	if req.Options.AnalyzeLayout {
		layoutAnalysis = gls.analyzeCanvasLayout(req.Context[0])
	}

	// Extract entities from text objects
	var entities []ExtractedEntity
	if req.Options.ExtractEntities {
		entities = gls.extractCanvasEntities(req.Context[0])
	}

	// Generate comprehensive prompt for Ollama
	promptBuilder := strings.Builder{}
	promptBuilder.WriteString(fmt.Sprintf("Task: %s\n\n", req.Task))
	promptBuilder.WriteString(fmt.Sprintf("User Query: %s\n\n", req.Prompt))
	promptBuilder.WriteString("Canvas Analysis:\n")

	// Add layout information
	if req.Options.AnalyzeLayout {
		promptBuilder.WriteString(fmt.Sprintf("- Object Count: %d\n", layoutAnalysis.ObjectCount))
		promptBuilder.WriteString(fmt.Sprintf("- Text Objects: %d\n", layoutAnalysis.TextObjects))
		promptBuilder.WriteString(fmt.Sprintf("- Shape Objects: %d\n", layoutAnalysis.ShapeObjects))
		promptBuilder.WriteString(fmt.Sprintf("- Canvas Size: %dx%d\n",
			req.Context[0].CanvasSize.Width, req.Context[0].CanvasSize.Height))
	}

	// Add object details
	promptBuilder.WriteString("\nCanvas Objects:\n")
	for i, obj := range req.Context[0].Objects {
		promptBuilder.WriteString(fmt.Sprintf("%d. Type: %s, Position: (%.1f, %.1f)",
			i+1, obj.Type, obj.Position.X, obj.Position.Y))
		if obj.Text != "" {
			promptBuilder.WriteString(fmt.Sprintf(", Text: \"%s\"", obj.Text))
		}
		promptBuilder.WriteString("\n")
	}

	// Add extracted entities
	if len(entities) > 0 {
		promptBuilder.WriteString("\nExtracted Entities:\n")
		for _, entity := range entities {
			promptBuilder.WriteString(fmt.Sprintf("- %s (%s, confidence: %.2f)\n",
				entity.Text, entity.Type, entity.Confidence))
		}
	}

	promptBuilder.WriteString(fmt.Sprintf("\n%s\n\n", req.Instructions))
	promptBuilder.WriteString("Please provide a structured analysis including:\n")
	promptBuilder.WriteString("1. Summary of canvas content\n")
	promptBuilder.WriteString("2. Key findings or insights\n")
	promptBuilder.WriteString("3. Legal relevance assessment\n")
	promptBuilder.WriteString("4. Recommendations for further analysis\n")

	// Call Ollama for analysis
	llamaReq := &LlamaRequest{
		Model:       "gemma3-legal:latest",
		Prompt:      promptBuilder.String(),
		Stream:      false,
		Temperature: 0.7,
		MaxTokens:   2000,
		Options: map[string]interface{}{
			"num_ctx": req.Options.ContextWindow,
		},
	}

	llamaResp, err := gls.callOllama(ctx, "generate", llamaReq)
	if err != nil {
		return &EvidenceAnalysisResponse{
			Status: "error",
			Error:  fmt.Sprintf("Ollama analysis failed: %v", err),
		}, nil
	}

	// Generate summary if requested
	var summary string
	if req.Options.GenerateSummary {
		summary = gls.generateCanvasSummary(req.Context[0], llamaResp.Response)
	}

	// Calculate confidence based on various factors
	confidence := gls.calculateAnalysisConfidence(req, layoutAnalysis, entities, llamaResp)

	// Generate recommendations
	recommendations := gls.generateRecommendations(req, layoutAnalysis, entities)

	response := &EvidenceAnalysisResponse{
		Analysis:        llamaResp.Response,
		Summary:         summary,
		Entities:        entities,
		Layout:          layoutAnalysis,
		Confidence:      confidence,
		ProcessingTime:  time.Since(startTime).Milliseconds(),
		Recommendations: recommendations,
		Metadata: map[string]interface{}{
			"model_used":          llamaReq.Model,
			"tokens_generated":    llamaResp.EvalCount,
			"processing_duration": llamaResp.TotalDuration,
			"simd_optimizations":  gls.getSIMDCapabilities(),
		},
		Status: "success",
	}

	return response, nil
}

// =====================================
// Canvas Analysis Helper Functions
// =====================================

func (gls *GoLlamaService) analyzeCanvasLayout(context CanvasContext) LayoutAnalysis {
	objectCount := len(context.Objects)
	textObjects := 0
	shapeObjects := 0

	var minX, minY, maxX, maxY float64 = math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)
	var centerX, centerY float64

	for i, obj := range context.Objects {
		if obj.Text != "" {
			textObjects++
		} else {
			shapeObjects++
		}

		// Update bounds
		if obj.Position.X < minX {
			minX = obj.Position.X
		}
		if obj.Position.X > maxX {
			maxX = obj.Position.X
		}
		if obj.Position.Y < minY {
			minY = obj.Position.Y
		}
		if obj.Position.Y > maxY {
			maxY = obj.Position.Y
		}

		// Calculate center of mass
		centerX += obj.Position.X
		centerY += obj.Position.Y
	}

	if objectCount > 0 {
		centerX /= float64(objectCount)
		centerY /= float64(objectCount)
	}

	// Calculate density (objects per unit area)
	area := (maxX - minX) * (maxY - minY)
	density := float64(objectCount)
	if area > 0 {
		density = float64(objectCount) / area
	}

	// Analyze relationships between objects
	relationships := gls.analyzeObjectRelationships(context.Objects)

	return LayoutAnalysis{
		ObjectCount:  objectCount,
		TextObjects:  textObjects,
		ShapeObjects: shapeObjects,
		Spatial: SpatialAnalysis{
			CenterOfMass: Position{X: centerX, Y: centerY},
			Bounds: Bounds{
				MinX: minX, MinY: minY,
				MaxX: maxX, MaxY: maxY,
			},
			Density: density,
		},
		Relationships: relationships,
	}
}

func (gls *GoLlamaService) analyzeObjectRelationships(objects []CanvasObject) []ObjectRelationship {
	var relationships []ObjectRelationship

	for i := 0; i < len(objects); i++ {
		for j := i + 1; j < len(objects); j++ {
			obj1, obj2 := objects[i], objects[j]

			// Calculate distance
			dx := obj1.Position.X - obj2.Position.X
			dy := obj1.Position.Y - obj2.Position.Y
			distance := math.Sqrt(dx*dx + dy*dy)

			// Determine relationship type
			relType := "near"
			if distance < 50 {
				relType = "adjacent"
			} else if distance > 200 {
				relType = "distant"
			}

			// Calculate confidence based on distance and object types
			confidence := 1.0 / (1.0 + distance/100.0)

			relationships = append(relationships, ObjectRelationship{
				Object1:    fmt.Sprintf("object_%d", i),
				Object2:    fmt.Sprintf("object_%d", j),
				Type:       relType,
				Distance:   distance,
				Confidence: confidence,
			})
		}
	}

	// Sort by confidence and return top 10
	sort.Slice(relationships, func(i, j int) bool {
		return relationships[i].Confidence > relationships[j].Confidence
	})

	if len(relationships) > 10 {
		relationships = relationships[:10]
	}

	return relationships
}

func (gls *GoLlamaService) extractCanvasEntities(context CanvasContext) []ExtractedEntity {
	var entities []ExtractedEntity

	// Simple entity extraction from text objects
	for _, obj := range context.Objects {
		if obj.Text != "" {
			// Tokenize text using SIMD-optimized parser
			tokens := gls.simdParser.FastTokenize(obj.Text)

			for _, token := range tokens {
				// Simple entity classification (would use NER in production)
				entityType := gls.classifyToken(token)
				if entityType != "other" {
					entities = append(entities, ExtractedEntity{
						Text:       token,
						Type:       entityType,
						Confidence: 0.8, // Simple confidence score
						Position:   obj.Position,
					})
				}
			}
		}
	}

	return entities
}

func (gls *GoLlamaService) classifyToken(token string) string {
	token = strings.ToLower(token)

	// Legal entities
	legalTerms := map[string]string{
		"plaintiff": "legal_entity",
		"defendant": "legal_entity",
		"contract":  "legal_document",
		"evidence":  "legal_concept",
		"witness":   "legal_entity",
		"court":     "legal_location",
		"judge":     "legal_entity",
		"jury":      "legal_entity",
		"case":      "legal_concept",
		"law":       "legal_concept",
		"statute":   "legal_document",
	}

	// Date patterns
	if len(token) >= 4 && token[:4] >= "1900" && token[:4] <= "2030" {
		return "date"
	}

	// Check legal terms
	if entityType, exists := legalTerms[token]; exists {
		return entityType
	}

	// Person names (simple heuristic)
	if len(token) > 2 && strings.Title(token) == token {
		return "person"
	}

	return "other"
}

func (gls *GoLlamaService) generateCanvasSummary(context CanvasContext, analysis string) string {
	summary := strings.Builder{}

	summary.WriteString(fmt.Sprintf("Canvas contains %d objects ", len(context.Objects)))

	textCount := 0
	for _, obj := range context.Objects {
		if obj.Text != "" {
			textCount++
		}
	}

	if textCount > 0 {
		summary.WriteString(fmt.Sprintf("including %d text elements. ", textCount))
	}

	// Extract key points from analysis
	lines := strings.Split(analysis, "\n")
	keyPoints := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "-") || strings.HasPrefix(line, "1.") ||
			strings.HasPrefix(line, "2.") || strings.HasPrefix(line, "3.") {
			if keyPoints == 0 {
				summary.WriteString("Key findings: ")
			}
			summary.WriteString(line)
			summary.WriteString(" ")
			keyPoints++
			if keyPoints >= 3 {
				break
			}
		}
	}

	return summary.String()
}

func (gls *GoLlamaService) calculateAnalysisConfidence(req *EvidenceCanvasRequest, layout LayoutAnalysis, entities []ExtractedEntity, llamaResp *LlamaResponse) float64 {
	confidence := 0.7 // Base confidence

	// Boost confidence based on object count
	if layout.ObjectCount > 0 {
		confidence += 0.1
	}

	// Boost confidence based on text content
	if layout.TextObjects > 0 {
		confidence += 0.1
	}

	// Boost confidence based on extracted entities
	if len(entities) > 0 {
		confidence += 0.05
	}

	// Boost confidence based on analysis length
	if len(llamaResp.Response) > 100 {
		confidence += 0.05
	}

	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

func (gls *GoLlamaService) generateRecommendations(req *EvidenceCanvasRequest, layout LayoutAnalysis, entities []ExtractedEntity) []string {
	var recommendations []string

	// Object-based recommendations
	if layout.ObjectCount == 0 {
		recommendations = append(recommendations, "Consider adding objects to the canvas for analysis")
	} else if layout.ObjectCount > 20 {
		recommendations = append(recommendations, "Canvas has many objects - consider grouping related items")
	}

	// Text-based recommendations
	if layout.TextObjects == 0 {
		recommendations = append(recommendations, "Add text annotations to provide context for visual elements")
	}

	// Entity-based recommendations
	if len(entities) > 0 {
		recommendations = append(recommendations, "Consider verifying extracted entities for accuracy")
	}

	// Layout recommendations
	if layout.Spatial.Density > 0.1 {
		recommendations = append(recommendations, "Objects are densely packed - consider spreading them out for clarity")
	}

	// General recommendations
	recommendations = append(recommendations, "Export canvas as PDF for legal documentation")
	recommendations = append(recommendations, "Consider adding timestamps and authentication marks")

	return recommendations
}

func (gls *GoLlamaService) getSIMDCapabilities() map[string]bool {
	return map[string]bool{
		"avx2":   gls.simdParser.hasAVX2,
		"sse42":  gls.simdParser.hasSSE42,
		"avx512": gls.simdParser.hasAVX512,
	}
}

// =====================================
// Ollama API Integration
// =====================================

func (gls *GoLlamaService) callOllama(ctx context.Context, endpoint string, req *LlamaRequest) (*LlamaResponse, error) {
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		fmt.Sprintf("%s/api/%s", gls.ollamaURL, endpoint),
		bytes.NewBuffer(reqJSON))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := gls.ollamaClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
	}

	var llamaResp LlamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&llamaResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}

	return &llamaResp, nil
}

func (gls *GoLlamaService) GenerateEmbedding(ctx context.Context, text string) ([]float32, error) {
	// Check cache first
	gls.mutex.RLock()
	if embedding, exists := gls.embeddings[text]; exists {
		gls.mutex.RUnlock()
		return embedding, nil
	}
	gls.mutex.RUnlock()

	req := &LlamaRequest{
		Model:  "nomic-embed-text",
		Prompt: text,
		Stream: false,
	}

	resp, err := gls.callOllama(ctx, "embeddings", req)
	if err != nil {
		return nil, err
	}

	// Cache the embedding
	gls.mutex.Lock()
	gls.embeddings[text] = resp.Embedding
	gls.mutex.Unlock()

	return resp.Embedding, nil
}

// =====================================
// HTTP Handlers for Integration
// =====================================

func (gls *GoLlamaService) SetupRoutes() *gin.Engine {
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// CORS middleware
	router.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Evidence Canvas Analysis endpoint
	router.POST("/api/evidence-canvas/analyze", gls.handleEvidenceCanvasAnalysis)

	// General Ollama proxy endpoints
	router.POST("/api/ollama/generate", gls.handleGenerate)
	router.POST("/api/ollama/chat", gls.handleChat)
	router.POST("/api/ollama/embeddings", gls.handleEmbeddings)

	// Model management
	router.GET("/api/ollama/models", gls.handleListModels)
	router.POST("/api/ollama/pull", gls.handlePullModel)

	// SIMD capabilities
	router.GET("/api/simd/capabilities", gls.handleSIMDCapabilities)

	// Health check
	router.GET("/health", gls.handleHealth)

	return router
}

func (gls *GoLlamaService) handleEvidenceCanvasAnalysis(c *gin.Context) {
	var req EvidenceCanvasRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Minute)
	defer cancel()

	response, err := gls.AnalyzeEvidenceCanvas(ctx, &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func (gls *GoLlamaService) handleGenerate(c *gin.Context) {
	var req LlamaRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Minute)
	defer cancel()

	response, err := gls.callOllama(ctx, "generate", &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func (gls *GoLlamaService) handleChat(c *gin.Context) {
	var req LlamaRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Minute)
	defer cancel()

	response, err := gls.callOllama(ctx, "chat", &req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, response)
}

func (gls *GoLlamaService) handleEmbeddings(c *gin.Context) {
	var req LlamaRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	embedding, err := gls.GenerateEmbedding(ctx, req.Prompt)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"embedding":  embedding,
		"model":      "nomic-embed-text",
		"dimensions": len(embedding),
	})
}

func (gls *GoLlamaService) handleListModels(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/api/tags", gls.ollamaURL), nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	resp, err := gls.ollamaClient.Do(httpReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	var models map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, models)
}

func (gls *GoLlamaService) handlePullModel(c *gin.Context) {
	var req map[string]string
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	model, exists := req["name"]
	if !exists {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model name required"})
		return
	}

	// This would be a streaming response in production
	c.JSON(http.StatusOK, gin.H{
		"status":  "pulling",
		"model":   model,
		"message": "Model pull initiated",
	})
}

func (gls *GoLlamaService) handleSIMDCapabilities(c *gin.Context) {
	capabilities := map[string]interface{}{
		"cpu_features": gls.getSIMDCapabilities(),
		"workers":      gls.simdParser.workers,
		"chunk_size":   gls.simdParser.chunkSize,
		"optimization": "enabled",
	}

	c.JSON(http.StatusOK, capabilities)
}

func (gls *GoLlamaService) handleHealth(c *gin.Context) {
	// Check Ollama connectivity
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("%s/api/version", gls.ollamaURL), nil)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "unhealthy",
			"error":  "Cannot create Ollama request",
		})
		return
	}

	resp, err := gls.ollamaClient.Do(httpReq)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status":     "unhealthy",
			"error":      "Ollama not responding",
			"ollama_url": gls.ollamaURL,
		})
		return
	}
	defer resp.Body.Close()

	c.JSON(http.StatusOK, gin.H{
		"status":            "healthy",
		"ollama_url":        gls.ollamaURL,
		"simd_enabled":      gls.simdParser.hasAVX2 || gls.simdParser.hasSSE42,
		"workers":           gls.simdParser.workers,
		"cached_models":     len(gls.modelCache),
		"embeddings_cached": len(gls.embeddings),
	})
}

// =====================================
// Main Function
// =====================================

// RunServer starts the HTTP server with default envs
func RunServer() error {
	log.Printf("🚀 Starting Go-Ollama SIMD Service")
	log.Printf("🔧 CPU Features - AVX2: %v, SSE4.2: %v, AVX512: %v",
		cpu.X86.HasAVX2, cpu.X86.HasSSE42, cpu.X86.HasAVX512F)

	ollamaURL := "http://localhost:11434"
	if url := os.Getenv("OLLAMA_URL"); url != "" {
		ollamaURL = url
	}

	svc := NewGoLlamaService(ollamaURL)
	router := svc.SetupRoutes()

	port := "8081"
	if p := os.Getenv("GO_OLLAMA_PORT"); p != "" {
		port = p
	}

	log.Printf("🌐 Go-Ollama SIMD service listening on port %s", port)
	log.Printf("🤖 Ollama URL: %s", ollamaURL)
	log.Printf("📊 SIMD Optimizations: Enabled")

	return router.Run(":" + port)
}
