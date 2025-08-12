package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/gin-gonic/gin"
	"golang.org/x/sys/cpu"
)

// SIMD parser and helpers omitted for brevity in this stub; keep scalar fallbacks

type SIMDParser struct{}

func NewSIMDParser() *SIMDParser { return &SIMDParser{} }
func (sp *SIMDParser) FastTokenize(text string) []string {
	if len(text) == 0 {
		return nil
	}
	tokens := []string{}
	start := 0
	for i, r := range text {
		if unicode.IsSpace(r) || unicode.IsPunct(r) {
			if start < i {
				tokens = append(tokens, strings.ToLower(text[start:i]))
			}
			start = i + 1
		}
	}
	if start < len(text) {
		tokens = append(tokens, strings.ToLower(text[start:]))
	}
	return tokens
}

// Service types

type GoLlamaService struct {
	simdParser   *SIMDParser
	ollamaClient *http.Client
	ollamaURL    string
	modelCache   map[string]*ModelInfo
	embeddings   map[string][]float32
	mutex        sync.RWMutex

	// metrics
	totalRequests        atomic.Uint64
	analyzeRequests      atomic.Uint64
	analyzeErrors        atomic.Uint64
	totalRequestDuration atomic.Uint64 // milliseconds aggregate
}

type ModelInfo struct {
	Name string `json:"name"`
}

type LlamaRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Stream  bool           `json:"stream"`
	Options map[string]any `json:"options,omitempty"`
}

type LlamaResponse struct {
	Response  string    `json:"response"`
	Done      bool      `json:"done"`
	Embedding []float32 `json:"embedding,omitempty"`
}

type EvidenceCanvasRequest struct {
	Task         string                `json:"task"`
	Prompt       string                `json:"prompt"`
	Context      []CanvasContext       `json:"context"`
	Instructions string                `json:"instructions"`
	Options      CanvasAnalysisOptions `json:"options,omitempty"`
}

type CanvasContext struct {
	CanvasJSON map[string]any `json:"canvas_json"`
	Objects    []CanvasObject `json:"objects"`
	CanvasSize CanvasSize     `json:"canvas_size"`
}

type CanvasObject struct {
	Type     string   `json:"type"`
	Position Position `json:"position"`
	Text     string   `json:"text,omitempty"`
}

type Position struct{ X, Y float64 }

type CanvasSize struct{ Width, Height int }

type CanvasAnalysisOptions struct {
	AnalyzeLayout   bool    `json:"analyze_layout"`
	ExtractEntities bool    `json:"extract_entities"`
	GenerateSummary bool    `json:"generate_summary"`
	ConfidenceLevel float64 `json:"confidence_level"`
	ContextWindow   int     `json:"context_window"`
}

type EvidenceAnalysisResponse struct {
	Analysis       string  `json:"analysis"`
	Summary        string  `json:"summary"`
	Confidence     float64 `json:"confidence"`
	ProcessingTime int64   `json:"processing_time_ms"`
	Status         string  `json:"status"`
	Error          string  `json:"error,omitempty"`
}

// Text summarization types
type TextSummarizeRequest struct {
	Content string `json:"content"`
	Prompt  string `json:"prompt,omitempty"`
}

type TextSummarizeResponse struct {
	Summary        string   `json:"summary"`
	Tokens         []string `json:"tokens"`
	TokenCount     int      `json:"token_count"`
	ProcessingTime int64    `json:"processing_time_ms"`
	Status         string   `json:"status"`
	Error          string   `json:"error,omitempty"`
}

func NewGoLlamaService(ollamaURL string) *GoLlamaService {
	return &GoLlamaService{
		simdParser:   NewSIMDParser(),
		ollamaClient: &http.Client{Timeout: 300 * time.Second},
		ollamaURL:    ollamaURL,
		modelCache:   map[string]*ModelInfo{},
		embeddings:   map[string][]float32{},
	}
}

func (gls *GoLlamaService) AnalyzeEvidenceCanvas(ctx context.Context, req *EvidenceCanvasRequest) (*EvidenceAnalysisResponse, error) {
	start := time.Now()
	if req.Task == "" || req.Prompt == "" || len(req.Context) == 0 {
		return &EvidenceAnalysisResponse{Status: "error", Error: "invalid request"}, nil
	}

	prompt := strings.Builder{}
	prompt.WriteString(req.Prompt)
	prompt.WriteString("\nObjects:\n")
	for i, o := range req.Context[0].Objects {
		_ = i
		if o.Text != "" {
			prompt.WriteString("- " + o.Text + "\n")
		}
	}

	llReq := &LlamaRequest{Model: "gemma3-legal:latest", Prompt: prompt.String(), Stream: false, Options: map[string]any{"num_ctx": req.Options.ContextWindow}}
	llResp, err := gls.callOllama(ctx, "generate", llReq)
	if err != nil {
		return &EvidenceAnalysisResponse{Status: "error", Error: err.Error()}, nil
	}

	return &EvidenceAnalysisResponse{
		Analysis:       llResp.Response,
		Summary:        llResp.Response,
		Confidence:     0.8,
		ProcessingTime: time.Since(start).Milliseconds(),
		Status:         "success",
	}, nil
}

func (gls *GoLlamaService) callOllama(ctx context.Context, endpoint string, req *LlamaRequest) (*LlamaResponse, error) {
	b, _ := json.Marshal(req)
	h, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/api/%s", gls.ollamaURL, endpoint), bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	h.Header.Set("Content-Type", "application/json")
	res, err := gls.ollamaClient.Do(h)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		return nil, fmt.Errorf("ollama status %d: %s", res.StatusCode, string(body))
	}
	var out LlamaResponse
	if err := json.NewDecoder(res.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (gls *GoLlamaService) SetupRoutes() *gin.Engine {
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())
	// Basic CORS + metrics middleware
	r.Use(func(c *gin.Context) {
		start := time.Now()
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		gls.totalRequests.Add(1)
		c.Next()
		dur := time.Since(start).Milliseconds()
		gls.totalRequestDuration.Add(uint64(dur))
	})
	r.POST("/api/evidence-canvas/analyze", func(c *gin.Context) {
		gls.analyzeRequests.Add(1)
		var req EvidenceCanvasRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			gls.analyzeErrors.Add(1)
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Minute)
		defer cancel()
		resp, err := gls.AnalyzeEvidenceCanvas(ctx, &req)
		if err != nil {
			gls.analyzeErrors.Add(1)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, resp)
	})
	// Text summarization with SIMD tokenization aid
	r.POST("/api/simd/summarize", func(c *gin.Context) {
		var req TextSummarizeRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		if strings.TrimSpace(req.Content) == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "content is required"})
			return
		}
		start := time.Now()
		tokens := gls.simdParser.FastTokenize(req.Content)
		// Build a simple prompt if none
		prompt := req.Prompt
		if strings.TrimSpace(prompt) == "" {
			prompt = "Summarize the following legal text focusing on key points, entities, and risks: \n\n" + req.Content
		} else {
			prompt = prompt + "\n\nText:" + req.Content
		}
		ctx, cancel := context.WithTimeout(c.Request.Context(), 120*time.Second)
		defer cancel()
		llReq := &LlamaRequest{Model: "gemma3-legal:latest", Prompt: prompt, Stream: false}
		llResp, err := gls.callOllama(ctx, "generate", llReq)
		if err != nil {
			c.JSON(http.StatusInternalServerError, TextSummarizeResponse{
				Summary:        "",
				Tokens:         tokens,
				TokenCount:     len(tokens),
				ProcessingTime: time.Since(start).Milliseconds(),
				Status:         "error",
				Error:          err.Error(),
			})
			return
		}
		c.JSON(http.StatusOK, TextSummarizeResponse{
			Summary:        llResp.Response,
			Tokens:         tokens,
			TokenCount:     len(tokens),
			ProcessingTime: time.Since(start).Milliseconds(),
			Status:         "success",
		})
	})
	// Capabilities endpoint for status checks
	r.GET("/api/simd/capabilities", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"cpu": gin.H{
				"avx2":   cpu.X86.HasAVX2,
				"sse4_2": cpu.X86.HasSSE42,
				"avx512": cpu.X86.HasAVX512F,
			},
			"ollama_url": gls.ollamaURL,
			"status":     "ok",
		})
	})
	r.GET("/health", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "healthy", "ollama_url": gls.ollamaURL}) })
	// Minimal metrics endpoint (Prometheus-like exposition)
	r.GET("/metrics", func(c *gin.Context) {
		total := gls.totalRequests.Load()
		analyze := gls.analyzeRequests.Load()
		errors := gls.analyzeErrors.Load()
		dur := gls.totalRequestDuration.Load()
		avg := float64(0)
		if total > 0 {
			avg = float64(dur) / float64(total)
		}
		c.Header("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(c.Writer, "go_ollama_simd_requests_total %d\n", total)
		fmt.Fprintf(c.Writer, "go_ollama_simd_analyze_requests_total %d\n", analyze)
		fmt.Fprintf(c.Writer, "go_ollama_simd_analyze_errors_total %d\n", errors)
		fmt.Fprintf(c.Writer, "go_ollama_simd_request_duration_ms_avg %.3f\n", avg)
	})
	return r
}

// RunServer entry
func RunServer() error {
	log.Printf("🚀 Starting Go-Ollama SIMD Service")
	log.Printf("🔧 CPU Features - AVX2: %v, SSE4.2: %v, AVX512: %v", cpu.X86.HasAVX2, cpu.X86.HasSSE42, cpu.X86.HasAVX512F)
	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}
	svc := NewGoLlamaService(ollamaURL)
	r := svc.SetupRoutes()
	port := os.Getenv("GO_OLLAMA_PORT")
	if port == "" {
		port = "8081"
	}
	return r.Run(":" + port)
}
