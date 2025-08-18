package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"
)

type ErrorLogProcessor struct {
	OllamaURL    string
	Context7URLs []string
}

type LogProcessRequest struct {
	LogFile   string `json:"logFile"`
	Content   string `json:"content"`
	Timestamp string `json:"timestamp"`
}

type ParsedError struct {
	File        string `json:"file"`
	Line        int    `json:"line"`
	Column      int    `json:"column"`
	Message     string `json:"message"`
	ErrorType   string `json:"errorType"`
	Severity    string `json:"severity"`
	Context     string `json:"context"`
	RawText     string `json:"rawText"`
}

type Recommendation struct {
	ErrorID     string `json:"errorId"`
	File        string `json:"file"`
	Issue       string `json:"issue"`
	Solution    string `json:"solution"`
	CodeFix     string `json:"codeFix"`
	Confidence  float64 `json:"confidence"`
	Category    string `json:"category"`
	AutoFixable bool   `json:"autoFixable"`
}

type ProcessResponse struct {
	Message            string           `json:"message"`
	RecommendationCount int             `json:"recommendationCount"`
	Errors             []ParsedError    `json:"errors"`
	Recommendations    []Recommendation `json:"recommendations"`
	ProcessingTime     string          `json:"processingTime"`
}

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func NewErrorLogProcessor() *ErrorLogProcessor {
	return &ErrorLogProcessor{
		OllamaURL: "http://localhost:11434/api/generate",
		Context7URLs: []string{
			"http://localhost:4100",
			"http://localhost:4101", 
			"http://localhost:4102",
			"http://localhost:4103",
			"http://localhost:4104",
			"http://localhost:4105",
			"http://localhost:4106",
			"http://localhost:4107",
		},
	}
}

func (e *ErrorLogProcessor) parseErrorLog(content string) ([]ParsedError, error) {
	var errors []ParsedError
	
	// TypeScript/Svelte error patterns
	tsErrorPattern := regexp.MustCompile(`(?m)^(.+\.(?:ts|js|svelte)):(\d+):(\d+) - error (TS\d+): (.+)$`)
	svelteErrorPattern := regexp.MustCompile(`(?m)^(.+\.svelte):(\d+):(\d+) (.+)$`)
	buildErrorPattern := regexp.MustCompile(`(?m)^Error: (.+) in (.+):(\d+):(\d+)`)
	
	lines := strings.Split(content, "\n")
	
	for i, line := range lines {
		// TypeScript errors
		if matches := tsErrorPattern.FindStringSubmatch(line); len(matches) > 5 {
			file := matches[1]
			line := parseIntSafe(matches[2])
			column := parseIntSafe(matches[3])
			errorCode := matches[4]
			message := matches[5]
			
			// Get surrounding context
			context := getContext(lines, i, 2)
			
			errors = append(errors, ParsedError{
				File:      file,
				Line:      line,
				Column:    column,
				Message:   message,
				ErrorType: errorCode,
				Severity:  "error",
				Context:   context,
				RawText:   matches[0],
			})
		}
		
		// Svelte errors
		if matches := svelteErrorPattern.FindStringSubmatch(line); len(matches) > 3 {
			file := matches[1]
			line := parseIntSafe(matches[2])
			column := parseIntSafe(matches[3])
			message := matches[4]
			
			context := getContext(lines, i, 2)
			
			errors = append(errors, ParsedError{
				File:      file,
				Line:      line,
				Column:    column,
				Message:   message,
				ErrorType: "SvelteError",
				Severity:  "error",
				Context:   context,
				RawText:   matches[0],
			})
		}
		
		// Build errors
		if matches := buildErrorPattern.FindStringSubmatch(line); len(matches) > 4 {
			message := matches[1]
			file := matches[2]
			line := parseIntSafe(matches[3])
			column := parseIntSafe(matches[4])
			
			context := getContext(lines, i, 2)
			
			errors = append(errors, ParsedError{
				File:      file,
				Line:      line,
				Column:    column,
				Message:   message,
				ErrorType: "BuildError",
				Severity:  "error",
				Context:   context,
				RawText:   matches[0],
			})
		}
	}
	
	log.Printf("📊 Parsed %d errors from log", len(errors))
	return errors, nil
}

func parseIntSafe(s string) int {
	if s == "" {
		return 0
	}
	var result int
	fmt.Sscanf(s, "%d", &result)
	return result
}

func getContext(lines []string, index, radius int) string {
	start := max(0, index-radius)
	end := min(len(lines), index+radius+1)
	
	var context []string
	for i := start; i < end; i++ {
		prefix := "  "
		if i == index {
			prefix = "→ "
		}
		context = append(context, prefix+lines[i])
	}
	
	return strings.Join(context, "\n")
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (e *ErrorLogProcessor) generateRecommendations(errors []ParsedError) ([]Recommendation, error) {
	var recommendations []Recommendation
	
	log.Printf("🤖 Generating recommendations for %d errors", len(errors))
	
	for i, err := range errors {
		// Create context-aware prompt for Ollama
		prompt := fmt.Sprintf(`You are an expert SvelteKit/TypeScript developer. Analyze this error and provide a specific solution:

Error Details:
- File: %s
- Line: %d, Column: %d  
- Error: %s
- Type: %s
- Context: %s

Please provide:
1. A clear explanation of what's wrong
2. The specific code fix needed
3. Whether this can be auto-fixed (true/false)
4. Category (types, imports, syntax, logic)
5. Confidence level (0.0-1.0)

Respond in JSON format:
{
  "issue": "brief explanation",
  "solution": "how to fix it", 
  "codeFix": "exact code to replace/add",
  "autoFixable": true/false,
  "category": "types|imports|syntax|logic",
  "confidence": 0.95
}`, err.File, err.Line, err.Column, err.Message, err.ErrorType, err.Context)

		recommendation, recErr := e.queryOllama(prompt)
		if recErr != nil {
			log.Printf("⚠️ Failed to get recommendation for error %d: %v", i, recErr)
			continue
		}
		
		recommendation.ErrorID = fmt.Sprintf("error_%d_%s_%d", i, strings.ReplaceAll(err.File, "/", "_"), err.Line)
		recommendation.File = err.File
		
		recommendations = append(recommendations, recommendation)
		
		// Small delay to not overwhelm Ollama
		time.Sleep(100 * time.Millisecond)
	}
	
	log.Printf("✅ Generated %d recommendations", len(recommendations))
	return recommendations, nil
}

func (e *ErrorLogProcessor) queryOllama(prompt string) (Recommendation, error) {
	reqData := OllamaRequest{
		Model:  "gemma2:2b", // Fast model for quick recommendations  
		Prompt: prompt,
		Stream: false,
	}
	
	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return Recommendation{}, err
	}
	
	resp, err := http.Post(e.OllamaURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return Recommendation{}, fmt.Errorf("ollama request failed: %v", err)
	}
	defer resp.Body.Close()
	
	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return Recommendation{}, err
	}
	
	// Parse JSON response from Ollama
	var rec Recommendation
	
	// Extract JSON from Ollama response (it might have extra text)
	response := ollamaResp.Response
	startIdx := strings.Index(response, "{")
	endIdx := strings.LastIndex(response, "}")
	
	if startIdx != -1 && endIdx != -1 && endIdx > startIdx {
		jsonStr := response[startIdx : endIdx+1]
		if err := json.Unmarshal([]byte(jsonStr), &rec); err != nil {
			// Fallback: create recommendation from raw response
			rec = Recommendation{
				Issue:       "Parse error occurred",
				Solution:    response,
				CodeFix:     "",
				Confidence:  0.5,
				Category:    "unknown",
				AutoFixable: false,
			}
		}
	} else {
		// Fallback: create recommendation from raw response  
		rec = Recommendation{
			Issue:       "Unable to parse structured response",
			Solution:    response,
			CodeFix:     "",
			Confidence:  0.3,
			Category:    "unknown",
			AutoFixable: false,
		}
	}
	
	return rec, nil
}

func (e *ErrorLogProcessor) handleProcessLog(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req LogProcessRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	log.Printf("📄 Processing log file: %s", req.LogFile)
	
	// Parse errors from log content
	errors, err := e.parseErrorLog(req.Content)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error parsing log: %v", err), http.StatusInternalServerError)
		return
	}
	
	// Generate recommendations
	recommendations, err := e.generateRecommendations(errors)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error generating recommendations: %v", err), http.StatusInternalServerError)
		return
	}
	
	processingTime := time.Since(startTime)
	
	// Send to Context7 pipeline for vector storage
	go e.sendToContext7Pipeline(errors, recommendations)
	
	response := ProcessResponse{
		Message:            "Error log processed successfully",
		RecommendationCount: len(recommendations),
		Errors:             errors,
		Recommendations:    recommendations,
		ProcessingTime:     processingTime.String(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
	
	log.Printf("✅ Processed %d errors, generated %d recommendations in %v", 
		len(errors), len(recommendations), processingTime)
}

func (e *ErrorLogProcessor) sendToContext7Pipeline(errors []ParsedError, recommendations []Recommendation) {
	// Send errors to our Context7 pipeline for vector storage
	pipelineData := map[string]interface{}{
		"errors":         errors,
		"recommendations": recommendations,
		"timestamp":      time.Now().Format(time.RFC3339),
		"source":         "recommendation-service",
	}
	
	jsonData, _ := json.Marshal(pipelineData)
	
	_, err := http.Post("http://localhost:8095/process", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("⚠️ Failed to send to Context7 pipeline: %v", err)
	} else {
		log.Printf("📡 Sent error data to Context7 pipeline for vector processing")
	}
}

func (e *ErrorLogProcessor) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":     "healthy",
		"service":    "recommendation-service",
		"ollama_url": e.OllamaURL,
		"context7_workers": len(e.Context7URLs),
	})
}

func (e *ErrorLogProcessor) handleStatus(w http.ResponseWriter, r *http.Request) {
	// Check Ollama connectivity
	ollamaHealthy := e.checkOllamaHealth()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"service":         "recommendation-service",
		"ollama_healthy":  ollamaHealthy,
		"context7_workers": len(e.Context7URLs),
		"endpoints": map[string]string{
			"process": "/api/process-error-log",
			"health":  "/health",
			"status":  "/status",
		},
	})
}

func (e *ErrorLogProcessor) checkOllamaHealth() bool {
	resp, err := http.Get("http://localhost:11434/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func main() {
	processor := NewErrorLogProcessor()
	
	http.HandleFunc("/api/process-error-log", processor.handleProcessLog)
	http.HandleFunc("/health", processor.handleHealth)
	http.HandleFunc("/status", processor.handleStatus)
	
	log.Printf("🚀 Recommendation Service starting on port 8096")
	log.Printf("🤖 Ollama URL: %s", processor.OllamaURL)
	log.Printf("📡 Context7 Workers: %d active", len(processor.Context7URLs))
	log.Printf("🌐 Endpoints:")
	log.Printf("   POST /api/process-error-log - Process error logs")
	log.Printf("   GET  /health - Health check")
	log.Printf("   GET  /status - Service status")
	
	log.Fatal(http.ListenAndServe(":8096", nil))
}