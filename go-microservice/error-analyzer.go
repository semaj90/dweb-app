package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/bytedance/sonic" // SIMD JSON parser
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
)

type TypeScriptError struct {
	File        string `json:"file"`
	Line        int    `json:"line"`
	Column      int    `json:"column"`
	ErrorCode   string `json:"errorCode"`
	Message     string `json:"message"`
	Category    string `json:"category"`
	Severity    string `json:"severity"`
	Context     string `json:"context,omitempty"`
	Suggestion  string `json:"suggestion,omitempty"`
}

type ErrorAnalysisResult struct {
	Timestamp    time.Time         `json:"timestamp"`
	TotalErrors  int               `json:"totalErrors"`
	Categories   map[string]int    `json:"categories"`
	Priorities   map[string]int    `json:"priorities"`
	Errors       []TypeScriptError `json:"errors"`
	Summary      string            `json:"summary"`
	Suggestions  []string          `json:"suggestions"`
	ClaudePrompt string            `json:"claudePrompt"`
}

type ClaudeRequest struct {
	Errors      []TypeScriptError `json:"errors"`
	Context     string            `json:"context"`
	ProjectType string            `json:"projectType"`
}

type ClaudeResponse struct {
	Analysis    string   `json:"analysis"`
	Priorities  []string `json:"priorities"`
	Suggestions []string `json:"suggestions"`
	FixPlan     []string `json:"fixPlan"`
}

var (
	// TypeScript error pattern regex
	tsErrorRegex = regexp.MustCompile(`^(.+?)\((\d+),(\d+)\): error (TS\d+): (.+)$`)
	
	// Error categorization map
	errorCategories = map[string]string{
		"TS2339": "property_access",     // Property does not exist
		"TS2345": "argument_type",       // Argument type mismatch
		"TS2322": "assignment_type",     // Type assignment error
		"TS2554": "argument_count",      // Wrong number of arguments
		"TS2307": "module_resolution",   // Cannot find module
		"TS2305": "export_member",       // Module has no exported member
		"TS2353": "object_literal",      // Object literal type error
		"TS2341": "private_property",    // Private property access
		"TS2484": "export_conflict",     // Export declaration conflicts
		"TS2698": "spread_type",         // Spread types error
		"TS2741": "missing_property",    // Property missing in type
		"TS2769": "overload_match",      // No overload matches
		"TS2559": "incompatible_type",   // Type incompatible
		"TS2551": "property_missing",    // Property missing, did you mean
		"TS18048": "undefined_access",   // Possibly undefined
	}
)

func main() {
	app := fiber.New(fiber.Config{
		JSONEncoder: sonic.Marshal,
		JSONDecoder: sonic.Unmarshal,
	})

	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowMethods: "GET,POST,PUT,DELETE,OPTIONS",
		AllowHeaders: "Origin,Content-Type,Accept,Authorization",
	}))

	// Parse TypeScript errors from npm run check output
	app.Post("/api/parse-errors", parseErrorsHandler)
	
	// Analyze errors with categorization and priorities
	app.Post("/api/analyze-errors", analyzeErrorsHandler)
	
	// Generate Claude AI prompt for error fixing
	app.Post("/api/claude-prompt", generateClaudePromptHandler)
	
	// Process Claude response and generate action items
	app.Post("/api/process-claude", processClaudeResponseHandler)

	// Health check
	app.Get("/api/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status": "healthy",
			"service": "error-analyzer",
			"timestamp": time.Now().UTC(),
			"features": []string{"simd-json", "error-parsing", "claude-integration"},
		})
	})

	fmt.Println("🔍 Error Analysis Service starting on http://localhost:8082")
	log.Fatal(app.Listen(":8082"))
}

func parseErrorsHandler(c *fiber.Ctx) error {
	var request struct {
		ErrorLog string `json:"errorLog"`
		Context  string `json:"context"`
	}

	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid JSON"})
	}

	errors := parseTypeScriptErrors(request.ErrorLog)
	
	result := ErrorAnalysisResult{
		Timestamp:   time.Now().UTC(),
		TotalErrors: len(errors),
		Errors:      errors,
		Categories:  categorizeErrors(errors),
		Priorities:  prioritizeErrors(errors),
	}

	return c.JSON(result)
}

func analyzeErrorsHandler(c *fiber.Ctx) error {
	var errors []TypeScriptError
	
	if err := c.BodyParser(&errors); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid JSON"})
	}

	analysis := analyzeErrorPatterns(errors)
	
	return c.JSON(analysis)
}

func generateClaudePromptHandler(c *fiber.Ctx) error {
	var request ClaudeRequest
	
	if err := c.BodyParser(&request); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid JSON"})
	}

	prompt := generateClaudePrompt(request.Errors, request.Context, request.ProjectType)
	
	return c.JSON(fiber.Map{
		"prompt": prompt,
		"errors": len(request.Errors),
		"timestamp": time.Now().UTC(),
	})
}

func processClaudeResponseHandler(c *fiber.Ctx) error {
	var response ClaudeResponse
	
	if err := c.BodyParser(&response); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid JSON"})
	}

	actionItems := processClaudeAnalysis(response)
	
	return c.JSON(actionItems)
}

func parseTypeScriptErrors(errorLog string) []TypeScriptError {
	var errors []TypeScriptError
	scanner := bufio.NewScanner(strings.NewReader(errorLog))
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if matches := tsErrorRegex.FindStringSubmatch(line); len(matches) == 6 {
			// Parse line and column numbers
			lineNum := 0
			colNum := 0
			fmt.Sscanf(matches[2], "%d", &lineNum)
			fmt.Sscanf(matches[3], "%d", &colNum)
			
			error := TypeScriptError{
				File:      matches[1],
				Line:      lineNum,
				Column:    colNum,
				ErrorCode: matches[4],
				Message:   matches[5],
				Category:  getErrorCategory(matches[4]),
				Severity:  getErrorSeverity(matches[4]),
				Context:   extractContext(matches[1], lineNum),
			}
			
			errors = append(errors, error)
		}
	}
	
	return errors
}

func getErrorCategory(errorCode string) string {
	if category, exists := errorCategories[errorCode]; exists {
		return category
	}
	return "unknown"
}

func getErrorSeverity(errorCode string) string {
	// Critical errors that block builds
	critical := []string{"TS2307", "TS2305", "TS2484", "TS2559"}
	for _, code := range critical {
		if errorCode == code {
			return "critical"
		}
	}
	
	// High priority errors
	high := []string{"TS2339", "TS2345", "TS2322", "TS2554"}
	for _, code := range high {
		if errorCode == code {
			return "high"
		}
	}
	
	return "medium"
}

func categorizeErrors(errors []TypeScriptError) map[string]int {
	categories := make(map[string]int)
	for _, err := range errors {
		categories[err.Category]++
	}
	return categories
}

func prioritizeErrors(errors []TypeScriptError) map[string]int {
	priorities := make(map[string]int)
	for _, err := range errors {
		priorities[err.Severity]++
	}
	return priorities
}

func extractContext(filePath string, lineNumber int) string {
	// Try to extract surrounding context from the file
	file, err := os.Open(filePath)
	if err != nil {
		return ""
	}
	defer file.Close()
	
	scanner := bufio.NewScanner(file)
	currentLine := 1
	var contextLines []string
	
	for scanner.Scan() {
		if currentLine >= lineNumber-2 && currentLine <= lineNumber+2 {
			contextLines = append(contextLines, fmt.Sprintf("%d: %s", currentLine, scanner.Text()))
		}
		currentLine++
		if currentLine > lineNumber+2 {
			break
		}
	}
	
	return strings.Join(contextLines, "\n")
}

func analyzeErrorPatterns(errors []TypeScriptError) ErrorAnalysisResult {
	categories := categorizeErrors(errors)
	priorities := prioritizeErrors(errors)
	
	// Generate summary
	summary := fmt.Sprintf("Found %d TypeScript errors across %d categories. Critical: %d, High: %d, Medium: %d", 
		len(errors), len(categories), priorities["critical"], priorities["high"], priorities["medium"])
	
	// Generate suggestions based on error patterns
	suggestions := generateSuggestions(categories, errors)
	
	return ErrorAnalysisResult{
		Timestamp:   time.Now().UTC(),
		TotalErrors: len(errors),
		Categories:  categories,
		Priorities:  priorities,
		Errors:      errors,
		Summary:     summary,
		Suggestions: suggestions,
	}
}

func generateSuggestions(categories map[string]int, errors []TypeScriptError) []string {
	var suggestions []string
	
	if categories["property_access"] > 5 {
		suggestions = append(suggestions, "High number of property access errors - check type definitions and imports")
	}
	
	if categories["module_resolution"] > 0 {
		suggestions = append(suggestions, "Module resolution errors found - verify imports and dependencies")
	}
	
	if categories["assignment_type"] > 3 {
		suggestions = append(suggestions, "Multiple type assignment errors - review type compatibility")
	}
	
	if categories["export_conflict"] > 0 {
		suggestions = append(suggestions, "Export conflicts detected - check for duplicate exports")
	}
	
	return suggestions
}

func generateClaudePrompt(errors []TypeScriptError, context string, projectType string) string {
	// Group errors by category for better analysis
	errorsByCategory := make(map[string][]TypeScriptError)
	for _, err := range errors {
		errorsByCategory[err.Category] = append(errorsByCategory[err.Category], err)
	}
	
	prompt := fmt.Sprintf(`# TypeScript Error Analysis Request

## Project Context
- Type: %s
- Total Errors: %d
- Analysis Timestamp: %s

## Error Summary by Category:
`, projectType, len(errors), time.Now().Format("2006-01-02 15:04:05"))

	for category, categoryErrors := range errorsByCategory {
		prompt += fmt.Sprintf("- %s: %d errors\n", category, len(categoryErrors))
	}
	
	prompt += "\n## Top Priority Errors (First 10):\n\n"
	
	// Show first 10 errors with context
	for i, err := range errors {
		if i >= 10 {
			break
		}
		prompt += fmt.Sprintf("### Error %d\n", i+1)
		prompt += fmt.Sprintf("- **File**: %s:%d:%d\n", err.File, err.Line, err.Column)
		prompt += fmt.Sprintf("- **Code**: %s\n", err.ErrorCode)
		prompt += fmt.Sprintf("- **Category**: %s\n", err.Category)
		prompt += fmt.Sprintf("- **Severity**: %s\n", err.Severity)
		prompt += fmt.Sprintf("- **Message**: %s\n", err.Message)
		if err.Context != "" {
			prompt += fmt.Sprintf("- **Context**:\n```\n%s\n```\n", err.Context)
		}
		prompt += "\n"
	}
	
	if context != "" {
		prompt += fmt.Sprintf("\n## Additional Context:\n%s\n", context)
	}
	
	prompt += `
## Please provide:

1. **Analysis**: Overall assessment of the error patterns and root causes
2. **Priorities**: Which errors should be fixed first and why
3. **Suggestions**: Specific recommendations for resolving each category of errors
4. **Fix Plan**: Step-by-step plan to systematically resolve these issues

Focus on:
- Schema/type definition issues
- Import/export problems  
- API endpoint type mismatches
- XState machine configurations
- Database schema alignment

Please format your response as structured JSON with the fields: analysis, priorities, suggestions, fixPlan.
`
	
	return prompt
}

func processClaudeAnalysis(response ClaudeResponse) map[string]interface{} {
	return map[string]interface{}{
		"timestamp":   time.Now().UTC(),
		"analysis":    response.Analysis,
		"priorities":  response.Priorities,
		"suggestions": response.Suggestions,
		"fixPlan":     response.FixPlan,
		"status":      "processed",
		"nextSteps": []string{
			"Implement fixes based on priority order",
			"Run npm run check after each fix",
			"Validate changes don't break existing functionality",
			"Update documentation if needed",
		},
	}
}