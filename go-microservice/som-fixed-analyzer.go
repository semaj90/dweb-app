//go:build legacy
// +build legacy

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/tidwall/buntdb"
)

// 🧠 SOM-based Intelligent Error Analyzer (Fixed Version)
// Uses Self-Organizing Maps for semantic clustering and simplified PageRank

type NPMError struct {
	Message      string    `json:"message"`
	File         string    `json:"file"`
	Line         int       `json:"line"`
	Severity     string    `json:"severity"`
	Category     string    `json:"category"`
	Type         string    `json:"type"`
	Timestamp    time.Time `json:"timestamp"`
	Context      []string  `json:"context"`
	Dependencies []string  `json:"dependencies"`
}

type IntelligentTodo struct {
	ID              string            `json:"id"`
	Priority        float64           `json:"priority"`
	Category        string            `json:"category"`
	Title           string            `json:"title"`
	Description     string            `json:"description"`
	EstimatedEffort time.Duration     `json:"estimated_effort"`
	Dependencies    []string          `json:"dependencies"`
	SuggestedFixes  []string          `json:"suggested_fixes"`
	RelatedErrors   []NPMError        `json:"related_errors"`
	Confidence      float64           `json:"confidence"`
	Tags            []string          `json:"tags"`
	CreatedAt       time.Time         `json:"created_at"`
	Metadata        map[string]interface{} `json:"metadata"`
}

type SOMNode struct {
	Weights      []float64  `json:"weights"`
	ErrorCluster []NPMError `json:"error_cluster"`
	Frequency    int        `json:"frequency"`
	Position     [2]int     `json:"position"`
	BMUCount     int        `json:"bmu_count"`
}

type SOMNetwork struct {
	Width        int         `json:"width"`
	Height       int         `json:"height"`
	Nodes        [][]SOMNode `json:"nodes"`
	LearningRate float64     `json:"learning_rate"`
	Radius       float64     `json:"radius"`
	Iterations   int         `json:"iterations"`
	FeatureDim   int         `json:"feature_dim"`
}

// Simplified PageRank for Todo Prioritization
type SimplePageRank struct {
	Nodes     map[string]*PRNode `json:"nodes"`
	Rankings  map[string]float64 `json:"rankings"`
	Damping   float64            `json:"damping"`
	Tolerance float64            `json:"tolerance"`
}

type PRNode struct {
	ID           string             `json:"id"`
	IncomingEdges map[string]float64 `json:"incoming_edges"`
	OutgoingEdges map[string]float64 `json:"outgoing_edges"`
	Rank         float64            `json:"rank"`
	Todo         IntelligentTodo    `json:"todo"`
}

type EnhancedSemanticAnalyzer struct {
	SOM          *SOMNetwork      `json:"som"`
	PageRank     *SimplePageRank  `json:"pagerank"`
	Cache        *buntdb.DB       `json:"-"`
	ErrorHistory []NPMError       `json:"error_history"`
	TodoHistory  []IntelligentTodo `json:"todo_history"`
	Config       AnalyzerConfig   `json:"config"`
}

type AnalyzerConfig struct {
	SOMGridSize      int           `json:"som_grid_size"`
	FeatureDimension int           `json:"feature_dimension"`
	LearningRate     float64       `json:"learning_rate"`
	TrainingEpochs   int           `json:"training_epochs"`
	PageRankDamping  float64       `json:"pagerank_damping"`
	CacheTimeout     time.Duration `json:"cache_timeout"`
}

func NewEnhancedSemanticAnalyzer(config AnalyzerConfig) (*EnhancedSemanticAnalyzer, error) {
	som := &SOMNetwork{
		Width:        config.SOMGridSize,
		Height:       config.SOMGridSize,
		LearningRate: config.LearningRate,
		Radius:       float64(config.SOMGridSize) / 2.0,
		Iterations:   config.TrainingEpochs,
		FeatureDim:   config.FeatureDimension,
	}
	
	// Initialize SOM grid
	som.Nodes = make([][]SOMNode, som.Width)
	for i := range som.Nodes {
		som.Nodes[i] = make([]SOMNode, som.Height)
		for j := range som.Nodes[i] {
			som.Nodes[i][j] = SOMNode{
				Weights:      randomVector(config.FeatureDimension),
				Position:     [2]int{i, j},
				ErrorCluster: make([]NPMError, 0),
			}
		}
	}
	
	// Initialize SimplePageRank
	pageRank := &SimplePageRank{
		Nodes:     make(map[string]*PRNode),
		Rankings:  make(map[string]float64),
		Damping:   config.PageRankDamping,
		Tolerance: 1e-6,
	}
	
	cache, err := buntdb.Open(":memory:")
	if err != nil {
		return nil, fmt.Errorf("failed to open cache: %v", err)
	}
	
	analyzer := &EnhancedSemanticAnalyzer{
		SOM:          som,
		PageRank:     pageRank,
		Cache:        cache,
		ErrorHistory: make([]NPMError, 0),
		TodoHistory:  make([]IntelligentTodo, 0),
		Config:       config,
	}
	
	return analyzer, nil
}

func (esa *EnhancedSemanticAnalyzer) ExtractErrorFeatures(npmError NPMError) []float64 {
	features := make([]float64, esa.Config.FeatureDimension)
	
	message := strings.ToLower(npmError.Message)
	words := strings.Fields(message)
	
	// Word frequency features (first 50 dimensions)
	wordMap := make(map[string]float64)
	for _, word := range words {
		wordMap[word]++
	}
	
	i := 0
	for _, freq := range wordMap {
		if i >= 50 { break }
		features[i] = freq / float64(len(words))
		i++
	}
	
	// Severity encoding
	severityMap := map[string]float64{
		"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0,
	}
	if i < len(features) {
		features[i] = severityMap[npmError.Severity]
		i++
	}
	
	// Category encoding
	categories := []string{"typescript", "service", "build", "import", "syntax"}
	for _, cat := range categories {
		if i >= len(features) { break }
		if strings.Contains(strings.ToLower(npmError.Category), cat) {
			features[i] = 1.0
		}
		i++
	}
	
	// Semantic patterns
	patterns := []string{"cannot find", "import", "undefined", "syntax", "error"}
	for _, pattern := range patterns {
		if i >= len(features) { break }
		if strings.Contains(message, pattern) {
			features[i] = 1.0
		}
		i++
	}
	
	return features
}

func (esa *EnhancedSemanticAnalyzer) TrainSOM(errors []NPMError) error {
	log.Printf("🧠 Training SOM on %d error patterns...", len(errors))
	
	errorFeatures := make([][]float64, len(errors))
	for i, err := range errors {
		errorFeatures[i] = esa.ExtractErrorFeatures(err)
	}
	
	for epoch := 0; epoch < esa.SOM.Iterations; epoch++ {
		learningRate := esa.SOM.LearningRate * math.Exp(-float64(epoch)/float64(esa.SOM.Iterations))
		radius := esa.SOM.Radius * math.Exp(-float64(epoch)/float64(esa.SOM.Iterations))
		
		for i, features := range errorFeatures {
			bmux, bmuy := esa.findBMU(features)
			esa.updateSOMWeights(bmux, bmuy, features, learningRate, radius)
			
			esa.SOM.Nodes[bmux][bmuy].ErrorCluster = append(
				esa.SOM.Nodes[bmux][bmuy].ErrorCluster, errors[i])
			esa.SOM.Nodes[bmux][bmuy].BMUCount++
		}
		
		if epoch%100 == 0 {
			log.Printf("   Training epoch %d/%d (lr=%.4f, r=%.2f)",
				epoch, esa.SOM.Iterations, learningRate, radius)
		}
	}
	
	log.Printf("✅ SOM training completed")
	return nil
}

func (esa *EnhancedSemanticAnalyzer) findBMU(features []float64) (int, int) {
	minDist := math.Inf(1)
	bmux, bmuy := 0, 0
	
	for i := 0; i < esa.SOM.Width; i++ {
		for j := 0; j < esa.SOM.Height; j++ {
			dist := euclideanDistance(features, esa.SOM.Nodes[i][j].Weights)
			if dist < minDist {
				minDist = dist
				bmux, bmuy = i, j
			}
		}
	}
	
	return bmux, bmuy
}

func (esa *EnhancedSemanticAnalyzer) updateSOMWeights(bmux, bmuy int, features []float64,
	learningRate, radius float64) {
	
	for i := 0; i < esa.SOM.Width; i++ {
		for j := 0; j < esa.SOM.Height; j++ {
			dist := math.Sqrt(float64((i-bmux)*(i-bmux) + (j-bmuy)*(j-bmuy)))
			
			if dist <= radius {
				influence := math.Exp(-(dist * dist) / (2 * radius * radius))
				
				for k := 0; k < len(features); k++ {
					delta := learningRate * influence * (features[k] - esa.SOM.Nodes[i][j].Weights[k])
					esa.SOM.Nodes[i][j].Weights[k] += delta
				}
			}
		}
	}
}

func (esa *EnhancedSemanticAnalyzer) GenerateIntelligentTodos(errors []NPMError) ([]IntelligentTodo, error) {
	log.Printf("🎯 Generating intelligent todos from %d errors...", len(errors))
	
	if err := esa.TrainSOM(errors); err != nil {
		return nil, err
	}
	
	todos := make([]IntelligentTodo, 0)
	
	for i := 0; i < esa.SOM.Width; i++ {
		for j := 0; j < esa.SOM.Height; j++ {
			node := &esa.SOM.Nodes[i][j]
			if len(node.ErrorCluster) == 0 {
				continue
			}
			
			todo := esa.generateTodoFromCluster(node.ErrorCluster, i, j)
			if todo.Confidence > 0.3 {
				todos = append(todos, todo)
			}
		}
	}
	
	// Apply PageRank prioritization
	rankedTodos := esa.applySimplePageRank(todos)
	
	log.Printf("✅ Generated %d intelligent todos", len(rankedTodos))
	return rankedTodos, nil
}

func (esa *EnhancedSemanticAnalyzer) generateTodoFromCluster(errors []NPMError, x, y int) IntelligentTodo {
	severityCount := make(map[string]int)
	categoryCount := make(map[string]int)
	fileCount := make(map[string]int)
	
	for _, err := range errors {
		severityCount[err.Severity]++
		categoryCount[err.Category]++
		fileCount[err.File]++
	}
	
	dominantSeverity := findDominant(severityCount)
	dominantCategory := findDominant(categoryCount)
	dominantFile := findDominant(fileCount)
	
	description := esa.generateTodoDescription(errors, dominantCategory, dominantFile)
	priority := esa.calculateTodoPriority(errors, dominantSeverity)
	effort := esa.estimateEffort(errors, dominantCategory)
	fixes := esa.generateSuggestedFixes(errors, dominantCategory)
	confidence := esa.calculateConfidence(len(errors), dominantCategory)
	
	todo := IntelligentTodo{
		ID:              fmt.Sprintf("som-cluster-%d-%d", x, y),
		Priority:        priority,
		Category:        dominantCategory,
		Title:           esa.generateTodoTitle(dominantCategory, len(errors)),
		Description:     description,
		EstimatedEffort: effort,
		Dependencies:    esa.findTodoDependencies(errors),
		SuggestedFixes:  fixes,
		RelatedErrors:   errors,
		Confidence:      confidence,
		Tags:            esa.generateTags(errors, dominantCategory),
		CreatedAt:       time.Now(),
		Metadata: map[string]interface{}{
			"cluster_position": [2]int{x, y},
			"error_count":      len(errors),
			"dominant_file":    dominantFile,
		},
	}
	
	return todo
}

func (esa *EnhancedSemanticAnalyzer) applySimplePageRank(todos []IntelligentTodo) []IntelligentTodo {
	if len(todos) == 0 {
		return todos
	}
	
	// Clear existing nodes
	esa.PageRank.Nodes = make(map[string]*PRNode)
	
	// Add nodes
	for _, todo := range todos {
		esa.PageRank.Nodes[todo.ID] = &PRNode{
			ID:            todo.ID,
			IncomingEdges: make(map[string]float64),
			OutgoingEdges: make(map[string]float64),
			Rank:          1.0 / float64(len(todos)),
			Todo:          todo,
		}
	}
	
	// Add edges based on relationships
	for _, todo := range todos {
		for _, otherTodo := range todos {
			if todo.ID != otherTodo.ID {
				similarity := esa.calculateSemanticSimilarity(todo, otherTodo)
				if similarity > 0.3 {
					esa.PageRank.Nodes[todo.ID].OutgoingEdges[otherTodo.ID] = similarity
					esa.PageRank.Nodes[otherTodo.ID].IncomingEdges[todo.ID] = similarity
				}
			}
		}
	}
	
	// Run PageRank iterations
	for iter := 0; iter < 50; iter++ {
		newRanks := make(map[string]float64)
		
		for nodeID := range esa.PageRank.Nodes {
			newRanks[nodeID] = (1.0 - esa.PageRank.Damping) / float64(len(todos))
		}
		
		for nodeID, node := range esa.PageRank.Nodes {
			for incomingID, weight := range node.IncomingEdges {
				incomingNode := esa.PageRank.Nodes[incomingID]
				outDegree := len(incomingNode.OutgoingEdges)
				if outDegree > 0 {
					contribution := esa.PageRank.Damping * incomingNode.Rank * weight / float64(outDegree)
					newRanks[nodeID] += contribution
				}
			}
		}
		
		// Update ranks
		for nodeID := range esa.PageRank.Nodes {
			esa.PageRank.Nodes[nodeID].Rank = newRanks[nodeID]
		}
	}
	
	// Apply rankings to todos
	for i := range todos {
		if node, exists := esa.PageRank.Nodes[todos[i].ID]; exists {
			todos[i].Priority = node.Rank
		}
	}
	
	// Sort by priority
	sort.Slice(todos, func(i, j int) bool {
		return todos[i].Priority > todos[j].Priority
	})
	
	return todos
}

// Helper functions
func randomVector(dim int) []float64 {
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rand.Float64()*2 - 1
	}
	return vec
}

func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func findDominant(counts map[string]int) string {
	maxCount := 0
	dominant := ""
	for key, count := range counts {
		if count > maxCount {
			maxCount = count
			dominant = key
		}
	}
	return dominant
}

func (esa *EnhancedSemanticAnalyzer) generateTodoDescription(errors []NPMError, category, file string) string {
	if len(errors) == 1 {
		return fmt.Sprintf("Fix %s error in %s: %s", category, file, errors[0].Message)
	}
	return fmt.Sprintf("Fix %d %s errors in %s and related files", len(errors), category, file)
}

func (esa *EnhancedSemanticAnalyzer) generateTodoTitle(category string, errorCount int) string {
	if errorCount == 1 {
		return fmt.Sprintf("Fix %s Error", strings.Title(category))
	}
	return fmt.Sprintf("Fix %d %s Errors", errorCount, strings.Title(category))
}

func (esa *EnhancedSemanticAnalyzer) calculateTodoPriority(errors []NPMError, severity string) float64 {
	severityWeights := map[string]float64{
		"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2,
	}
	
	base := severityWeights[severity]
	count := float64(len(errors))
	
	return base * (1.0 + math.Log(1.0+count)/10.0)
}

func (esa *EnhancedSemanticAnalyzer) estimateEffort(errors []NPMError, category string) time.Duration {
	categoryEffort := map[string]time.Duration{
		"typescript": 15 * time.Minute,
		"import":     5 * time.Minute,
		"syntax":     10 * time.Minute,
		"service":    30 * time.Minute,
		"build":      20 * time.Minute,
	}
	
	base := categoryEffort[category]
	if base == 0 {
		base = 20 * time.Minute
	}
	
	scale := 1.0 + float64(len(errors))*0.2
	return time.Duration(float64(base) * scale)
}

func (esa *EnhancedSemanticAnalyzer) generateSuggestedFixes(errors []NPMError, category string) []string {
	fixes := []string{}
	
	switch category {
	case "typescript":
		fixes = append(fixes, "Add missing type declarations", "Fix import statements", "Update tsconfig.json")
	case "service":
		fixes = append(fixes, "Check service connectivity", "Verify configuration", "Restart services")
	case "build":
		fixes = append(fixes, "Clear build cache", "Update dependencies", "Fix configuration")
	default:
		fixes = append(fixes, "Review error messages", "Check documentation", "Apply standard fixes")
	}
	
	return fixes
}

func (esa *EnhancedSemanticAnalyzer) findTodoDependencies(errors []NPMError) []string {
	deps := []string{}
	files := make(map[string]bool)
	
	for _, err := range errors {
		for _, dep := range err.Dependencies {
			if !files[dep] {
				deps = append(deps, dep)
				files[dep] = true
			}
		}
	}
	
	return deps
}

func (esa *EnhancedSemanticAnalyzer) calculateConfidence(errorCount int, category string) float64 {
	categoryConfidence := map[string]float64{
		"typescript": 0.9, "syntax": 0.95, "import": 0.85,
		"service": 0.7, "build": 0.8,
	}
	
	base := categoryConfidence[category]
	if base == 0 {
		base = 0.6
	}
	
	countBonus := math.Min(0.2, float64(errorCount)*0.05)
	return math.Min(1.0, base+countBonus)
}

func (esa *EnhancedSemanticAnalyzer) generateTags(errors []NPMError, category string) []string {
	tags := []string{category}
	
	severities := make(map[string]bool)
	for _, err := range errors {
		severities[err.Severity] = true
	}
	
	for severity := range severities {
		tags = append(tags, severity)
	}
	
	if len(errors) > 5 {
		tags = append(tags, "bulk")
	}
	
	return tags
}

func (esa *EnhancedSemanticAnalyzer) calculateSemanticSimilarity(todo1, todo2 IntelligentTodo) float64 {
	words1 := strings.Fields(strings.ToLower(todo1.Description))
	words2 := strings.Fields(strings.ToLower(todo2.Description))
	
	wordSet1 := make(map[string]bool)
	for _, word := range words1 {
		wordSet1[word] = true
	}
	
	intersection := 0
	for _, word := range words2 {
		if wordSet1[word] {
			intersection++
		}
	}
	
	union := len(words1) + len(words2) - intersection
	if union == 0 {
		return 0
	}
	
	return float64(intersection) / float64(union)
}

func main() {
	log.Println("🚀 Enhanced SOM-based Intelligent Error Analyzer (Fixed)")
	
	config := AnalyzerConfig{
		SOMGridSize:      8,
		FeatureDimension: 64,
		LearningRate:     0.5,
		TrainingEpochs:   200,
		PageRankDamping:  0.85,
		CacheTimeout:     5 * time.Minute,
	}
	
	analyzer, err := NewEnhancedSemanticAnalyzer(config)
	if err != nil {
		log.Fatalf("Failed to initialize analyzer: %v", err)
	}
	defer analyzer.Cache.Close()
	
	// Sample npm errors for testing
	sampleErrors := []NPMError{
		{
			Message:   "Cannot find module '@types/node'",
			File:      "src/app.ts",
			Line:      1,
			Severity:  "high",
			Category:  "typescript",
			Type:      "error",
			Timestamp: time.Now(),
			Context:   []string{"import { Server } from 'http'"},
		},
		{
			Message:   "Property 'foo' does not exist on type 'Object'",
			File:      "src/utils.ts",
			Line:      15,
			Severity:  "medium",
			Category:  "typescript",
			Type:      "error",
			Timestamp: time.Now(),
			Context:   []string{"const result = obj.foo"},
		},
		{
			Message:      "Service unavailable: http://localhost:8080",
			File:         "src/api.ts",
			Line:         25,
			Severity:     "critical",
			Category:     "service",
			Type:         "error",
			Timestamp:    time.Now(),
			Dependencies: []string{"service-config.json"},
		},
		{
			Message:   "Unexpected token ';'",
			File:      "src/parser.ts",
			Line:      42,
			Severity:  "high",
			Category:  "syntax",
			Type:      "error",
			Timestamp: time.Now(),
			Context:   []string{"function parse() {;"},
		},
		{
			Message:   "Module not found: Can't resolve './missing'",
			File:      "src/index.ts",
			Line:      8,
			Severity:  "high", 
			Category:  "import",
			Type:      "error",
			Timestamp: time.Now(),
			Context:   []string{"import { data } from './missing'"},
		},
	}
	
	// Generate intelligent todos
	todos, err := analyzer.GenerateIntelligentTodos(sampleErrors)
	if err != nil {
		log.Fatalf("Failed to generate todos: %v", err)
	}
	
	// Display results
	log.Printf("\n🎯 Generated %d Intelligent Todos:\n", len(todos))
	
	for i, todo := range todos {
		log.Printf("%d. [%s] %s", i+1, strings.ToUpper(todo.Category), todo.Title)
		log.Printf("   Priority: %.3f | Confidence: %.1f%% | Effort: %v",
			todo.Priority, todo.Confidence*100, todo.EstimatedEffort)
		log.Printf("   Description: %s", todo.Description)
		log.Printf("   Fixes: %s", strings.Join(todo.SuggestedFixes, ", "))
		log.Printf("   Tags: %s", strings.Join(todo.Tags, ", "))
		log.Printf("   Related Errors: %d\n", len(todo.RelatedErrors))
	}
	
	// Output JSON for integration
	output, _ := json.MarshalIndent(todos, "", "  ")
	fmt.Printf("\n📄 JSON Output for Frontend Integration:\n%s\n", output)
	
	log.Println("✅ SOM Analysis complete! Ready for WebGPU caching integration.")
}