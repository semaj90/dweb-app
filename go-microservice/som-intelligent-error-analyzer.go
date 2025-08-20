//go:build legacy
// +build legacy

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/dominikbraun/graph"
	"github.com/RoaringBitmap/roaring"
	"github.com/tidwall/buntdb"
	"gorgonia.org/tensor"
)

// 🧠 SOM-based Intelligent Error Analyzer
// Uses Self-Organizing Maps for semantic clustering and PageRank for prioritization

type NPMError struct {
	Message    string    `json:"message"`
	File       string    `json:"file"`
	Line       int       `json:"line"`
	Severity   string    `json:"severity"`   // low, medium, high, critical
	Category   string    `json:"category"`   // typescript, service, build, etc.
	Type       string    `json:"type"`       // error, warning, info
	Timestamp  time.Time `json:"timestamp"`
	Context    []string  `json:"context"`    // surrounding code context
	Dependencies []string `json:"dependencies"` // related files/modules
}

type IntelligentTodo struct {
	ID               string          `json:"id"`
	Priority         float64         `json:"priority"`         // 0.0 - 1.0 (PageRank score)
	Category         string          `json:"category"`
	Title            string          `json:"title"`
	Description      string          `json:"description"`
	EstimatedEffort  time.Duration   `json:"estimated_effort"` // time estimate
	Dependencies     []string        `json:"dependencies"`     // prerequisite todos
	SuggestedFixes   []string        `json:"suggested_fixes"`
	RelatedErrors    []NPMError      `json:"related_errors"`
	Confidence       float64         `json:"confidence"`       // AI confidence in solution
	Tags             []string        `json:"tags"`
	CreatedAt        time.Time       `json:"created_at"`
	Metadata         map[string]interface{} `json:"metadata"`
}

// Self-Organizing Map Node
type SOMNode struct {
	Weights      []float64 `json:"weights"`       // Feature vector
	ErrorCluster []NPMError `json:"error_cluster"` // Assigned errors
	Frequency    int       `json:"frequency"`     // Activation frequency
	Position     [2]int    `json:"position"`      // Grid position (x, y)
	BMUCount     int       `json:"bmu_count"`     // Best Matching Unit count
}

// SOM Network for Error Clustering
type SOMNetwork struct {
	Width          int          `json:"width"`           // Grid width
	Height         int          `json:"height"`          // Grid height
	Nodes          [][]SOMNode  `json:"nodes"`           // 2D grid of nodes
	LearningRate   float64      `json:"learning_rate"`   // Initial learning rate
	Radius         float64      `json:"radius"`          // Initial neighborhood radius
	Iterations     int          `json:"iterations"`      // Training iterations
	FeatureDim     int          `json:"feature_dim"`     // Feature vector dimension
}

// PageRank Graph for Todo Prioritization  
type TodoPageRank struct {
	Graph     graph.Graph[string, IntelligentTodo] `json:"-"`
	Cache     *buntdb.DB                           `json:"-"`
	Rankings  map[string]float64                   `json:"rankings"`
	Iteration int                                  `json:"iteration"`
}

// Enhanced Semantic Error Analyzer
type EnhancedSemanticAnalyzer struct {
	SOM           *SOMNetwork       `json:"som"`
	PageRank      *TodoPageRank     `json:"pagerank"`  
	Cache         *buntdb.DB        `json:"-"`
	ErrorHistory  []NPMError        `json:"error_history"`
	TodoHistory   []IntelligentTodo `json:"todo_history"`
	Bitmap        *roaring.Bitmap   `json:"-"` // Fast set operations
	Config        AnalyzerConfig    `json:"config"`
}

type AnalyzerConfig struct {
	SOMGridSize      int     `json:"som_grid_size"`
	FeatureDimension int     `json:"feature_dimension"`
	LearningRate     float64 `json:"learning_rate"`
	TrainingEpochs   int     `json:"training_epochs"`
	PageRankDamping  float64 `json:"pagerank_damping"`
	CacheTimeout     time.Duration `json:"cache_timeout"`
}

// Initialize the Enhanced Semantic Analyzer
func NewEnhancedSemanticAnalyzer(config AnalyzerConfig) (*EnhancedSemanticAnalyzer, error) {
	// Initialize SOM Network
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
	
	// Initialize PageRank graph
	pageRankGraph := graph.New(graph.StringHash, graph.Directed(), graph.Weighted())
	
	// Initialize BuntDB cache
	cache, err := buntdb.Open(":memory:")
	if err != nil {
		return nil, fmt.Errorf("failed to open cache: %v", err)
	}
	
	analyzer := &EnhancedSemanticAnalyzer{
		SOM: som,
		PageRank: &TodoPageRank{
			Graph:    pageRankGraph,
			Cache:    cache,
			Rankings: make(map[string]float64),
		},
		Cache:        cache,
		ErrorHistory: make([]NPMError, 0),
		TodoHistory:  make([]IntelligentTodo, 0),
		Bitmap:       roaring.New(),
		Config:       config,
	}
	
	return analyzer, nil
}

// Extract semantic features from npm errors
func (esa *EnhancedSemanticAnalyzer) ExtractErrorFeatures(npmError NPMError) []float64 {
	features := make([]float64, esa.Config.FeatureDimension)
	
	// Text-based features
	message := strings.ToLower(npmError.Message)
	words := strings.Fields(message)
	
	// Word frequency features (first 50 dimensions)
	wordMap := make(map[string]float64)
	for _, word := range words {
		wordMap[word]++
	}
	
	i := 0
	for word, freq := range wordMap {
		if i >= 50 { break }
		features[i] = freq / float64(len(words)) // Normalize
		i++
	}
	
	// Severity encoding (dimensions 50-54)
	severityMap := map[string]float64{
		"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0,
	}
	if i < len(features) {
		features[i] = severityMap[npmError.Severity]
		i++
	}
	
	// Category encoding (dimensions 54-64)
	categories := []string{"typescript", "service", "build", "import", "syntax", 
	                     "runtime", "network", "database", "cache", "auth"}
	for _, cat := range categories {
		if i >= len(features) { break }
		if strings.Contains(strings.ToLower(npmError.Category), cat) {
			features[i] = 1.0
		}
		i++
	}
	
	// Semantic embeddings (dimensions 64+)
	// Simple semantic features based on error patterns
	semanticPatterns := []string{
		"cannot find", "import", "export", "undefined", "null", "reference",
		"syntax", "missing", "failed", "error", "warning", "type",
	}
	
	for _, pattern := range semanticPatterns {
		if i >= len(features) { break }
		if strings.Contains(message, pattern) {
			features[i] = 1.0
		}
		i++
	}
	
	return features
}

// Train SOM on error patterns
func (esa *EnhancedSemanticAnalyzer) TrainSOM(errors []NPMError) error {
	log.Printf("🧠 Training SOM on %d error patterns...", len(errors))
	
	// Extract features for all errors
	errorFeatures := make([][]float64, len(errors))
	for i, err := range errors {
		errorFeatures[i] = esa.ExtractErrorFeatures(err)
	}
	
	// Training loop
	for epoch := 0; epoch < esa.SOM.Iterations; epoch++ {
		learningRate := esa.SOM.LearningRate * math.Exp(-float64(epoch)/float64(esa.SOM.Iterations))
		radius := esa.SOM.Radius * math.Exp(-float64(epoch)/float64(esa.SOM.Iterations))
		
		for i, features := range errorFeatures {
			// Find Best Matching Unit (BMU)
			bmux, bmuy := esa.findBMU(features)
			
			// Update BMU and neighbors
			esa.updateSOMWeights(bmux, bmuy, features, learningRate, radius)
			
			// Assign error to BMU
			esa.SOM.Nodes[bmux][bmuy].ErrorCluster = append(
				esa.SOM.Nodes[bmux][bmuy].ErrorCluster, errors[i])
			esa.SOM.Nodes[bmux][bmuy].BMUCount++
		}
		
		if epoch % 100 == 0 {
			log.Printf("   Training epoch %d/%d (lr=%.4f, r=%.2f)", 
				epoch, esa.SOM.Iterations, learningRate, radius)
		}
	}
	
	log.Printf("✅ SOM training completed")
	return nil
}

// Find Best Matching Unit in SOM
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

// Update SOM weights using neighborhood function
func (esa *EnhancedSemanticAnalyzer) updateSOMWeights(bmux, bmuy int, features []float64, 
	learningRate, radius float64) {
	
	for i := 0; i < esa.SOM.Width; i++ {
		for j := 0; j < esa.SOM.Height; j++ {
			// Calculate distance from BMU
			dist := math.Sqrt(float64((i-bmux)*(i-bmux) + (j-bmuy)*(j-bmuy)))
			
			// Neighborhood function (Gaussian)
			if dist <= radius {
				influence := math.Exp(-(dist * dist) / (2 * radius * radius))
				
				// Update weights
				for k := 0; k < len(features); k++ {
					delta := learningRate * influence * (features[k] - esa.SOM.Nodes[i][j].Weights[k])
					esa.SOM.Nodes[i][j].Weights[k] += delta
				}
			}
		}
	}
}

// Generate intelligent todos from clustered errors
func (esa *EnhancedSemanticAnalyzer) GenerateIntelligentTodos(errors []NPMError) ([]IntelligentTodo, error) {
	log.Printf("🎯 Generating intelligent todos from %d errors...", len(errors))
	
	// Train SOM if not already trained
	if len(esa.ErrorHistory) == 0 {
		if err := esa.TrainSOM(errors); err != nil {
			return nil, err
		}
	}
	
	// Generate todos from SOM clusters
	todos := make([]IntelligentTodo, 0)
	
	for i := 0; i < esa.SOM.Width; i++ {
		for j := 0; j < esa.SOM.Height; j++ {
			node := &esa.SOM.Nodes[i][j]
			if len(node.ErrorCluster) == 0 {
				continue
			}
			
			// Analyze error cluster
			todo := esa.generateTodoFromCluster(node.ErrorCluster, i, j)
			if todo.Confidence > 0.3 { // Only include confident recommendations
				todos = append(todos, todo)
			}
		}
	}
	
	// Apply PageRank prioritization
	rankedTodos, err := esa.applyPageRankPrioritization(todos)
	if err != nil {
		log.Printf("⚠️  PageRank prioritization failed: %v", err)
		return todos, nil // Return unranked todos
	}
	
	log.Printf("✅ Generated %d intelligent todos", len(rankedTodos))
	return rankedTodos, nil
}

// Generate todo from error cluster
func (esa *EnhancedSemanticAnalyzer) generateTodoFromCluster(errors []NPMError, x, y int) IntelligentTodo {
	// Analyze error patterns
	severityCount := make(map[string]int)
	categoryCount := make(map[string]int)
	fileCount := make(map[string]int)
	
	for _, err := range errors {
		severityCount[err.Severity]++
		categoryCount[err.Category]++
		fileCount[err.File]++
	}
	
	// Find dominant patterns
	dominantSeverity := findDominant(severityCount)
	dominantCategory := findDominant(categoryCount)
	dominantFile := findDominant(fileCount)
	
	// Generate intelligent description
	description := esa.generateTodoDescription(errors, dominantCategory, dominantFile)
	
	// Calculate priority based on error patterns
	priority := esa.calculateTodoPriority(errors, dominantSeverity)
	
	// Estimate effort
	effort := esa.estimateEffort(errors, dominantCategory)
	
	// Generate suggested fixes
	fixes := esa.generateSuggestedFixes(errors, dominantCategory)
	
	// Calculate confidence
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
			"error_count":     len(errors),
			"dominant_file":   dominantFile,
		},
	}
	
	return todo
}

// Apply PageRank prioritization to todos
func (esa *EnhancedSemanticAnalyzer) applyPageRankPrioritization(todos []IntelligentTodo) ([]IntelligentTodo, error) {
	if len(todos) == 0 {
		return todos, nil
	}
	
	// Clear existing graph
	esa.PageRank.Graph = graph.New(graph.StringHash, graph.Directed(), graph.Weighted())
	
	// Add todos as vertices
	for _, todo := range todos {
		esa.PageRank.Graph.AddVertex(todo.ID, todo)
	}
	
	// Add edges based on dependencies and relationships
	for _, todo := range todos {
		for _, dep := range todo.Dependencies {
			// Find dependency todo
			for _, depTodo := range todos {
				if strings.Contains(depTodo.Title, dep) {
					weight := esa.calculateEdgeWeight(todo, depTodo)
					esa.PageRank.Graph.AddEdge(depTodo.ID, todo.ID, graph.EdgeWeight(weight))
				}
			}
		}
		
		// Add semantic similarity edges
		for _, otherTodo := range todos {
			if todo.ID != otherTodo.ID {
				similarity := esa.calculateSemanticSimilarity(todo, otherTodo)
				if similarity > 0.5 {
					weight := similarity
					esa.PageRank.Graph.AddEdge(todo.ID, otherTodo.ID, graph.EdgeWeight(weight))
				}
			}
		}
	}
	
	// Run PageRank algorithm
	rankings := esa.runPageRank()
	
	// Apply rankings to todos
	for i := range todos {
		if rank, exists := rankings[todos[i].ID]; exists {
			todos[i].Priority = rank
		}
	}
	
	// Sort by priority (descending)
	sort.Slice(todos, func(i, j int) bool {
		return todos[i].Priority > todos[j].Priority
	})
	
	return todos, nil
}

// Run PageRank algorithm
func (esa *EnhancedSemanticAnalyzer) runPageRank() map[string]float64 {
	damping := esa.Config.PageRankDamping
	iterations := 100
	tolerance := 1e-6
	
	// Get all vertices
	vertices, _ := esa.PageRank.Graph.AdjacencyMap()
	n := len(vertices)
	
	if n == 0 {
		return make(map[string]float64)
	}
	
	// Initialize ranks
	ranks := make(map[string]float64)
	for vertex := range vertices {
		ranks[vertex] = 1.0 / float64(n)
	}
	
	// PageRank iterations
	for iter := 0; iter < iterations; iter++ {
		newRanks := make(map[string]float64)
		
		// Initialize with random walk probability
		for vertex := range vertices {
			newRanks[vertex] = (1.0 - damping) / float64(n)
		}
		
		// Add contributions from incoming links
		for vertex, edges := range vertices {
			outDegree := len(edges)
			if outDegree > 0 {
				for _, edge := range edges {
					target := edge.Target
					weight := 1.0 // Default weight
					if edge.Properties != nil {
						if w, ok := edge.Properties.(graph.EdgeWeight); ok {
							weight = float64(w)
						}
					}
					contribution := damping * ranks[vertex] * weight / float64(outDegree)
					newRanks[target] += contribution
				}
			}
		}
		
		// Check convergence
		diff := 0.0
		for vertex := range vertices {
			diff += math.Abs(newRanks[vertex] - ranks[vertex])
		}
		
		ranks = newRanks
		
		if diff < tolerance {
			log.Printf("📊 PageRank converged after %d iterations", iter+1)
			break
		}
	}
	
	return ranks
}

// Helper functions

func randomVector(dim int) []float64 {
	vec := make([]float64, dim)
	for i := range vec {
		vec[i] = rand.Float64()*2 - 1 // Random values between -1 and 1
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
	
	// Priority increases with error count but with diminishing returns
	return base * (1.0 + math.Log(1.0+count)/10.0)
}

func (esa *EnhancedSemanticAnalyzer) estimateEffort(errors []NPMError, category string) time.Duration {
	categoryEffort := map[string]time.Duration{
		"typescript": 15 * time.Minute,
		"import":     5 * time.Minute,
		"syntax":     10 * time.Minute,
		"service":    30 * time.Minute,
		"build":      20 * time.Minute,
		"network":    45 * time.Minute,
		"database":   60 * time.Minute,
	}
	
	base := categoryEffort[category]
	if base == 0 {
		base = 20 * time.Minute // Default
	}
	
	// Scale by error count
	scale := 1.0 + float64(len(errors))*0.2
	return time.Duration(float64(base) * scale)
}

func (esa *EnhancedSemanticAnalyzer) generateSuggestedFixes(errors []NPMError, category string) []string {
	fixes := []string{}
	
	switch category {
	case "typescript":
		fixes = append(fixes, "Add missing type declarations")
		fixes = append(fixes, "Fix import statements") 
		fixes = append(fixes, "Update tsconfig.json")
	case "service":
		fixes = append(fixes, "Check service connectivity")
		fixes = append(fixes, "Verify service configuration")
		fixes = append(fixes, "Restart affected services")
	case "build":
		fixes = append(fixes, "Clear build cache")
		fixes = append(fixes, "Update dependencies")
		fixes = append(fixes, "Fix build configuration")
	default:
		fixes = append(fixes, "Review error messages")
		fixes = append(fixes, "Check documentation")
		fixes = append(fixes, "Apply standard fixes")
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
	// Base confidence based on category
	categoryConfidence := map[string]float64{
		"typescript": 0.9,
		"syntax":     0.95,
		"import":     0.85,
		"service":    0.7,
		"build":      0.8,
		"network":    0.6,
		"database":   0.65,
	}
	
	base := categoryConfidence[category]
	if base == 0 {
		base = 0.6 // Default
	}
	
	// Confidence increases with more errors (better pattern recognition)
	countBonus := math.Min(0.2, float64(errorCount)*0.05)
	
	return math.Min(1.0, base+countBonus)
}

func (esa *EnhancedSemanticAnalyzer) generateTags(errors []NPMError, category string) []string {
	tags := []string{category}
	
	// Add severity tags
	severities := make(map[string]bool)
	for _, err := range errors {
		severities[err.Severity] = true
	}
	
	for severity := range severities {
		tags = append(tags, severity)
	}
	
	// Add size tag
	if len(errors) > 5 {
		tags = append(tags, "bulk")
	}
	
	return tags
}

func (esa *EnhancedSemanticAnalyzer) calculateEdgeWeight(todo1, todo2 IntelligentTodo) float64 {
	// Calculate weight based on file overlap, category similarity, etc.
	weight := 0.0
	
	// Category similarity
	if todo1.Category == todo2.Category {
		weight += 0.5
	}
	
	// File overlap
	files1 := make(map[string]bool)
	for _, err := range todo1.RelatedErrors {
		files1[err.File] = true
	}
	
	overlap := 0
	total := len(files1)
	for _, err := range todo2.RelatedErrors {
		if files1[err.File] {
			overlap++
		} else {
			total++
		}
	}
	
	if total > 0 {
		weight += float64(overlap) / float64(total)
	}
	
	return weight
}

func (esa *EnhancedSemanticAnalyzer) calculateSemanticSimilarity(todo1, todo2 IntelligentTodo) float64 {
	// Simple semantic similarity based on shared keywords
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
	
	return float64(intersection) / float64(union) // Jaccard similarity
}

// Main execution function
func main() {
	log.Println("🚀 Enhanced SOM-based Intelligent Error Analyzer")
	
	// Configuration
	config := AnalyzerConfig{
		SOMGridSize:      10,
		FeatureDimension: 128,
		LearningRate:     0.5,
		TrainingEpochs:   500,
		PageRankDamping:  0.85,
		CacheTimeout:     5 * time.Minute,
	}
	
	// Initialize analyzer
	analyzer, err := NewEnhancedSemanticAnalyzer(config)
	if err != nil {
		log.Fatalf("Failed to initialize analyzer: %v", err)
	}
	defer analyzer.Cache.Close()
	
	// Sample npm errors for testing
	sampleErrors := []NPMError{
		{
			Message:  "Cannot find module '@types/node'",
			File:     "src/app.ts",
			Line:     1,
			Severity: "high",
			Category: "typescript",
			Type:     "error",
			Timestamp: time.Now(),
			Context:  []string{"import { Server } from 'http'"},
		},
		{
			Message:  "Property 'foo' does not exist on type 'Object'",
			File:     "src/utils.ts", 
			Line:     15,
			Severity: "medium",
			Category: "typescript",
			Type:     "error",
			Timestamp: time.Now(),
			Context:  []string{"const result = obj.foo"},
		},
		{
			Message:  "Service unavailable: http://localhost:8080",
			File:     "src/api.ts",
			Line:     25,
			Severity: "critical",
			Category: "service",
			Type:     "error",
			Timestamp: time.Now(),
			Dependencies: []string{"service-config.json"},
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
		log.Printf("   Tags: %s\n", strings.Join(todo.Tags, ", "))
	}
	
	// Output JSON for integration
	output, _ := json.MarshalIndent(todos, "", "  ")
	log.Printf("📄 JSON Output:\n%s", output)
	
	log.Println("✅ Analysis complete!")
}