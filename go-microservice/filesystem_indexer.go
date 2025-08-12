//go:build legacy
// +build legacy

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/pgvector/pgvector-go"
	"github.com/jackc/pgx/v5/pgxpool"
)

// FileIndexer handles filesystem indexing and analysis
type FileIndexer struct {
	mu            sync.RWMutex
	index         map[string]*IndexEntry
	neo4jDriver   neo4j.DriverWithContext
	pgPool        *pgxpool.Pool
	rootPath      string
	filePatterns  []string
	excludePaths  []string
	totalFiles    int
	totalImports  int
	totalExports  int
	typeRegistry  map[string]*TypeDefinition
}

// IndexEntry represents a parsed file with its dependencies
type IndexEntry struct {
	FilePath       string                 `json:"filePath"`
	RelativePath   string                 `json:"relativePath"`
	FileType       string                 `json:"fileType"`
	Size           int64                  `json:"size"`
	ModTime        time.Time              `json:"modTime"`
	Hash           string                 `json:"hash"`
	Exports        []ExportInfo           `json:"exports"`
	Imports        []ImportInfo           `json:"imports"`
	TypeDefs       []TypeDefinition       `json:"typeDefs"`
	Components     []ComponentInfo        `json:"components,omitempty"`
	Functions      []FunctionInfo         `json:"functions"`
	Dependencies   []string               `json:"dependencies"`
	Errors         []ErrorInfo            `json:"errors,omitempty"`
	Embedding      []float32              `json:"-"`
	Summary        string                 `json:"summary"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// ExportInfo describes an exported symbol
type ExportInfo struct {
	Name        string   `json:"name"`
	Type        string   `json:"type"` // "function", "class", "type", "const", "component"
	Line        int      `json:"line"`
	Column      int      `json:"column"`
	IsDefault   bool     `json:"isDefault"`
	Description string   `json:"description,omitempty"`
	Params      []string `json:"params,omitempty"`
	ReturnType  string   `json:"returnType,omitempty"`
}

// ImportInfo describes an imported module
type ImportInfo struct {
	Source      string   `json:"source"`
	Specifiers  []string `json:"specifiers"`
	Line        int      `json:"line"`
	IsType      bool     `json:"isType"`
	IsNamespace bool     `json:"isNamespace"`
	Alias       string   `json:"alias,omitempty"`
}

// TypeDefinition represents a TypeScript type or interface
type TypeDefinition struct {
	Name       string                 `json:"name"`
	Kind       string                 `json:"kind"` // "interface", "type", "enum", "class"
	FilePath   string                 `json:"filePath"`
	Line       int                    `json:"line"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Extends    []string               `json:"extends,omitempty"`
	Implements []string               `json:"implements,omitempty"`
	Generic    []string               `json:"generic,omitempty"`
	Definition string                 `json:"definition"`
}

// ComponentInfo for Svelte/React components
type ComponentInfo struct {
	Name      string   `json:"name"`
	Props     []string `json:"props"`
	Events    []string `json:"events,omitempty"`
	Slots     []string `json:"slots,omitempty"`
	Stores    []string `json:"stores,omitempty"`
	FilePath  string   `json:"filePath"`
	Framework string   `json:"framework"` // "svelte", "react", "vue"
}

// FunctionInfo describes a function
type FunctionInfo struct {
	Name       string   `json:"name"`
	Async      bool     `json:"async"`
	Generator  bool     `json:"generator"`
	Params     []string `json:"params"`
	ReturnType string   `json:"returnType,omitempty"`
	Line       int      `json:"line"`
}

// ErrorInfo represents TypeScript/build errors
type ErrorInfo struct {
	Code     string `json:"code"`
	Message  string `json:"message"`
	Line     int    `json:"line"`
	Column   int    `json:"column"`
	Severity string `json:"severity"` // "error", "warning", "info"
	Source   string `json:"source"`   // "typescript", "eslint", "svelte"
}

// AnalysisResult contains the complete analysis
type AnalysisResult struct {
	TotalFiles      int                       `json:"totalFiles"`
	TotalErrors     int                       `json:"totalErrors"`
	FilesByType     map[string]int            `json:"filesByType"`
	ErrorPatterns   []ErrorPattern            `json:"errorPatterns"`
	DependencyGraph map[string][]string       `json:"dependencyGraph"`
	TypeRegistry    map[string]TypeDefinition `json:"typeRegistry"`
	Recommendations []Recommendation          `json:"recommendations"`
	TodoList        []TodoItem                `json:"todoList"`
	FixStrategy     FixStrategy               `json:"fixStrategy"`
}

// ErrorPattern identifies common error patterns
type ErrorPattern struct {
	Pattern     string   `json:"pattern"`
	Count       int      `json:"count"`
	Files       []string `json:"files"`
	Severity    string   `json:"severity"`
	Category    string   `json:"category"`
	Description string   `json:"description"`
}

// Recommendation for fixing issues
type Recommendation struct {
	Priority    int      `json:"priority"`
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Files       []string `json:"files"`
	Commands    []string `json:"commands,omitempty"`
	CodeChanges []string `json:"codeChanges,omitempty"`
}

// TodoItem for action items
type TodoItem struct {
	ID          string   `json:"id"`
	Task        string   `json:"task"`
	Priority    int      `json:"priority"`
	Category    string   `json:"category"`
	Files       []string `json:"files"`
	Completed   bool     `json:"completed"`
	Dependencies []string `json:"dependencies,omitempty"`
}

// FixStrategy provides strategic approach
type FixStrategy struct {
	Phases []Phase `json:"phases"`
}

// Phase of the fix strategy
type Phase struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Steps       []string   `json:"steps"`
	TodoItems   []TodoItem `json:"todoItems"`
}

// NewFileIndexer creates a new file indexer
func NewFileIndexer(rootPath string, neo4jDriver neo4j.DriverWithContext, pgPool *pgxpool.Pool) *FileIndexer {
	return &FileIndexer{
		index:        make(map[string]*IndexEntry),
		neo4jDriver:  neo4jDriver,
		pgPool:       pgPool,
		rootPath:     rootPath,
		filePatterns: []string{".ts", ".tsx", ".svelte", ".js", ".jsx", ".json"},
		excludePaths: []string{"node_modules", ".git", "dist", "build", ".svelte-kit"},
		typeRegistry: make(map[string]*TypeDefinition),
	}
}

// IndexFileSystem performs complete filesystem indexing
func (fi *FileIndexer) IndexFileSystem() (*AnalysisResult, error) {
	log.Printf("Starting filesystem indexing from: %s", fi.rootPath)
	startTime := time.Now()

	// Walk the filesystem and build index
	err := filepath.WalkDir(fi.rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // Skip files we can't read
		}

		// Check if should exclude
		for _, exclude := range fi.excludePaths {
			if strings.Contains(path, exclude) {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}

		if d.IsDir() {
			return nil
		}

		// Check file extension
		ext := filepath.Ext(path)
		shouldIndex := false
		for _, pattern := range fi.filePatterns {
			if ext == pattern {
				shouldIndex = true
				break
			}
		}

		if !shouldIndex {
			return nil
		}

		// Index the file
		entry, err := fi.indexFile(path)
		if err != nil {
			log.Printf("Error indexing %s: %v", path, err)
			return nil
		}

		fi.mu.Lock()
		fi.index[path] = entry
		fi.totalFiles++
		fi.mu.Unlock()

		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("filesystem walk failed: %v", err)
	}

	// Analyze the indexed data
	analysis := fi.analyzeIndex()
	
	// Store in databases
	fi.storeInNeo4j(analysis)
	fi.storeInPostgres(analysis)

	log.Printf("Indexing complete: %d files in %v", fi.totalFiles, time.Since(startTime))
	
	return analysis, nil
}

// indexFile parses and indexes a single file
func (fi *FileIndexer) indexFile(path string) (*IndexEntry, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}

	relPath, _ := filepath.Rel(fi.rootPath, path)
	
	entry := &IndexEntry{
		FilePath:     path,
		RelativePath: relPath,
		FileType:     filepath.Ext(path),
		Size:         info.Size(),
		ModTime:      info.ModTime(),
		Metadata:     make(map[string]interface{}),
	}

	// Parse based on file type
	switch filepath.Ext(path) {
	case ".ts", ".tsx":
		fi.parseTypeScriptFile(entry, content)
	case ".svelte":
		fi.parseSvelteFile(entry, content)
	case ".js", ".jsx":
		fi.parseJavaScriptFile(entry, content)
	case ".json":
		fi.parseJSONFile(entry, content)
	}

	// Get AI analysis if available
	if cudaManager != nil && cudaManager.initialized {
		// Use GPU-accelerated embedding
		entry.Embedding = fi.generateEmbedding(string(content))
	}

	// Get summary from Ollama
	if summary, err := fi.generateSummary(string(content)); err == nil {
		entry.Summary = summary
	}

	return entry, nil
}

// parseTypeScriptFile extracts TypeScript-specific information
func (fi *FileIndexer) parseTypeScriptFile(entry *IndexEntry, content []byte) {
	text := string(content)
	lines := strings.Split(text, "\n")

	for i, line := range lines {
		// Extract exports
		if strings.Contains(line, "export") {
			export := fi.extractExport(line, i+1)
			if export != nil {
				entry.Exports = append(entry.Exports, *export)
				fi.totalExports++
			}
		}

		// Extract imports
		if strings.Contains(line, "import") {
			imp := fi.extractImport(line, i+1)
			if imp != nil {
				entry.Imports = append(entry.Imports, *imp)
				fi.totalImports++
			}
		}

		// Extract type definitions
		if strings.Contains(line, "interface") || strings.Contains(line, "type ") {
			typeDef := fi.extractTypeDefinition(lines, i)
			if typeDef != nil {
				entry.TypeDefs = append(entry.TypeDefs, *typeDef)
				fi.typeRegistry[typeDef.Name] = typeDef
			}
		}

		// Extract functions
		if strings.Contains(line, "function") || strings.Contains(line, "=>") {
			fn := fi.extractFunction(line, i+1)
			if fn != nil {
				entry.Functions = append(entry.Functions, *fn)
			}
		}
	}
}

// parseSvelteFile extracts Svelte-specific information
func (fi *FileIndexer) parseSvelteFile(entry *IndexEntry, content []byte) {
	text := string(content)
	
	// Extract script content
	scriptStart := strings.Index(text, "<script")
	scriptEnd := strings.Index(text, "</script>")
	
	if scriptStart != -1 && scriptEnd != -1 {
		scriptContent := text[scriptStart:scriptEnd]
		fi.parseTypeScriptFile(entry, []byte(scriptContent))
	}

	// Extract component info
	component := &ComponentInfo{
		Name:      filepath.Base(entry.FilePath),
		Framework: "svelte",
		FilePath:  entry.FilePath,
	}

	// Extract props
	if strings.Contains(text, "$props()") || strings.Contains(text, "export let") {
		component.Props = fi.extractSvelteProps(text)
	}

	// Extract stores
	if strings.Contains(text, "$:") || strings.Contains(text, "writable") || strings.Contains(text, "readable") {
		component.Stores = fi.extractSvelteStores(text)
	}

	entry.Components = append(entry.Components, *component)
}

// analyzeIndex performs analysis on the indexed data
func (fi *FileIndexer) analyzeIndex() *AnalysisResult {
	fi.mu.RLock()
	defer fi.mu.RUnlock()

	result := &AnalysisResult{
		TotalFiles:      fi.totalFiles,
		FilesByType:     make(map[string]int),
		DependencyGraph: make(map[string][]string),
		TypeRegistry:    make(map[string]TypeDefinition),
		ErrorPatterns:   []ErrorPattern{},
		Recommendations: []Recommendation{},
		TodoList:        []TodoItem{},
	}

	// Count files by type
	for _, entry := range fi.index {
		result.FilesByType[entry.FileType]++
		
		// Build dependency graph
		for _, imp := range entry.Imports {
			result.DependencyGraph[entry.RelativePath] = append(
				result.DependencyGraph[entry.RelativePath],
				imp.Source,
			)
		}

		// Count errors
		result.TotalErrors += len(entry.Errors)
	}

	// Copy type registry
	for name, typeDef := range fi.typeRegistry {
		result.TypeRegistry[name] = *typeDef
	}

	// Identify error patterns
	result.ErrorPatterns = fi.identifyErrorPatterns()

	// Generate recommendations
	result.Recommendations = fi.generateRecommendations(result.ErrorPatterns)

	// Create fix strategy
	result.FixStrategy = fi.createFixStrategy(result)

	return result
}

// identifyErrorPatterns finds common error patterns
func (fi *FileIndexer) identifyErrorPatterns() []ErrorPattern {
	patterns := make(map[string]*ErrorPattern)

	for _, entry := range fi.index {
		for _, err := range entry.Errors {
			key := fmt.Sprintf("%s:%s", err.Code, err.Source)
			
			if pattern, exists := patterns[key]; exists {
				pattern.Count++
				pattern.Files = append(pattern.Files, entry.RelativePath)
			} else {
				patterns[key] = &ErrorPattern{
					Pattern:  key,
					Count:    1,
					Files:    []string{entry.RelativePath},
					Severity: err.Severity,
					Category: err.Source,
				}
			}
		}
	}

	// Convert to slice and sort by count
	result := make([]ErrorPattern, 0, len(patterns))
	for _, pattern := range patterns {
		result = append(result, *pattern)
	}

	return result
}

// generateRecommendations creates actionable recommendations
func (fi *FileIndexer) generateRecommendations(patterns []ErrorPattern) []Recommendation {
	recommendations := []Recommendation{}

	// Check for type mismatches
	typeMismatchCount := 0
	for _, pattern := range patterns {
		if strings.Contains(pattern.Pattern, "TS2322") || strings.Contains(pattern.Pattern, "type") {
			typeMismatchCount += pattern.Count
		}
	}

	if typeMismatchCount > 10 {
		recommendations = append(recommendations, Recommendation{
			Priority:    1,
			Title:       "Fix Systemic Type Inconsistencies",
			Description: "Multiple type mismatches detected across the codebase. Create unified type definitions.",
			Commands: []string{
				"npm run type-check",
				"npx tsc --noEmit",
			},
		})
	}

	// Check for missing exports
	missingExports := 0
	for _, pattern := range patterns {
		if strings.Contains(pattern.Pattern, "TS2305") {
			missingExports += pattern.Count
		}
	}

	if missingExports > 5 {
		recommendations = append(recommendations, Recommendation{
			Priority:    2,
			Title:       "Fix Missing Exports",
			Description: "Multiple modules have missing exports. Review and add proper export statements.",
		})
	}

	return recommendations
}

// createFixStrategy generates a phased approach to fixing issues
func (fi *FileIndexer) createFixStrategy(analysis *AnalysisResult) FixStrategy {
	strategy := FixStrategy{
		Phases: []Phase{},
	}

	// Phase 1: Critical type fixes
	if analysis.TotalErrors > 50 {
		phase1 := Phase{
			Name:        "Critical Type Fixes",
			Description: "Fix type mismatches and missing type definitions",
			Steps: []string{
				"Create unified types file (src/lib/types/unified.ts)",
				"Fix XState machine type definitions",
				"Resolve schema type conflicts",
			},
			TodoItems: []TodoItem{
				{
					ID:       "type-1",
					Task:     "Create unified type definitions file",
					Priority: 1,
					Category: "types",
				},
				{
					ID:       "type-2",
					Task:     "Fix XState context and event types",
					Priority: 1,
					Category: "types",
				},
			},
		}
		strategy.Phases = append(strategy.Phases, phase1)
	}

	// Phase 2: Module resolution
	phase2 := Phase{
		Name:        "Module Resolution",
		Description: "Fix import/export issues and module boundaries",
		Steps: []string{
			"Fix missing exports",
			"Update import paths",
			"Create barrel exports",
		},
	}
	strategy.Phases = append(strategy.Phases, phase2)

	// Phase 3: Schema alignment
	phase3 := Phase{
		Name:        "Schema Alignment",
		Description: "Align database schemas and API contracts",
		Steps: []string{
			"Sync Drizzle schema with database",
			"Update Zod schemas",
			"Fix form validation schemas",
		},
	}
	strategy.Phases = append(strategy.Phases, phase3)

	return strategy
}

// storeInNeo4j stores the analysis in Neo4j
func (fi *FileIndexer) storeInNeo4j(analysis *AnalysisResult) error {
	ctx := context.Background()
	session := fi.neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// Store file nodes and relationships
	for path, entry := range fi.index {
		_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			// Create file node
			_, err := tx.Run(ctx,
				`MERGE (f:File {path: $path})
				 SET f.type = $type,
				     f.size = $size,
				     f.lastModified = $modTime,
				     f.exportCount = $exports,
				     f.importCount = $imports,
				     f.errorCount = $errors,
				     f.summary = $summary,
				     f.indexed = datetime()`,
				map[string]any{
					"path":     path,
					"type":     entry.FileType,
					"size":     entry.Size,
					"modTime":  entry.ModTime,
					"exports":  len(entry.Exports),
					"imports":  len(entry.Imports),
					"errors":   len(entry.Errors),
					"summary":  entry.Summary,
				})
			if err != nil {
				return nil, err
			}

			// Create import relationships
			for _, imp := range entry.Imports {
				_, err = tx.Run(ctx,
					`MATCH (f:File {path: $from})
					 MERGE (m:Module {name: $module})
					 MERGE (f)-[:IMPORTS]->(m)`,
					map[string]any{
						"from":   path,
						"module": imp.Source,
					})
				if err != nil {
					return nil, err
				}
			}

			// Create type nodes
			for _, typeDef := range entry.TypeDefs {
				_, err = tx.Run(ctx,
					`MATCH (f:File {path: $file})
					 MERGE (t:Type {name: $name})
					 SET t.kind = $kind,
					     t.definition = $def
					 MERGE (f)-[:DEFINES]->(t)`,
					map[string]any{
						"file": path,
						"name": typeDef.Name,
						"kind": typeDef.Kind,
						"def":  typeDef.Definition,
					})
				if err != nil {
					return nil, err
				}
			}

			return nil, nil
		})

		if err != nil {
			log.Printf("Error storing %s in Neo4j: %v", path, err)
		}
	}

	// Store analysis summary
	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		analysisJSON, _ := json.Marshal(analysis)
		_, err := tx.Run(ctx,
			`CREATE (a:Analysis {
				timestamp: datetime(),
				totalFiles: $totalFiles,
				totalErrors: $totalErrors,
				data: $data
			})`,
			map[string]any{
				"totalFiles":  analysis.TotalFiles,
				"totalErrors": analysis.TotalErrors,
				"data":        string(analysisJSON),
			})
		return nil, err
	})

	return err
}

// storeInPostgres stores the analysis in PostgreSQL with pgvector
func (fi *FileIndexer) storeInPostgres(analysis *AnalysisResult) error {
	ctx := context.Background()

	// Store each file entry
	for path, entry := range fi.index {
		embeddingStr := pgvector.NewVector(entry.Embedding).String()
		
		_, err := fi.pgPool.Exec(ctx,
			`INSERT INTO indexed_files (
				file_path, relative_path, file_type, content_hash,
				size, modified_at, exports, imports, errors,
				embedding, summary, metadata
			) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
			ON CONFLICT (file_path) DO UPDATE SET
				modified_at = EXCLUDED.modified_at,
				exports = EXCLUDED.exports,
				imports = EXCLUDED.imports,
				errors = EXCLUDED.errors,
				embedding = EXCLUDED.embedding,
				summary = EXCLUDED.summary,
				metadata = EXCLUDED.metadata,
				updated_at = NOW()`,
			path, entry.RelativePath, entry.FileType, entry.Hash,
			entry.Size, entry.ModTime, len(entry.Exports), len(entry.Imports),
			len(entry.Errors), embeddingStr, entry.Summary, entry.Metadata,
		)

		if err != nil {
			log.Printf("Error storing %s in PostgreSQL: %v", path, err)
		}
	}

	// Store analysis results
	analysisJSON, _ := json.Marshal(analysis)
	_, err := fi.pgPool.Exec(ctx,
		`INSERT INTO analysis_results (
			total_files, total_errors, analysis_data, created_at
		) VALUES ($1, $2, $3, NOW())`,
		analysis.TotalFiles, analysis.TotalErrors, analysisJSON,
	)

	return err
}

// Helper methods

func (fi *FileIndexer) extractExport(line string, lineNum int) *ExportInfo {
	// Simple extraction - in production use proper AST parsing
	export := &ExportInfo{Line: lineNum}
	
	if strings.Contains(line, "export default") {
		export.IsDefault = true
		export.Type = "default"
	} else if strings.Contains(line, "export function") {
		export.Type = "function"
		// Extract function name
		parts := strings.Split(line, "function")
		if len(parts) > 1 {
			name := strings.TrimSpace(strings.Split(parts[1], "(")[0])
			export.Name = name
		}
	} else if strings.Contains(line, "export const") {
		export.Type = "const"
		// Extract const name
		parts := strings.Split(line, "const")
		if len(parts) > 1 {
			name := strings.TrimSpace(strings.Split(parts[1], "=")[0])
			export.Name = name
		}
	}

	if export.Name != "" || export.IsDefault {
		return export
	}
	return nil
}

func (fi *FileIndexer) extractImport(line string, lineNum int) *ImportInfo {
	imp := &ImportInfo{Line: lineNum}
	
	// Extract source
	if strings.Contains(line, "from") {
		parts := strings.Split(line, "from")
		if len(parts) > 1 {
			source := strings.Trim(parts[1], " '\";")
			imp.Source = source
		}
	}

	// Check if type import
	if strings.Contains(line, "import type") {
		imp.IsType = true
	}

	if imp.Source != "" {
		return imp
	}
	return nil
}

func (fi *FileIndexer) extractTypeDefinition(lines []string, startLine int) *TypeDefinition {
	if startLine >= len(lines) {
		return nil
	}

	line := lines[startLine]
	typeDef := &TypeDefinition{Line: startLine + 1}

	if strings.Contains(line, "interface") {
		typeDef.Kind = "interface"
		parts := strings.Split(line, "interface")
		if len(parts) > 1 {
			name := strings.TrimSpace(strings.Split(parts[1], " ")[0])
			name = strings.TrimSuffix(name, "{")
			typeDef.Name = name
		}
	} else if strings.Contains(line, "type ") {
		typeDef.Kind = "type"
		parts := strings.Split(line, "type ")
		if len(parts) > 1 {
			name := strings.TrimSpace(strings.Split(parts[1], "=")[0])
			typeDef.Name = name
		}
	}

	// Extract full definition (simplified)
	definition := line
	if strings.Contains(line, "{") && !strings.Contains(line, "}") {
		// Multi-line definition
		for i := startLine + 1; i < len(lines) && i < startLine+50; i++ {
			definition += "\n" + lines[i]
			if strings.Contains(lines[i], "}") {
				break
			}
		}
	}
	typeDef.Definition = definition

	if typeDef.Name != "" {
		return typeDef
	}
	return nil
}

func (fi *FileIndexer) extractFunction(line string, lineNum int) *FunctionInfo {
	fn := &FunctionInfo{Line: lineNum}

	if strings.Contains(line, "async") {
		fn.Async = true
	}
	if strings.Contains(line, "function*") {
		fn.Generator = true
	}

	// Extract function name (simplified)
	if strings.Contains(line, "function") {
		parts := strings.Split(line, "function")
		if len(parts) > 1 {
			namePart := strings.TrimSpace(parts[1])
			if idx := strings.Index(namePart, "("); idx > 0 {
				fn.Name = strings.TrimSpace(namePart[:idx])
			}
		}
	}

	if fn.Name != "" {
		return fn
	}
	return nil
}

func (fi *FileIndexer) extractSvelteProps(content string) []string {
	props := []string{}
	lines := strings.Split(content, "\n")
	
	for _, line := range lines {
		if strings.Contains(line, "export let") {
			parts := strings.Split(line, "export let")
			if len(parts) > 1 {
				propName := strings.TrimSpace(strings.Split(parts[1], "=")[0])
				propName = strings.TrimSuffix(propName, ";")
				propName = strings.TrimSuffix(propName, ":")
				props = append(props, propName)
			}
		}
	}
	
	return props
}

func (fi *FileIndexer) extractSvelteStores(content string) []string {
	stores := []string{}
	// Simplified extraction
	if strings.Contains(content, "writable(") {
		stores = append(stores, "writable")
	}
	if strings.Contains(content, "readable(") {
		stores = append(stores, "readable")
	}
	if strings.Contains(content, "derived(") {
		stores = append(stores, "derived")
	}
	return stores
}

func (fi *FileIndexer) generateEmbedding(content string) []float32 {
	// Use Ollama or GPU-accelerated embedding
	// This is a placeholder - integrate with your embedding service
	embedding := make([]float32, 384)
	// Fill with mock data for now
	for i := range embedding {
		embedding[i] = float32(i) / 384.0
	}
	return embedding
}

func (fi *FileIndexer) generateSummary(content string) (string, error) {
	// Use Ollama to generate summary
	// This is a placeholder - integrate with your LLM service
	if len(content) > 200 {
		return content[:200] + "...", nil
	}
	return content, nil
}

// HTTP Handlers

func handleIndexRequest(c *gin.Context) {
	var req struct {
		RootPath string   `json:"rootPath"`
		Patterns []string `json:"patterns,omitempty"`
		Exclude  []string `json:"exclude,omitempty"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if req.RootPath == "" {
		req.RootPath = "./sveltekit-frontend"
	}

	indexer := NewFileIndexer(req.RootPath, neo4jDriver, nil)
	
	if len(req.Patterns) > 0 {
		indexer.filePatterns = req.Patterns
	}
	if len(req.Exclude) > 0 {
		indexer.excludePaths = req.Exclude
	}

	// Run indexing in background
	go func() {
		analysis, err := indexer.IndexFileSystem()
		if err != nil {
			log.Printf("Indexing failed: %v", err)
			return
		}

		// Save analysis to file
		outputPath := filepath.Join("./analysis", fmt.Sprintf("analysis_%d.json", time.Now().Unix()))
		os.MkdirAll("./analysis", 0755)
		
		data, _ := json.MarshalIndent(analysis, "", "  ")
		os.WriteFile(outputPath, data, 0644)
		
		log.Printf("Analysis saved to: %s", outputPath)
	}()

	c.JSON(202, gin.H{
		"status": "indexing_started",
		"path":   req.RootPath,
	})
}

func handleAnalyzeErrors(c *gin.Context) {
	var req struct {
		Errors []string `json:"errors"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	// Analyze error patterns
	errorMap := make(map[string]int)
	for _, err := range req.Errors {
		// Extract error code
		if idx := strings.Index(err, "TS"); idx != -1 && idx+6 < len(err) {
			code := err[idx : idx+6]
			errorMap[code]++
		}
	}

	// Generate recommendations based on error patterns
	recommendations := []string{}
	
	if errorMap["TS2322"] > 10 {
		recommendations = append(recommendations, 
			"Fix type mismatches: Create unified type definitions file")
	}
	if errorMap["TS2305"] > 5 {
		recommendations = append(recommendations,
			"Fix missing exports: Review module boundaries")
	}
	if errorMap["TS2339"] > 5 {
		recommendations = append(recommendations,
			"Fix property access: Update type definitions")
	}

	c.JSON(200, gin.H{
		"status":          "analysis_complete",
		"total_errors":    len(req.Errors),
		"error_breakdown": errorMap,
		"recommendations": recommendations,
		"strategy": gin.H{
			"phase1": "Fix critical type errors (XState, schemas)",
			"phase2": "Resolve module exports/imports",
			"phase3": "Update form validation (Zod/Superforms)",
			"phase4": "Clean up remaining warnings",
		},
	})
}

func handleQueryIndex(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(400, gin.H{"error": "query parameter required"})
		return
	}

	// Search in Neo4j
	ctx := context.Background()
	session := neo4jDriver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.Run(ctx,
		`MATCH (f:File)
		 WHERE f.path CONTAINS $query OR f.summary CONTAINS $query
		 RETURN f.path as path, f.type as type, f.summary as summary
		 LIMIT 20`,
		map[string]any{"query": query})

	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	files := []gin.H{}
	for result.Next(ctx) {
		record := result.Record()
		files = append(files, gin.H{
			"path":    record.Values[0],
			"type":    record.Values[1],
			"summary": record.Values[2],
		})
	}

	c.JSON(200, gin.H{
		"query":   query,
		"results": files,
		"count":   len(files),
	})
}
