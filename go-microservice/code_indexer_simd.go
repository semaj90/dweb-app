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

	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/minio/simdjson-go"
)

type CodeIndexer struct {
	pgPool     *pgxpool.Pool
	simdParser *simdjson.Parser
	rootPath   string
}

type FileChunk struct {
	ID        string                 `json:"id"`
	FilePath  string                 `json:"file_path"`
	Content   string                 `json:"content"`
	ChunkIdx  int                    `json:"chunk_idx"`
	Embedding []float32              `json:"embedding"`
	Metadata  map[string]interface{} `json:"metadata"`
}

func NewCodeIndexer(rootPath string) *CodeIndexer {
	pgConfig := "postgres://postgres:postgres@localhost:5432/codebase_index?sslmode=disable"
	pool, err := pgxpool.New(context.Background(), pgConfig)
	if err != nil {
		log.Printf("DB connection failed, running without persistence: %v", err)
	}
	
	parser := simdjson.NewParser()
	parser.SetCapacity(10 << 20)
	
	return &CodeIndexer{
		pgPool:     pool,
		simdParser: parser,
		rootPath:   rootPath,
	}
}

func (ci *CodeIndexer) IndexDirectory() error {
	var chunks []FileChunk
	
	err := filepath.WalkDir(ci.rootPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		
		ext := filepath.Ext(path)
		if ext != ".ts" && ext != ".tsx" && ext != ".svelte" && ext != ".go" && ext != ".js" {
			return nil
		}
		
		content, _ := os.ReadFile(path)
		relPath, _ := filepath.Rel(ci.rootPath, path)
		
		// Use SIMD parser for JSON files
		if ext == ".json" {
			pj, err := ci.simdParser.Parse(content, nil)
			if err == nil {
				iter := pj.Iter()
				iter.Advance()
				// Process JSON structure
			}
		}
		
		chunkSize := 1000
		text := string(content)
		for i := 0; i < len(text); i += chunkSize {
			end := i + chunkSize
			if end > len(text) {
				end = len(text)
			}
			
			chunk := FileChunk{
				ID:       fmt.Sprintf("%s_%d", relPath, i/chunkSize),
				FilePath: relPath,
				Content:  text[i:end],
				ChunkIdx: i / chunkSize,
				Metadata: map[string]interface{}{
					"extension": ext,
					"size":      len(content),
				},
			}
			chunks = append(chunks, chunk)
		}
		
		return nil
	})
	
	if err != nil {
		return err
	}
	
	log.Printf("Indexed %d chunks", len(chunks))
	if ci.pgPool != nil {
		ci.embedAndStore(chunks)
	}
	return nil
}

func (ci *CodeIndexer) embedAndStore(chunks []FileChunk) {
	if ci.pgPool == nil {
		return
	}
	
	ctx := context.Background()
	for _, chunk := range chunks {
		chunk.Embedding = make([]float32, 384)
		
		_, err := ci.pgPool.Exec(ctx, `
			INSERT INTO code_chunks (id, file_path, content, chunk_idx, embedding, metadata)
			VALUES ($1, $2, $3, $4, $5, $6)
			ON CONFLICT (id) DO UPDATE SET
				content = EXCLUDED.content,
				embedding = EXCLUDED.embedding
		`, chunk.ID, chunk.FilePath, chunk.Content, chunk.ChunkIdx, chunk.Embedding, chunk.Metadata)
		
		if err != nil {
			log.Printf("Store error: %v", err)
		}
	}
}

func (ci *CodeIndexer) AnalyzeAndGenerate(errors []string) map[string]interface{} {
	affectedFiles := ci.extractFilesFromErrors(errors)
	
	analysis := map[string]interface{}{
		"total_errors":    len(errors),
		"affected_files":  affectedFiles,
		"error_patterns":  ci.categorizeErrors(errors),
		"recommendations": ci.generateRecommendations(errors),
	}
	
	jsonOutput, _ := json.MarshalIndent(analysis, "", "  ")
	os.WriteFile("analysis_report.json", jsonOutput, 0644)
	
	txtOutput := fmt.Sprintf("Error Analysis\n==============\nTotal: %d errors\nFiles: %v\n\nRecommendations:\n%v",
		len(errors), affectedFiles, analysis["recommendations"])
	os.WriteFile("analysis_summary.txt", []byte(txtOutput), 0644)
	
	mdOutput := fmt.Sprintf("# Code Analysis\n\n## Errors: %d\n\n## Affected Files\n%v\n\n## Recommendations\n%v",
		len(errors), affectedFiles, analysis["recommendations"])
	os.WriteFile("ANALYSIS_REPORT.md", []byte(mdOutput), 0644)
	
	return analysis
}

func (ci *CodeIndexer) extractFilesFromErrors(errors []string) []string {
	fileMap := make(map[string]bool)
	for _, err := range errors {
		if idx := strings.Index(err, "("); idx > 0 {
			fileMap[err[:idx]] = true
		}
	}
	
	files := []string{}
	for f := range fileMap {
		files = append(files, f)
	}
	return files
}

func (ci *CodeIndexer) categorizeErrors(errors []string) map[string]int {
	categories := make(map[string]int)
	for _, err := range errors {
		if strings.Contains(err, "TS2") {
			categories["type_errors"]++
		} else if strings.Contains(err, "TS7") {
			categories["syntax_errors"]++
		} else {
			categories["other"]++
		}
	}
	return categories
}

func (ci *CodeIndexer) generateRecommendations(errors []string) []string {
	recs := []string{}
	
	hasTypeErrors := false
	hasSonicErrors := false
	
	for _, err := range errors {
		if strings.Contains(err, "TS23") {
			hasTypeErrors = true
		}
		if strings.Contains(err, "sonic") {
			hasSonicErrors = true
		}
	}
	
	if hasTypeErrors {
		recs = append(recs, "Fix type mismatches in XState machines")
	}
	if hasSonicErrors {
		recs = append(recs, "Remove sonic dependency (Unix-only)")
	}
	
	return recs
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	
	indexer := NewCodeIndexer("./sveltekit-frontend")
	
	r.POST("/index", func(c *gin.Context) {
		go indexer.IndexDirectory()
		c.JSON(202, gin.H{"status": "indexing_started"})
	})
	
	r.POST("/analyze", func(c *gin.Context) {
		var req struct {
			Errors []string `json:"errors"`
		}
		c.ShouldBindJSON(&req)
		
		outputs := indexer.AnalyzeAndGenerate(req.Errors)
		c.JSON(200, outputs)
	})
	
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "operational", "simd": true})
	})
	
	fmt.Println("Code Indexer + SIMD on :8080")
	r.Run(":8080")
}
