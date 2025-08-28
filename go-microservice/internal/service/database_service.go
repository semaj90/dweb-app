package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// DocumentType represents the type of legal document
type DocumentType int32

const (
	DocumentTypeContract DocumentType = iota
	DocumentTypeEvidence
	DocumentTypeBrief
	DocumentTypeCitation
	DocumentTypeReport
)

// DatabaseConfig holds configuration for database service
type DatabaseConfig struct {
	PgPool *pgxpool.Pool
	Logger *log.Logger
}

// VectorOperationRecord represents a vector operation record
type VectorOperationRecord struct {
	RequestID         string
	UserID            string
	Operation         string
	InputDimensions   int
	OutputDimensions  int
	ProcessingTimeMs  int64
	Success           bool
}

// LegalDocumentProcessingRequest represents a legal document processing request
type LegalDocumentProcessingRequest struct {
	DocumentID   string
	DocumentType DocumentType
	Content      string
	Metadata     map[string]string
	UserID       string
}

// LegalDocumentProcessingResult represents the result of document processing
type LegalDocumentProcessingResult struct {
	DocumentID        string
	ExtractedEntities []string
	Summary           string
	ConfidenceScore   float32
}

// DatabaseService provides database operations for native Windows deployment
type DatabaseService struct {
	pgPool *pgxpool.Pool
	logger *log.Logger
}

// NewDatabaseService creates a new database service
func NewDatabaseService(config *DatabaseConfig) (*DatabaseService, error) {
	if config == nil {
		return nil, fmt.Errorf("config is nil")
	}
	if config.PgPool == nil {
		return nil, fmt.Errorf("pgPool is nil")
	}
	if config.Logger == nil {
		return nil, fmt.Errorf("logger is nil")
	}

	service := &DatabaseService{
		pgPool: config.PgPool,
		logger: config.Logger,
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := service.pingDB(ctx); err != nil {
		return nil, fmt.Errorf("database ping failed: %w", err)
	}

	service.logger.Printf("Database service initialized successfully; deployment_type=%s; database_type=%s", "native_windows", "postgresql_pgvector")

	return service, nil
}

// RecordVectorOperation records a vector operation in the database
func (d *DatabaseService) RecordVectorOperation(ctx context.Context, record *VectorOperationRecord) error {
	if record == nil {
		return fmt.Errorf("record is nil")
	}
	startTime := time.Now()

	query := `
		INSERT INTO vector_operations (
			request_id, user_id, operation, input_dimensions,
			output_dimensions, processing_time_ms, success, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`

	_, err := d.pgPool.Exec(ctx, query,
		record.RequestID,
		record.UserID,
		record.Operation,
		record.InputDimensions,
		record.OutputDimensions,
		record.ProcessingTimeMs,
		record.Success,
		time.Now(),
	)

	if err != nil {
		d.logger.Printf("Failed to record vector operation: %v; request_id=%s; operation=%s", err, record.RequestID, record.Operation)
		return fmt.Errorf("failed to record vector operation: %w", err)
	}

	d.logger.Printf("Vector operation recorded successfully; request_id=%s; operation=%s; db_operation_time=%s", record.RequestID, record.Operation, time.Since(startTime))
	return nil
}
// ProcessLegalDocument processes a legal document and optionally stores it
func (d *DatabaseService) ProcessLegalDocument(ctx context.Context, req *LegalDocumentProcessingRequest) (*LegalDocumentProcessingResult, error) {
	if d == nil {
		return nil, fmt.Errorf("database service is nil")
	}
	if req == nil {
		return nil, fmt.Errorf("request is nil")
	}

	// Ensure we have a logger to avoid nil dereference in logging paths
	if d.logger == nil {
		d.logger = log.New(io.Discard, "", 0)
	}

	// Ensure metadata map exists to avoid panics when accessing keys
	if req.Metadata == nil {
		req.Metadata = map[string]string{}
	}

	startTime := time.Now()

	d.logger.Printf("Processing legal document; document_id=%s; document_type=%s; content_size=%d; user_id=%s",
		req.DocumentID, d.documentTypeString(req.DocumentType), len(req.Content), req.UserID)

	// Simulate document processing
	extractedEntities := d.extractEntities(req.Content)
	summary := d.generateSummary(req.Content)
	confidenceScore := d.calculateConfidenceScore(req.Content, extractedEntities)

	result := &LegalDocumentProcessingResult{
		DocumentID:        req.DocumentID,
		ExtractedEntities: extractedEntities,
		Summary:           summary,
		ConfidenceScore:   confidenceScore,
	}

	d.logger.Printf("Legal document processed successfully; document_id=%s; entities_count=%d; confidence_score=%.3f; processing_time=%s",
		req.DocumentID, len(extractedEntities), confidenceScore, time.Since(startTime))

	// Try to store into database; log but do not fail the whole operation on storage error
	if err := d.storeDocument(ctx, req, extractedEntities, summary, confidenceScore); err != nil {
		d.logger.Printf("Warning: failed to store document: %v; document_id=%s", err, req.DocumentID)
	}

	return result, nil
}

// storeDocument stores the processed document in the database (best-effort, no transaction used)
func (d *DatabaseService) storeDocument(ctx context.Context, req *LegalDocumentProcessingRequest, entities []string, summary string, confidence float32) error {
	if d == nil {
		return fmt.Errorf("database service is nil")
	}

	// If there is no pgPool, skip storing (best-effort) and log a warning.
	if d.pgPool == nil {
		if d.logger != nil {
			d.logger.Printf("pgPool is nil; skipping document store; document_id=%s", req.DocumentID)
		} else {
			log.Printf("pgPool is nil; skipping document store; document_id=%s", req.DocumentID)
		}
		return nil
	}

	if req == nil {
		return fmt.Errorf("request is nil")
	}

	documentQuery := `
		INSERT INTO documents (
			id, title, content, document_type, metadata,
			summary, confidence_score, user_id, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		ON CONFLICT (id) DO UPDATE SET
			content = EXCLUDED.content,
			summary = EXCLUDED.summary,
			confidence_score = EXCLUDED.confidence_score,
			updated_at = NOW()
	`

	// Marshal metadata to JSON so it can be stored in a JSON/JSONB column
	metaJSON, err := json.Marshal(req.Metadata)
	if err != nil {
		if d.logger != nil {
			d.logger.Printf("Failed to marshal metadata: %v; document_id=%s", err, req.DocumentID)
		} else {
			log.Printf("Failed to marshal metadata: %v; document_id=%s", err, req.DocumentID)
		}
		// fallback to empty object
		metaJSON = []byte("{}")
	}

	_, err = d.pgPool.Exec(ctx, documentQuery,
		req.DocumentID,
		req.Metadata["title"],
		req.Content,
		d.documentTypeString(req.DocumentType),
		metaJSON,
		summary,
		confidence,
		req.UserID,
		time.Now(),
	)
	if err != nil {
		return fmt.Errorf("failed to insert/update document: %w", err)
	}

	// Insert entities (best-effort)
	entityQuery := `
		INSERT INTO document_entities (
			document_id, entity_type, entity_name, confidence_score, created_at
		) VALUES ($1, $2, $3, $4, $5)
	`
	for _, entity := range entities {
		_, err := d.pgPool.Exec(ctx, entityQuery,
			req.DocumentID,
			"extracted",
			entity,
			confidence,
			time.Now(),
		)
		if err != nil {
			// Log and continue with other entities
			if d.logger != nil {
				d.logger.Printf("Failed to insert entity: %v; entity=%s; document_id=%s", err, entity, req.DocumentID)
			} else {
				log.Printf("Failed to insert entity: %v; entity=%s; document_id=%s", err, entity, req.DocumentID)
			}
		}
	}

	return nil
}

// extractEntities extracts legal entities from document content
func (d *DatabaseService) extractEntities(content string) []string {
	// Simulate entity extraction using rule-based approach
	entities := []string{}
	if content == "" {
		return entities
	}

	contentLower := strings.ToLower(content)

	// Look for common legal terms
	legalTerms := []string{
		"plaintiff", "defendant", "contract", "agreement", "clause", "party", "witness", "judgment",
	}

	for _, term := range legalTerms {
		if strings.Contains(contentLower, term) {
			entities = append(entities, term)
		}
	}

	// Simulate more sophisticated NLP extraction
	if len(content) > 1000 {
		entities = append(entities, "complex_document")
	}
	if strings.Contains(contentLower, "whereas") {
		entities = append(entities, "formal_agreement")
	}
	if strings.Contains(contentLower, "hereby") {
		entities = append(entities, "legal_declaration")
	}

	return entities
}

// generateSummary generates a summary of the document
func (d *DatabaseService) generateSummary(content string) string {
	// Simulate summary generation
	if len(content) == 0 {
		return "Empty document"
	}

	if len(content) < 100 {
		return "Brief document with minimal content"
	}

	if len(content) < 500 {
		return "Short document containing legal text and references"
	}

	if len(content) < 2000 {
		return "Medium-length legal document with detailed provisions and clauses"
	}

	return "Comprehensive legal document containing extensive provisions, clauses, and legal references requiring detailed analysis"
}

// calculateConfidenceScore calculates confidence score for processing
func (d *DatabaseService) calculateConfidenceScore(content string, entities []string) float32 {
	baseScore := float32(0.5)

	// Increase confidence based on content length
	if len(content) > 100 {
		baseScore += 0.1
	}
	if len(content) > 500 {
		baseScore += 0.1
	}
	if len(content) > 1000 {
		baseScore += 0.1
	}

	// Increase confidence based on extracted entities
	entityBonus := float32(len(entities)) * 0.02
	if entityBonus > 0.2 {
		entityBonus = 0.2
	}
	baseScore += entityBonus

	// Cap at 0.95
	if baseScore > 0.95 {
		baseScore = 0.95
	}

	return baseScore
}

// documentTypeString converts DocumentType to string
func (d *DatabaseService) documentTypeString(dt DocumentType) string {
	switch dt {
	case DocumentTypeContract:
		return "contract"
	case DocumentTypeEvidence:
		return "evidence"
	case DocumentTypeBrief:
		return "brief"
	case DocumentTypeCitation:
		return "citation"
	case DocumentTypeReport:
		return "report"
	default:
		return "unknown"
	}
}

// pingDB acquires a connection from the pool and runs a simple query to verify connectivity.
func (d *DatabaseService) pingDB(ctx context.Context) error {
	if d == nil || d.pgPool == nil {
		return fmt.Errorf("pgPool is nil")
	}

	conn, err := d.pgPool.Acquire(ctx)
	if err != nil {
		return fmt.Errorf("failed to acquire connection: %w", err)
	}
	defer conn.Release()

	var one int
	if err := conn.QueryRow(ctx, "SELECT 1").Scan(&one); err != nil {
		return fmt.Errorf("ping query failed: %w", err)
	}

	return nil
}

// GetHealthStatus returns database health status
func (d *DatabaseService) GetHealthStatus(ctx context.Context) map[string]interface{} {
	healthStatus := map[string]interface{}{
		"status":          "healthy",
		"deployment_type": "native_windows",
	}

	if d == nil {
		healthStatus["status"] = "unhealthy"
		healthStatus["error"] = "database service is nil"
		return healthStatus
	}

	if d.pgPool == nil {
		healthStatus["status"] = "unhealthy"
		healthStatus["error"] = "pgPool is nil"
		return healthStatus
	}

	// Test database connection
	start := time.Now()
	err := d.pingDB(ctx)
	pingTime := time.Since(start)

	if err != nil {
		healthStatus["status"] = "unhealthy"
		healthStatus["error"] = err.Error()
	}

	healthStatus["ping_time"] = pingTime.String()
	healthStatus["connection_pool_size"] = d.pgPool.Stat().TotalConns
	healthStatus["active_connections"] = d.pgPool.Stat().AcquiredConns

	return healthStatus
}