package service

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"legal-ai-production/internal/observability"
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
	Logger *observability.ELKLogger
}

// VectorOperationRecord represents a vector operation record
type VectorOperationRecord struct {
	RequestID        string
	UserID          string
	Operation       string
	InputDimensions int
	OutputDimensions int
	ProcessingTimeMs int64
	Success         bool
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
	Summary          string
	ConfidenceScore  float32
}

// DatabaseService provides database operations for native Windows deployment
type DatabaseService struct {
	pgPool *pgxpool.Pool
	logger *observability.ELKLogger
}

// NewDatabaseService creates a new database service
func NewDatabaseService(config *DatabaseConfig) (*DatabaseService, error) {
	service := &DatabaseService{
		pgPool: config.PgPool,
		logger: config.Logger,
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	if err := service.pgPool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("database ping failed: %w", err)
	}

	config.Logger.Info("Database service initialized successfully").
		WithString("deployment_type", "native_windows").
		WithString("database_type", "postgresql_pgvector").
		Log()

	return service, nil
}

// RecordVectorOperation records a vector operation in the database
func (d *DatabaseService) RecordVectorOperation(ctx context.Context, record *VectorOperationRecord) error {
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
		d.logger.Error("Failed to record vector operation").
			WithError(err).
			WithString("request_id", record.RequestID).
			WithString("operation", record.Operation).
			Log()
		return fmt.Errorf("failed to record vector operation: %w", err)
	}

	d.logger.Debug("Vector operation recorded successfully").
		WithString("request_id", record.RequestID).
		WithString("operation", record.Operation).
		WithDuration("db_operation_time", time.Since(startTime)).
		Log()

	return nil
}

// ProcessLegalDocument processes a legal document and stores the results
func (d *DatabaseService) ProcessLegalDocument(ctx context.Context, req *LegalDocumentProcessingRequest) (*LegalDocumentProcessingResult, error) {
	startTime := time.Now()

	d.logger.Info("Processing legal document").
		WithString("document_id", req.DocumentID).
		WithString("document_type", d.documentTypeString(req.DocumentType)).
		WithInt("content_size", len(req.Content)).
		WithString("user_id", req.UserID).
		Log()

	// Simulate document processing
	extractedEntities := d.extractEntities(req.Content)
	summary := d.generateSummary(req.Content)
	confidenceScore := d.calculateConfidenceScore(req.Content, extractedEntities)

	// Store document in database
	if err := d.storeDocument(ctx, req, extractedEntities, summary, confidenceScore); err != nil {
		return nil, fmt.Errorf("failed to store document: %w", err)
	}

	result := &LegalDocumentProcessingResult{
		DocumentID:        req.DocumentID,
		ExtractedEntities: extractedEntities,
		Summary:          summary,
		ConfidenceScore:  confidenceScore,
	}

	d.logger.Info("Legal document processed successfully").
		WithString("document_id", req.DocumentID).
		WithInt("entities_count", len(extractedEntities)).
		WithFloat32("confidence_score", confidenceScore).
		WithDuration("processing_time", time.Since(startTime)).
		Log()

	return result, nil
}

// storeDocument stores the processed document in the database
func (d *DatabaseService) storeDocument(ctx context.Context, req *LegalDocumentProcessingRequest, entities []string, summary string, confidence float32) error {
	tx, err := d.pgPool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	// Insert document
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

	_, err = tx.Exec(ctx, documentQuery,
		req.DocumentID,
		req.Metadata["title"],
		req.Content,
		d.documentTypeString(req.DocumentType),
		req.Metadata,
		summary,
		confidence,
		req.UserID,
		time.Now(),
	)

	if err != nil {
		return fmt.Errorf("failed to insert document: %w", err)
	}

	// Insert extracted entities
	for _, entity := range entities {
		entityQuery := `
			INSERT INTO legal_entities (
				document_id, entity_type, entity_name, confidence_score, created_at
			) VALUES ($1, $2, $3, $4, $5)
		`

		_, err = tx.Exec(ctx, entityQuery,
			req.DocumentID,
			"extracted",
			entity,
			confidence,
			time.Now(),
		)

		if err != nil {
			d.logger.Warning("Failed to insert entity").
				WithError(err).
				WithString("entity", entity).
				Log()
			// Continue with other entities
		}
	}

	if err = tx.Commit(ctx); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// extractEntities extracts legal entities from document content
func (d *DatabaseService) extractEntities(content string) []string {
	// Simulate entity extraction using rule-based approach
	entities := []string{}

	// Look for common legal terms
	legalTerms := []string{
		"plaintiff", "defendant", "contract", "agreement", 
		"liability", "damages", "breach", "violation",
		"jurisdiction", "court", "judge", "attorney",
		"evidence", "witness", "testimony", "precedent",
	}

	contentLower := strings.ToLower(content)
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

// GetHealthStatus returns database health status
func (d *DatabaseService) GetHealthStatus(ctx context.Context) map[string]interface{} {
	healthStatus := map[string]interface{}{
		"status": "healthy",
		"deployment_type": "native_windows",
	}

	// Test database connection
	start := time.Now()
	err := d.pgPool.Ping(ctx)
	pingTime := time.Since(start)

	if err != nil {
		healthStatus["status"] = "unhealthy"
		healthStatus["error"] = err.Error()
	}

	healthStatus["ping_time"] = pingTime.String()
	healthStatus["connection_pool_size"] = d.pgPool.Stat().TotalConns()
	healthStatus["active_connections"] = d.pgPool.Stat().AcquiredConns()

	return healthStatus
}