package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/lib/pq"
	_ "github.com/lib/pq"
	"github.com/gorilla/mux"
	"github.com/rs/cors"

	minioClient "github.com/deeds-web/deeds-web-app/go-microservice/pkg/minio"
)

type UploadService struct {
	minio *minioClient.Client
	db    *sql.DB
}

type DocumentMetadata struct {
	ID           string                 `json:"id"`
	CaseID       string                 `json:"caseId"`
	Filename     string                 `json:"filename"`
	ObjectName   string                 `json:"objectName"`
	ContentType  string                 `json:"contentType"`
	Size         int64                  `json:"size"`
	UploadTime   time.Time              `json:"uploadTime"`
	DocumentType string                 `json:"documentType"`
	Tags         map[string]string      `json:"tags"`
	Metadata     map[string]interface{} `json:"metadata"`
	ProcessingStatus string             `json:"processingStatus"`
	Embedding    []float32              `json:"embedding,omitempty"`
	ExtractedText string                `json:"extractedText,omitempty"`
}

type UploadResponse struct {
	Success      bool              `json:"success"`
	DocumentID   string            `json:"documentId"`
	URL          string            `json:"url"`
	ObjectName   string            `json:"objectName"`
	Message      string            `json:"message"`
	Metadata     *DocumentMetadata `json:"metadata,omitempty"`
}

func NewUploadService() (*UploadService, error) {
	// Initialize MinIO client
	endpoint := os.Getenv("MINIO_ENDPOINT")
	if endpoint == "" {
		endpoint = "localhost:9000"
	}
	
	accessKey := os.Getenv("MINIO_ACCESS_KEY")
	if accessKey == "" {
		accessKey = "minioadmin"
	}
	
	secretKey := os.Getenv("MINIO_SECRET_KEY")
	if secretKey == "" {
		secretKey = "minioadmin"
	}
	
	bucketName := os.Getenv("MINIO_BUCKET")
	if bucketName == "" {
		bucketName = "legal-documents"
	}
	
	useSSL := os.Getenv("MINIO_USE_SSL") == "true"

	minioClient, err := minioClient.NewClient(endpoint, accessKey, secretKey, bucketName, useSSL)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize MinIO client: %w", err)
	}

	// Initialize PostgreSQL connection
	var db *sql.DB
	if dbURL := os.Getenv("DATABASE_URL"); dbURL != "" {
		db, err = sql.Open("postgres", dbURL)
		if err != nil {
			log.Printf("Warning: Failed to connect to PostgreSQL: %v", err)
		} else {
			if err := db.Ping(); err != nil {
				log.Printf("Warning: PostgreSQL ping failed: %v", err)
				db = nil
			} else {
				log.Println("✅ PostgreSQL connected successfully")
				if err := initDatabase(db); err != nil {
					log.Printf("Warning: Database initialization failed: %v", err)
				}
			}
		}
	}

	return &UploadService{
		minio: minioClient,
		db:    db,
	}, nil
}

func initDatabase(db *sql.DB) error {
	schema := `
	CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
	CREATE EXTENSION IF NOT EXISTS vector;

	CREATE TABLE IF NOT EXISTS document_metadata (
		id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
		case_id VARCHAR(255) NOT NULL,
		filename VARCHAR(500) NOT NULL,
		object_name VARCHAR(1000) NOT NULL UNIQUE,
		content_type VARCHAR(100),
		size_bytes BIGINT,
		upload_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		document_type VARCHAR(100),
		tags JSONB,
		metadata JSONB,
		processing_status VARCHAR(50) DEFAULT 'uploaded',
		embedding vector(384), -- nomic-embed-text dimensions
		extracted_text TEXT,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	CREATE INDEX IF NOT EXISTS idx_document_case_id ON document_metadata(case_id);
	CREATE INDEX IF NOT EXISTS idx_document_type ON document_metadata(document_type);
	CREATE INDEX IF NOT EXISTS idx_document_status ON document_metadata(processing_status);
	CREATE INDEX IF NOT EXISTS idx_document_embedding ON document_metadata USING ivfflat (embedding vector_cosine_ops);
	CREATE INDEX IF NOT EXISTS idx_document_tags ON document_metadata USING gin(tags);
	CREATE INDEX IF NOT EXISTS idx_document_metadata ON document_metadata USING gin(metadata);
	`

	_, err := db.Exec(schema)
	return err
}

func (s *UploadService) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form
	err := r.ParseMultipartForm(100 << 20) // 100MB max
	if err != nil {
		http.Error(w, "Failed to parse form", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "No file provided", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Get form parameters
	caseID := r.FormValue("caseId")
	documentType := r.FormValue("documentType")
	if caseID == "" || documentType == "" {
		http.Error(w, "caseId and documentType are required", http.StatusBadRequest)
		return
	}

	// Parse tags and metadata
	tags := make(map[string]string)
	if tagsStr := r.FormValue("tags"); tagsStr != "" {
		json.Unmarshal([]byte(tagsStr), &tags)
	}

	metadata := make(map[string]string)
	if metadataStr := r.FormValue("metadata"); metadataStr != "" {
		json.Unmarshal([]byte(metadataStr), &metadata)
	}

	// Upload to MinIO
	uploadResult, err := s.minio.UploadFile(r.Context(), file, header, minioClient.UploadOptions{
		CaseID:       caseID,
		DocumentType: documentType,
		Tags:         tags,
		Metadata:     metadata,
	})
	if err != nil {
		log.Printf("MinIO upload failed: %v", err)
		http.Error(w, "Upload failed", http.StatusInternalServerError)
		return
	}

	// Save metadata to PostgreSQL
	var documentID string
	if s.db != nil {
		documentID, err = s.saveMetadata(r.Context(), uploadResult, caseID, documentType, tags, metadata)
		if err != nil {
			log.Printf("Database save failed: %v", err)
			// Don't fail the upload, just log the error
		}
	}

	response := UploadResponse{
		Success:    true,
		DocumentID: documentID,
		URL:        uploadResult.URL,
		ObjectName: uploadResult.ObjectName,
		Message:    "File uploaded successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)

	// Trigger background processing
	go s.processDocument(context.Background(), documentID, uploadResult.ObjectName)
}

func (s *UploadService) saveMetadata(ctx context.Context, result *minioClient.UploadResult, caseID, documentType string, tags map[string]string, metadata map[string]string) (string, error) {
	var documentID string
	
	tagsJSON, _ := json.Marshal(tags)
	metadataJSON, _ := json.Marshal(metadata)

	query := `
		INSERT INTO document_metadata 
		(case_id, filename, object_name, content_type, size_bytes, document_type, tags, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING id
	`

	err := s.db.QueryRowContext(ctx, query,
		caseID,
		result.Metadata["filename"],
		result.ObjectName,
		result.Metadata["content-type"],
		result.Size,
		documentType,
		tagsJSON,
		metadataJSON,
	).Scan(&documentID)

	return documentID, err
}

func (s *UploadService) processDocument(ctx context.Context, documentID, objectName string) {
	if s.db == nil {
		return
	}

	log.Printf("🔄 Processing document: %s", documentID)

	// Update status to processing
	s.updateProcessingStatus(ctx, documentID, "processing")

	// 1. Extract text (placeholder - integrate with your text extraction service)
	extractedText := s.extractText(ctx, objectName)
	
	// 2. Generate embeddings via RAG service
	embedding := s.generateEmbedding(ctx, extractedText)
	
	// 3. Update database with results
	if err := s.updateDocumentProcessing(ctx, documentID, extractedText, embedding); err != nil {
		log.Printf("❌ Failed to update document processing: %v", err)
		s.updateProcessingStatus(ctx, documentID, "failed")
		return
	}

	s.updateProcessingStatus(ctx, documentID, "completed")
	log.Printf("✅ Document processing completed: %s", documentID)
}

func (s *UploadService) extractText(ctx context.Context, objectName string) string {
	// Placeholder - integrate with your text extraction service
	// You could call another service here or implement OCR/PDF parsing
	return "Extracted text placeholder for " + objectName
}

func (s *UploadService) generateEmbedding(ctx context.Context, text string) []float32 {
	// Call your RAG service for embeddings
	ragURL := os.Getenv("RAG_SERVICE_URL")
	if ragURL == "" {
		ragURL = "http://localhost:8092"
	}

	// Make HTTP request to /embed endpoint
	payload := map[string]interface{}{
		"texts": []string{text},
	}

	payloadBytes, _ := json.Marshal(payload)
	
	resp, err := http.Post(ragURL+"/embed", "application/json", 
		http.NewRequest("POST", ragURL+"/embed", nil).Body)
	if err != nil {
		log.Printf("Embedding generation failed: %v", err)
		return make([]float32, 384) // Return zero vector
	}
	defer resp.Body.Close()

	var embedResp struct {
		Vectors [][]float32 `json:"vectors"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		log.Printf("Embedding response parsing failed: %v", err)
		return make([]float32, 384)
	}

	if len(embedResp.Vectors) > 0 {
		return embedResp.Vectors[0]
	}

	return make([]float32, 384)
}

func (s *UploadService) updateProcessingStatus(ctx context.Context, documentID, status string) {
	if s.db == nil {
		return
	}

	_, err := s.db.ExecContext(ctx, 
		"UPDATE document_metadata SET processing_status = $1, updated_at = NOW() WHERE id = $2",
		status, documentID)
	if err != nil {
		log.Printf("Failed to update processing status: %v", err)
	}
}

func (s *UploadService) updateDocumentProcessing(ctx context.Context, documentID, extractedText string, embedding []float32) error {
	query := `
		UPDATE document_metadata 
		SET extracted_text = $1, embedding = $2, updated_at = NOW()
		WHERE id = $3
	`

	_, err := s.db.ExecContext(ctx, query, extractedText, pq.Array(embedding), documentID)
	return err
}

func (s *UploadService) handleSearch(w http.ResponseWriter, r *http.Request) {
	if s.db == nil {
		http.Error(w, "Database not available", http.StatusServiceUnavailable)
		return
	}

	query := r.URL.Query().Get("q")
	caseID := r.URL.Query().Get("caseId")
	
	if query == "" {
		http.Error(w, "Query parameter 'q' is required", http.StatusBadRequest)
		return
	}

	// Generate embedding for search query
	embedding := s.generateEmbedding(r.Context(), query)

	// Perform vector search
	sqlQuery := `
		SELECT id, case_id, filename, object_name, document_type, 
			   extracted_text, (embedding <=> $1) as distance
		FROM document_metadata 
		WHERE ($2 = '' OR case_id = $2) AND embedding IS NOT NULL
		ORDER BY embedding <=> $1
		LIMIT 10
	`

	rows, err := s.db.QueryContext(r.Context(), sqlQuery, pq.Array(embedding), caseID)
	if err != nil {
		log.Printf("Search query failed: %v", err)
		http.Error(w, "Search failed", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var results []map[string]interface{}
	for rows.Next() {
		var id, caseId, filename, objectName, docType, text string
		var distance float64

		err := rows.Scan(&id, &caseId, &filename, &objectName, &docType, &text, &distance)
		if err != nil {
			continue
		}

		results = append(results, map[string]interface{}{
			"id":           id,
			"caseId":       caseId,
			"filename":     filename,
			"objectName":   objectName,
			"documentType": docType,
			"extractedText": text,
			"similarity":   1.0 - distance, // Convert distance to similarity
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"query":   query,
		"results": results,
	})
}

func main() {
	service, err := NewUploadService()
	if err != nil {
		log.Fatalf("Failed to create upload service: %v", err)
	}

	r := mux.NewRouter()

	// Upload endpoint
	r.HandleFunc("/upload", service.handleUpload).Methods("POST", "OPTIONS")
	
	// Search endpoint
	r.HandleFunc("/search", service.handleSearch).Methods("GET", "OPTIONS")
	
	// Health check
	r.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "healthy",
			"time":   time.Now().Format(time.RFC3339),
			"minio":  service.minio != nil,
			"db":     service.db != nil,
		})
	}).Methods("GET")

	// CORS setup
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"http://localhost:5173", "http://localhost:3000"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"*"},
		AllowCredentials: true,
	})

	handler := c.Handler(r)

	port := os.Getenv("UPLOAD_SERVICE_PORT")
	if port == "" {
		port = "8093"
	}

	fmt.Printf("🚀 Upload service starting on port %s\n", port)
	fmt.Printf("📁 MinIO endpoint: %s\n", os.Getenv("MINIO_ENDPOINT"))
	fmt.Printf("🗄️  Database: %v\n", service.db != nil)
	
	log.Fatal(http.ListenAndServe(":"+port, handler))
}