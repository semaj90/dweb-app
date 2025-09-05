package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/rs/cors"
)

// DocumentVector represents a document with its embedding and 3D projection
type DocumentVector struct {
	ID             string             `json:"id"`
	Title          string             `json:"title"`
	ContentSnippet string             `json:"content_snippet"`
	Embedding      []float64          `json:"embedding"`
	Projected3D    ProjectedPoint     `json:"projected_3d"`
	DocumentType   string             `json:"document_type"`
	Metadata       map[string]string  `json:"metadata"`
}

// ProjectedPoint represents 3D coordinates for visualization
type ProjectedPoint struct {
	X          float64 `json:"x"`
	Y          float64 `json:"y"`
	Z          float64 `json:"z"`
	Confidence float64 `json:"confidence"`
}

// DocumentCluster represents a cluster of related documents
type DocumentCluster struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Centroid    ProjectedPoint `json:"centroid"`
	DocumentIDs []string       `json:"document_ids"`
	Density     float64        `json:"density"`
	ClusterType string         `json:"cluster_type"`
}

// VectorSearchResponse is the main API response
type VectorSearchResponse struct {
	Documents []DocumentVector  `json:"documents"`
	Clusters  []DocumentCluster `json:"clusters"`
	Stats     QueryStatistics   `json:"stats"`
	SessionID string            `json:"session_id"`
}

// QueryStatistics provides performance metrics
type QueryStatistics struct {
	TotalDocuments     int     `json:"total_documents"`
	ProcessedDocuments int     `json:"processed_documents"`
	ProcessingTimeMS   float64 `json:"processing_time_ms"`
	EmbeddingTimeMS    float64 `json:"embedding_time_ms"`
	SearchTimeMS       float64 `json:"search_time_ms"`
	AlgorithmUsed      string  `json:"algorithm_used"`
}

// Global document store
var documentStore []DocumentVector
var clusters []DocumentCluster

// Deterministic random generator
var deterministicRand *rand.Rand

func init() {
	// Initialize deterministic random generator
	deterministicRand = rand.New(rand.NewSource(42))
	
	// Generate initial document dataset
	generateDeterministicDocuments()
	
	// Generate clusters
	generateDocumentClusters()
}

// generateDeterministicDocuments creates a consistent set of documents with embeddings
func generateDeterministicDocuments() {
	legalDocTypes := []string{"contract", "evidence", "brief", "citation", "statute", "case_law", "motion", "deposition"}
	
	sampleTitles := []string{
		"Employment Contract Agreement 2024",
		"Criminal Evidence Analysis Report",
		"Motion for Summary Judgment", 
		"Witness Deposition Transcript",
		"Property Purchase Agreement",
		"Patent Infringement Analysis",
		"Corporate Merger Documentation",
		"Immigration Status Review",
		"Personal Injury Case Brief",
		"Tax Liability Assessment",
		"Environmental Compliance Report",
		"Securities Fraud Investigation",
		"Intellectual Property License",
		"Medical Malpractice Review",
		"Construction Contract Dispute",
		"Data Privacy Impact Assessment",
		"Antitrust Compliance Audit",
		"Labor Relations Agreement",
		"Real Estate Transaction Records",
		"Bankruptcy Proceedings Summary",
	}
	
	documentStore = make([]DocumentVector, 0, len(sampleTitles))
	
	for i, title := range sampleTitles {
		// Deterministic embedding generation based on title
		embedding := generateDeterministicEmbedding(title, 384) // 384-dimensional like nomic-embed-text
		
		// 3D projection using deterministic PCA-like transformation
		projected := projectTo3D(embedding, i)
		
		doc := DocumentVector{
			ID:             fmt.Sprintf("doc_%03d", i+1),
			Title:          title,
			ContentSnippet: generateContentSnippet(title, i),
			Embedding:      embedding,
			Projected3D:    projected,
			DocumentType:   legalDocTypes[i%len(legalDocTypes)],
			Metadata: map[string]string{
				"created_date":  fmt.Sprintf("2024-%02d-%02d", (i%12)+1, (i%28)+1),
				"importance":    []string{"low", "medium", "high", "critical"}[i%4],
				"jurisdiction":  []string{"federal", "state", "local", "international"}[i%4],
				"complexity":    strconv.Itoa((i%5) + 1),
			},
		}
		
		documentStore = append(documentStore, doc)
	}
	
	log.Printf("Generated %d deterministic documents", len(documentStore))
}

// generateDeterministicEmbedding creates consistent embeddings based on text content
func generateDeterministicEmbedding(text string, dimensions int) []float64 {
	// Use text content to seed deterministic generation
	textSeed := int64(0)
	for _, r := range text {
		textSeed = textSeed*31 + int64(r)
	}
	
	textRand := rand.New(rand.NewSource(textSeed))
	embedding := make([]float64, dimensions)
	
	// Generate embedding components based on text characteristics
	words := strings.Fields(strings.ToLower(text))
	
	for i := 0; i < dimensions; i++ {
		// Base random component
		val := textRand.NormFloat64() * 0.1
		
		// Add word-based components
		if len(words) > 0 {
			wordIndex := i % len(words)
			word := words[wordIndex]
			
			// Different dimensions capture different word characteristics
			switch i % 4 {
			case 0: // Word length influence
				val += float64(len(word)) * 0.02
			case 1: // First letter influence  
				if len(word) > 0 {
					val += float64(word[0]) * 0.001
				}
			case 2: // Vowel density influence
				vowels := countVowels(word)
				val += float64(vowels) * 0.03
			case 3: // Position in text influence
				val += float64(wordIndex) * 0.001
			}
		}
		
		// Normalize to reasonable embedding range
		embedding[i] = math.Tanh(val)
	}
	
	// L2 normalize the embedding
	return normalizeVector(embedding)
}

// projectTo3D creates 3D visualization coordinates from high-dimensional embedding
func projectTo3D(embedding []float64, docIndex int) ProjectedPoint {
	// Deterministic PCA-like projection
	// Use first few embedding dimensions with rotation
	
	x := 0.0
	y := 0.0
	z := 0.0
	
	// Weighted sum of embedding dimensions
	for i, val := range embedding {
		if i >= 20 { // Use first 20 dimensions for projection
			break
		}
		
		weight := 1.0 / (float64(i) + 1) // Decreasing weights
		angle := float64(i) * 0.3        // Rotation factor
		
		x += val * weight * math.Cos(angle)
		y += val * weight * math.Sin(angle)
		z += val * weight * math.Cos(angle*1.5)
	}
	
	// Scale and offset for good visualization
	scale := 100.0
	x = x*scale + deterministicRand.Float64()*20 - 10 // Add some spread
	y = y*scale + deterministicRand.Float64()*20 - 10
	z = z*scale + deterministicRand.Float64()*20 - 10
	
	// Confidence based on embedding magnitude
	magnitude := vectorMagnitude(embedding)
	confidence := math.Min(magnitude*2, 1.0)
	
	return ProjectedPoint{
		X:          x,
		Y:          y,
		Z:          z,
		Confidence: confidence,
	}
}

// generateDocumentClusters creates meaningful clusters from the documents
func generateDocumentClusters() {
	// Group documents by type and similarity
	clusterMap := make(map[string][]string)
	
	for _, doc := range documentStore {
		clusterType := doc.DocumentType
		clusterMap[clusterType] = append(clusterMap[clusterType], doc.ID)
	}
	
	clusters = make([]DocumentCluster, 0)
	clusterID := 1
	
	for docType, docIDs := range clusterMap {
		if len(docIDs) > 1 { // Only create clusters with multiple documents
			// Calculate centroid from document positions
			centroid := calculateClusterCentroid(docIDs)
			
			cluster := DocumentCluster{
				ID:          fmt.Sprintf("cluster_%02d", clusterID),
				Name:        strings.Title(strings.ReplaceAll(docType, "_", " ")) + " Documents",
				Centroid:    centroid,
				DocumentIDs: docIDs,
				Density:     calculateClusterDensity(docIDs),
				ClusterType: docType,
			}
			
			clusters = append(clusters, cluster)
			clusterID++
		}
	}
	
	log.Printf("Generated %d document clusters", len(clusters))
}

// API Handlers

func handleInitialData(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()
	
	response := VectorSearchResponse{
		Documents: documentStore,
		Clusters:  clusters,
		Stats: QueryStatistics{
			TotalDocuments:     len(documentStore),
			ProcessedDocuments: len(documentStore),
			ProcessingTimeMS:   float64(time.Since(startTime).Nanoseconds()) / 1e6,
			EmbeddingTimeMS:    0, // No embedding needed for initial data
			SearchTimeMS:       0, // No search performed
			AlgorithmUsed:      "deterministic_generation",
		},
		SessionID: fmt.Sprintf("session_%d", time.Now().Unix()),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handleVectorSearch(w http.ResponseWriter, r *http.Request) {
	var query struct {
		QueryText string `json:"query_text"`
		Limit     int    `json:"limit"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&query); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	if query.Limit <= 0 {
		query.Limit = 10
	}
	
	startTime := time.Now()
	embeddingStart := time.Now()
	
	// Generate embedding for query
	queryEmbedding := generateDeterministicEmbedding(query.QueryText, 384)
	embeddingTime := time.Since(embeddingStart)
	
	searchStart := time.Now()
	
	// Calculate similarities and sort
	type docSimilarity struct {
		doc        DocumentVector
		similarity float64
	}
	
	similarities := make([]docSimilarity, 0, len(documentStore))
	
	for _, doc := range documentStore {
		similarity := cosineSimilarity(queryEmbedding, doc.Embedding)
		similarities = append(similarities, docSimilarity{doc, similarity})
	}
	
	// Sort by similarity (highest first)
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].similarity > similarities[j].similarity
	})
	
	// Take top results
	if query.Limit > len(similarities) {
		query.Limit = len(similarities)
	}
	
	results := make([]DocumentVector, query.Limit)
	for i := 0; i < query.Limit; i++ {
		results[i] = similarities[i].doc
	}
	
	searchTime := time.Since(searchStart)
	totalTime := time.Since(startTime)
	
	response := VectorSearchResponse{
		Documents: results,
		Clusters:  clusters, // Include all clusters
		Stats: QueryStatistics{
			TotalDocuments:     len(documentStore),
			ProcessedDocuments: len(results),
			ProcessingTimeMS:   float64(totalTime.Nanoseconds()) / 1e6,
			EmbeddingTimeMS:    float64(embeddingTime.Nanoseconds()) / 1e6,
			SearchTimeMS:       float64(searchTime.Nanoseconds()) / 1e6,
			AlgorithmUsed:      "cosine_similarity",
		},
		SessionID: fmt.Sprintf("search_%d", time.Now().Unix()),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"healthy":            true,
		"version":           "1.0.0",
		"total_documents":   len(documentStore),
		"active_clusters":   len(clusters),
		"uptime_seconds":    time.Since(startTime).Seconds(),
		"memory_usage_mb":   "~50MB", // Estimated
		"algorithm":         "deterministic_embeddings",
		"gpu_available":     false, // CPU-only for now
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// Utility Functions

func countVowels(word string) int {
	count := 0
	vowels := "aeiouAEIOU"
	for _, r := range word {
		if strings.ContainsRune(vowels, r) {
			count++
		}
	}
	return count
}

func normalizeVector(vec []float64) []float64 {
	magnitude := vectorMagnitude(vec)
	if magnitude == 0 {
		return vec
	}
	
	normalized := make([]float64, len(vec))
	for i, val := range vec {
		normalized[i] = val / magnitude
	}
	return normalized
}

func vectorMagnitude(vec []float64) float64 {
	sum := 0.0
	for _, val := range vec {
		sum += val * val
	}
	return math.Sqrt(sum)
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0
	
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0 || normB == 0 {
		return 0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func calculateClusterCentroid(docIDs []string) ProjectedPoint {
	if len(docIDs) == 0 {
		return ProjectedPoint{}
	}
	
	var sumX, sumY, sumZ, sumConf float64
	
	for _, docID := range docIDs {
		for _, doc := range documentStore {
			if doc.ID == docID {
				sumX += doc.Projected3D.X
				sumY += doc.Projected3D.Y
				sumZ += doc.Projected3D.Z
				sumConf += doc.Projected3D.Confidence
				break
			}
		}
	}
	
	count := float64(len(docIDs))
	return ProjectedPoint{
		X:          sumX / count,
		Y:          sumY / count,
		Z:          sumZ / count,
		Confidence: sumConf / count,
	}
}

func calculateClusterDensity(docIDs []string) float64 {
	if len(docIDs) <= 1 {
		return 1.0
	}
	
	// Calculate average distance between documents in cluster
	totalDistance := 0.0
	comparisons := 0
	
	for i, id1 := range docIDs {
		for j, id2 := range docIDs {
			if i >= j {
				continue
			}
			
			doc1 := findDocumentByID(id1)
			doc2 := findDocumentByID(id2)
			
			if doc1 != nil && doc2 != nil {
				distance := euclideanDistance3D(doc1.Projected3D, doc2.Projected3D)
				totalDistance += distance
				comparisons++
			}
		}
	}
	
	if comparisons == 0 {
		return 1.0
	}
	
	avgDistance := totalDistance / float64(comparisons)
	// Density is inverse of average distance (scaled)
	return 1.0 / (1.0 + avgDistance/100.0)
}

func findDocumentByID(id string) *DocumentVector {
	for i := range documentStore {
		if documentStore[i].ID == id {
			return &documentStore[i]
		}
	}
	return nil
}

func euclideanDistance3D(a, b ProjectedPoint) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	dz := a.Z - b.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func generateContentSnippet(title string, index int) string {
	snippets := []string{
		"This document outlines the key terms and conditions...",
		"Evidence collected during the investigation reveals...",
		"The legal analysis demonstrates clear precedent for...",
		"Witness testimony confirms the following details...",
		"Contract provisions specify the obligations of each party...",
		"Prior case law establishes the framework for...",
		"The motion requests the court to consider...",
		"Deposition transcript excerpt showing relevant facts...",
		"Property documentation includes all necessary details...",
		"Patent claims cover the following innovative aspects...",
		"Merger agreement terms define the structure of...",
		"Immigration status review indicates compliance with...",
		"Personal injury assessment reveals significant damages...",
		"Tax documentation supports the claimed deductions...",
		"Environmental impact study concludes minimal risk...",
		"Securities analysis indicates potential fraud indicators...",
		"Intellectual property license grants specific rights...",
		"Medical records support the malpractice claim...",
		"Construction specifications detail the required standards...",
		"Privacy assessment identifies data protection measures...",
	}
	
	return snippets[index%len(snippets)]
}

var startTime = time.Now()

func main() {
	r := mux.NewRouter()
	
	// API routes
	r.HandleFunc("/api/v1/initial-data", handleInitialData).Methods("GET")
	r.HandleFunc("/api/v1/search", handleVectorSearch).Methods("POST")
	r.HandleFunc("/api/v1/health", handleHealthCheck).Methods("GET")
	
	// Enable CORS
	c := cors.New(cors.Options{
		AllowedOrigins: []string{"http://localhost:5173", "http://localhost:3000", "*"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"*"},
	})
	
	handler := c.Handler(r)
	
	fmt.Println("= Cyber Elephant Backend (Psychic Mind) starting on :8080")
	fmt.Printf("=Ê Initialized with %d documents and %d clusters\n", len(documentStore), len(clusters))
	fmt.Println(">à Using deterministic embeddings for consistent results")
	fmt.Println("= CORS enabled for frontend integration")
	
	log.Fatal(http.ListenAndServe(":8080", handler))
}