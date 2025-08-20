//go:build legacy
// +build legacy

// Self-Organizing Map (SOM) Implementation for Legal Document Clustering
// Real-time clustering and pattern recognition for legal document embeddings
// Integrates with tensor tiling system for efficient processing

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

// SOM (Self-Organizing Map) structure
type SOM struct {
	ID           string              `json:"id"`
	Width        int                 `json:"width"`
	Height       int                 `json:"height"`
	InputDim     int                 `json:"input_dim"`
	Neurons      [][]Neuron          `json:"neurons"`
	LearningRate float64             `json:"learning_rate"`
	Radius       float64             `json:"radius"`
	Iteration    int                 `json:"iteration"`
	MaxIter      int                 `json:"max_iter"`
	Metadata     SOMMetadata         `json:"metadata"`
	CreatedAt    time.Time           `json:"created_at"`
	UpdatedAt    time.Time           `json:"updated_at"`
}

// Individual neuron in the SOM
type Neuron struct {
	Position    [2]int    `json:"position"`     // Position in the SOM grid
	Weights     []float32 `json:"weights"`      // Weight vector
	Activations int       `json:"activations"`  // Number of times activated
	Documents   []string  `json:"documents"`    // Associated document IDs
	Labels      []string  `json:"labels"`       // Legal category labels
	UpdateCount int       `json:"update_count"` // Update frequency
}

// SOM metadata for legal context
type SOMMetadata struct {
	Domain          string                 `json:"domain"`           // "legal", "contracts", "litigation"
	PracticeAreas   []string               `json:"practice_areas"`   // Legal practice areas
	Purpose         string                 `json:"purpose"`          // "classification", "similarity", "recommendation"
	TrainingData    SOMTrainingInfo        `json:"training_data"`    // Training dataset info
	Performance     SOMPerformance         `json:"performance"`      // Performance metrics
	Configuration   SOMConfiguration       `json:"configuration"`    // SOM hyperparameters
	LastOptimized   time.Time              `json:"last_optimized"`   // Last optimization run
	Context         map[string]interface{} `json:"context"`          // Additional context
}

// SOM training information
type SOMTrainingInfo struct {
	DocumentCount    int       `json:"document_count"`
	EmbeddingModel   string    `json:"embedding_model"`
	DatasetVersion   string    `json:"dataset_version"`
	TrainingStarted  time.Time `json:"training_started"`
	TrainingFinished time.Time `json:"training_finished"`
	Epochs           int       `json:"epochs"`
	BatchSize        int       `json:"batch_size"`
}

// SOM performance metrics
type SOMPerformance struct {
	QuantizationError   float64                `json:"quantization_error"`
	TopographicError    float64                `json:"topographic_error"`
	ClusterPurity       float64                `json:"cluster_purity"`
	Silhouette          float64                `json:"silhouette_score"`
	DocumentCoverage    float64                `json:"document_coverage"`
	CategoryDistribution map[string]int        `json:"category_distribution"`
	PerformanceHistory  []PerformanceSnapshot `json:"performance_history"`
}

// Performance snapshot for tracking
type PerformanceSnapshot struct {
	Timestamp         time.Time `json:"timestamp"`
	Iteration         int       `json:"iteration"`
	QuantizationError float64   `json:"quantization_error"`
	LearningRate      float64   `json:"learning_rate"`
	Radius            float64   `json:"radius"`
}

// SOM configuration parameters
type SOMConfiguration struct {
	InitialLearningRate float64 `json:"initial_learning_rate"`
	FinalLearningRate   float64 `json:"final_learning_rate"`
	InitialRadius       float64 `json:"initial_radius"`
	FinalRadius         float64 `json:"final_radius"`
	NeighborhoodFunc    string  `json:"neighborhood_func"`    // "gaussian", "mexican_hat", "bubble"
	LearningRateDecay   string  `json:"learning_rate_decay"`  // "linear", "exponential", "inverse"
	DistanceMetric      string  `json:"distance_metric"`      // "euclidean", "cosine", "manhattan"
	Topology            string  `json:"topology"`             // "rectangular", "hexagonal"
	InitializationMethod string `json:"initialization_method"` // "random", "pca", "sample"
}

// Document clustering request
type ClusteringRequest struct {
	DocumentEmbeddings []DocumentEmbedding `json:"document_embeddings"`
	SOMParameters      SOMConfiguration    `json:"som_parameters"`
	ClusteringMode     string              `json:"clustering_mode"`  // "train", "predict", "update"
	ForceRetrain       bool                `json:"force_retrain"`
}

// Document embedding with metadata
type DocumentEmbedding struct {
	DocumentID   string                 `json:"document_id"`
	Embedding    []float32              `json:"embedding"`
	DocumentType string                 `json:"document_type"`
	PracticeArea string                 `json:"practice_area"`
	Jurisdiction string                 `json:"jurisdiction"`
	Metadata     map[string]interface{} `json:"metadata"`
	Timestamp    time.Time              `json:"timestamp"`
}

// Clustering result
type ClusteringResult struct {
	DocumentID      string    `json:"document_id"`
	ClusterPosition [2]int    `json:"cluster_position"`
	Distance        float64   `json:"distance"`
	Confidence      float64   `json:"confidence"`
	SimilarDocs     []string  `json:"similar_docs"`
	ClusterLabels   []string  `json:"cluster_labels"`
	Timestamp       time.Time `json:"timestamp"`
}

// SOM service for managing multiple SOMs
type SOMService struct {
	redis   *redis.Client
	ctx     context.Context
	config  *SOMServiceConfig
	soms    sync.Map              // Active SOMs cache
	workers []*SOMWorker          // Worker pool for training
	jobs    chan SOMJob           // Job queue
	mu      sync.RWMutex
}

// SOM service configuration
type SOMServiceConfig struct {
	DefaultSOMSize    [2]int        `json:"default_som_size"`
	MaxSOMSize        [2]int        `json:"max_som_size"`
	CacheExpiration   time.Duration `json:"cache_expiration"`
	WorkerCount       int           `json:"worker_count"`
	JobQueueSize      int           `json:"job_queue_size"`
	RedisKeyPrefix    string        `json:"redis_key_prefix"`
	AutoOptimization  bool          `json:"auto_optimization"`
	OptimizationInterval time.Duration `json:"optimization_interval"`
}

// SOM processing job
type SOMJob struct {
	ID          string      `json:"id"`
	Type        string      `json:"type"`        // "train", "update", "optimize", "predict"
	SOMID       string      `json:"som_id"`
	Data        interface{} `json:"data"`
	Parameters  interface{} `json:"parameters"`
	Priority    int         `json:"priority"`
	SubmittedAt time.Time   `json:"submitted_at"`
	Status      string      `json:"status"`
	Result      interface{} `json:"result,omitempty"`
	Error       string      `json:"error,omitempty"`
}

// SOM worker for processing jobs
type SOMWorker struct {
	ID      int
	service *SOMService
	jobs    chan SOMJob
	quit    chan bool
}

// Initialize SOM service
func NewSOMService(redis *redis.Client, config *SOMServiceConfig) *SOMService {
	service := &SOMService{
		redis:   redis,
		ctx:     context.Background(),
		config:  config,
		jobs:    make(chan SOMJob, config.JobQueueSize),
	}
	
	// Start worker pool
	service.startWorkers()
	
	// Start auto-optimization if enabled
	if config.AutoOptimization {
		go service.autoOptimizationLoop()
	}
	
	return service
}

// Start SOM workers
func (ss *SOMService) startWorkers() {
	ss.workers = make([]*SOMWorker, ss.config.WorkerCount)
	
	for i := 0; i < ss.config.WorkerCount; i++ {
		worker := &SOMWorker{
			ID:      i,
			service: ss,
			jobs:    make(chan SOMJob, 10),
			quit:    make(chan bool),
		}
		ss.workers[i] = worker
		go worker.start()
	}
	
	log.Printf("🧠 Started %d SOM workers", ss.config.WorkerCount)
}

// Create new SOM for legal document clustering
func (ss *SOMService) CreateSOM(width, height, inputDim int, metadata SOMMetadata) (*SOM, error) {
	somID := fmt.Sprintf("som_%d", time.Now().UnixNano())
	
	// Initialize neurons with random weights
	neurons := make([][]Neuron, height)
	for i := 0; i < height; i++ {
		neurons[i] = make([]Neuron, width)
		for j := 0; j < width; j++ {
			weights := make([]float32, inputDim)
			for k := 0; k < inputDim; k++ {
				weights[k] = float32(rand.NormFloat64() * 0.1) // Small random initialization
			}
			
			neurons[i][j] = Neuron{
				Position:    [2]int{i, j},
				Weights:     weights,
				Activations: 0,
				Documents:   []string{},
				Labels:      []string{},
				UpdateCount: 0,
			}
		}
	}
	
	// Default configuration
	defaultConfig := SOMConfiguration{
		InitialLearningRate:  0.5,
		FinalLearningRate:    0.01,
		InitialRadius:        float64(max(width, height)) / 2.0,
		FinalRadius:          1.0,
		NeighborhoodFunc:     "gaussian",
		LearningRateDecay:    "exponential",
		DistanceMetric:       "cosine", // Better for document embeddings
		Topology:             "rectangular",
		InitializationMethod: "random",
	}
	
	som := &SOM{
		ID:           somID,
		Width:        width,
		Height:       height,
		InputDim:     inputDim,
		Neurons:      neurons,
		LearningRate: defaultConfig.InitialLearningRate,
		Radius:       defaultConfig.InitialRadius,
		Iteration:    0,
		MaxIter:      1000, // Default max iterations
		Metadata:     metadata,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}
	
	som.Metadata.Configuration = defaultConfig
	som.Metadata.Performance = SOMPerformance{
		CategoryDistribution: make(map[string]int),
		PerformanceHistory:   []PerformanceSnapshot{},
	}
	
	// Store SOM in Redis
	if err := ss.storeSOM(som); err != nil {
		return nil, fmt.Errorf("failed to store SOM: %w", err)
	}
	
	// Cache in memory
	ss.soms.Store(somID, som)
	
	log.Printf("🧠 Created SOM %s (%dx%d) for %s", somID, width, height, metadata.Domain)
	return som, nil
}

// Train SOM with document embeddings
func (ss *SOMService) TrainSOM(somID string, embeddings []DocumentEmbedding, epochs int) error {
	som, err := ss.loadSOM(somID)
	if err != nil {
		return fmt.Errorf("failed to load SOM: %w", err)
	}
	
	log.Printf("🎓 Training SOM %s with %d documents for %d epochs", somID, len(embeddings), epochs)
	
	// Update training info
	som.Metadata.TrainingData.DocumentCount = len(embeddings)
	som.Metadata.TrainingData.TrainingStarted = time.Now()
	som.Metadata.TrainingData.Epochs = epochs
	som.MaxIter = epochs * len(embeddings)
	
	startTime := time.Now()
	
	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle embeddings for each epoch
		shuffled := make([]DocumentEmbedding, len(embeddings))
		copy(shuffled, embeddings)
		rand.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})
		
		epochError := 0.0
		
		for _, embedding := range shuffled {
			// Find best matching unit (BMU)
			bmuPos, distance := ss.findBMU(som, embedding.Embedding)
			epochError += distance
			
			// Update BMU and neighbors
			ss.updateNeighborhood(som, bmuPos, embedding)
			
			som.Iteration++
			
			// Update learning parameters
			ss.updateLearningParameters(som)
		}
		
		// Calculate and record performance metrics
		avgError := epochError / float64(len(embeddings))
		ss.recordPerformance(som, avgError)
		
		if epoch%10 == 0 {
			log.Printf("📈 Epoch %d/%d - Avg Error: %.6f", epoch+1, epochs, avgError)
		}
	}
	
	// Finalize training
	som.Metadata.TrainingData.TrainingFinished = time.Now()
	som.UpdatedAt = time.Now()
	
	// Calculate final performance metrics
	ss.calculateFinalMetrics(som, embeddings)
	
	// Store updated SOM
	if err := ss.storeSOM(som); err != nil {
		return fmt.Errorf("failed to store trained SOM: %w", err)
	}
	
	trainingTime := time.Since(startTime)
	log.Printf("✅ SOM %s training completed in %v", somID, trainingTime)
	
	return nil
}

// Find Best Matching Unit (BMU)
func (ss *SOMService) findBMU(som *SOM, input []float32) ([2]int, float64) {
	minDistance := math.Inf(1)
	var bmuPos [2]int
	
	for i := 0; i < som.Height; i++ {
		for j := 0; j < som.Width; j++ {
			distance := ss.calculateDistance(som.Neurons[i][j].Weights, input, som.Metadata.Configuration.DistanceMetric)
			if distance < minDistance {
				minDistance = distance
				bmuPos = [2]int{i, j}
			}
		}
	}
	
	return bmuPos, minDistance
}

// Calculate distance between vectors
func (ss *SOMService) calculateDistance(weights []float32, input []float32, metric string) float64 {
	switch metric {
	case "cosine":
		return ss.cosineDistance(weights, input)
	case "manhattan":
		return ss.manhattanDistance(weights, input)
	default: // euclidean
		return ss.euclideanDistance(weights, input)
	}
}

// Cosine distance (better for document embeddings)
func (ss *SOMService) cosineDistance(a, b []float32) float64 {
	dotProduct := float64(0)
	normA := float64(0)
	normB := float64(0)
	
	for i := 0; i < len(a) && i < len(b); i++ {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	
	if normA == 0 || normB == 0 {
		return 1.0
	}
	
	similarity := dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
	return 1.0 - similarity // Convert similarity to distance
}

// Euclidean distance
func (ss *SOMService) euclideanDistance(a, b []float32) float64 {
	sum := float64(0)
	for i := 0; i < len(a) && i < len(b); i++ {
		diff := float64(a[i]) - float64(b[i])
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// Manhattan distance
func (ss *SOMService) manhattanDistance(a, b []float32) float64 {
	sum := float64(0)
	for i := 0; i < len(a) && i < len(b); i++ {
		sum += math.Abs(float64(a[i]) - float64(b[i]))
	}
	return sum
}

// Update neighborhood around BMU
func (ss *SOMService) updateNeighborhood(som *SOM, bmuPos [2]int, embedding DocumentEmbedding) {
	for i := 0; i < som.Height; i++ {
		for j := 0; j < som.Width; j++ {
			// Calculate distance from BMU
			distance := ss.gridDistance(bmuPos, [2]int{i, j})
			
			// Calculate neighborhood influence
			influence := ss.neighborhoodFunction(distance, som.Radius, som.Metadata.Configuration.NeighborhoodFunc)
			
			if influence > 0.01 { // Only update if influence is significant
				// Update neuron weights
				for k := 0; k < len(som.Neurons[i][j].Weights) && k < len(embedding.Embedding); k++ {
					delta := som.LearningRate * influence * (float64(embedding.Embedding[k]) - float64(som.Neurons[i][j].Weights[k]))
					som.Neurons[i][j].Weights[k] += float32(delta)
				}
				
				// Update neuron metadata if this is the BMU
				if i == bmuPos[0] && j == bmuPos[1] {
					som.Neurons[i][j].Activations++
					som.Neurons[i][j].Documents = append(som.Neurons[i][j].Documents, embedding.DocumentID)
					
					// Add legal category labels
					if embedding.PracticeArea != "" {
						som.Neurons[i][j].Labels = ss.addUniqueLabel(som.Neurons[i][j].Labels, embedding.PracticeArea)
					}
					if embedding.DocumentType != "" {
						som.Neurons[i][j].Labels = ss.addUniqueLabel(som.Neurons[i][j].Labels, embedding.DocumentType)
					}
				}
				
				som.Neurons[i][j].UpdateCount++
			}
		}
	}
}

// Grid distance between neurons
func (ss *SOMService) gridDistance(pos1, pos2 [2]int) float64 {
	dx := float64(pos1[0] - pos2[0])
	dy := float64(pos1[1] - pos2[1])
	return math.Sqrt(dx*dx + dy*dy)
}

// Neighborhood function
func (ss *SOMService) neighborhoodFunction(distance, radius float64, funcType string) float64 {
	switch funcType {
	case "mexican_hat":
		// Mexican hat function
		if distance <= radius {
			sigma := radius / 3.0
			factor := distance / sigma
			return (1.0 - factor*factor) * math.Exp(-factor*factor/2.0)
		}
		return 0.0
	case "bubble":
		// Simple bubble function
		if distance <= radius {
			return 1.0
		}
		return 0.0
	default: // gaussian
		// Gaussian neighborhood function
		sigma := radius / 3.0
		return math.Exp(-(distance * distance) / (2.0 * sigma * sigma))
	}
}

// Update learning parameters
func (ss *SOMService) updateLearningParameters(som *SOM) {
	progress := float64(som.Iteration) / float64(som.MaxIter)
	
	config := som.Metadata.Configuration
	
	// Update learning rate
	switch config.LearningRateDecay {
	case "linear":
		som.LearningRate = config.InitialLearningRate * (1.0 - progress)
	case "inverse":
		som.LearningRate = config.InitialLearningRate / (1.0 + progress*10.0)
	default: // exponential
		som.LearningRate = config.InitialLearningRate * math.Exp(-progress*3.0)
	}
	
	// Ensure minimum learning rate
	if som.LearningRate < config.FinalLearningRate {
		som.LearningRate = config.FinalLearningRate
	}
	
	// Update radius
	som.Radius = config.InitialRadius * math.Exp(-progress*2.0)
	if som.Radius < config.FinalRadius {
		som.Radius = config.FinalRadius
	}
}

// Record performance metrics
func (ss *SOMService) recordPerformance(som *SOM, quantizationError float64) {
	snapshot := PerformanceSnapshot{
		Timestamp:         time.Now(),
		Iteration:         som.Iteration,
		QuantizationError: quantizationError,
		LearningRate:      som.LearningRate,
		Radius:            som.Radius,
	}
	
	som.Metadata.Performance.PerformanceHistory = append(som.Metadata.Performance.PerformanceHistory, snapshot)
	som.Metadata.Performance.QuantizationError = quantizationError
}

// Calculate final performance metrics
func (ss *SOMService) calculateFinalMetrics(som *SOM, embeddings []DocumentEmbedding) {
	// Calculate cluster purity
	purity := ss.calculateClusterPurity(som)
	som.Metadata.Performance.ClusterPurity = purity
	
	// Calculate document coverage
	coverage := ss.calculateDocumentCoverage(som)
	som.Metadata.Performance.DocumentCoverage = coverage
	
	// Update category distribution
	distribution := make(map[string]int)
	for _, embedding := range embeddings {
		if embedding.PracticeArea != "" {
			distribution[embedding.PracticeArea]++
		}
	}
	som.Metadata.Performance.CategoryDistribution = distribution
	
	log.Printf("📊 SOM %s metrics - Purity: %.3f, Coverage: %.3f", som.ID, purity, coverage)
}

// Calculate cluster purity
func (ss *SOMService) calculateClusterPurity(som *SOM) float64 {
	totalPurity := 0.0
	activeClusters := 0
	
	for i := 0; i < som.Height; i++ {
		for j := 0; j < som.Width; j++ {
			neuron := &som.Neurons[i][j]
			if len(neuron.Labels) > 0 {
				// Calculate label distribution
				labelCounts := make(map[string]int)
				for _, label := range neuron.Labels {
					labelCounts[label]++
				}
				
				// Find majority label
				maxCount := 0
				for _, count := range labelCounts {
					if count > maxCount {
						maxCount = count
					}
				}
				
				// Calculate purity for this cluster
				if len(neuron.Labels) > 0 {
					purity := float64(maxCount) / float64(len(neuron.Labels))
					totalPurity += purity
					activeClusters++
				}
			}
		}
	}
	
	if activeClusters == 0 {
		return 0.0
	}
	
	return totalPurity / float64(activeClusters)
}

// Calculate document coverage
func (ss *SOMService) calculateDocumentCoverage(som *SOM) float64 {
	activeNeurons := 0
	totalNeurons := som.Width * som.Height
	
	for i := 0; i < som.Height; i++ {
		for j := 0; j < som.Width; j++ {
			if som.Neurons[i][j].Activations > 0 {
				activeNeurons++
			}
		}
	}
	
	return float64(activeNeurons) / float64(totalNeurons)
}

// Predict cluster for new document
func (ss *SOMService) PredictCluster(somID string, embedding []float32) (*ClusteringResult, error) {
	som, err := ss.loadSOM(somID)
	if err != nil {
		return nil, fmt.Errorf("failed to load SOM: %w", err)
	}
	
	// Find best matching unit
	bmuPos, distance := ss.findBMU(som, embedding)
	
	// Get BMU neuron
	bmu := &som.Neurons[bmuPos[0]][bmuPos[1]]
	
	// Calculate confidence based on distance and neuron activation frequency
	maxDistance := float64(len(embedding)) // Theoretical maximum for normalized vectors
	confidence := math.Max(0.0, 1.0-distance/maxDistance)
	
	// Boost confidence if neuron has been activated frequently
	if bmu.Activations > 10 {
		confidence *= 1.1
	}
	
	result := &ClusteringResult{
		ClusterPosition: bmuPos,
		Distance:        distance,
		Confidence:      math.Min(confidence, 1.0),
		SimilarDocs:     bmu.Documents,
		ClusterLabels:   bmu.Labels,
		Timestamp:       time.Now(),
	}
	
	return result, nil
}

// Load SOM from Redis
func (ss *SOMService) loadSOM(somID string) (*SOM, error) {
	// Check memory cache first
	if cached, ok := ss.soms.Load(somID); ok {
		return cached.(*SOM), nil
	}
	
	// Load from Redis
	key := fmt.Sprintf("%s:som:%s", ss.config.RedisKeyPrefix, somID)
	data, err := ss.redis.Get(ss.ctx, key).Result()
	if err != nil {
		return nil, fmt.Errorf("SOM not found: %w", err)
	}
	
	var som SOM
	if err := json.Unmarshal([]byte(data), &som); err != nil {
		return nil, fmt.Errorf("failed to unmarshal SOM: %w", err)
	}
	
	// Cache in memory
	ss.soms.Store(somID, &som)
	
	return &som, nil
}

// Store SOM in Redis
func (ss *SOMService) storeSOM(som *SOM) error {
	key := fmt.Sprintf("%s:som:%s", ss.config.RedisKeyPrefix, som.ID)
	
	data, err := json.Marshal(som)
	if err != nil {
		return fmt.Errorf("failed to marshal SOM: %w", err)
	}
	
	// Store with expiration
	if err := ss.redis.Set(ss.ctx, key, data, ss.config.CacheExpiration).Err(); err != nil {
		return fmt.Errorf("failed to store SOM in Redis: %w", err)
	}
	
	// Update memory cache
	ss.soms.Store(som.ID, som)
	
	return nil
}

// Auto-optimization loop
func (ss *SOMService) autoOptimizationLoop() {
	ticker := time.NewTicker(ss.config.OptimizationInterval)
	defer ticker.Stop()
	
	for range ticker.C {
		ss.optimizeAllSOMs()
	}
}

// Optimize all SOMs
func (ss *SOMService) optimizeAllSOMs() {
	log.Printf("🔄 Running auto-optimization for SOMs")
	
	// This would implement automatic hyperparameter tuning
	// and performance optimization for all active SOMs
}

// Add unique label to neuron
func (ss *SOMService) addUniqueLabel(labels []string, newLabel string) []string {
	for _, label := range labels {
		if label == newLabel {
			return labels
		}
	}
	return append(labels, newLabel)
}

// SOM worker processing
func (sw *SOMWorker) start() {
	log.Printf("🧠 SOM worker %d started", sw.ID)
	
	for {
		select {
		case job := <-sw.jobs:
			sw.processJob(job)
		case <-sw.quit:
			log.Printf("🛑 SOM worker %d stopped", sw.ID)
			return
		}
	}
}

// Process SOM job
func (sw *SOMWorker) processJob(job SOMJob) {
	log.Printf("⚙️  SOM worker %d processing job %s (%s)", sw.ID, job.ID, job.Type)
	
	start := time.Now()
	
	switch job.Type {
	case "train":
		sw.processTrainingJob(job)
	case "predict":
		sw.processPredictionJob(job)
	case "optimize":
		sw.processOptimizationJob(job)
	default:
		log.Printf("❌ Unknown SOM job type: %s", job.Type)
	}
	
	duration := time.Since(start)
	log.Printf("✅ SOM worker %d completed job %s in %v", sw.ID, job.ID, duration)
}

// Process training job
func (sw *SOMWorker) processTrainingJob(job SOMJob) {
	// Implementation for SOM training job
	log.Printf("🎓 Processing SOM training job for %s", job.SOMID)
}

// Process prediction job
func (sw *SOMWorker) processPredictionJob(job SOMJob) {
	// Implementation for SOM prediction job
	log.Printf("🔮 Processing SOM prediction job for %s", job.SOMID)
}

// Process optimization job
func (sw *SOMWorker) processOptimizationJob(job SOMJob) {
	// Implementation for SOM optimization job
	log.Printf("⚡ Processing SOM optimization job for %s", job.SOMID)
}

// API endpoints for SOM operations
func (ss *SOMService) addSOMRoutes(router *gin.Engine) {
	som := router.Group("/api/som")
	{
		// Create new SOM
		som.POST("/create", ss.createSOMEndpoint)
		
		// Train SOM
		som.POST("/:somId/train", ss.trainSOMEndpoint)
		
		// Predict cluster
		som.POST("/:somId/predict", ss.predictEndpoint)
		
		// Get SOM info
		som.GET("/:somId", ss.getSOMEndpoint)
		
		// Get SOM visualization data
		som.GET("/:somId/visualization", ss.getVisualizationEndpoint)
		
		// Update SOM
		som.PUT("/:somId/update", ss.updateSOMEndpoint)
	}
}

// Create SOM endpoint
func (ss *SOMService) createSOMEndpoint(c *gin.Context) {
	var req struct {
		Width        int         `json:"width"`
		Height       int         `json:"height"`
		InputDim     int         `json:"input_dim"`
		Metadata     SOMMetadata `json:"metadata"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	som, err := ss.CreateSOM(req.Width, req.Height, req.InputDim, req.Metadata)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, gin.H{
		"som_id":     som.ID,
		"dimensions": [2]int{som.Width, som.Height},
		"input_dim":  som.InputDim,
		"created":    som.CreatedAt,
	})
}

// Train SOM endpoint
func (ss *SOMService) trainSOMEndpoint(c *gin.Context) {
	somID := c.Param("somId")
	
	var req struct {
		Embeddings []DocumentEmbedding `json:"embeddings"`
		Epochs     int                 `json:"epochs"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	// Submit training job
	job := SOMJob{
		ID:          fmt.Sprintf("train_%d", time.Now().UnixNano()),
		Type:        "train",
		SOMID:       somID,
		Data:        req.Embeddings,
		Parameters:  req.Epochs,
		Priority:    1,
		SubmittedAt: time.Now(),
		Status:      "pending",
	}
	
	select {
	case ss.jobs <- job:
		c.JSON(200, gin.H{
			"job_id":     job.ID,
			"status":     "submitted",
			"embeddings": len(req.Embeddings),
			"epochs":     req.Epochs,
		})
	default:
		c.JSON(503, gin.H{"error": "job queue full"})
	}
}

// Predict endpoint
func (ss *SOMService) predictEndpoint(c *gin.Context) {
	somID := c.Param("somId")
	
	var req struct {
		Embedding []float32 `json:"embedding"`
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	result, err := ss.PredictCluster(somID, req.Embedding)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	
	c.JSON(200, result)
}

// Get SOM endpoint
func (ss *SOMService) getSOMEndpoint(c *gin.Context) {
	somID := c.Param("somId")
	
	som, err := ss.loadSOM(somID)
	if err != nil {
		c.JSON(404, gin.H{"error": "SOM not found"})
		return
	}
	
	// Return SOM info without full neuron data (too large)
	c.JSON(200, gin.H{
		"id":         som.ID,
		"dimensions": [2]int{som.Width, som.Height},
		"input_dim":  som.InputDim,
		"iteration":  som.Iteration,
		"metadata":   som.Metadata,
		"created":    som.CreatedAt,
		"updated":    som.UpdatedAt,
	})
}

// Get visualization endpoint
func (ss *SOMService) getVisualizationEndpoint(c *gin.Context) {
	somID := c.Param("somId")
	
	som, err := ss.loadSOM(somID)
	if err != nil {
		c.JSON(404, gin.H{"error": "SOM not found"})
		return
	}
	
	// Create visualization data
	visualization := ss.createVisualization(som)
	
	c.JSON(200, visualization)
}

// Create visualization data
func (ss *SOMService) createVisualization(som *SOM) map[string]interface{} {
	grid := make([][]map[string]interface{}, som.Height)
	
	for i := 0; i < som.Height; i++ {
		grid[i] = make([]map[string]interface{}, som.Width)
		for j := 0; j < som.Width; j++ {
			neuron := &som.Neurons[i][j]
			grid[i][j] = map[string]interface{}{
				"position":    neuron.Position,
				"activations": neuron.Activations,
				"labels":      neuron.Labels,
				"doc_count":   len(neuron.Documents),
				"update_count": neuron.UpdateCount,
			}
		}
	}
	
	return map[string]interface{}{
		"som_id":      som.ID,
		"dimensions":  [2]int{som.Width, som.Height},
		"grid":        grid,
		"performance": som.Metadata.Performance,
		"timestamp":   time.Now(),
	}
}

// Update SOM endpoint
func (ss *SOMService) updateSOMEndpoint(c *gin.Context) {
	somID := c.Param("somId")
	
	var req struct {
		NewEmbeddings []DocumentEmbedding `json:"new_embeddings"`
		UpdateMode    string              `json:"update_mode"` // "incremental", "batch"
	}
	
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}
	
	// Submit update job
	job := SOMJob{
		ID:          fmt.Sprintf("update_%d", time.Now().UnixNano()),
		Type:        "update",
		SOMID:       somID,
		Data:        req.NewEmbeddings,
		Parameters:  req.UpdateMode,
		Priority:    2,
		SubmittedAt: time.Now(),
		Status:      "pending",
	}
	
	select {
	case ss.jobs <- job:
		c.JSON(200, gin.H{
			"job_id":     job.ID,
			"status":     "submitted",
			"mode":       req.UpdateMode,
			"embeddings": len(req.NewEmbeddings),
		})
	default:
		c.JSON(503, gin.H{"error": "job queue full"})
	}
}