// performance-monitor.go
// Comprehensive performance monitoring and metrics collection system
// Real-time monitoring of go-llama, GPU acceleration, and TypeScript error processing

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// PerformanceMonitor manages comprehensive performance tracking
type PerformanceMonitor struct {
	mu                    sync.RWMutex
	startTime             time.Time
	collectors            map[string]MetricsCollector
	aggregatedMetrics     *AggregatedMetrics
	realtimeMetrics       *RealtimeMetrics
	historicalData        *HistoricalMetrics
	alertSystem           *AlertSystem
	reportGenerator       *ReportGenerator
	isRunning             bool
	updateTicker          *time.Ticker
	exportTicker          *time.Ticker
	done                  chan bool
}

// MetricsCollector interface for different metric collection strategies
type MetricsCollector interface {
	CollectMetrics() (*MetricsBatch, error)
	GetCollectorID() string
	GetCollectorType() string
	Reset()
}

// AggregatedMetrics contains all performance metrics
type AggregatedMetrics struct {
	SystemMetrics      *SystemMetrics      `json:"system_metrics"`
	LlamaMetrics       *LlamaMetrics       `json:"llama_metrics"`
	GPUMetrics         *GPUMetrics         `json:"gpu_metrics"`
	TypeScriptMetrics  *TypeScriptMetrics  `json:"typescript_metrics"`
	APIMetrics         *APIMetrics         `json:"api_metrics"`
	CacheMetrics       *CacheMetrics       `json:"cache_metrics"`
	WorkerPoolMetrics  *WorkerPoolMetrics  `json:"worker_pool_metrics"`
	LastUpdated        time.Time           `json:"last_updated"`
	CollectionDuration time.Duration       `json:"collection_duration"`
}

// RealtimeMetrics contains real-time performance data
type RealtimeMetrics struct {
	CurrentRPS           float64              `json:"current_rps"`
	AverageLatency       time.Duration        `json:"average_latency"`
	ErrorRate            float64              `json:"error_rate"`
	ActiveConnections    int                  `json:"active_connections"`
	QueueDepth           int                  `json:"queue_depth"`
	GPUUtilization       float64              `json:"gpu_utilization"`
	MemoryUtilization    float64              `json:"memory_utilization"`
	TokensPerSecond      float64              `json:"tokens_per_second"`
	FixesPerSecond       float64              `json:"fixes_per_second"`
	CacheHitRate         float64              `json:"cache_hit_rate"`
	Timestamp            time.Time            `json:"timestamp"`
	Alerts               []PerformanceAlert   `json:"alerts"`
}

// SystemMetrics tracks overall system performance
type SystemMetrics struct {
	CPUUsagePercent      float64   `json:"cpu_usage_percent"`
	MemoryUsageMB        int64     `json:"memory_usage_mb"`
	MemoryTotalMB        int64     `json:"memory_total_mb"`
	GoroutineCount       int       `json:"goroutine_count"`
	GCPauseTime          time.Duration `json:"gc_pause_time"`
	HeapAllocMB          float64   `json:"heap_alloc_mb"`
	HeapSysMB            float64   `json:"heap_sys_mb"`
	NumGC                uint32    `json:"num_gc"`
	LoadAverage          float64   `json:"load_average"`
	OpenFiles            int       `json:"open_files"`
	NetworkConnections   int       `json:"network_connections"`
	DiskIOReadMB         float64   `json:"disk_io_read_mb"`
	DiskIOWriteMB        float64   `json:"disk_io_write_mb"`
}

// LlamaMetrics tracks go-llama engine performance
type LlamaMetrics struct {
	IsLoaded             bool          `json:"is_loaded"`
	ModelName            string        `json:"model_name"`
	ModelSizeMB          int64         `json:"model_size_mb"`
	TotalInferences      int64         `json:"total_inferences"`
	SuccessfulInferences int64         `json:"successful_inferences"`
	FailedInferences     int64         `json:"failed_inferences"`
	AverageInferenceTime time.Duration `json:"average_inference_time"`
	TokensGenerated      int64         `json:"tokens_generated"`
	TokensPerSecond      float64       `json:"tokens_per_second"`
	ContextSize          int           `json:"context_size"`
	BatchSize            int           `json:"batch_size"`
	GPULayers            int           `json:"gpu_layers"`
	MemoryUsageMB        int64         `json:"memory_usage_mb"`
	QueueLength          int           `json:"queue_length"`
	WorkerUtilization    float64       `json:"worker_utilization"`
}

// GPUMetrics tracks GPU acceleration performance
type GPUMetrics struct {
	IsAvailable          bool          `json:"is_available"`
	DeviceName           string        `json:"device_name"`
	TotalMemoryMB        int64         `json:"total_memory_mb"`
	UsedMemoryMB         int64         `json:"used_memory_mb"`
	FreeMemoryMB         int64         `json:"free_memory_mb"`
	MemoryUtilization    float64       `json:"memory_utilization"`
	GPUUtilization       float64       `json:"gpu_utilization"`
	Temperature          int           `json:"temperature"`
	PowerUsage           float64       `json:"power_usage"`
	ClockSpeed           int           `json:"clock_speed"`
	CUDAKernelLaunches   int64         `json:"cuda_kernel_launches"`
	CUDAErrors           int64         `json:"cuda_errors"`
	TotalJobs            int64         `json:"total_jobs"`
	CompletedJobs        int64         `json:"completed_jobs"`
	FailedJobs           int64         `json:"failed_jobs"`
	AverageJobTime       time.Duration `json:"average_job_time"`
	ThroughputPerSec     float64       `json:"throughput_per_sec"`
}

// TypeScriptMetrics tracks TypeScript error processing
type TypeScriptMetrics struct {
	TotalErrors          int64         `json:"total_errors"`
	FixedErrors          int64         `json:"fixed_errors"`
	FailedFixes          int64         `json:"failed_fixes"`
	FixSuccessRate       float64       `json:"fix_success_rate"`
	AverageFixTime       time.Duration `json:"average_fix_time"`
	TemplateMatches      int64         `json:"template_matches"`
	LlamaInferences      int64         `json:"llama_inferences"`
	GPUAccelerations     int64         `json:"gpu_accelerations"`
	CacheHits            int64         `json:"cache_hits"`
	CacheMisses          int64         `json:"cache_misses"`
	CacheHitRate         float64       `json:"cache_hit_rate"`
	ErrorTypeDistribution map[string]int64 `json:"error_type_distribution"`
	ConfidenceDistribution map[string]int64 `json:"confidence_distribution"`
	ProcessingStrategies map[string]int64 `json:"processing_strategies"`
}

// APIMetrics tracks API endpoint performance
type APIMetrics struct {
	TotalRequests        int64                    `json:"total_requests"`
	SuccessfulRequests   int64                    `json:"successful_requests"`
	FailedRequests       int64                    `json:"failed_requests"`
	AverageResponseTime  time.Duration            `json:"average_response_time"`
	MedianResponseTime   time.Duration            `json:"median_response_time"`
	P95ResponseTime      time.Duration            `json:"p95_response_time"`
	P99ResponseTime      time.Duration            `json:"p99_response_time"`
	RequestsPerSecond    float64                  `json:"requests_per_second"`
	ErrorRate            float64                  `json:"error_rate"`
	ActiveConnections    int                      `json:"active_connections"`
	EndpointMetrics      map[string]*EndpointMetric `json:"endpoint_metrics"`
	StatusCodeDistribution map[int]int64          `json:"status_code_distribution"`
}

// EndpointMetric tracks individual endpoint performance
type EndpointMetric struct {
	Path                 string        `json:"path"`
	Method               string        `json:"method"`
	TotalRequests        int64         `json:"total_requests"`
	AverageResponseTime  time.Duration `json:"average_response_time"`
	ErrorRate            float64       `json:"error_rate"`
	LastAccessed         time.Time     `json:"last_accessed"`
}

// CacheMetrics tracks caching system performance
type CacheMetrics struct {
	TotalEntries         int64         `json:"total_entries"`
	HitCount             int64         `json:"hit_count"`
	MissCount            int64         `json:"miss_count"`
	HitRate              float64       `json:"hit_rate"`
	EvictionCount        int64         `json:"eviction_count"`
	MemoryUsageMB        float64       `json:"memory_usage_mb"`
	AverageAccessTime    time.Duration `json:"average_access_time"`
	MostAccessedKeys     []string      `json:"most_accessed_keys"`
	RecentlyAddedKeys    []string      `json:"recently_added_keys"`
	ExpirationCount      int64         `json:"expiration_count"`
}

// WorkerPoolMetrics tracks worker pool performance
type WorkerPoolMetrics struct {
	TotalPools           int                        `json:"total_pools"`
	ActiveWorkers        int                        `json:"active_workers"`
	IdleWorkers          int                        `json:"idle_workers"`
	TotalJobs            int64                      `json:"total_jobs"`
	CompletedJobs        int64                      `json:"completed_jobs"`
	FailedJobs           int64                      `json:"failed_jobs"`
	QueueLength          int                        `json:"queue_length"`
	AverageWaitTime      time.Duration              `json:"average_wait_time"`
	AverageProcessTime   time.Duration              `json:"average_process_time"`
	WorkerUtilization    float64                    `json:"worker_utilization"`
	PoolMetrics          map[string]*PoolMetric     `json:"pool_metrics"`
}

// PoolMetric tracks individual pool performance
type PoolMetric struct {
	PoolID               string        `json:"pool_id"`
	MaxWorkers           int           `json:"max_workers"`
	ActiveWorkers        int           `json:"active_workers"`
	CompletedJobs        int64         `json:"completed_jobs"`
	AverageJobTime       time.Duration `json:"average_job_time"`
	QueueLength          int           `json:"queue_length"`
}

// HistoricalMetrics stores time-series performance data
type HistoricalMetrics struct {
	mu               sync.RWMutex
	DataPoints       []*HistoricalDataPoint `json:"data_points"`
	MaxDataPoints    int                    `json:"max_data_points"`
	RetentionPeriod  time.Duration          `json:"retention_period"`
	AggregationWindow time.Duration         `json:"aggregation_window"`
}

// HistoricalDataPoint represents a single metrics snapshot
type HistoricalDataPoint struct {
	Timestamp         time.Time               `json:"timestamp"`
	AggregatedMetrics *AggregatedMetrics      `json:"aggregated_metrics"`
	Summary           *MetricsSummary         `json:"summary"`
}

// MetricsSummary provides high-level performance summary
type MetricsSummary struct {
	TotalRequests     int64         `json:"total_requests"`
	AverageLatency    time.Duration `json:"average_latency"`
	ErrorRate         float64       `json:"error_rate"`
	ThroughputRPS     float64       `json:"throughput_rps"`
	GPUUtilization    float64       `json:"gpu_utilization"`
	MemoryUtilization float64       `json:"memory_utilization"`
	FixSuccessRate    float64       `json:"fix_success_rate"`
	CacheHitRate      float64       `json:"cache_hit_rate"`
}

// AlertSystem manages performance alerts and notifications
type AlertSystem struct {
	mu             sync.RWMutex
	alerts         []*PerformanceAlert
	thresholds     *AlertThresholds
	lastAlertTime  map[string]time.Time
	cooldownPeriod time.Duration
}

// PerformanceAlert represents a performance alert
type PerformanceAlert struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Severity    string                 `json:"severity"`
	Message     string                 `json:"message"`
	Metric      string                 `json:"metric"`
	Value       float64                `json:"value"`
	Threshold   float64                `json:"threshold"`
	Timestamp   time.Time              `json:"timestamp"`
	Resolved    bool                   `json:"resolved"`
	ResolvedAt  *time.Time             `json:"resolved_at,omitempty"`
	Context     map[string]interface{} `json:"context"`
}

// AlertThresholds defines performance alert thresholds
type AlertThresholds struct {
	MaxLatencyMs         int     `json:"max_latency_ms"`
	MaxErrorRate         float64 `json:"max_error_rate"`
	MaxGPUUtilization    float64 `json:"max_gpu_utilization"`
	MaxMemoryUtilization float64 `json:"max_memory_utilization"`
	MinCacheHitRate      float64 `json:"min_cache_hit_rate"`
	MaxQueueDepth        int     `json:"max_queue_depth"`
	MaxGPUTemperature    int     `json:"max_gpu_temperature"`
	MinFixSuccessRate    float64 `json:"min_fix_success_rate"`
}

// ReportGenerator generates performance reports
type ReportGenerator struct {
	mu                   sync.RWMutex
	reportTemplates      map[string]*ReportTemplate
	scheduledReports     map[string]*ScheduledReport
	lastReportTime       map[string]time.Time
}

// ReportTemplate defines report generation template
type ReportTemplate struct {
	ID              string        `json:"id"`
	Name            string        `json:"name"`
	Description     string        `json:"description"`
	Metrics         []string      `json:"metrics"`
	Format          string        `json:"format"`
	AggregationPeriod time.Duration `json:"aggregation_period"`
	Filters         map[string]interface{} `json:"filters"`
}

// ScheduledReport represents a scheduled performance report
type ScheduledReport struct {
	ID              string        `json:"id"`
	TemplateID      string        `json:"template_id"`
	Schedule        string        `json:"schedule"`
	NextRun         time.Time     `json:"next_run"`
	LastRun         *time.Time    `json:"last_run,omitempty"`
	IsEnabled       bool          `json:"is_enabled"`
	Recipients      []string      `json:"recipients"`
}

// MetricsBatch contains a batch of collected metrics
type MetricsBatch struct {
	CollectorID   string                 `json:"collector_id"`
	CollectorType string                 `json:"collector_type"`
	Timestamp     time.Time              `json:"timestamp"`
	Metrics       map[string]interface{} `json:"metrics"`
	Duration      time.Duration          `json:"duration"`
}

// NewPerformanceMonitor creates a new performance monitor
func NewPerformanceMonitor() *PerformanceMonitor {
	log.Printf("🚀 Initializing Performance Monitor...")

	monitor := &PerformanceMonitor{
		startTime:         time.Now(),
		collectors:        make(map[string]MetricsCollector),
		aggregatedMetrics: &AggregatedMetrics{},
		realtimeMetrics:   &RealtimeMetrics{},
		historicalData: &HistoricalMetrics{
			DataPoints:        make([]*HistoricalDataPoint, 0),
			MaxDataPoints:     1000,
			RetentionPeriod:   24 * time.Hour,
			AggregationWindow: 1 * time.Minute,
		},
		alertSystem: &AlertSystem{
			alerts:         make([]*PerformanceAlert, 0),
			lastAlertTime:  make(map[string]time.Time),
			cooldownPeriod: 5 * time.Minute,
			thresholds: &AlertThresholds{
				MaxLatencyMs:         5000,  // 5 seconds
				MaxErrorRate:         0.05,  // 5%
				MaxGPUUtilization:    0.90,  // 90%
				MaxMemoryUtilization: 0.85,  // 85%
				MinCacheHitRate:      0.70,  // 70%
				MaxQueueDepth:        100,
				MaxGPUTemperature:    85,    // 85°C
				MinFixSuccessRate:    0.80,  // 80%
			},
		},
		reportGenerator: &ReportGenerator{
			reportTemplates:  make(map[string]*ReportTemplate),
			scheduledReports: make(map[string]*ScheduledReport),
			lastReportTime:   make(map[string]time.Time),
		},
		done: make(chan bool),
	}

	// Register default collectors
	monitor.RegisterCollector(NewSystemMetricsCollector())
	monitor.RegisterCollector(NewLlamaMetricsCollector())
	monitor.RegisterCollector(NewGPUMetricsCollector())
	monitor.RegisterCollector(NewTypeScriptMetricsCollector())
	monitor.RegisterCollector(NewAPIMetricsCollector())

	// Load default report templates
	monitor.loadDefaultReportTemplates()

	log.Printf("✅ Performance Monitor initialized with %d collectors", len(monitor.collectors))
	return monitor
}

// RegisterCollector registers a new metrics collector
func (pm *PerformanceMonitor) RegisterCollector(collector MetricsCollector) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	pm.collectors[collector.GetCollectorID()] = collector
	log.Printf("📊 Registered metrics collector: %s (%s)", collector.GetCollectorID(), collector.GetCollectorType())
}

// Start begins performance monitoring
func (pm *PerformanceMonitor) Start() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.isRunning {
		return fmt.Errorf("performance monitor already running")
	}

	pm.isRunning = true

	// Start metrics collection (every 5 seconds)
	pm.updateTicker = time.NewTicker(5 * time.Second)
	go pm.metricsCollectionLoop()

	// Start data export/archival (every 1 minute)
	pm.exportTicker = time.NewTicker(1 * time.Minute)
	go pm.dataExportLoop()

	// Start alert monitoring
	go pm.alertMonitoringLoop()

	// Start report generation
	go pm.reportGenerationLoop()

	log.Printf("🚀 Performance Monitor started successfully")
	return nil
}

// metricsCollectionLoop runs the main metrics collection loop
func (pm *PerformanceMonitor) metricsCollectionLoop() {
	for {
		select {
		case <-pm.updateTicker.C:
			pm.collectAllMetrics()
		case <-pm.done:
			return
		}
	}
}

// collectAllMetrics collects metrics from all registered collectors
func (pm *PerformanceMonitor) collectAllMetrics() {
	collectionStart := time.Now()

	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Collect from all collectors
	for id, collector := range pm.collectors {
		batch, err := collector.CollectMetrics()
		if err != nil {
			log.Printf("❌ Failed to collect metrics from %s: %v", id, err)
			continue
		}

		pm.processMetricsBatch(batch)
	}

	// Update aggregated metrics
	pm.updateAggregatedMetrics()

	// Update real-time metrics
	pm.updateRealtimeMetrics()

	// Check for alerts
	pm.checkAlerts()

	pm.aggregatedMetrics.CollectionDuration = time.Since(collectionStart)
	pm.aggregatedMetrics.LastUpdated = time.Now()
}

// processMetricsBatch processes a batch of metrics from a collector
func (pm *PerformanceMonitor) processMetricsBatch(batch *MetricsBatch) {
	switch batch.CollectorType {
	case "system":
		pm.processSystemMetrics(batch.Metrics)
	case "llama":
		pm.processLlamaMetrics(batch.Metrics)
	case "gpu":
		pm.processGPUMetrics(batch.Metrics)
	case "typescript":
		pm.processTypeScriptMetrics(batch.Metrics)
	case "api":
		pm.processAPIMetrics(batch.Metrics)
	}
}

// updateAggregatedMetrics updates the aggregated metrics structure
func (pm *PerformanceMonitor) updateAggregatedMetrics() {
	// This would aggregate metrics from all collectors
	// For now, we'll use placeholder values
	
	if pm.aggregatedMetrics.SystemMetrics == nil {
		pm.aggregatedMetrics.SystemMetrics = &SystemMetrics{}
	}
	if pm.aggregatedMetrics.LlamaMetrics == nil {
		pm.aggregatedMetrics.LlamaMetrics = &LlamaMetrics{}
	}
	if pm.aggregatedMetrics.GPUMetrics == nil {
		pm.aggregatedMetrics.GPUMetrics = &GPUMetrics{}
	}
	if pm.aggregatedMetrics.TypeScriptMetrics == nil {
		pm.aggregatedMetrics.TypeScriptMetrics = &TypeScriptMetrics{
			ErrorTypeDistribution:  make(map[string]int64),
			ConfidenceDistribution: make(map[string]int64),
			ProcessingStrategies:   make(map[string]int64),
		}
	}
	if pm.aggregatedMetrics.APIMetrics == nil {
		pm.aggregatedMetrics.APIMetrics = &APIMetrics{
			EndpointMetrics:         make(map[string]*EndpointMetric),
			StatusCodeDistribution:  make(map[int]int64),
		}
	}
	if pm.aggregatedMetrics.CacheMetrics == nil {
		pm.aggregatedMetrics.CacheMetrics = &CacheMetrics{}
	}
	if pm.aggregatedMetrics.WorkerPoolMetrics == nil {
		pm.aggregatedMetrics.WorkerPoolMetrics = &WorkerPoolMetrics{
			PoolMetrics: make(map[string]*PoolMetric),
		}
	}
}

// updateRealtimeMetrics updates real-time performance metrics
func (pm *PerformanceMonitor) updateRealtimeMetrics() {
	now := time.Now()
	
	// Calculate real-time values from aggregated metrics
	if pm.aggregatedMetrics.APIMetrics != nil {
		duration := now.Sub(pm.realtimeMetrics.Timestamp).Seconds()
		if duration > 0 {
			pm.realtimeMetrics.CurrentRPS = float64(pm.aggregatedMetrics.APIMetrics.TotalRequests) / duration
		}
		pm.realtimeMetrics.AverageLatency = pm.aggregatedMetrics.APIMetrics.AverageResponseTime
		pm.realtimeMetrics.ErrorRate = pm.aggregatedMetrics.APIMetrics.ErrorRate
		pm.realtimeMetrics.ActiveConnections = pm.aggregatedMetrics.APIMetrics.ActiveConnections
	}

	if pm.aggregatedMetrics.GPUMetrics != nil {
		pm.realtimeMetrics.GPUUtilization = pm.aggregatedMetrics.GPUMetrics.GPUUtilization
	}

	if pm.aggregatedMetrics.SystemMetrics != nil {
		pm.realtimeMetrics.MemoryUtilization = pm.aggregatedMetrics.SystemMetrics.CPUUsagePercent
	}

	if pm.aggregatedMetrics.LlamaMetrics != nil {
		pm.realtimeMetrics.TokensPerSecond = pm.aggregatedMetrics.LlamaMetrics.TokensPerSecond
	}

	if pm.aggregatedMetrics.TypeScriptMetrics != nil {
		pm.realtimeMetrics.FixesPerSecond = float64(pm.aggregatedMetrics.TypeScriptMetrics.FixedErrors) / time.Since(pm.startTime).Seconds()
		pm.realtimeMetrics.CacheHitRate = pm.aggregatedMetrics.TypeScriptMetrics.CacheHitRate
	}

	pm.realtimeMetrics.Timestamp = now
}

// checkAlerts checks for performance threshold violations
func (pm *PerformanceMonitor) checkAlerts() {
	thresholds := pm.alertSystem.thresholds

	// Check latency threshold
	if pm.realtimeMetrics.AverageLatency > time.Duration(thresholds.MaxLatencyMs)*time.Millisecond {
		pm.triggerAlert("high_latency", "warning", 
			fmt.Sprintf("Average latency (%v) exceeds threshold (%dms)", 
				pm.realtimeMetrics.AverageLatency, thresholds.MaxLatencyMs),
			"average_latency",
			float64(pm.realtimeMetrics.AverageLatency.Milliseconds()),
			float64(thresholds.MaxLatencyMs))
	}

	// Check error rate threshold
	if pm.realtimeMetrics.ErrorRate > thresholds.MaxErrorRate {
		pm.triggerAlert("high_error_rate", "critical",
			fmt.Sprintf("Error rate (%.2f%%) exceeds threshold (%.2f%%)",
				pm.realtimeMetrics.ErrorRate*100, thresholds.MaxErrorRate*100),
			"error_rate",
			pm.realtimeMetrics.ErrorRate,
			thresholds.MaxErrorRate)
	}

	// Check GPU utilization
	if pm.realtimeMetrics.GPUUtilization > thresholds.MaxGPUUtilization {
		pm.triggerAlert("high_gpu_utilization", "warning",
			fmt.Sprintf("GPU utilization (%.1f%%) exceeds threshold (%.1f%%)",
				pm.realtimeMetrics.GPUUtilization*100, thresholds.MaxGPUUtilization*100),
			"gpu_utilization",
			pm.realtimeMetrics.GPUUtilization,
			thresholds.MaxGPUUtilization)
	}

	// Check cache hit rate
	if pm.realtimeMetrics.CacheHitRate < thresholds.MinCacheHitRate {
		pm.triggerAlert("low_cache_hit_rate", "warning",
			fmt.Sprintf("Cache hit rate (%.1f%%) below threshold (%.1f%%)",
				pm.realtimeMetrics.CacheHitRate*100, thresholds.MinCacheHitRate*100),
			"cache_hit_rate",
			pm.realtimeMetrics.CacheHitRate,
			thresholds.MinCacheHitRate)
	}
}

// triggerAlert creates and registers a new performance alert
func (pm *PerformanceMonitor) triggerAlert(alertType, severity, message, metric string, value, threshold float64) {
	alertID := fmt.Sprintf("%s_%d", alertType, time.Now().Unix())
	
	// Check cooldown period
	if lastAlert, exists := pm.alertSystem.lastAlertTime[alertType]; exists {
		if time.Since(lastAlert) < pm.alertSystem.cooldownPeriod {
			return // Still in cooldown period
		}
	}

	alert := &PerformanceAlert{
		ID:        alertID,
		Type:      alertType,
		Severity:  severity,
		Message:   message,
		Metric:    metric,
		Value:     value,
		Threshold: threshold,
		Timestamp: time.Now(),
		Resolved:  false,
		Context: map[string]interface{}{
			"realtime_metrics": pm.realtimeMetrics,
		},
	}

	pm.alertSystem.mu.Lock()
	pm.alertSystem.alerts = append(pm.alertSystem.alerts, alert)
	pm.alertSystem.lastAlertTime[alertType] = time.Now()
	pm.alertSystem.mu.Unlock()

	// Add to real-time alerts
	pm.realtimeMetrics.Alerts = append(pm.realtimeMetrics.Alerts, *alert)

	log.Printf("🚨 Performance alert triggered: %s - %s", alertType, message)
}

// GetAggregatedMetrics returns current aggregated metrics
func (pm *PerformanceMonitor) GetAggregatedMetrics() *AggregatedMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Return copy of metrics
	metrics := *pm.aggregatedMetrics
	return &metrics
}

// GetRealtimeMetrics returns current real-time metrics
func (pm *PerformanceMonitor) GetRealtimeMetrics() *RealtimeMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Return copy of real-time metrics
	metrics := *pm.realtimeMetrics
	return &metrics
}

// GetHistoricalMetrics returns historical performance data
func (pm *PerformanceMonitor) GetHistoricalMetrics(from, to time.Time) []*HistoricalDataPoint {
	pm.historicalData.mu.RLock()
	defer pm.historicalData.mu.RUnlock()

	var filtered []*HistoricalDataPoint
	for _, point := range pm.historicalData.DataPoints {
		if point.Timestamp.After(from) && point.Timestamp.Before(to) {
			filtered = append(filtered, point)
		}
	}

	return filtered
}

// GetActiveAlerts returns currently active performance alerts
func (pm *PerformanceMonitor) GetActiveAlerts() []*PerformanceAlert {
	pm.alertSystem.mu.RLock()
	defer pm.alertSystem.mu.RUnlock()

	var active []*PerformanceAlert
	for _, alert := range pm.alertSystem.alerts {
		if !alert.Resolved {
			active = append(active, alert)
		}
	}

	return active
}

// ResolveAlert marks an alert as resolved
func (pm *PerformanceMonitor) ResolveAlert(alertID string) error {
	pm.alertSystem.mu.Lock()
	defer pm.alertSystem.mu.Unlock()

	for _, alert := range pm.alertSystem.alerts {
		if alert.ID == alertID && !alert.Resolved {
			alert.Resolved = true
			now := time.Now()
			alert.ResolvedAt = &now
			log.Printf("✅ Resolved performance alert: %s", alertID)
			return nil
		}
	}

	return fmt.Errorf("alert %s not found or already resolved", alertID)
}

// dataExportLoop handles periodic data export and cleanup
func (pm *PerformanceMonitor) dataExportLoop() {
	for {
		select {
		case <-pm.exportTicker.C:
			pm.exportHistoricalData()
			pm.cleanupOldData()
		case <-pm.done:
			return
		}
	}
}

// exportHistoricalData exports current metrics to historical storage
func (pm *PerformanceMonitor) exportHistoricalData() {
	pm.mu.RLock()
	metrics := *pm.aggregatedMetrics
	pm.mu.RUnlock()

	// Create historical data point
	dataPoint := &HistoricalDataPoint{
		Timestamp:         time.Now(),
		AggregatedMetrics: &metrics,
		Summary: &MetricsSummary{
			TotalRequests:     metrics.APIMetrics.TotalRequests,
			AverageLatency:    metrics.APIMetrics.AverageResponseTime,
			ErrorRate:         metrics.APIMetrics.ErrorRate,
			ThroughputRPS:     metrics.APIMetrics.RequestsPerSecond,
			GPUUtilization:    metrics.GPUMetrics.GPUUtilization,
			MemoryUtilization: metrics.SystemMetrics.CPUUsagePercent,
			FixSuccessRate:    metrics.TypeScriptMetrics.FixSuccessRate,
			CacheHitRate:      metrics.TypeScriptMetrics.CacheHitRate,
		},
	}

	// Add to historical data
	pm.historicalData.mu.Lock()
	pm.historicalData.DataPoints = append(pm.historicalData.DataPoints, dataPoint)
	
	// Limit data points
	if len(pm.historicalData.DataPoints) > pm.historicalData.MaxDataPoints {
		pm.historicalData.DataPoints = pm.historicalData.DataPoints[1:]
	}
	pm.historicalData.mu.Unlock()
}

// cleanupOldData removes old historical data points
func (pm *PerformanceMonitor) cleanupOldData() {
	cutoff := time.Now().Add(-pm.historicalData.RetentionPeriod)

	pm.historicalData.mu.Lock()
	defer pm.historicalData.mu.Unlock()

	var filtered []*HistoricalDataPoint
	for _, point := range pm.historicalData.DataPoints {
		if point.Timestamp.After(cutoff) {
			filtered = append(filtered, point)
		}
	}

	if len(filtered) < len(pm.historicalData.DataPoints) {
		pm.historicalData.DataPoints = filtered
		log.Printf("🧹 Cleaned up %d old historical data points", 
			len(pm.historicalData.DataPoints)-len(filtered))
	}
}

// alertMonitoringLoop runs the alert monitoring system
func (pm *PerformanceMonitor) alertMonitoringLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.processAlerts()
		case <-pm.done:
			return
		}
	}
}

// processAlerts processes and potentially auto-resolves alerts
func (pm *PerformanceMonitor) processAlerts() {
	pm.alertSystem.mu.Lock()
	defer pm.alertSystem.mu.Unlock()

	for _, alert := range pm.alertSystem.alerts {
		if !alert.Resolved {
			// Check if alert condition is no longer met
			if pm.shouldAutoResolveAlert(alert) {
				alert.Resolved = true
				now := time.Now()
				alert.ResolvedAt = &now
				log.Printf("✅ Auto-resolved alert: %s", alert.ID)
			}
		}
	}
}

// shouldAutoResolveAlert checks if an alert should be auto-resolved
func (pm *PerformanceMonitor) shouldAutoResolveAlert(alert *PerformanceAlert) bool {
	switch alert.Type {
	case "high_latency":
		return pm.realtimeMetrics.AverageLatency < time.Duration(alert.Threshold)*time.Millisecond
	case "high_error_rate":
		return pm.realtimeMetrics.ErrorRate < alert.Threshold
	case "high_gpu_utilization":
		return pm.realtimeMetrics.GPUUtilization < alert.Threshold
	case "low_cache_hit_rate":
		return pm.realtimeMetrics.CacheHitRate > alert.Threshold
	}
	return false
}

// reportGenerationLoop handles scheduled report generation
func (pm *PerformanceMonitor) reportGenerationLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.checkScheduledReports()
		case <-pm.done:
			return
		}
	}
}

// checkScheduledReports checks and generates scheduled reports
func (pm *PerformanceMonitor) checkScheduledReports() {
	pm.reportGenerator.mu.RLock()
	defer pm.reportGenerator.mu.RUnlock()

	now := time.Now()
	for _, report := range pm.reportGenerator.scheduledReports {
		if report.IsEnabled && now.After(report.NextRun) {
			go pm.generateScheduledReport(report)
		}
	}
}

// generateScheduledReport generates a scheduled performance report
func (pm *PerformanceMonitor) generateScheduledReport(report *ScheduledReport) {
	log.Printf("📊 Generating scheduled report: %s", report.ID)
	
	// This would generate the actual report
	// For now, just update the schedule
	pm.reportGenerator.mu.Lock()
	now := time.Now()
	report.LastRun = &now
	// Calculate next run based on schedule (simplified)
	report.NextRun = now.Add(24 * time.Hour) // Daily by default
	pm.reportGenerator.mu.Unlock()
}

// Placeholder methods for processing different metric types
func (pm *PerformanceMonitor) processSystemMetrics(metrics map[string]interface{}) {
	// Process system metrics
}

func (pm *PerformanceMonitor) processLlamaMetrics(metrics map[string]interface{}) {
	// Process Llama metrics
}

func (pm *PerformanceMonitor) processGPUMetrics(metrics map[string]interface{}) {
	// Process GPU metrics
}

func (pm *PerformanceMonitor) processTypeScriptMetrics(metrics map[string]interface{}) {
	// Process TypeScript metrics
}

func (pm *PerformanceMonitor) processAPIMetrics(metrics map[string]interface{}) {
	// Process API metrics
}

// loadDefaultReportTemplates loads default report templates
func (pm *PerformanceMonitor) loadDefaultReportTemplates() {
	templates := []*ReportTemplate{
		{
			ID:          "daily_summary",
			Name:        "Daily Performance Summary",
			Description: "Daily performance overview with key metrics",
			Metrics:     []string{"throughput", "latency", "error_rate", "gpu_utilization"},
			Format:      "json",
			AggregationPeriod: 24 * time.Hour,
		},
		{
			ID:          "typescript_analysis",
			Name:        "TypeScript Error Analysis",
			Description: "Analysis of TypeScript error processing performance",
			Metrics:     []string{"fix_success_rate", "processing_time", "error_types"},
			Format:      "json",
			AggregationPeriod: time.Hour,
		},
	}

	pm.reportGenerator.mu.Lock()
	for _, template := range templates {
		pm.reportGenerator.reportTemplates[template.ID] = template
	}
	pm.reportGenerator.mu.Unlock()
}

// Stop stops the performance monitor
func (pm *PerformanceMonitor) Stop() error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if !pm.isRunning {
		return fmt.Errorf("performance monitor not running")
	}

	log.Printf("🛑 Stopping Performance Monitor...")

	// Stop tickers
	if pm.updateTicker != nil {
		pm.updateTicker.Stop()
	}
	if pm.exportTicker != nil {
		pm.exportTicker.Stop()
	}

	// Signal goroutines to stop
	close(pm.done)

	pm.isRunning = false

	log.Printf("✅ Performance Monitor stopped successfully")
	return nil
}

// Placeholder collector implementations
type SystemMetricsCollector struct{}

func NewSystemMetricsCollector() *SystemMetricsCollector {
	return &SystemMetricsCollector{}
}

func (smc *SystemMetricsCollector) CollectMetrics() (*MetricsBatch, error) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	metrics := map[string]interface{}{
		"goroutine_count": runtime.NumGoroutine(),
		"heap_alloc_mb":   float64(m.Alloc) / 1024 / 1024,
		"heap_sys_mb":     float64(m.Sys) / 1024 / 1024,
		"num_gc":          m.NumGC,
		"gc_pause_time":   time.Duration(m.PauseNs[(m.NumGC+255)%256]),
	}

	return &MetricsBatch{
		CollectorID:   "system",
		CollectorType: "system",
		Timestamp:     time.Now(),
		Metrics:       metrics,
	}, nil
}

func (smc *SystemMetricsCollector) GetCollectorID() string   { return "system" }
func (smc *SystemMetricsCollector) GetCollectorType() string { return "system" }
func (smc *SystemMetricsCollector) Reset()                  {}

// Additional placeholder collectors
type LlamaMetricsCollector struct{}

func NewLlamaMetricsCollector() *LlamaMetricsCollector     { return &LlamaMetricsCollector{} }
func (lmc *LlamaMetricsCollector) CollectMetrics() (*MetricsBatch, error) {
	return &MetricsBatch{CollectorID: "llama", CollectorType: "llama", Timestamp: time.Now(), Metrics: make(map[string]interface{})}, nil
}
func (lmc *LlamaMetricsCollector) GetCollectorID() string   { return "llama" }
func (lmc *LlamaMetricsCollector) GetCollectorType() string { return "llama" }
func (lmc *LlamaMetricsCollector) Reset()                   {}

type GPUMetricsCollector struct{}

func NewGPUMetricsCollector() *GPUMetricsCollector     { return &GPUMetricsCollector{} }
func (gmc *GPUMetricsCollector) CollectMetrics() (*MetricsBatch, error) {
	return &MetricsBatch{CollectorID: "gpu", CollectorType: "gpu", Timestamp: time.Now(), Metrics: make(map[string]interface{})}, nil
}
func (gmc *GPUMetricsCollector) GetCollectorID() string   { return "gpu" }
func (gmc *GPUMetricsCollector) GetCollectorType() string { return "gpu" }
func (gmc *GPUMetricsCollector) Reset()                   {}

type TypeScriptMetricsCollector struct{}

func NewTypeScriptMetricsCollector() *TypeScriptMetricsCollector { return &TypeScriptMetricsCollector{} }
func (tmc *TypeScriptMetricsCollector) CollectMetrics() (*MetricsBatch, error) {
	return &MetricsBatch{CollectorID: "typescript", CollectorType: "typescript", Timestamp: time.Now(), Metrics: make(map[string]interface{})}, nil
}
func (tmc *TypeScriptMetricsCollector) GetCollectorID() string   { return "typescript" }
func (tmc *TypeScriptMetricsCollector) GetCollectorType() string { return "typescript" }
func (tmc *TypeScriptMetricsCollector) Reset()                   {}

type APIMetricsCollector struct{}

func NewAPIMetricsCollector() *APIMetricsCollector     { return &APIMetricsCollector{} }
func (amc *APIMetricsCollector) CollectMetrics() (*MetricsBatch, error) {
	return &MetricsBatch{CollectorID: "api", CollectorType: "api", Timestamp: time.Now(), Metrics: make(map[string]interface{})}, nil
}
func (amc *APIMetricsCollector) GetCollectorID() string   { return "api" }
func (amc *APIMetricsCollector) GetCollectorType() string { return "api" }
func (amc *APIMetricsCollector) Reset()                   {}