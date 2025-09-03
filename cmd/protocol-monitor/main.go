// Protocol Performance Monitor
// Real-time monitoring and metrics collection for multi-protocol communication
// Provides detailed insights into protocol performance and health

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

// Performance Metrics Structure
type ProtocolMetrics struct {
	Protocol           string            `json:"protocol"`
	RequestCount       int64             `json:"request_count"`
	SuccessCount       int64             `json:"success_count"`
	ErrorCount         int64             `json:"error_count"`
	SuccessRate        float64           `json:"success_rate"`
	AverageLatency     time.Duration     `json:"average_latency"`
	MinLatency         time.Duration     `json:"min_latency"`
	MaxLatency         time.Duration     `json:"max_latency"`
	P50Latency         time.Duration     `json:"p50_latency"`
	P95Latency         time.Duration     `json:"p95_latency"`
	P99Latency         time.Duration     `json:"p99_latency"`
	ThroughputRPS      float64           `json:"throughput_rps"`
	BytesTransferred   int64             `json:"bytes_transferred"`
	ActiveConnections  int64             `json:"active_connections"`
	CircuitBreakerTrips int64            `json:"circuit_breaker_trips"`
	FallbackCount      int64             `json:"fallback_count"`
	LastUpdated        time.Time         `json:"last_updated"`
	LatencyHistogram   []LatencyBucket   `json:"latency_histogram"`
	ErrorTypes         map[string]int64  `json:"error_types"`
}

type LatencyBucket struct {
	UpperBound time.Duration `json:"upper_bound"`
	Count      int64         `json:"count"`
}

// Service Performance Metrics
type ServiceMetrics struct {
	ServiceName        string                       `json:"service_name"`
	ProtocolMetrics    map[string]*ProtocolMetrics  `json:"protocol_metrics"`
	TotalRequests      int64                        `json:"total_requests"`
	TotalSuccesses     int64                        `json:"total_successes"`
	TotalErrors        int64                        `json:"total_errors"`
	OverallSuccessRate float64                      `json:"overall_success_rate"`
	PreferredProtocol  string                       `json:"preferred_protocol"`
	HealthScore        float64                      `json:"health_score"`
	LastUpdated        time.Time                    `json:"last_updated"`
}

// Performance Monitor
type ProtocolPerformanceMonitor struct {
	redis              *redis.Client
	ctx                context.Context
	metrics            sync.Map // map[string]*ServiceMetrics
	protocolMetrics    sync.Map // map[string]*ProtocolMetrics  
	latencyCollector   *LatencyCollector
	metricsAggregator  *MetricsAggregator
	alertManager       *AlertManager
	config             *MonitorConfig
	shutdown           chan struct{}
	httpServer         *http.Server
}

type MonitorConfig struct {
	RedisAddr          string        `json:"redis_addr"`
	HTTPPort           int           `json:"http_port"`
	MetricsInterval    time.Duration `json:"metrics_interval"`
	RetentionPeriod    time.Duration `json:"retention_period"`
	AlertThresholds    AlertConfig   `json:"alert_thresholds"`
	EnableAlerts       bool          `json:"enable_alerts"`
	EnableMetricsPush  bool          `json:"enable_metrics_push"`
}

type AlertConfig struct {
	ErrorRateThreshold       float64       `json:"error_rate_threshold"`        // 0.05 = 5%
	LatencyThreshold         time.Duration `json:"latency_threshold"`           // 1s
	CircuitBreakerThreshold  int64         `json:"circuit_breaker_threshold"`   // 10 trips
	FallbackRateThreshold    float64       `json:"fallback_rate_threshold"`     // 0.1 = 10%
	HealthScoreThreshold     float64       `json:"health_score_threshold"`      // 0.8 = 80%
}

// Latency Collector for histogram data
type LatencyCollector struct {
	buckets    []time.Duration
	data       sync.Map // map[string][]int64 (protocol -> bucket counts)
	sampleRate float64  // 0.1 = 10% sampling
}

// Metrics Aggregator for time-series data
type MetricsAggregator struct {
	windowSize     time.Duration
	windows        sync.Map // map[string]*TimeWindow
	aggregateFunc  func([]float64) float64
}

type TimeWindow struct {
	StartTime time.Time     `json:"start_time"`
	EndTime   time.Time     `json:"end_time"`
	Values    []float64     `json:"values"`
	Count     int64         `json:"count"`
	mutex     sync.RWMutex
}

// Alert Manager
type AlertManager struct {
	alerts       sync.Map // map[string]*Alert
	webhookURL   string
	alertChannel chan Alert
	config       AlertConfig
}

type Alert struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Severity    string                 `json:"severity"`
	Service     string                 `json:"service"`
	Protocol    string                 `json:"protocol"`
	Message     string                 `json:"message"`
	Threshold   interface{}           `json:"threshold"`
	CurrentValue interface{}          `json:"current_value"`
	Timestamp   time.Time             `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Performance Event for real-time updates
type PerformanceEvent struct {
	Type        string                 `json:"type"`
	Service     string                 `json:"service"`
	Protocol    string                 `json:"protocol"`
	Latency     time.Duration          `json:"latency"`
	Success     bool                   `json:"success"`
	ErrorType   string                 `json:"error_type,omitempty"`
	BytesIn     int64                  `json:"bytes_in"`
	BytesOut    int64                  `json:"bytes_out"`
	Timestamp   time.Time              `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// Initialize Performance Monitor
func NewProtocolPerformanceMonitor() (*ProtocolPerformanceMonitor, error) {
	config := &MonitorConfig{
		RedisAddr:       getEnv("REDIS_ADDR", "localhost:6379"),
		HTTPPort:        getEnvInt("MONITOR_HTTP_PORT", 8240),
		MetricsInterval: time.Duration(getEnvInt("METRICS_INTERVAL", 10)) * time.Second,
		RetentionPeriod: time.Duration(getEnvInt("RETENTION_PERIOD", 24)) * time.Hour,
		EnableAlerts:    getEnvBool("ENABLE_ALERTS", true),
		EnableMetricsPush: getEnvBool("ENABLE_METRICS_PUSH", true),
		AlertThresholds: AlertConfig{
			ErrorRateThreshold:      getEnvFloat("ERROR_RATE_THRESHOLD", 0.05),
			LatencyThreshold:        time.Duration(getEnvInt("LATENCY_THRESHOLD", 1000)) * time.Millisecond,
			CircuitBreakerThreshold: int64(getEnvInt("CIRCUIT_BREAKER_THRESHOLD", 10)),
			FallbackRateThreshold:   getEnvFloat("FALLBACK_RATE_THRESHOLD", 0.1),
			HealthScoreThreshold:    getEnvFloat("HEALTH_SCORE_THRESHOLD", 0.8),
		},
	}

	ctx := context.Background()
	
	// Initialize Redis connection
	rdb := redis.NewClient(&redis.Options{
		Addr: config.RedisAddr,
		DB:   3, // Use DB 3 for metrics
	})

	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	monitor := &ProtocolPerformanceMonitor{
		redis:    rdb,
		ctx:      ctx,
		config:   config,
		shutdown: make(chan struct{}),
	}

	// Initialize components
	monitor.latencyCollector = &LatencyCollector{
		buckets: []time.Duration{
			1 * time.Millisecond,
			5 * time.Millisecond,
			10 * time.Millisecond,
			25 * time.Millisecond,
			50 * time.Millisecond,
			100 * time.Millisecond,
			250 * time.Millisecond,
			500 * time.Millisecond,
			1 * time.Second,
			2 * time.Second,
			5 * time.Second,
		},
		sampleRate: 0.1, // 10% sampling
	}

	monitor.metricsAggregator = &MetricsAggregator{
		windowSize:    1 * time.Minute,
		aggregateFunc: calculateP95,
	}

	monitor.alertManager = &AlertManager{
		webhookURL:   getEnv("ALERT_WEBHOOK_URL", ""),
		alertChannel: make(chan Alert, 100),
		config:       config.AlertThresholds,
	}

	log.Printf("🔍 Protocol Performance Monitor initialized")
	log.Printf("📊 Metrics HTTP Port: %d", config.HTTPPort)
	log.Printf("⏱️ Metrics Interval: %s", config.MetricsInterval)
	log.Printf("🚨 Alerts Enabled: %v", config.EnableAlerts)

	return monitor, nil
}

// Start Performance Monitor
func (ppm *ProtocolPerformanceMonitor) Start() error {
	// Start metrics collection
	go ppm.startMetricsCollection()
	
	// Start alert processing
	if ppm.config.EnableAlerts {
		go ppm.startAlertProcessing()
	}
	
	// Start HTTP metrics server
	go ppm.startHTTPServer()
	
	// Start Redis metrics subscriber
	go ppm.startRedisSubscriber()

	log.Printf("🚀 Protocol Performance Monitor started")
	
	// Wait for shutdown
	<-ppm.shutdown
	return nil
}

// Record Performance Event
func (ppm *ProtocolPerformanceMonitor) RecordEvent(event PerformanceEvent) {
	// Update protocol metrics
	ppm.updateProtocolMetrics(event)
	
	// Update service metrics
	ppm.updateServiceMetrics(event)
	
	// Collect latency data for histogram
	if event.Latency > 0 {
		ppm.latencyCollector.collectLatency(event.Protocol, event.Latency)
	}
	
	// Check for alerts
	if ppm.config.EnableAlerts {
		ppm.checkForAlerts(event)
	}
	
	// Push to Redis for persistence
	if ppm.config.EnableMetricsPush {
		ppm.pushToRedis(event)
	}
}

// Update Protocol Metrics
func (ppm *ProtocolPerformanceMonitor) updateProtocolMetrics(event PerformanceEvent) {
	key := event.Protocol
	
	var metrics *ProtocolMetrics
	if val, exists := ppm.protocolMetrics.Load(key); exists {
		metrics = val.(*ProtocolMetrics)
	} else {
		metrics = &ProtocolMetrics{
			Protocol:         event.Protocol,
			MinLatency:       time.Duration(^uint64(0) >> 1), // Max duration
			ErrorTypes:       make(map[string]int64),
			LatencyHistogram: make([]LatencyBucket, len(ppm.latencyCollector.buckets)),
		}
		ppm.protocolMetrics.Store(key, metrics)
	}

	// Update counters
	atomic.AddInt64(&metrics.RequestCount, 1)
	atomic.AddInt64(&metrics.BytesTransferred, event.BytesIn+event.BytesOut)
	
	if event.Success {
		atomic.AddInt64(&metrics.SuccessCount, 1)
	} else {
		atomic.AddInt64(&metrics.ErrorCount, 1)
		if event.ErrorType != "" {
			metrics.ErrorTypes[event.ErrorType]++
		}
	}

	// Update latency metrics
	if event.Latency > 0 {
		// Update min/max latency
		for {
			current := time.Duration(atomic.LoadInt64((*int64)(&metrics.MinLatency)))
			if event.Latency >= current || atomic.CompareAndSwapInt64((*int64)(&metrics.MinLatency), int64(current), int64(event.Latency)) {
				break
			}
		}
		
		for {
			current := time.Duration(atomic.LoadInt64((*int64)(&metrics.MaxLatency)))
			if event.Latency <= current || atomic.CompareAndSwapInt64((*int64)(&metrics.MaxLatency), int64(current), int64(event.Latency)) {
				break
			}
		}

		// Update average latency (simple moving average)
		currentAvg := time.Duration(atomic.LoadInt64((*int64)(&metrics.AverageLatency)))
		requestCount := atomic.LoadInt64(&metrics.RequestCount)
		newAvg := time.Duration((int64(currentAvg)*(requestCount-1) + int64(event.Latency)) / requestCount)
		atomic.StoreInt64((*int64)(&metrics.AverageLatency), int64(newAvg))
	}

	// Calculate success rate
	successCount := atomic.LoadInt64(&metrics.SuccessCount)
	requestCount := atomic.LoadInt64(&metrics.RequestCount)
	if requestCount > 0 {
		metrics.SuccessRate = float64(successCount) / float64(requestCount)
	}

	// Calculate throughput (requests per second)
	// This is simplified - in production you'd use a sliding window
	if metrics.LastUpdated.IsZero() {
		metrics.LastUpdated = event.Timestamp
	} else {
		elapsed := event.Timestamp.Sub(metrics.LastUpdated).Seconds()
		if elapsed > 0 {
			metrics.ThroughputRPS = float64(requestCount) / elapsed
		}
	}
	
	metrics.LastUpdated = event.Timestamp
}

// Update Service Metrics
func (ppm *ProtocolPerformanceMonitor) updateServiceMetrics(event PerformanceEvent) {
	key := event.Service
	
	var metrics *ServiceMetrics
	if val, exists := ppm.metrics.Load(key); exists {
		metrics = val.(*ServiceMetrics)
	} else {
		metrics = &ServiceMetrics{
			ServiceName:     event.Service,
			ProtocolMetrics: make(map[string]*ProtocolMetrics),
		}
		ppm.metrics.Store(key, metrics)
	}

	// Update total counters
	atomic.AddInt64(&metrics.TotalRequests, 1)
	
	if event.Success {
		atomic.AddInt64(&metrics.TotalSuccesses, 1)
	} else {
		atomic.AddInt64(&metrics.TotalErrors, 1)
	}

	// Calculate overall success rate
	totalRequests := atomic.LoadInt64(&metrics.TotalRequests)
	totalSuccesses := atomic.LoadInt64(&metrics.TotalSuccesses)
	if totalRequests > 0 {
		metrics.OverallSuccessRate = float64(totalSuccesses) / float64(totalRequests)
	}

	// Calculate health score (composite of success rate, latency, and availability)
	if protocolMetrics, exists := ppm.protocolMetrics.Load(event.Protocol); exists {
		pm := protocolMetrics.(*ProtocolMetrics)
		latencyScore := calculateLatencyScore(pm.AverageLatency)
		metrics.HealthScore = (metrics.OverallSuccessRate * 0.5) + (latencyScore * 0.3) + (0.2) // 0.2 for availability assumption
	}

	metrics.LastUpdated = event.Timestamp
}

// Latency Collection for Histogram
func (lc *LatencyCollector) collectLatency(protocol string, latency time.Duration) {
	// Sample based on sample rate to reduce overhead
	if rand.Float64() > lc.sampleRate {
		return
	}

	// Find appropriate bucket
	bucketIndex := len(lc.buckets) - 1 // Default to last bucket
	for i, bucket := range lc.buckets {
		if latency <= bucket {
			bucketIndex = i
			break
		}
	}

	// Update bucket count
	key := fmt.Sprintf("%s:bucket:%d", protocol, bucketIndex)
	if counters, exists := lc.data.LoadOrStore(key, make([]int64, len(lc.buckets))); exists {
		bucketCounts := counters.([]int64)
		atomic.AddInt64(&bucketCounts[bucketIndex], 1)
	}
}

// Alert Checking
func (ppm *ProtocolPerformanceMonitor) checkForAlerts(event PerformanceEvent) {
	// Check error rate
	if serviceMetrics, exists := ppm.metrics.Load(event.Service); exists {
		sm := serviceMetrics.(*ServiceMetrics)
		
		// Error rate alert
		if sm.OverallSuccessRate < (1.0 - ppm.config.AlertThresholds.ErrorRateThreshold) {
			alert := Alert{
				ID:           fmt.Sprintf("error_rate_%s_%d", event.Service, time.Now().Unix()),
				Type:         "error_rate",
				Severity:     "warning",
				Service:      event.Service,
				Protocol:     event.Protocol,
				Message:      fmt.Sprintf("High error rate detected for service %s", event.Service),
				Threshold:    ppm.config.AlertThresholds.ErrorRateThreshold,
				CurrentValue: 1.0 - sm.OverallSuccessRate,
				Timestamp:    time.Now(),
			}
			ppm.sendAlert(alert)
		}

		// Health score alert
		if sm.HealthScore < ppm.config.AlertThresholds.HealthScoreThreshold {
			alert := Alert{
				ID:           fmt.Sprintf("health_score_%s_%d", event.Service, time.Now().Unix()),
				Type:         "health_score",
				Severity:     "critical",
				Service:      event.Service,
				Message:      fmt.Sprintf("Low health score for service %s", event.Service),
				Threshold:    ppm.config.AlertThresholds.HealthScoreThreshold,
				CurrentValue: sm.HealthScore,
				Timestamp:    time.Now(),
			}
			ppm.sendAlert(alert)
		}
	}

	// Check latency
	if protocolMetrics, exists := ppm.protocolMetrics.Load(event.Protocol); exists {
		pm := protocolMetrics.(*ProtocolMetrics)
		
		if pm.AverageLatency > ppm.config.AlertThresholds.LatencyThreshold {
			alert := Alert{
				ID:           fmt.Sprintf("latency_%s_%d", event.Protocol, time.Now().Unix()),
				Type:         "latency",
				Severity:     "warning",
				Protocol:     event.Protocol,
				Message:      fmt.Sprintf("High latency detected for protocol %s", event.Protocol),
				Threshold:    ppm.config.AlertThresholds.LatencyThreshold,
				CurrentValue: pm.AverageLatency,
				Timestamp:    time.Now(),
			}
			ppm.sendAlert(alert)
		}
	}
}

// Send Alert
func (ppm *ProtocolPerformanceMonitor) sendAlert(alert Alert) {
	select {
	case ppm.alertManager.alertChannel <- alert:
		// Alert queued successfully
	default:
		// Alert channel full, log error
		log.Printf("⚠️ Alert channel full, dropping alert: %s", alert.ID)
	}
}

// Start Alert Processing
func (ppm *ProtocolPerformanceMonitor) startAlertProcessing() {
	for {
		select {
		case alert := <-ppm.alertManager.alertChannel:
			ppm.processAlert(alert)
		case <-ppm.shutdown:
			return
		}
	}
}

// Process Alert
func (ppm *ProtocolPerformanceMonitor) processAlert(alert Alert) {
	// Store alert
	ppm.alertManager.alerts.Store(alert.ID, &alert)
	
	// Log alert
	log.Printf("🚨 ALERT [%s]: %s", alert.Severity, alert.Message)
	
	// Send webhook if configured
	if ppm.alertManager.webhookURL != "" {
		go ppm.sendWebhookAlert(alert)
	}
	
	// Store in Redis for persistence
	alertJSON, _ := json.Marshal(alert)
	ppm.redis.LPush(ppm.ctx, "protocol_alerts", alertJSON)
	ppm.redis.LTrim(ppm.ctx, "protocol_alerts", 0, 100) // Keep last 100 alerts
}

// Send Webhook Alert
func (ppm *ProtocolPerformanceMonitor) sendWebhookAlert(alert Alert) {
	payload, err := json.Marshal(alert)
	if err != nil {
		log.Printf("❌ Failed to marshal alert: %v", err)
		return
	}

	resp, err := http.Post(ppm.alertManager.webhookURL, "application/json", bytes.NewBuffer(payload))
	if err != nil {
		log.Printf("❌ Failed to send webhook alert: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("⚠️ Webhook alert returned status: %d", resp.StatusCode)
	}
}

// Start HTTP Server for Metrics API
func (ppm *ProtocolPerformanceMonitor) startHTTPServer() {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// Metrics API
	api := router.Group("/api/metrics")
	{
		api.GET("/protocols", ppm.getProtocolMetrics)
		api.GET("/protocols/:protocol", ppm.getSpecificProtocolMetrics)
		api.GET("/services", ppm.getServiceMetrics)
		api.GET("/services/:service", ppm.getSpecificServiceMetrics)
		api.GET("/health", ppm.getHealthMetrics)
		api.GET("/alerts", ppm.getAlerts)
		api.DELETE("/alerts/:id", ppm.dismissAlert)
	}

	// Real-time metrics endpoint
	router.GET("/metrics/stream", ppm.streamMetrics)

	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "Protocol Performance Monitor",
			"version": "1.0.0",
			"endpoints": gin.H{
				"protocol_metrics": "/api/metrics/protocols",
				"service_metrics":  "/api/metrics/services",
				"health_metrics":   "/api/metrics/health",
				"alerts":          "/api/metrics/alerts",
				"stream":          "/metrics/stream",
			},
		})
	})

	addr := fmt.Sprintf(":%d", ppm.config.HTTPPort)
	ppm.httpServer = &http.Server{
		Addr:    addr,
		Handler: router,
	}

	log.Printf("📊 Metrics HTTP server starting on port %d", ppm.config.HTTPPort)
	if err := ppm.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("❌ Metrics HTTP server failed: %v", err)
	}
}

// HTTP Handlers
func (ppm *ProtocolPerformanceMonitor) getProtocolMetrics(c *gin.Context) {
	metrics := make(map[string]*ProtocolMetrics)
	
	ppm.protocolMetrics.Range(func(key, value interface{}) bool {
		protocol := key.(string)
		metric := value.(*ProtocolMetrics)
		metrics[protocol] = metric
		return true
	})
	
	c.JSON(http.StatusOK, gin.H{"protocols": metrics})
}

func (ppm *ProtocolPerformanceMonitor) getSpecificProtocolMetrics(c *gin.Context) {
	protocol := c.Param("protocol")
	
	if val, exists := ppm.protocolMetrics.Load(protocol); exists {
		c.JSON(http.StatusOK, val.(*ProtocolMetrics))
	} else {
		c.JSON(http.StatusNotFound, gin.H{"error": "Protocol not found"})
	}
}

func (ppm *ProtocolPerformanceMonitor) getServiceMetrics(c *gin.Context) {
	metrics := make(map[string]*ServiceMetrics)
	
	ppm.metrics.Range(func(key, value interface{}) bool {
		service := key.(string)
		metric := value.(*ServiceMetrics)
		metrics[service] = metric
		return true
	})
	
	c.JSON(http.StatusOK, gin.H{"services": metrics})
}

func (ppm *ProtocolPerformanceMonitor) getSpecificServiceMetrics(c *gin.Context) {
	service := c.Param("service")
	
	if val, exists := ppm.metrics.Load(service); exists {
		c.JSON(http.StatusOK, val.(*ServiceMetrics))
	} else {
		c.JSON(http.StatusNotFound, gin.H{"error": "Service not found"})
	}
}

func (ppm *ProtocolPerformanceMonitor) getHealthMetrics(c *gin.Context) {
	health := gin.H{
		"overall_health": "healthy",
		"protocol_health": gin.H{},
		"service_health": gin.H{},
		"alerts_active": 0,
	}
	
	// Collect protocol health
	ppm.protocolMetrics.Range(func(key, value interface{}) bool {
		protocol := key.(string)
		metric := value.(*ProtocolMetrics)
		
		status := "healthy"
		if metric.SuccessRate < 0.95 {
			status = "degraded"
		}
		if metric.SuccessRate < 0.8 {
			status = "unhealthy"
		}
		
		health["protocol_health"].(gin.H)[protocol] = gin.H{
			"status": status,
			"success_rate": metric.SuccessRate,
			"avg_latency": metric.AverageLatency,
		}
		return true
	})
	
	// Collect service health
	ppm.metrics.Range(func(key, value interface{}) bool {
		service := key.(string)
		metric := value.(*ServiceMetrics)
		
		status := "healthy"
		if metric.HealthScore < 0.8 {
			status = "degraded"
		}
		if metric.HealthScore < 0.6 {
			status = "unhealthy"
		}
		
		health["service_health"].(gin.H)[service] = gin.H{
			"status": status,
			"health_score": metric.HealthScore,
			"success_rate": metric.OverallSuccessRate,
		}
		return true
	})
	
	// Count active alerts
	alertCount := 0
	ppm.alertManager.alerts.Range(func(key, value interface{}) bool {
		alertCount++
		return true
	})
	health["alerts_active"] = alertCount
	
	c.JSON(http.StatusOK, health)
}

func (ppm *ProtocolPerformanceMonitor) getAlerts(c *gin.Context) {
	alerts := make(map[string]*Alert)
	
	ppm.alertManager.alerts.Range(func(key, value interface{}) bool {
		id := key.(string)
		alert := value.(*Alert)
		alerts[id] = alert
		return true
	})
	
	c.JSON(http.StatusOK, gin.H{"alerts": alerts})
}

func (ppm *ProtocolPerformanceMonitor) dismissAlert(c *gin.Context) {
	alertID := c.Param("id")
	
	if _, exists := ppm.alertManager.alerts.LoadAndDelete(alertID); exists {
		c.JSON(http.StatusOK, gin.H{"message": "Alert dismissed"})
	} else {
		c.JSON(http.StatusNotFound, gin.H{"error": "Alert not found"})
	}
}

// Stream real-time metrics via Server-Sent Events
func (ppm *ProtocolPerformanceMonitor) streamMetrics(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Collect current metrics
			data := gin.H{
				"timestamp": time.Now(),
				"protocols": gin.H{},
				"services": gin.H{},
			}
			
			ppm.protocolMetrics.Range(func(key, value interface{}) bool {
				protocol := key.(string)
				metric := value.(*ProtocolMetrics)
				data["protocols"].(gin.H)[protocol] = gin.H{
					"success_rate": metric.SuccessRate,
					"avg_latency": metric.AverageLatency.Milliseconds(),
					"throughput": metric.ThroughputRPS,
				}
				return true
			})
			
			ppm.metrics.Range(func(key, value interface{}) bool {
				service := key.(string)
				metric := value.(*ServiceMetrics)
				data["services"].(gin.H)[service] = gin.H{
					"health_score": metric.HealthScore,
					"success_rate": metric.OverallSuccessRate,
				}
				return true
			})
			
			jsonData, _ := json.Marshal(data)
			fmt.Fprintf(c.Writer, "data: %s\n\n", jsonData)
			c.Writer.Flush()
			
		case <-c.Request.Context().Done():
			return
		case <-ppm.shutdown:
			return
		}
	}
}

// Additional utility functions...

func calculateLatencyScore(latency time.Duration) float64 {
	// Convert latency to a score between 0 and 1 (higher is better)
	ms := latency.Milliseconds()
	if ms <= 10 {
		return 1.0
	} else if ms <= 100 {
		return 0.8
	} else if ms <= 500 {
		return 0.6
	} else if ms <= 1000 {
		return 0.4
	} else if ms <= 2000 {
		return 0.2
	} else {
		return 0.1
	}
}

func calculateP95(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	// Simple P95 calculation - in production use a proper percentile library
	index := int(0.95 * float64(len(values)))
	if index >= len(values) {
		index = len(values) - 1
	}
	return values[index]
}

// Start Redis subscriber for cross-service metrics
func (ppm *ProtocolPerformanceMonitor) startRedisSubscriber() {
	pubsub := ppm.redis.Subscribe(ppm.ctx, "protocol_events")
	defer pubsub.Close()

	for {
		select {
		case msg := <-pubsub.Channel():
			var event PerformanceEvent
			if err := json.Unmarshal([]byte(msg.Payload), &event); err == nil {
				ppm.RecordEvent(event)
			}
		case <-ppm.shutdown:
			return
		}
	}
}

// Start metrics collection background task
func (ppm *ProtocolPerformanceMonitor) startMetricsCollection() {
	ticker := time.NewTicker(ppm.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			ppm.collectAndPersistMetrics()
		case <-ppm.shutdown:
			return
		}
	}
}

func (ppm *ProtocolPerformanceMonitor) collectAndPersistMetrics() {
	// Persist current metrics to Redis for long-term storage
	timestamp := time.Now().Unix()
	
	ppm.protocolMetrics.Range(func(key, value interface{}) bool {
		protocol := key.(string)
		metric := value.(*ProtocolMetrics)
		
		data, _ := json.Marshal(metric)
		ppm.redis.ZAdd(ppm.ctx, fmt.Sprintf("metrics:protocol:%s", protocol), redis.Z{
			Score:  float64(timestamp),
			Member: data,
		})
		
		// Keep only last 24 hours of data
		cutoff := timestamp - int64(ppm.config.RetentionPeriod.Seconds())
		ppm.redis.ZRemRangeByScore(ppm.ctx, fmt.Sprintf("metrics:protocol:%s", protocol), "-inf", fmt.Sprintf("%d", cutoff))
		
		return true
	})
}

func (ppm *ProtocolPerformanceMonitor) pushToRedis(event PerformanceEvent) {
	data, err := json.Marshal(event)
	if err != nil {
		return
	}
	
	// Publish event for other subscribers
	ppm.redis.Publish(ppm.ctx, "protocol_events", data)
	
	// Store in time-series for analysis
	timestamp := event.Timestamp.Unix()
	ppm.redis.ZAdd(ppm.ctx, "protocol_events_timeseries", redis.Z{
		Score:  float64(timestamp),
		Member: data,
	})
}

// Environment helpers (same as previous files)
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatValue, err := strconv.ParseFloat(value, 64); err == nil {
			return floatValue
		}
	}
	return defaultValue
}

// Main function
func main() {
	monitor, err := NewProtocolPerformanceMonitor()
	if err != nil {
		log.Fatalf("❌ Failed to initialize performance monitor: %v", err)
	}
	defer monitor.redis.Close()

	log.Printf("🚀 Starting Protocol Performance Monitor...")
	
	if err := monitor.Start(); err != nil {
		log.Fatalf("❌ Failed to start performance monitor: %v", err)
	}
}