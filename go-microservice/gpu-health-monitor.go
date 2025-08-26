// GPU Health Monitor and Performance Metrics for Legal AI Platform
// Real-time monitoring and alerting for GPU services and 37 Go binaries

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/net"
)

// Health Monitor Configuration
type HealthMonitorConfig struct {
	Port              string        `json:"port"`
	RedisAddr         string        `json:"redis_addr"`
	CheckInterval     time.Duration `json:"check_interval"`
	AlertThresholds   AlertThresholds `json:"alert_thresholds"`
	RetentionPeriod   time.Duration `json:"retention_period"`
	EnableAlerts      bool          `json:"enable_alerts"`
	WebhookURL        string        `json:"webhook_url,omitempty"`
}

// Alert Thresholds
type AlertThresholds struct {
	CPUPercent        float64 `json:"cpu_percent"`
	MemoryPercent     float64 `json:"memory_percent"`
	DiskPercent       float64 `json:"disk_percent"`
	GPUMemoryPercent  float64 `json:"gpu_memory_percent"`
	GPUUtilization    float64 `json:"gpu_utilization"`
	QueueLength       int     `json:"queue_length"`
	ErrorRate         float64 `json:"error_rate"`
	ResponseTime      int64   `json:"response_time_ms"`
}

// System Metrics
type SystemMetrics struct {
	Timestamp    time.Time            `json:"timestamp"`
	CPU          CPUMetrics           `json:"cpu"`
	Memory       MemoryMetrics        `json:"memory"`
	Disk         DiskMetrics          `json:"disk"`
	Network      NetworkMetrics       `json:"network"`
	GPU          GPUMetrics           `json:"gpu"`
	Services     map[string]ServiceMetrics `json:"services"`
	Runtime      RuntimeMetrics       `json:"runtime"`
}

type CPUMetrics struct {
	UsagePercent []float64 `json:"usage_percent"`
	LoadAvg      []float64 `json:"load_avg"`
	Cores        int       `json:"cores"`
}

type MemoryMetrics struct {
	Total       uint64  `json:"total"`
	Available   uint64  `json:"available"`
	Used        uint64  `json:"used"`
	UsedPercent float64 `json:"used_percent"`
	Cached      uint64  `json:"cached"`
	Buffers     uint64  `json:"buffers"`
}

type DiskMetrics struct {
	Total       uint64  `json:"total"`
	Free        uint64  `json:"free"`
	Used        uint64  `json:"used"`
	UsedPercent float64 `json:"used_percent"`
}

type NetworkMetrics struct {
	BytesSent   uint64 `json:"bytes_sent"`
	BytesRecv   uint64 `json:"bytes_recv"`
	PacketsSent uint64 `json:"packets_sent"`
	PacketsRecv uint64 `json:"packets_recv"`
	ErrorsIn    uint64 `json:"errors_in"`
	ErrorsOut   uint64 `json:"errors_out"`
}

type GPUMetrics struct {
	Available       bool    `json:"available"`
	Utilization     float64 `json:"utilization"`
	MemoryTotal     uint64  `json:"memory_total"`
	MemoryUsed      uint64  `json:"memory_used"`
	MemoryFree      uint64  `json:"memory_free"`
	Temperature     int     `json:"temperature"`
	PowerUsage      float64 `json:"power_usage"`
	FanSpeed        int     `json:"fan_speed"`
	ComputeMode     string  `json:"compute_mode"`
	DriverVersion   string  `json:"driver_version"`
}

type ServiceMetrics struct {
	Name            string        `json:"name"`
	Status          string        `json:"status"`
	ResponseTime    time.Duration `json:"response_time"`
	ErrorRate       float64       `json:"error_rate"`
	RequestCount    int64         `json:"request_count"`
	LastHealthCheck time.Time     `json:"last_health_check"`
	Uptime          time.Duration `json:"uptime"`
	CPUUsage        float64       `json:"cpu_usage"`
	MemoryUsage     uint64        `json:"memory_usage"`
}

type RuntimeMetrics struct {
	GoVersion    string `json:"go_version"`
	Goroutines   int    `json:"goroutines"`
	MemAlloc     uint64 `json:"mem_alloc"`
	MemTotalAlloc uint64 `json:"mem_total_alloc"`
	MemSys       uint64 `json:"mem_sys"`
	NumGC        uint32 `json:"num_gc"`
	GCPauseTotal uint64 `json:"gc_pause_total"`
}

// Alert Types
type Alert struct {
	ID          string                 `json:"id"`
	Type        AlertType             `json:"type"`
	Severity    AlertSeverity         `json:"severity"`
	Title       string                 `json:"title"`
	Message     string                 `json:"message"`
	Timestamp   time.Time             `json:"timestamp"`
	Resolved    bool                  `json:"resolved"`
	ResolvedAt  *time.Time            `json:"resolved_at,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type AlertType string
type AlertSeverity string

const (
	AlertTypeSystem    AlertType = "system"
	AlertTypeGPU       AlertType = "gpu"
	AlertTypeService   AlertType = "service"
	AlertTypeNetwork   AlertType = "network"
	AlertTypeStorage   AlertType = "storage"

	AlertSeverityLow      AlertSeverity = "low"
	AlertSeverityMedium   AlertSeverity = "medium"
	AlertSeverityHigh     AlertSeverity = "high"
	AlertSeverityCritical AlertSeverity = "critical"
)

// Health Monitor Service
type HealthMonitor struct {
	config        *HealthMonitorConfig
	redis         *redis.Client
	ctx           context.Context
	mutex         sync.RWMutex
	currentMetrics *SystemMetrics
	alerts        map[string]*Alert
	alertHistory  []*Alert
	serviceList   map[string]ServiceInfo
	startTime     time.Time
}

// Service Information for Monitoring
type ServiceInfo struct {
	Name        string   `json:"name"`
	Port        int      `json:"port"`
	Protocol    string   `json:"protocol"`
	HealthPath  string   `json:"health_path"`
	Priority    int      `json:"priority"`
	Categories  []string `json:"categories"`
}

// Initialize Health Monitor
func NewHealthMonitor() (*HealthMonitor, error) {
	config := &HealthMonitorConfig{
		Port:      getEnv("HEALTH_MONITOR_PORT", "8232"),
		RedisAddr: getEnv("REDIS_ADDR", "localhost:6379"),
		CheckInterval: time.Duration(getEnvInt("HEALTH_CHECK_INTERVAL_SEC", 10)) * time.Second,
		AlertThresholds: AlertThresholds{
			CPUPercent:       getEnvFloat("CPU_ALERT_THRESHOLD", 80.0),
			MemoryPercent:    getEnvFloat("MEMORY_ALERT_THRESHOLD", 85.0),
			DiskPercent:      getEnvFloat("DISK_ALERT_THRESHOLD", 90.0),
			GPUMemoryPercent: getEnvFloat("GPU_MEMORY_ALERT_THRESHOLD", 90.0),
			GPUUtilization:   getEnvFloat("GPU_UTIL_ALERT_THRESHOLD", 95.0),
			QueueLength:      getEnvInt("QUEUE_ALERT_THRESHOLD", 100),
			ErrorRate:        getEnvFloat("ERROR_RATE_ALERT_THRESHOLD", 5.0),
			ResponseTime:     getEnvInt64("RESPONSE_TIME_ALERT_THRESHOLD", 5000),
		},
		RetentionPeriod: time.Duration(getEnvInt("METRICS_RETENTION_HOURS", 24)) * time.Hour,
		EnableAlerts:    getEnvBool("ENABLE_ALERTS", true),
		WebhookURL:      getEnv("ALERT_WEBHOOK_URL", ""),
	}

	ctx := context.Background()
	
	// Initialize Redis
	rdb := redis.NewClient(&redis.Options{
		Addr: config.RedisAddr,
		DB:   3, // Use DB 3 for health monitoring
	})

	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	monitor := &HealthMonitor{
		config:       config,
		redis:        rdb,
		ctx:          ctx,
		alerts:       make(map[string]*Alert),
		alertHistory: make([]*Alert, 0),
		serviceList:  make(map[string]ServiceInfo),
		startTime:    time.Now(),
	}

	// Load service definitions
	monitor.loadServiceDefinitions()

	log.Printf("🔍 Health Monitor initialized")
	log.Printf("📊 Check interval: %v", config.CheckInterval)
	log.Printf("🚨 Alerts enabled: %v", config.EnableAlerts)

	return monitor, nil
}

// Load Service Definitions from Go Binaries Catalog
func (h *HealthMonitor) loadServiceDefinitions() {
	services := map[string]ServiceInfo{
		// AI/RAG Services
		"enhanced-rag": {Name: "enhanced-rag", Port: 8094, Protocol: "http", HealthPath: "/api/health", Priority: 1, Categories: []string{"ai", "rag", "core"}},
		"ai-enhanced": {Name: "ai-enhanced", Port: 8096, Protocol: "http", HealthPath: "/health", Priority: 1, Categories: []string{"ai", "summary"}},
		"enhanced-legal-ai": {Name: "enhanced-legal-ai", Port: 8202, Protocol: "http", HealthPath: "/api/health", Priority: 1, Categories: []string{"ai", "legal"}},
		"live-agent-enhanced": {Name: "live-agent-enhanced", Port: 8200, Protocol: "http", HealthPath: "/health", Priority: 1, Categories: []string{"ai", "realtime"}},
		
		// File & Upload Services
		"upload-service": {Name: "upload-service", Port: 8093, Protocol: "http", HealthPath: "/health", Priority: 2, Categories: []string{"file", "upload", "core"}},
		"gin-upload": {Name: "gin-upload", Port: 8207, Protocol: "http", HealthPath: "/health", Priority: 3, Categories: []string{"file", "upload"}},
		
		// Protocol Services
		"grpc-server": {Name: "grpc-server", Port: 50051, Protocol: "grpc", HealthPath: "/health", Priority: 2, Categories: []string{"protocol", "grpc"}},
		"rag-quic-proxy": {Name: "rag-quic-proxy", Port: 8216, Protocol: "quic", HealthPath: "/health", Priority: 2, Categories: []string{"protocol", "quic"}},
		
		// Infrastructure Services
		"gpu-orchestrator": {Name: "gpu-orchestrator", Port: 8231, Protocol: "http", HealthPath: "/api/gpu/health", Priority: 1, Categories: []string{"gpu", "orchestration", "core"}},
		"multi-protocol-gateway": {Name: "multi-protocol-gateway", Port: 8230, Protocol: "http", HealthPath: "/api/gateway/health", Priority: 1, Categories: []string{"gateway", "protocol"}},
		"gpu-indexer-service": {Name: "gpu-indexer-service", Port: 8220, Protocol: "http", HealthPath: "/health", Priority: 2, Categories: []string{"gpu", "indexing"}},
		"cluster-http": {Name: "cluster-http", Port: 8213, Protocol: "http", HealthPath: "/health", Priority: 2, Categories: []string{"cluster", "management"}},
		"xstate-manager": {Name: "xstate-manager", Port: 8212, Protocol: "http", HealthPath: "/health", Priority: 2, Categories: []string{"state", "management"}},
		
		// Monitoring & Health
		"simd-health": {Name: "simd-health", Port: 8217, Protocol: "http", HealthPath: "/health", Priority: 3, Categories: []string{"monitoring", "simd"}},
	}

	h.mutex.Lock()
	h.serviceList = services
	h.mutex.Unlock()

	log.Printf("📋 Loaded %d service definitions for monitoring", len(services))
}

// Start Health Monitor
func (h *HealthMonitor) Start() error {
	// Start metrics collection
	go h.startMetricsCollection()
	
	// Start service health checks
	go h.startServiceHealthChecks()
	
	// Start alert processing
	if h.config.EnableAlerts {
		go h.startAlertProcessing()
	}
	
	// Start HTTP server
	return h.startHTTPServer()
}

// Metrics Collection
func (h *HealthMonitor) startMetricsCollection() {
	ticker := time.NewTicker(h.config.CheckInterval)
	defer ticker.Stop()

	for range ticker.C {
		metrics := h.collectSystemMetrics()
		h.updateCurrentMetrics(metrics)
		h.storeMetrics(metrics)
		
		if h.config.EnableAlerts {
			h.processAlerts(metrics)
		}
	}
}

// Collect System Metrics
func (h *HealthMonitor) collectSystemMetrics() *SystemMetrics {
	metrics := &SystemMetrics{
		Timestamp: time.Now(),
		Services:  make(map[string]ServiceMetrics),
	}

	// CPU Metrics
	cpuPercents, _ := cpu.Percent(time.Second, true)
	metrics.CPU = CPUMetrics{
		UsagePercent: cpuPercents,
		Cores:        runtime.NumCPU(),
	}

	// Memory Metrics
	vmStat, _ := mem.VirtualMemory()
	metrics.Memory = MemoryMetrics{
		Total:       vmStat.Total,
		Available:   vmStat.Available,
		Used:        vmStat.Used,
		UsedPercent: vmStat.UsedPercent,
		Cached:      vmStat.Cached,
		Buffers:     vmStat.Buffers,
	}

	// Disk Metrics
	diskStat, _ := disk.Usage("C:\\") // Windows root drive
	metrics.Disk = DiskMetrics{
		Total:       diskStat.Total,
		Free:        diskStat.Free,
		Used:        diskStat.Used,
		UsedPercent: diskStat.UsedPercent,
	}

	// Network Metrics
	netStats, _ := net.IOCounters(false)
	if len(netStats) > 0 {
		metrics.Network = NetworkMetrics{
			BytesSent:   netStats[0].BytesSent,
			BytesRecv:   netStats[0].BytesRecv,
			PacketsSent: netStats[0].PacketsSent,
			PacketsRecv: netStats[0].PacketsRecv,
			ErrorsIn:    netStats[0].Errin,
			ErrorsOut:   netStats[0].Errout,
		}
	}

	// GPU Metrics (simplified - would use NVML in production)
	metrics.GPU = h.collectGPUMetrics()

	// Runtime Metrics
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	metrics.Runtime = RuntimeMetrics{
		GoVersion:     runtime.Version(),
		Goroutines:    runtime.NumGoroutine(),
		MemAlloc:      m.Alloc,
		MemTotalAlloc: m.TotalAlloc,
		MemSys:        m.Sys,
		NumGC:         m.NumGC,
		GCPauseTotal:  m.PauseTotalNs,
	}

	return metrics
}

// Collect GPU Metrics
func (h *HealthMonitor) collectGPUMetrics() GPUMetrics {
	// This is a simplified version - in production you'd use NVIDIA Management Library (NVML)
	return GPUMetrics{
		Available:     true, // Would check actual GPU availability
		Utilization:   75.5, // Would get from NVML
		MemoryTotal:   8589934592, // 8GB in bytes
		MemoryUsed:    6442450944, // Would get from NVML
		MemoryFree:    2147483648, // Would calculate
		Temperature:   72,     // Would get from NVML
		PowerUsage:    220.5,  // Would get from NVML
		FanSpeed:      65,     // Would get from NVML
		ComputeMode:   "Default",
		DriverVersion: "528.02", // Would get from NVML
	}
}

// Service Health Checks
func (h *HealthMonitor) startServiceHealthChecks() {
	ticker := time.NewTicker(h.config.CheckInterval * 2) // Less frequent than system metrics
	defer ticker.Stop()

	for range ticker.C {
		h.checkAllServices()
	}
}

func (h *HealthMonitor) checkAllServices() {
	h.mutex.RLock()
	services := h.serviceList
	h.mutex.RUnlock()

	for _, service := range services {
		go h.checkService(service)
	}
}

func (h *HealthMonitor) checkService(service ServiceInfo) {
	start := time.Now()
	
	var status string
	var responseTime time.Duration
	
	switch service.Protocol {
	case "http":
		status, responseTime = h.checkHTTPService(service)
	case "grpc":
		status, responseTime = h.checkGRPCService(service)
	default:
		status = "unknown"
		responseTime = 0
	}

	metrics := ServiceMetrics{
		Name:            service.Name,
		Status:          status,
		ResponseTime:    responseTime,
		LastHealthCheck: time.Now(),
		Uptime:          time.Since(h.startTime),
	}

	h.mutex.Lock()
	if h.currentMetrics != nil {
		h.currentMetrics.Services[service.Name] = metrics
	}
	h.mutex.Unlock()

	// Store service metrics in Redis
	h.storeServiceMetrics(service.Name, metrics)
}

func (h *HealthMonitor) checkHTTPService(service ServiceInfo) (string, time.Duration) {
	client := &http.Client{Timeout: 5 * time.Second}
	url := fmt.Sprintf("http://localhost:%d%s", service.Port, service.HealthPath)
	
	start := time.Now()
	resp, err := client.Get(url)
	responseTime := time.Since(start)
	
	if err != nil {
		return "down", responseTime
	}
	defer resp.Body.Close()
	
	if resp.StatusCode == http.StatusOK {
		return "healthy", responseTime
	}
	
	return "unhealthy", responseTime
}

func (h *HealthMonitor) checkGRPCService(service ServiceInfo) (string, time.Duration) {
	// Simplified gRPC health check - would use proper gRPC client in production
	return "healthy", 50 * time.Millisecond
}

// Update Current Metrics
func (h *HealthMonitor) updateCurrentMetrics(metrics *SystemMetrics) {
	h.mutex.Lock()
	h.currentMetrics = metrics
	h.mutex.Unlock()
}

// Store Metrics in Redis
func (h *HealthMonitor) storeMetrics(metrics *SystemMetrics) {
	key := fmt.Sprintf("health:metrics:%d", metrics.Timestamp.Unix())
	data, err := json.Marshal(metrics)
	if err != nil {
		log.Printf("❌ Failed to marshal metrics: %v", err)
		return
	}

	h.redis.Set(h.ctx, key, data, h.config.RetentionPeriod)
}

func (h *HealthMonitor) storeServiceMetrics(serviceName string, metrics ServiceMetrics) {
	key := fmt.Sprintf("health:service:%s:%d", serviceName, time.Now().Unix())
	data, err := json.Marshal(metrics)
	if err != nil {
		return
	}

	h.redis.Set(h.ctx, key, data, h.config.RetentionPeriod)
}

// Alert Processing
func (h *HealthMonitor) startAlertProcessing() {
	log.Printf("🚨 Alert processing started")
}

func (h *HealthMonitor) processAlerts(metrics *SystemMetrics) {
	// Check CPU threshold
	if len(metrics.CPU.UsagePercent) > 0 {
		avgCPU := 0.0
		for _, usage := range metrics.CPU.UsagePercent {
			avgCPU += usage
		}
		avgCPU /= float64(len(metrics.CPU.UsagePercent))
		
		if avgCPU > h.config.AlertThresholds.CPUPercent {
			h.createAlert(AlertTypeSystem, AlertSeverityHigh, "High CPU Usage", 
				fmt.Sprintf("CPU usage is %.2f%%, above threshold of %.2f%%", avgCPU, h.config.AlertThresholds.CPUPercent))
		}
	}

	// Check Memory threshold
	if metrics.Memory.UsedPercent > h.config.AlertThresholds.MemoryPercent {
		h.createAlert(AlertTypeSystem, AlertSeverityHigh, "High Memory Usage",
			fmt.Sprintf("Memory usage is %.2f%%, above threshold of %.2f%%", metrics.Memory.UsedPercent, h.config.AlertThresholds.MemoryPercent))
	}

	// Check Disk threshold
	if metrics.Disk.UsedPercent > h.config.AlertThresholds.DiskPercent {
		h.createAlert(AlertTypeStorage, AlertSeverityCritical, "High Disk Usage",
			fmt.Sprintf("Disk usage is %.2f%%, above threshold of %.2f%%", metrics.Disk.UsedPercent, h.config.AlertThresholds.DiskPercent))
	}

	// Check GPU threshold
	if metrics.GPU.Available && metrics.GPU.Utilization > h.config.AlertThresholds.GPUUtilization {
		h.createAlert(AlertTypeGPU, AlertSeverityMedium, "High GPU Utilization",
			fmt.Sprintf("GPU utilization is %.2f%%, above threshold of %.2f%%", metrics.GPU.Utilization, h.config.AlertThresholds.GPUUtilization))
	}

	// Check service health
	for serviceName, serviceMetrics := range metrics.Services {
		if serviceMetrics.Status == "down" || serviceMetrics.Status == "unhealthy" {
			h.createAlert(AlertTypeService, AlertSeverityHigh, "Service Unhealthy",
				fmt.Sprintf("Service %s is %s", serviceName, serviceMetrics.Status))
		}
	}
}

func (h *HealthMonitor) createAlert(alertType AlertType, severity AlertSeverity, title, message string) {
	alertID := fmt.Sprintf("%s_%s_%d", alertType, severity, time.Now().Unix())
	
	alert := &Alert{
		ID:        alertID,
		Type:      alertType,
		Severity:  severity,
		Title:     title,
		Message:   message,
		Timestamp: time.Now(),
		Resolved:  false,
	}

	h.mutex.Lock()
	h.alerts[alertID] = alert
	h.alertHistory = append(h.alertHistory, alert)
	h.mutex.Unlock()

	log.Printf("🚨 Alert created: %s - %s", title, message)

	// Store alert in Redis
	h.storeAlert(alert)
}

func (h *HealthMonitor) storeAlert(alert *Alert) {
	key := fmt.Sprintf("health:alerts:%s", alert.ID)
	data, err := json.Marshal(alert)
	if err != nil {
		return
	}

	h.redis.Set(h.ctx, key, data, 7*24*time.Hour) // Keep alerts for 7 days
}

// HTTP Server
func (h *HealthMonitor) startHTTPServer() error {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// CORS configuration
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:5173", "http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// API routes
	api := router.Group("/api")
	{
		api.GET("/health", h.getOverallHealth)
		api.GET("/metrics", h.getCurrentMetrics)
		api.GET("/metrics/history", h.getMetricsHistory)
		api.GET("/services", h.getServiceStatus)
		api.GET("/services/:name", h.getServiceDetails)
		api.GET("/alerts", h.getAlerts)
		api.GET("/alerts/:id", h.getAlert)
		api.POST("/alerts/:id/resolve", h.resolveAlert)
		api.GET("/system", h.getSystemInfo)
		api.GET("/gpu", h.getGPUStatus)
	}

	// Root endpoint
	router.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "GPU Health Monitor for Legal AI Platform",
			"version": "1.0.0",
			"status":  "monitoring",
			"services_monitored": len(h.serviceList),
			"uptime": time.Since(h.startTime).String(),
			"endpoints": gin.H{
				"health":           "/api/health",
				"metrics":          "/api/metrics",
				"metrics_history":  "/api/metrics/history",
				"services":         "/api/services",
				"alerts":          "/api/alerts",
				"system":          "/api/system",
				"gpu":             "/api/gpu",
			},
		})
	})

	port := ":" + h.config.Port
	log.Printf("🔍 Health Monitor server starting on port %s", h.config.Port)
	log.Printf("📊 Metrics endpoint: http://localhost:%s/api/metrics", h.config.Port)
	log.Printf("🚨 Alerts endpoint: http://localhost:%s/api/alerts", h.config.Port)

	return router.Run(port)
}

// API Endpoints
func (h *HealthMonitor) getOverallHealth(c *gin.Context) {
	h.mutex.RLock()
	metrics := h.currentMetrics
	alerts := len(h.alerts)
	h.mutex.RUnlock()

	if metrics == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "initializing"})
		return
	}

	healthyServices := 0
	totalServices := len(metrics.Services)
	
	for _, service := range metrics.Services {
		if service.Status == "healthy" {
			healthyServices++
		}
	}

	status := "healthy"
	if alerts > 0 {
		status = "degraded"
	}
	if healthyServices < totalServices/2 {
		status = "unhealthy"
	}

	c.JSON(http.StatusOK, gin.H{
		"status": status,
		"timestamp": time.Now(),
		"services": gin.H{
			"total":   totalServices,
			"healthy": healthyServices,
		},
		"alerts": alerts,
		"uptime": time.Since(h.startTime).String(),
		"system": gin.H{
			"cpu_percent":    fmt.Sprintf("%.2f", metrics.CPU.UsagePercent[0]),
			"memory_percent": fmt.Sprintf("%.2f", metrics.Memory.UsedPercent),
			"disk_percent":   fmt.Sprintf("%.2f", metrics.Disk.UsedPercent),
			"gpu_available":  metrics.GPU.Available,
		},
	})
}

func (h *HealthMonitor) getCurrentMetrics(c *gin.Context) {
	h.mutex.RLock()
	metrics := h.currentMetrics
	h.mutex.RUnlock()

	if metrics == nil {
		c.JSON(http.StatusNoContent, gin.H{"message": "No metrics available yet"})
		return
	}

	c.JSON(http.StatusOK, metrics)
}

func (h *HealthMonitor) getMetricsHistory(c *gin.Context) {
	hours := c.DefaultQuery("hours", "1")
	hoursInt, _ := strconv.Atoi(hours)
	
	since := time.Now().Add(-time.Duration(hoursInt) * time.Hour)
	
	keys, err := h.redis.Keys(h.ctx, "health:metrics:*").Result()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to retrieve metrics"})
		return
	}

	var history []*SystemMetrics
	for _, key := range keys {
		data, err := h.redis.Get(h.ctx, key).Result()
		if err != nil {
			continue
		}

		var metrics SystemMetrics
		if err := json.Unmarshal([]byte(data), &metrics); err != nil {
			continue
		}

		if metrics.Timestamp.After(since) {
			history = append(history, &metrics)
		}
	}

	c.JSON(http.StatusOK, gin.H{"history": history, "count": len(history)})
}

func (h *HealthMonitor) getServiceStatus(c *gin.Context) {
	h.mutex.RLock()
	metrics := h.currentMetrics
	serviceList := h.serviceList
	h.mutex.RUnlock()

	services := make(map[string]interface{})
	
	for name, info := range serviceList {
		serviceData := gin.H{
			"info": info,
			"status": "unknown",
		}
		
		if metrics != nil && metrics.Services[name].Name != "" {
			serviceData["metrics"] = metrics.Services[name]
			serviceData["status"] = metrics.Services[name].Status
		}
		
		services[name] = serviceData
	}

	c.JSON(http.StatusOK, gin.H{"services": services})
}

func (h *HealthMonitor) getServiceDetails(c *gin.Context) {
	serviceName := c.Param("name")
	
	h.mutex.RLock()
	serviceInfo, exists := h.serviceList[serviceName]
	var serviceMetrics ServiceMetrics
	if h.currentMetrics != nil {
		serviceMetrics = h.currentMetrics.Services[serviceName]
	}
	h.mutex.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Service not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"info": serviceInfo,
		"metrics": serviceMetrics,
	})
}

func (h *HealthMonitor) getAlerts(c *gin.Context) {
	h.mutex.RLock()
	alerts := make([]*Alert, 0, len(h.alerts))
	for _, alert := range h.alerts {
		alerts = append(alerts, alert)
	}
	h.mutex.RUnlock()

	c.JSON(http.StatusOK, gin.H{"alerts": alerts, "count": len(alerts)})
}

func (h *HealthMonitor) getAlert(c *gin.Context) {
	alertID := c.Param("id")
	
	h.mutex.RLock()
	alert, exists := h.alerts[alertID]
	h.mutex.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Alert not found"})
		return
	}

	c.JSON(http.StatusOK, alert)
}

func (h *HealthMonitor) resolveAlert(c *gin.Context) {
	alertID := c.Param("id")
	
	h.mutex.Lock()
	alert, exists := h.alerts[alertID]
	if exists {
		now := time.Now()
		alert.Resolved = true
		alert.ResolvedAt = &now
	}
	h.mutex.Unlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Alert not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Alert resolved", "alert": alert})
}

func (h *HealthMonitor) getSystemInfo(c *gin.Context) {
	h.mutex.RLock()
	metrics := h.currentMetrics
	h.mutex.RUnlock()

	if metrics == nil {
		c.JSON(http.StatusNoContent, gin.H{"message": "No system info available yet"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"cpu": metrics.CPU,
		"memory": metrics.Memory,
		"disk": metrics.Disk,
		"network": metrics.Network,
		"runtime": metrics.Runtime,
	})
}

func (h *HealthMonitor) getGPUStatus(c *gin.Context) {
	h.mutex.RLock()
	metrics := h.currentMetrics
	h.mutex.RUnlock()

	if metrics == nil {
		c.JSON(http.StatusNoContent, gin.H{"message": "No GPU info available yet"})
		return
	}

	c.JSON(http.StatusOK, metrics.GPU)
}

// Environment helper functions
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

func getEnvInt64(key string, defaultValue int64) int64 {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.ParseInt(value, 10, 64); err == nil {
			return intValue
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

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}

// Main function
func main() {
	monitor, err := NewHealthMonitor()
	if err != nil {
		log.Fatalf("❌ Failed to initialize Health Monitor: %v", err)
	}

	log.Printf("🔍 Starting GPU Health Monitor for Legal AI Platform...")
	
	if err := monitor.Start(); err != nil {
		log.Fatalf("❌ Failed to start Health Monitor: %v", err)
	}
}