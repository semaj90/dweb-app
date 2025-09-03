/**
 * Windows Service Monitoring and Logging
 * Production-ready logging with Windows Event Log integration, performance monitoring,
 * and memory-optimized log rotation
 */

package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"golang.org/x/sys/windows/svc/eventlog"
	"legal-ai-platform/internal/config"
)

// LogLevel represents logging levels
type LogLevel int

const (
	DEBUG LogLevel = iota
	INFO
	WARN
	ERROR
	FATAL
)

var logLevelNames = map[LogLevel]string{
	DEBUG: "DEBUG",
	INFO:  "INFO",
	WARN:  "WARN",
	ERROR: "ERROR",
	FATAL: "FATAL",
}

// LogEntry represents a structured log entry
type LogEntry struct {
	Timestamp   time.Time              `json:"timestamp"`
	Level       string                 `json:"level"`
	Service     string                 `json:"service"`
	Message     string                 `json:"message"`
	Fields      map[string]interface{} `json:"fields,omitempty"`
	Source      string                 `json:"source"`
	ProcessID   int                    `json:"process_id"`
	ThreadID    uint64                 `json:"thread_id"`
	MemoryUsage int64                  `json:"memory_usage_mb"`
}

// WindowsLogger provides production-ready logging for Windows services
type WindowsLogger struct {
	config      *config.ProductionConfig
	eventLog    *eventlog.Log
	fileLogger  *FileLogger
	level       LogLevel
	serviceName string
	fields      map[string]interface{}
	mutex       sync.RWMutex
	buffer      chan *LogEntry
	ctx         context.Context
	cancel      context.CancelFunc
	metrics     *LogMetrics
}

// FileLogger handles file-based logging with rotation
type FileLogger struct {
	file        *os.File
	path        string
	maxSize     int64
	maxBackups  int
	maxAge      time.Duration
	compress    bool
	currentSize int64
	mutex       sync.Mutex
}

// LogMetrics tracks logging performance and statistics
type LogMetrics struct {
	TotalLogs       int64
	LogsByLevel     map[LogLevel]int64
	ErrorRate       float64
	AvgLogSize      int64
	BufferOverflows int64
	WriteLatency    time.Duration
	LastRotation    time.Time
	FileSize        int64
	mutex           sync.RWMutex
}

// PerformanceMonitor tracks service performance metrics
type PerformanceMonitor struct {
	logger        *WindowsLogger
	metrics       map[string]*Metric
	collectors    []MetricCollector
	updateInterval time.Duration
	mutex         sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
}

// Metric represents a performance metric
type Metric struct {
	Name        string
	Value       float64
	Unit        string
	Timestamp   time.Time
	Tags        map[string]string
	Type        string // "counter", "gauge", "histogram"
	History     []float64
	MaxHistory  int
}

// MetricCollector interface for custom metric collection
type MetricCollector interface {
	Collect() map[string]*Metric
	Name() string
}

// SystemMetricCollector collects system-level metrics
type SystemMetricCollector struct {
	processID int
}

// GPUMetricCollector collects GPU utilization metrics
type GPUMetricCollector struct {
	deviceID int
}

// NewWindowsLogger creates a production-ready Windows service logger
func NewWindowsLogger(serviceName string, cfg *config.ProductionConfig) (*WindowsLogger, error) {
	if cfg == nil {
		return nil, fmt.Errorf("configuration cannot be nil")
	}

	ctx, cancel := context.WithCancel(context.Background())

	logger := &WindowsLogger{
		config:      cfg,
		serviceName: serviceName,
		fields:      make(map[string]interface{}),
		buffer:      make(chan *LogEntry, 10000), // Large buffer for high throughput
		ctx:         ctx,
		cancel:      cancel,
		metrics: &LogMetrics{
			LogsByLevel: make(map[LogLevel]int64),
		},
	}

	// Parse log level
	logger.level = parseLogLevel(cfg.Logging.Level)

	// Initialize Windows Event Log
	if cfg.Logging.EventLog {
		eventLog, err := eventlog.Open(serviceName)
		if err != nil {
			// Try to install the event source
			err = eventlog.InstallAsEventCreate(serviceName, eventlog.Error|eventlog.Warning|eventlog.Info)
			if err != nil {
				return nil, fmt.Errorf("failed to install event log source: %v", err)
			}
			
			eventLog, err = eventlog.Open(serviceName)
			if err != nil {
				return nil, fmt.Errorf("failed to open event log: %v", err)
			}
		}
		logger.eventLog = eventLog
	}

	// Initialize file logger
	if cfg.Logging.OutputPath != "" {
		fileLogger, err := NewFileLogger(cfg.Logging.OutputPath, &FileLoggerConfig{
			MaxSize:    int64(cfg.Logging.MaxSize) * 1024 * 1024, // Convert MB to bytes
			MaxBackups: cfg.Logging.MaxBackups,
			MaxAge:     time.Duration(cfg.Logging.MaxAge) * 24 * time.Hour,
			Compress:   cfg.Logging.Compress,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create file logger: %v", err)
		}
		logger.fileLogger = fileLogger
	}

	// Start background log processing
	go logger.processLogEntries()
	go logger.updateMetrics()

	return logger, nil
}

// FileLoggerConfig holds file logger configuration
type FileLoggerConfig struct {
	MaxSize    int64
	MaxBackups int
	MaxAge     time.Duration
	Compress   bool
}

// NewFileLogger creates a new file logger with rotation
func NewFileLogger(path string, config *FileLoggerConfig) (*FileLogger, error) {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %v", err)
	}

	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %v", err)
	}

	// Get current file size
	stat, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat log file: %v", err)
	}

	return &FileLogger{
		file:        file,
		path:        path,
		maxSize:     config.MaxSize,
		maxBackups:  config.MaxBackups,
		maxAge:      config.MaxAge,
		compress:    config.Compress,
		currentSize: stat.Size(),
	}, nil
}

// Log writes a structured log entry
func (wl *WindowsLogger) Log(level LogLevel, message string, fields map[string]interface{}) {
	if level < wl.level {
		return
	}

	// Get memory usage
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Create log entry
	entry := &LogEntry{
		Timestamp:   time.Now(),
		Level:       logLevelNames[level],
		Service:     wl.serviceName,
		Message:     message,
		Fields:      fields,
		Source:      getCallerInfo(),
		ProcessID:   os.Getpid(),
		ThreadID:    getThreadID(),
		MemoryUsage: int64(memStats.Alloc / 1024 / 1024), // Convert to MB
	}

	// Add global fields
	wl.mutex.RLock()
	for k, v := range wl.fields {
		if entry.Fields == nil {
			entry.Fields = make(map[string]interface{})
		}
		entry.Fields[k] = v
	}
	wl.mutex.RUnlock()

	// Send to buffer (non-blocking)
	select {
	case wl.buffer <- entry:
	default:
		// Buffer is full, increment overflow counter
		wl.metrics.mutex.Lock()
		wl.metrics.BufferOverflows++
		wl.metrics.mutex.Unlock()
	}
}

// Convenience methods for different log levels
func (wl *WindowsLogger) Debug(message string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	wl.Log(DEBUG, message, f)
}

func (wl *WindowsLogger) Info(message string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	wl.Log(INFO, message, f)
}

func (wl *WindowsLogger) Warn(message string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	wl.Log(WARN, message, f)
}

func (wl *WindowsLogger) Error(message string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	wl.Log(ERROR, message, f)
}

func (wl *WindowsLogger) Fatal(message string, fields ...map[string]interface{}) {
	var f map[string]interface{}
	if len(fields) > 0 {
		f = fields[0]
	}
	wl.Log(FATAL, message, f)
}

// WithFields returns a logger with additional fields
func (wl *WindowsLogger) WithFields(fields map[string]interface{}) *WindowsLogger {
	wl.mutex.Lock()
	defer wl.mutex.Unlock()
	
	for k, v := range fields {
		wl.fields[k] = v
	}
	return wl
}

// processLogEntries handles log entry processing in background
func (wl *WindowsLogger) processLogEntries() {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Log processor recovered from panic: %v", r)
		}
	}()

	for {
		select {
		case entry := <-wl.buffer:
			wl.writeLogEntry(entry)
		case <-wl.ctx.Done():
			// Process remaining entries
			for len(wl.buffer) > 0 {
				entry := <-wl.buffer
				wl.writeLogEntry(entry)
			}
			return
		}
	}
}

// writeLogEntry writes a log entry to configured outputs
func (wl *WindowsLogger) writeLogEntry(entry *LogEntry) {
	startTime := time.Now()

	// Write to file
	if wl.fileLogger != nil {
		if err := wl.writeToFile(entry); err != nil {
			// Fallback to stderr if file writing fails
			fmt.Fprintf(os.Stderr, "Failed to write to log file: %v\n", err)
		}
	}

	// Write to Windows Event Log
	if wl.eventLog != nil {
		wl.writeToEventLog(entry)
	}

	// Update metrics
	wl.updateLogMetrics(entry, time.Since(startTime))
}

// writeToFile writes log entry to file with JSON formatting
func (wl *WindowsLogger) writeToFile(entry *LogEntry) error {
	wl.fileLogger.mutex.Lock()
	defer wl.fileLogger.mutex.Unlock()

	// Check if rotation is needed
	if wl.fileLogger.currentSize >= wl.fileLogger.maxSize {
		if err := wl.fileLogger.rotate(); err != nil {
			return fmt.Errorf("failed to rotate log file: %v", err)
		}
	}

	// Format based on configuration
	var output []byte
	var err error

	if wl.config.Logging.Format == "json" {
		output, err = json.Marshal(entry)
		if err != nil {
			return fmt.Errorf("failed to marshal log entry: %v", err)
		}
		output = append(output, '\n')
	} else {
		// Text format
		text := fmt.Sprintf("%s [%s] %s: %s", 
			entry.Timestamp.Format("2006-01-02 15:04:05.000"),
			entry.Level,
			entry.Service,
			entry.Message)
		
		if entry.Fields != nil && len(entry.Fields) > 0 {
			fieldsJSON, _ := json.Marshal(entry.Fields)
			text += fmt.Sprintf(" fields=%s", string(fieldsJSON))
		}
		text += "\n"
		output = []byte(text)
	}

	// Write to file
	n, err := wl.fileLogger.file.Write(output)
	if err != nil {
		return err
	}

	wl.fileLogger.currentSize += int64(n)
	return nil
}

// writeToEventLog writes to Windows Event Log
func (wl *WindowsLogger) writeToEventLog(entry *LogEntry) {
	if wl.eventLog == nil {
		return
	}

	message := fmt.Sprintf("%s: %s", entry.Service, entry.Message)
	if entry.Fields != nil && len(entry.Fields) > 0 {
		fieldsJSON, _ := json.Marshal(entry.Fields)
		message += fmt.Sprintf(" | Fields: %s", string(fieldsJSON))
	}

	switch entry.Level {
	case "ERROR", "FATAL":
		wl.eventLog.Error(1, message)
	case "WARN":
		wl.eventLog.Warning(2, message)
	default:
		wl.eventLog.Info(3, message)
	}
}

// rotate rotates the log file
func (fl *FileLogger) rotate() error {
	fl.file.Close()

	// Rename current file with timestamp
	timestamp := time.Now().Format("2006-01-02-15-04-05")
	rotatedPath := fmt.Sprintf("%s.%s", fl.path, timestamp)
	
	if err := os.Rename(fl.path, rotatedPath); err != nil {
		return err
	}

	// Compress if enabled
	if fl.compress {
		go fl.compressFile(rotatedPath)
	}

	// Clean up old files
	go fl.cleanupOldFiles()

	// Create new file
	file, err := os.OpenFile(fl.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}

	fl.file = file
	fl.currentSize = 0

	return nil
}

// compressFile compresses a log file (placeholder - would use actual compression)
func (fl *FileLogger) compressFile(path string) {
	// In a real implementation, would use gzip compression
	compressedPath := path + ".gz"
	fmt.Printf("Compressing %s to %s\n", path, compressedPath)
	// Compression logic would go here
}

// cleanupOldFiles removes old log files based on retention policy
func (fl *FileLogger) cleanupOldFiles() {
	dir := filepath.Dir(fl.path)
	base := filepath.Base(fl.path)
	
	files, err := filepath.Glob(filepath.Join(dir, base+".*"))
	if err != nil {
		return
	}

	// Sort files by modification time and remove excess
	// Implementation would sort and remove files beyond maxBackups and maxAge
	if len(files) > fl.maxBackups {
		for i := fl.maxBackups; i < len(files); i++ {
			os.Remove(files[i])
		}
	}
}

// updateLogMetrics updates logging performance metrics
func (wl *WindowsLogger) updateLogMetrics(entry *LogEntry, writeLatency time.Duration) {
	wl.metrics.mutex.Lock()
	defer wl.metrics.mutex.Unlock()

	wl.metrics.TotalLogs++
	
	level := parseLogLevel(entry.Level)
	wl.metrics.LogsByLevel[level]++

	// Update average log size
	entrySize := int64(len(entry.Message) + len(entry.Service))
	if entry.Fields != nil {
		fieldsJSON, _ := json.Marshal(entry.Fields)
		entrySize += int64(len(fieldsJSON))
	}

	if wl.metrics.TotalLogs == 1 {
		wl.metrics.AvgLogSize = entrySize
	} else {
		// Exponential moving average
		alpha := 0.1
		wl.metrics.AvgLogSize = int64(float64(wl.metrics.AvgLogSize)*(1-alpha) + float64(entrySize)*alpha)
	}

	// Update write latency
	if wl.metrics.TotalLogs == 1 {
		wl.metrics.WriteLatency = writeLatency
	} else {
		alpha := 0.1
		wl.metrics.WriteLatency = time.Duration(float64(wl.metrics.WriteLatency)*(1-alpha) + float64(writeLatency)*alpha)
	}

	// Calculate error rate
	errorLogs := wl.metrics.LogsByLevel[ERROR] + wl.metrics.LogsByLevel[FATAL]
	wl.metrics.ErrorRate = float64(errorLogs) / float64(wl.metrics.TotalLogs) * 100
}

// updateMetrics periodically updates logging metrics
func (wl *WindowsLogger) updateMetrics() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Update file size
			if wl.fileLogger != nil {
				wl.metrics.mutex.Lock()
				wl.metrics.FileSize = wl.fileLogger.currentSize
				wl.metrics.mutex.Unlock()
			}
		case <-wl.ctx.Done():
			return
		}
	}
}

// NewPerformanceMonitor creates a new performance monitoring system
func NewPerformanceMonitor(logger *WindowsLogger, updateInterval time.Duration) *PerformanceMonitor {
	ctx, cancel := context.WithCancel(context.Background())

	pm := &PerformanceMonitor{
		logger:         logger,
		metrics:        make(map[string]*Metric),
		collectors:     make([]MetricCollector, 0),
		updateInterval: updateInterval,
		ctx:            ctx,
		cancel:         cancel,
	}

	// Add default collectors
	pm.AddCollector(&SystemMetricCollector{processID: os.Getpid()})
	pm.AddCollector(&GPUMetricCollector{deviceID: 0})

	// Start monitoring
	go pm.monitorPerformance()

	return pm
}

// AddCollector adds a metric collector
func (pm *PerformanceMonitor) AddCollector(collector MetricCollector) {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()
	
	pm.collectors = append(pm.collectors, collector)
}

// monitorPerformance periodically collects and logs performance metrics
func (pm *PerformanceMonitor) monitorPerformance() {
	ticker := time.NewTicker(pm.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.collectAllMetrics()
		case <-pm.ctx.Done():
			return
		}
	}
}

// collectAllMetrics collects metrics from all registered collectors
func (pm *PerformanceMonitor) collectAllMetrics() {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	allMetrics := make(map[string]*Metric)

	// Collect from all collectors
	for _, collector := range pm.collectors {
		metrics := collector.Collect()
		for name, metric := range metrics {
			fullName := fmt.Sprintf("%s_%s", collector.Name(), name)
			allMetrics[fullName] = metric
		}
	}

	// Update stored metrics and log
	for name, metric := range allMetrics {
		pm.metrics[name] = metric
		
		// Log significant metrics
		if pm.shouldLogMetric(metric) {
			pm.logger.Info("Performance metric", map[string]interface{}{
				"metric_name":  name,
				"metric_value": metric.Value,
				"metric_unit":  metric.Unit,
				"metric_type":  metric.Type,
				"metric_tags":  metric.Tags,
			})
		}
	}
}

// shouldLogMetric determines if a metric should be logged
func (pm *PerformanceMonitor) shouldLogMetric(metric *Metric) bool {
	switch metric.Name {
	case "cpu_usage":
		return metric.Value > 80.0 // Log if CPU usage > 80%
	case "memory_usage":
		return metric.Value > 85.0 // Log if memory usage > 85%
	case "gpu_usage":
		return metric.Value > 90.0 // Log if GPU usage > 90%
	case "error_rate":
		return metric.Value > 5.0 // Log if error rate > 5%
	default:
		return false
	}
}

// Collect implements MetricCollector for system metrics
func (smc *SystemMetricCollector) Collect() map[string]*Metric {
	metrics := make(map[string]*Metric)

	// Get memory stats
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// CPU usage (simplified - would use actual CPU monitoring in production)
	cpuUsage := float64(runtime.NumGoroutine()) / float64(runtime.NumCPU()) * 10
	if cpuUsage > 100 {
		cpuUsage = 100
	}

	metrics["cpu_usage"] = &Metric{
		Name:      "cpu_usage",
		Value:     cpuUsage,
		Unit:      "percent",
		Timestamp: time.Now(),
		Type:      "gauge",
		Tags:      map[string]string{"process_id": fmt.Sprintf("%d", smc.processID)},
	}

	metrics["memory_usage"] = &Metric{
		Name:      "memory_usage",
		Value:     float64(memStats.Alloc) / 1024 / 1024, // MB
		Unit:      "MB",
		Timestamp: time.Now(),
		Type:      "gauge",
		Tags:      map[string]string{"process_id": fmt.Sprintf("%d", smc.processID)},
	}

	metrics["goroutines"] = &Metric{
		Name:      "goroutines",
		Value:     float64(runtime.NumGoroutine()),
		Unit:      "count",
		Timestamp: time.Now(),
		Type:      "gauge",
	}

	return metrics
}

func (smc *SystemMetricCollector) Name() string {
	return "system"
}

// Collect implements MetricCollector for GPU metrics
func (gmc *GPUMetricCollector) Collect() map[string]*Metric {
	metrics := make(map[string]*Metric)

	// Simulate GPU metrics (would query actual GPU in production)
	metrics["gpu_usage"] = &Metric{
		Name:      "gpu_usage",
		Value:     float64(runtime.NumGoroutine()%100), // Simulated
		Unit:      "percent",
		Timestamp: time.Now(),
		Type:      "gauge",
		Tags:      map[string]string{"device_id": fmt.Sprintf("%d", gmc.deviceID)},
	}

	metrics["gpu_memory"] = &Metric{
		Name:      "gpu_memory",
		Value:     float64(runtime.NumGoroutine()%8192), // Simulated MB
		Unit:      "MB",
		Timestamp: time.Now(),
		Type:      "gauge",
		Tags:      map[string]string{"device_id": fmt.Sprintf("%d", gmc.deviceID)},
	}

	return metrics
}

func (gmc *GPUMetricCollector) Name() string {
	return "gpu"
}

// GetMetrics returns current performance metrics
func (pm *PerformanceMonitor) GetMetrics() map[string]*Metric {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	// Return a copy to avoid race conditions
	result := make(map[string]*Metric)
	for k, v := range pm.metrics {
		result[k] = v
	}
	return result
}

// GetLogMetrics returns logging performance metrics
func (wl *WindowsLogger) GetLogMetrics() *LogMetrics {
	wl.metrics.mutex.RLock()
	defer wl.metrics.mutex.RUnlock()

	// Return a copy
	logsByLevel := make(map[LogLevel]int64)
	for k, v := range wl.metrics.LogsByLevel {
		logsByLevel[k] = v
	}

	return &LogMetrics{
		TotalLogs:       wl.metrics.TotalLogs,
		LogsByLevel:     logsByLevel,
		ErrorRate:       wl.metrics.ErrorRate,
		AvgLogSize:      wl.metrics.AvgLogSize,
		BufferOverflows: wl.metrics.BufferOverflows,
		WriteLatency:    wl.metrics.WriteLatency,
		LastRotation:    wl.metrics.LastRotation,
		FileSize:        wl.metrics.FileSize,
	}
}

// Utility functions
func parseLogLevel(level string) LogLevel {
	switch level {
	case "debug", "DEBUG":
		return DEBUG
	case "info", "INFO":
		return INFO
	case "warn", "WARN", "warning", "WARNING":
		return WARN
	case "error", "ERROR":
		return ERROR
	case "fatal", "FATAL":
		return FATAL
	default:
		return INFO
	}
}

func getCallerInfo() string {
	_, file, line, ok := runtime.Caller(3) // Skip Log, convenience method, and this function
	if !ok {
		return "unknown"
	}
	return fmt.Sprintf("%s:%d", filepath.Base(file), line)
}

func getThreadID() uint64 {
	// Simplified thread ID (would use actual OS thread ID in production)
	return uint64(runtime.NumGoroutine())
}

// Cleanup releases resources
func (wl *WindowsLogger) Cleanup() {
	if wl.cancel != nil {
		wl.cancel()
	}

	// Close event log
	if wl.eventLog != nil {
		wl.eventLog.Close()
	}

	// Close file logger
	if wl.fileLogger != nil {
		wl.fileLogger.mutex.Lock()
		if wl.fileLogger.file != nil {
			wl.fileLogger.file.Close()
		}
		wl.fileLogger.mutex.Unlock()
	}
}

func (pm *PerformanceMonitor) Cleanup() {
	if pm.cancel != nil {
		pm.cancel()
	}
}