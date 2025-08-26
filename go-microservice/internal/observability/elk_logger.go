// elk_logger.go - Enterprise ELK Stack Observability Integration
// Version 2.0 - Structured logging for Elasticsearch, Logstash, and Kibana
package observability

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

// LogLevel represents the severity of log entries
type LogLevel string

const (
	LogLevelTrace LogLevel = "trace"
	LogLevelDebug LogLevel = "debug"
	LogLevelInfo  LogLevel = "info"
	LogLevelWarn  LogLevel = "warn"
	LogLevelError LogLevel = "error"
	LogLevelFatal LogLevel = "fatal"
	LogLevelPanic LogLevel = "panic"
)

// LogEntry represents a structured log entry for ELK Stack
type LogEntry struct {
	// Standard fields
	Timestamp    time.Time   `json:"@timestamp"`
	Level        LogLevel    `json:"level"`
	Message      string      `json:"message"`
	Service      string      `json:"service"`
	Version      string      `json:"version"`
	Environment  string      `json:"environment"`
	
	// Request context
	TraceID      string      `json:"trace_id,omitempty"`
	SpanID       string      `json:"span_id,omitempty"`
	RequestID    string      `json:"request_id,omitempty"`
	UserID       string      `json:"user_id,omitempty"`
	SessionID    string      `json:"session_id,omitempty"`
	
	// gRPC specific fields
	GRPCMethod   string      `json:"grpc_method,omitempty"`
	GRPCCode     codes.Code  `json:"grpc_code,omitempty"`
	GRPCMessage  string      `json:"grpc_message,omitempty"`
	
	// HTTP/Network fields
	ClientIP     string      `json:"client_ip,omitempty"`
	UserAgent    string      `json:"user_agent,omitempty"`
	Duration     float64     `json:"duration_ms,omitempty"`
	
	// Performance metrics
	CPUUsage     float64     `json:"cpu_usage_percent,omitempty"`
	MemoryUsage  int64       `json:"memory_usage_bytes,omitempty"`
	GPUUsage     float64     `json:"gpu_usage_percent,omitempty"`
	GPUMemory    int64       `json:"gpu_memory_bytes,omitempty"`
	
	// Error details
	ErrorType    string      `json:"error_type,omitempty"`
	ErrorCode    string      `json:"error_code,omitempty"`
	StackTrace   string      `json:"stack_trace,omitempty"`
	
	// Business logic fields
	JobID        string      `json:"job_id,omitempty"`
	JobType      string      `json:"job_type,omitempty"`
	DocumentID   string      `json:"document_id,omitempty"`
	
	// Custom fields
	Fields       map[string]interface{} `json:"fields,omitempty"`
	
	// Metadata
	Host         string      `json:"host"`
	PID          int         `json:"pid"`
	GoVersion    string      `json:"go_version"`
	Goroutines   int         `json:"goroutines"`
}

// ELKLoggerConfig holds configuration for the ELK logger
type ELKLoggerConfig struct {
	ServiceName     string    `json:"service_name"`
	ServiceVersion  string    `json:"service_version"`
	Environment     string    `json:"environment"`
	LogLevel        LogLevel  `json:"log_level"`
	EnableConsole   bool      `json:"enable_console"`
	EnableFile      bool      `json:"enable_file"`
	LogFilePath     string    `json:"log_file_path"`
	EnableMetrics   bool      `json:"enable_metrics"`
	EnableTracing   bool      `json:"enable_tracing"`
	
	// ELK specific configuration
	ElasticsearchURL    string `json:"elasticsearch_url"`
	ElasticsearchIndex  string `json:"elasticsearch_index"`
	LogstashURL         string `json:"logstash_url"`
	EnableDirectES      bool   `json:"enable_direct_es"`  // Send logs directly to Elasticsearch
	EnableLogstash      bool   `json:"enable_logstash"`   // Send logs via Logstash
	
	// Performance tuning
	BufferSize          int           `json:"buffer_size"`
	FlushInterval       time.Duration `json:"flush_interval"`
	MaxConcurrentWrites int           `json:"max_concurrent_writes"`
}

// ELKLogger provides enterprise-grade structured logging for ELK Stack
type ELKLogger struct {
	config    ELKLoggerConfig
	logger    zerolog.Logger
	hostname  string
	pid       int
	
	// Performance monitoring
	stats     LoggerStats
	statsLock sync.RWMutex
	
	// Buffer for batch processing
	logBuffer chan LogEntry
	wg        sync.WaitGroup
}

type LoggerStats struct {
	TotalLogs       int64     `json:"total_logs"`
	LogsByLevel     map[LogLevel]int64 `json:"logs_by_level"`
	ErrorsLogged    int64     `json:"errors_logged"`
	AverageLatency  float64   `json:"average_latency_ms"`
	LastLogTime     time.Time `json:"last_log_time"`
	BufferUtilization float64 `json:"buffer_utilization"`
}

// NewELKLogger creates a new ELK Stack logger instance
func NewELKLogger(config ELKLoggerConfig) (*ELKLogger, error) {
	// Set up zerolog
	var logger zerolog.Logger
	
	// Configure output
	if config.EnableConsole {
		consoleWriter := zerolog.ConsoleWriter{
			Out:        os.Stdout,
			TimeFormat: time.RFC3339,
		}
		logger = zerolog.New(consoleWriter)
	} else {
		logger = zerolog.New(os.Stdout)
	}
	
	// Set log level
	switch config.LogLevel {
	case LogLevelTrace:
		logger = logger.Level(zerolog.TraceLevel)
	case LogLevelDebug:
		logger = logger.Level(zerolog.DebugLevel)
	case LogLevelInfo:
		logger = logger.Level(zerolog.InfoLevel)
	case LogLevelWarn:
		logger = logger.Level(zerolog.WarnLevel)
	case LogLevelError:
		logger = logger.Level(zerolog.ErrorLevel)
	case LogLevelFatal:
		logger = logger.Level(zerolog.FatalLevel)
	case LogLevelPanic:
		logger = logger.Level(zerolog.PanicLevel)
	default:
		logger = logger.Level(zerolog.InfoLevel)
	}
	
	logger = logger.With().Timestamp().Caller().Logger()
	
	// Get hostname
	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown"
	}
	
	elkLogger := &ELKLogger{
		config:    config,
		logger:    logger,
		hostname:  hostname,
		pid:       os.Getpid(),
		logBuffer: make(chan LogEntry, config.BufferSize),
		stats: LoggerStats{
			LogsByLevel: make(map[LogLevel]int64),
		},
	}
	
	// Start background log processing
	elkLogger.startLogProcessor()
	
	log.Info().
		Str("service", config.ServiceName).
		Str("version", config.ServiceVersion).
		Str("environment", config.Environment).
		Msg("ELK Logger initialized")
	
	return elkLogger, nil
}

// Log creates and processes a structured log entry
func (e *ELKLogger) Log(level LogLevel, message string) *LogEntryBuilder {
	return &LogEntryBuilder{
		logger: e,
		entry: LogEntry{
			Timestamp:   time.Now(),
			Level:       level,
			Message:     message,
			Service:     e.config.ServiceName,
			Version:     e.config.ServiceVersion,
			Environment: e.config.Environment,
			Host:        e.hostname,
			PID:         e.pid,
			GoVersion:   runtime.Version(),
			Goroutines:  runtime.NumGoroutine(),
			Fields:      make(map[string]interface{}),
		},
	}
}

// Convenience methods for different log levels
func (e *ELKLogger) Trace(message string) *LogEntryBuilder { return e.Log(LogLevelTrace, message) }
func (e *ELKLogger) Debug(message string) *LogEntryBuilder { return e.Log(LogLevelDebug, message) }
func (e *ELKLogger) Info(message string) *LogEntryBuilder  { return e.Log(LogLevelInfo, message) }
func (e *ELKLogger) Warn(message string) *LogEntryBuilder  { return e.Log(LogLevelWarn, message) }
func (e *ELKLogger) Error(message string) *LogEntryBuilder { return e.Log(LogLevelError, message) }
func (e *ELKLogger) Fatal(message string) *LogEntryBuilder { return e.Log(LogLevelFatal, message) }
func (e *ELKLogger) Panic(message string) *LogEntryBuilder { return e.Log(LogLevelPanic, message) }

// LogEntryBuilder provides a fluent interface for building log entries
type LogEntryBuilder struct {
	logger *ELKLogger
	entry  LogEntry
}

func (b *LogEntryBuilder) TraceID(traceID string) *LogEntryBuilder {
	b.entry.TraceID = traceID
	return b
}

func (b *LogEntryBuilder) SpanID(spanID string) *LogEntryBuilder {
	b.entry.SpanID = spanID
	return b
}

func (b *LogEntryBuilder) RequestID(requestID string) *LogEntryBuilder {
	b.entry.RequestID = requestID
	return b
}

func (b *LogEntryBuilder) UserID(userID string) *LogEntryBuilder {
	b.entry.UserID = userID
	return b
}

func (b *LogEntryBuilder) SessionID(sessionID string) *LogEntryBuilder {
	b.entry.SessionID = sessionID
	return b
}

func (b *LogEntryBuilder) GRPCMethod(method string) *LogEntryBuilder {
	b.entry.GRPCMethod = method
	return b
}

func (b *LogEntryBuilder) GRPCStatus(code codes.Code, message string) *LogEntryBuilder {
	b.entry.GRPCCode = code
	b.entry.GRPCMessage = message
	return b
}

func (b *LogEntryBuilder) ClientIP(ip string) *LogEntryBuilder {
	b.entry.ClientIP = ip
	return b
}

func (b *LogEntryBuilder) Duration(duration time.Duration) *LogEntryBuilder {
	b.entry.Duration = float64(duration.Nanoseconds()) / 1e6 // Convert to milliseconds
	return b
}

func (b *LogEntryBuilder) Error(err error) *LogEntryBuilder {
	if err != nil {
		b.entry.ErrorType = fmt.Sprintf("%T", err)
		b.entry.Fields["error"] = err.Error()
		
		// Extract stack trace if available
		if b.logger.config.LogLevel == LogLevelTrace || b.logger.config.LogLevel == LogLevelDebug {
			b.entry.StackTrace = getStackTrace()
		}
	}
	return b
}

func (b *LogEntryBuilder) JobID(jobID string) *LogEntryBuilder {
	b.entry.JobID = jobID
	return b
}

func (b *LogEntryBuilder) JobType(jobType string) *LogEntryBuilder {
	b.entry.JobType = jobType
	return b
}

func (b *LogEntryBuilder) DocumentID(documentID string) *LogEntryBuilder {
	b.entry.DocumentID = documentID
	return b
}

func (b *LogEntryBuilder) Field(key string, value interface{}) *LogEntryBuilder {
	if b.entry.Fields == nil {
		b.entry.Fields = make(map[string]interface{})
	}
	b.entry.Fields[key] = value
	return b
}

func (b *LogEntryBuilder) Fields(fields map[string]interface{}) *LogEntryBuilder {
	if b.entry.Fields == nil {
		b.entry.Fields = make(map[string]interface{})
	}
	for k, v := range fields {
		b.entry.Fields[k] = v
	}
	return b
}

func (b *LogEntryBuilder) PerformanceMetrics(cpuUsage float64, memoryUsage int64, gpuUsage float64, gpuMemory int64) *LogEntryBuilder {
	b.entry.CPUUsage = cpuUsage
	b.entry.MemoryUsage = memoryUsage
	b.entry.GPUUsage = gpuUsage
	b.entry.GPUMemory = gpuMemory
	return b
}

// Send finalizes and sends the log entry
func (b *LogEntryBuilder) Send() {
	start := time.Now()
	
	// Send to buffer for processing
	select {
	case b.logger.logBuffer <- b.entry:
		// Successfully buffered
	default:
		// Buffer is full, log synchronously
		b.logger.processLogEntry(b.entry)
	}
	
	// Update statistics
	b.logger.updateStats(b.entry.Level, time.Since(start))
}

// processLogEntry handles the actual log processing and forwarding
func (e *ELKLogger) processLogEntry(entry LogEntry) {
	// Log to zerolog first (for local debugging and monitoring)
	var zerologEvent *zerolog.Event
	
	switch entry.Level {
	case LogLevelTrace:
		zerologEvent = e.logger.Trace()
	case LogLevelDebug:
		zerologEvent = e.logger.Debug()
	case LogLevelInfo:
		zerologEvent = e.logger.Info()
	case LogLevelWarn:
		zerologEvent = e.logger.Warn()
	case LogLevelError:
		zerologEvent = e.logger.Error()
	case LogLevelFatal:
		zerologEvent = e.logger.Fatal()
	case LogLevelPanic:
		zerologEvent = e.logger.Panic()
	default:
		zerologEvent = e.logger.Info()
	}
	
	// Add structured fields
	zerologEvent = zerologEvent.
		Str("service", entry.Service).
		Str("version", entry.Version).
		Str("environment", entry.Environment)
	
	if entry.TraceID != "" {
		zerologEvent = zerologEvent.Str("trace_id", entry.TraceID)
	}
	if entry.UserID != "" {
		zerologEvent = zerologEvent.Str("user_id", entry.UserID)
	}
	if entry.GRPCMethod != "" {
		zerologEvent = zerologEvent.Str("grpc_method", entry.GRPCMethod)
	}
	if entry.Duration > 0 {
		zerologEvent = zerologEvent.Float64("duration_ms", entry.Duration)
	}
	if entry.JobID != "" {
		zerologEvent = zerologEvent.Str("job_id", entry.JobID)
	}
	
	// Add custom fields
	for key, value := range entry.Fields {
		zerologEvent = zerologEvent.Interface(key, value)
	}
	
	zerologEvent.Msg(entry.Message)
	
	// Send to external systems (Elasticsearch, Logstash) if configured
	if e.config.EnableDirectES || e.config.EnableLogstash {
		go e.sendToExternalSystems(entry)
	}
}

// sendToExternalSystems forwards logs to Elasticsearch or Logstash
func (e *ELKLogger) sendToExternalSystems(entry LogEntry) {
	// This would be implemented to send logs to Elasticsearch or Logstash
	// For brevity, this is a placeholder implementation
	
	// Convert to JSON
	jsonData, err := json.Marshal(entry)
	if err != nil {
		e.logger.Error().Err(err).Msg("Failed to marshal log entry for external systems")
		return
	}
	
	// Send to Elasticsearch directly (if enabled)
	if e.config.EnableDirectES && e.config.ElasticsearchURL != "" {
		// Implementation would use Elasticsearch client to send data
		// This is a placeholder
		fmt.Printf("Would send to Elasticsearch: %s\n", string(jsonData))
	}
	
	// Send to Logstash (if enabled)
	if e.config.EnableLogstash && e.config.LogstashURL != "" {
		// Implementation would use HTTP client to send data to Logstash
		// This is a placeholder
		fmt.Printf("Would send to Logstash: %s\n", string(jsonData))
	}
}

// startLogProcessor starts background goroutines for processing logs
func (e *ELKLogger) startLogProcessor() {
	// Start buffer processor
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		for entry := range e.logBuffer {
			e.processLogEntry(entry)
		}
	}()
}

// updateStats updates logger performance statistics
func (e *ELKLogger) updateStats(level LogLevel, latency time.Duration) {
	e.statsLock.Lock()
	defer e.statsLock.Unlock()
	
	e.stats.TotalLogs++
	e.stats.LogsByLevel[level]++
	e.stats.LastLogTime = time.Now()
	
	if level == LogLevelError || level == LogLevelFatal || level == LogLevelPanic {
		e.stats.ErrorsLogged++
	}
	
	// Update average latency (simple moving average)
	latencyMs := float64(latency.Nanoseconds()) / 1e6
	e.stats.AverageLatency = (e.stats.AverageLatency*float64(e.stats.TotalLogs-1) + latencyMs) / float64(e.stats.TotalLogs)
	
	// Update buffer utilization
	e.stats.BufferUtilization = float64(len(e.logBuffer)) / float64(cap(e.logBuffer)) * 100
}

// GetStats returns current logger statistics
func (e *ELKLogger) GetStats() LoggerStats {
	e.statsLock.RLock()
	defer e.statsLock.RUnlock()
	return e.stats
}

// gRPC Interceptors for automatic logging
func (e *ELKLogger) UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		start := time.Now()
		
		// Extract metadata
		traceID := e.extractTraceID(ctx)
		userID := e.extractUserID(ctx)
		clientIP := e.extractClientIP(ctx)
		
		// Call the handler
		resp, err := handler(ctx, req)
		
		duration := time.Since(start)
		
		// Build log entry
		logBuilder := e.Info("gRPC request completed").
			GRPCMethod(info.FullMethod).
			Duration(duration).
			ClientIP(clientIP)
		
		if traceID != "" {
			logBuilder = logBuilder.TraceID(traceID)
		}
		if userID != "" {
			logBuilder = logBuilder.UserID(userID)
		}
		
		if err != nil {
			st := status.Convert(err)
			logBuilder = logBuilder.
				GRPCStatus(st.Code(), st.Message()).
				Error(err).
				Field("grpc_details", st.Details())
			
			if st.Code() != codes.OK {
				logBuilder = e.Error("gRPC request failed").
					GRPCMethod(info.FullMethod).
					Duration(duration).
					Error(err)
			}
		} else {
			logBuilder = logBuilder.GRPCStatus(codes.OK, "OK")
		}
		
		logBuilder.Send()
		
		return resp, err
	}
}

// Helper methods for extracting context information
func (e *ELKLogger) extractTraceID(ctx context.Context) string {
	if md, ok := metadata.FromIncomingContext(ctx); ok {
		if values := md.Get("x-trace-id"); len(values) > 0 {
			return values[0]
		}
		if values := md.Get("trace-id"); len(values) > 0 {
			return values[0]
		}
	}
	return ""
}

func (e *ELKLogger) extractUserID(ctx context.Context) string {
	// Extract from context values (set by auth interceptor)
	if userID, ok := ctx.Value("user_id").(string); ok {
		return userID
	}
	return ""
}

func (e *ELKLogger) extractClientIP(ctx context.Context) string {
	if peer, ok := peer.FromContext(ctx); ok {
		return peer.Addr.String()
	}
	return ""
}

// getStackTrace returns the current stack trace
func getStackTrace() string {
	buf := make([]byte, 4096)
	n := runtime.Stack(buf, false)
	return string(buf[:n])
}

// Close gracefully shuts down the logger
func (e *ELKLogger) Close() {
	close(e.logBuffer)
	e.wg.Wait()
}