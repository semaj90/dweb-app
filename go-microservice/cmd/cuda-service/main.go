//go:build experimental
// +build experimental

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"

	// Prometheus
	"github.com/prometheus/client_golang/prometheus"
	promhttp "github.com/prometheus/client_golang/prometheus/promhttp"

	// Optional NVML (graceful fallback)
	"sync/atomic"

	"runtime"
	"sync"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/load"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

type CudaService struct {
	db          *gorm.DB
	redis       *redis.Client
	workerPath  string
	healthCache map[string]bool
	startTime   time.Time
	// GPU / CPU ring buffers
	bufMu sync.RWMutex
	gpuSamples []GpuPoint
	cpuSamples []CpuPoint
	cacheSamples []CachePoint
	// Alert history (in-memory) and thresholds
	alerts []Alert
	thresholds AlertThresholds
	lastAlertEmitted map[string]int64
	// Rolling stats for anomaly detection (Welford algorithm)
	gpuCount int64
	gpuMean float64
	gpuM2 float64
	jobsCount int64
	jobsMean float64
	jobsM2 float64
	// Worker / process monitoring
	workerNames []string
	workerStats map[int32]ProcStat
	// Profiling snapshot ring buffer (in-memory) for quick UI (recent N)
	profilingSnapshots []profilingSnapshot
}

type ProcStat struct {
	PID        int32   `json:"pid"`
	Name       string  `json:"name"`
	CPUPercent float64 `json:"cpu_percent"`
	RSSBytes   uint64  `json:"rss_bytes"`
	VMSBytes   uint64  `json:"vms_bytes"`
	NumThreads int32   `json:"num_threads"`
	CreateTime int64   `json:"create_time"`
}

type GpuPoint struct {
	T int64 `json:"t"`
	Util float64 `json:"util"`
	MemUsed uint64 `json:"mem_used"`
	MemTotal uint64 `json:"mem_total"`
	TempC int `json:"temp_c,omitempty"`
	PowerMilliW int `json:"power_mw,omitempty"`
	ClockSMMHz int `json:"clock_sm_mhz,omitempty"`
}
type CpuPoint struct { T int64 `json:"t"`; Goroutines int `json:"goroutines"`; GOMAXPROCS int `json:"gomaxprocs"` }
type CachePoint struct { T int64 `json:"t"`; Size int `json:"size"` }

// Alert & threshold definitions
type Alert struct {
	Timestamp int64   `json:"ts"`
	Level     string  `json:"level"` // warn | crit
	Type      string  `json:"type"`
	Message   string  `json:"message"`
	Anomaly   bool    `json:"anomaly,omitempty"`
	ZScore    *float64 `json:"zscore,omitempty"`
}
type AlertThresholds struct {
	GpuTempWarn         int
	GpuTempCrit         int
	MemUsedWarnPct      float64
	MemUsedCritPct      float64
	LoadWarnMultiplier  float64
	LoadCritMultiplier  float64
	RedisMemWarnMB      int64
	RedisMemCritMB      int64
	JobsPerMinWarn      int
	JobsPerMinCrit      int
	HistoryTTLSeconds   int
	HistoryMaxEntries   int
	GpuUtilAnomZ       float64
	JobsRateAnomZ      float64
	AlertRetentionSeconds int
	SuppressWindowSec   int
}

type CudaRequest struct {
	JobID string    `json:"jobId"`
	Type  string    `json:"type"` // "embedding", "similarity", "som_train", "autoindex"
	Data  []float32 `json:"data"`
}

type CudaResponse struct {
	JobID     string    `json:"jobId"`
	Type      string    `json:"type"`
	Vector    []float32 `json:"vector"`
	Status    string    `json:"status"`
	Timestamp int64     `json:"timestamp"`
	Error     string    `json:"error,omitempty"`
}

type EmbeddingJob struct {
	ID        uint      `json:"id" gorm:"primaryKey"`
	JobID     string    `json:"job_id" gorm:"uniqueIndex"`
	Type      string    `json:"type"`
	Status    string    `json:"status"` // "pending", "processing", "completed", "failed"
	InputData string    `json:"input_data"`
	Result    string    `json:"result"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

func NewCudaService() *CudaService {
	// Database connection
	dsn := os.Getenv("POSTGRES_DSN")
	if dsn == "" {
		dsn = "host=localhost user=postgres password=123456 dbname=legal_ai_db port=5432 sslmode=disable"
	}

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Printf("Failed to connect to database: %v", err)
	} else {
		// Auto-migrate the schema
		db.AutoMigrate(&EmbeddingJob{})
	}

	// Redis connection
	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}

	rdb := redis.NewClient(&redis.Options{
		Addr: redisAddr,
	})

	// CUDA worker path
	workerPath := os.Getenv("CUDA_WORKER_PATH")
	if workerPath == "" {
		workerPath = "../cuda-worker/cuda-worker.exe"
	}

	return &CudaService{
		db:          db,
		redis:       rdb,
		workerPath:  workerPath,
		healthCache: make(map[string]bool),
		startTime:   time.Now(),
	gpuSamples:  make([]GpuPoint,0,720),
	cpuSamples:  make([]CpuPoint,0,720),
	cacheSamples: make([]CachePoint,0,720),
	alerts:      make([]Alert,0,100),
	thresholds: AlertThresholds{
		GpuTempWarn: 80, GpuTempCrit: 85,
		MemUsedWarnPct: 85, MemUsedCritPct: 92,
		LoadWarnMultiplier: 1.0, LoadCritMultiplier: 1.2,
		RedisMemWarnMB: 700, RedisMemCritMB: 900,
		JobsPerMinWarn: 300, JobsPerMinCrit: 500,
		HistoryTTLSeconds: 0, // if >0 could add expiry metadata
		HistoryMaxEntries: 720,
		GpuUtilAnomZ: 3.0, JobsRateAnomZ: 3.0, AlertRetentionSeconds: 86400,
		SuppressWindowSec: 30,
	},
	lastAlertEmitted: make(map[string]int64),
		workerNames: parseWorkerNames(),
		workerStats: make(map[int32]ProcStat),
	}
}

// parseWorkerNames reads MONITOR_WORKER_NAMES (comma-separated)
func parseWorkerNames() []string {
	val := os.Getenv("MONITOR_WORKER_NAMES")
	if val == "" { return nil }
	parts := strings.Split(val, ",")
	out := make([]string,0,len(parts))
	for _, p := range parts { p = strings.TrimSpace(p); if p != "" { out = append(out,p) } }
	return out
}

// After construction, override thresholds from environment variables if present
func (cs *CudaService) applyThresholdEnv() {
	// helper
	setInt := func(env string, apply func(int)) { if v := os.Getenv(env); v != "" { if n, err := strconv.Atoi(v); err==nil { apply(n) } } }
	setFloat := func(env string, apply func(float64)) { if v := os.Getenv(env); v != "" { if f, err := strconv.ParseFloat(v,64); err==nil { apply(f) } } }
	setInt("ALERT_GPU_TEMP_WARN", func(n int){ cs.thresholds.GpuTempWarn = n })
	setInt("ALERT_GPU_TEMP_CRIT", func(n int){ cs.thresholds.GpuTempCrit = n })
	setFloat("ALERT_MEM_USED_WARN_PCT", func(f float64){ cs.thresholds.MemUsedWarnPct = f })
	setFloat("ALERT_MEM_USED_CRIT_PCT", func(f float64){ cs.thresholds.MemUsedCritPct = f })
	setFloat("ALERT_LOAD_WARN_MULT", func(f float64){ cs.thresholds.LoadWarnMultiplier = f })
	setFloat("ALERT_LOAD_CRIT_MULT", func(f float64){ cs.thresholds.LoadCritMultiplier = f })
	setInt("ALERT_REDIS_MEM_WARN_MB", func(n int){ cs.thresholds.RedisMemWarnMB = int64(n) })
	setInt("ALERT_REDIS_MEM_CRIT_MB", func(n int){ cs.thresholds.RedisMemCritMB = int64(n) })
	setInt("ALERT_JOBS_PER_MIN_WARN", func(n int){ cs.thresholds.JobsPerMinWarn = n })
	setInt("ALERT_JOBS_PER_MIN_CRIT", func(n int){ cs.thresholds.JobsPerMinCrit = n })
	setInt("ALERT_HISTORY_MAX_ENTRIES", func(n int){ if n>0 { cs.thresholds.HistoryMaxEntries = n } })
	setFloat("ALERT_GPU_UTIL_ANOM_Z", func(f float64){ cs.thresholds.GpuUtilAnomZ = f })
	setFloat("ALERT_JOBS_RATE_ANOM_Z", func(f float64){ cs.thresholds.JobsRateAnomZ = f })
	setInt("ALERT_RETENTION_SECONDS", func(n int){ if n>0 { cs.thresholds.AlertRetentionSeconds = n } })
	setInt("ALERT_SUPPRESS_WINDOW_SEC", func(n int){ if n>0 { cs.thresholds.SuppressWindowSec = n } })
}

func (cs *CudaService) healthCheck() gin.HandlerFunc {
	return func(c *gin.Context) {
		health := map[string]interface{}{
			"service":   "cuda-service",
			"timestamp": time.Now().Unix(),
			"status":    "healthy",
			"checks":    map[string]bool{},
		}

		// Check database
		if cs.db != nil {
			sqlDB, err := cs.db.DB()
			if err == nil && sqlDB.Ping() == nil {
				health["checks"].(map[string]bool)["database"] = true
			} else {
				health["checks"].(map[string]bool)["database"] = false
				health["status"] = "degraded"
			}
		} else {
			health["checks"].(map[string]bool)["database"] = false
		}

		// Check Redis
		if cs.redis != nil {
			_, err := cs.redis.Ping(context.Background()).Result()
			health["checks"].(map[string]bool)["redis"] = err == nil
			if err != nil {
				health["status"] = "degraded"
			}
		}

		// Check CUDA worker
		cudaHealthy := cs.testCudaWorker()
		health["checks"].(map[string]bool)["cuda_worker"] = cudaHealthy
		if !cudaHealthy {
			health["status"] = "degraded"
		}

		if health["status"] == "healthy" {
			c.JSON(200, health)
		} else {
			c.JSON(503, health)
		}
	}
}

func (cs *CudaService) testCudaWorker() bool {
	// Test with simple embedding request
	testReq := CudaRequest{
		JobID: "health-check",
		Type:  "embedding",
		Data:  []float32{1.0, 2.0, 3.0, 4.0},
	}

	response, err := cs.executeCudaWorker(testReq)
	return err == nil && response.Status == "success"
}

func (cs *CudaService) executeCudaWorker(req CudaRequest) (*CudaResponse, error) {
	// Convert request to JSON
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Execute CUDA worker
	cmd := exec.Command(cs.workerPath)
	cmd.Stdin = strings.NewReader(string(reqJSON))

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("cuda worker execution failed: %v", err)
	}

	// Parse response
	var response CudaResponse
	if err := json.Unmarshal(output, &response); err != nil {
		return nil, fmt.Errorf("failed to parse cuda worker response: %v", err)
	}

	return &response, nil
}

func (cs *CudaService) processEmbedding() gin.HandlerFunc {
	return func(c *gin.Context) {
	reqCounter.Inc()
		var req CudaRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(400, gin.H{"error": "Invalid request format"})
			return
		}

		// Generate job ID if not provided
		if req.JobID == "" {
			req.JobID = fmt.Sprintf("emb_%d", time.Now().UnixNano())
		}

		// Save job to database (if available)
		if cs.db != nil {
			inputJSON, _ := json.Marshal(req.Data)
			job := EmbeddingJob{
				JobID:     req.JobID,
				Type:      req.Type,
				Status:    "processing",
				InputData: string(inputJSON),
			}
			cs.db.Create(&job)
		}

		// Execute CUDA worker
		start := time.Now()
		response, err := cs.executeCudaWorker(req)
		if err != nil {
			// Update job status if DB available
			if cs.db != nil {
				cs.db.Model(&EmbeddingJob{}).Where("job_id = ?", req.JobID).Updates(map[string]interface{}{
					"status": "failed",
					"result": err.Error(),
				})
			}
			errorCounter.Inc()
			c.JSON(500, gin.H{
				"error":  "CUDA processing failed",
				"detail": err.Error(),
				"jobId":  req.JobID,
			})
			return
		}
		dur := time.Since(start)
		jobDuration.Observe(dur.Seconds())

		// Update job status and save result
		if cs.db != nil {
			resultJSON, _ := json.Marshal(response)
			cs.db.Model(&EmbeddingJob{}).Where("job_id = ?", req.JobID).Updates(map[string]interface{}{
				"status": "completed",
				"result": string(resultJSON),
			})
		}

		// Cache result in Redis (if available)
		if cs.redis != nil {
			resultJSON, _ := json.Marshal(response)
			cs.redis.Set(context.Background(), fmt.Sprintf("cuda:result:%s", req.JobID), resultJSON, time.Hour)
		}

	// GPU metrics update (best-effort)
	updateGPUMetrics()
	c.JSON(200, response)
	}
}

func (cs *CudaService) getJobStatus() gin.HandlerFunc {
	return func(c *gin.Context) {
		jobID := c.Param("jobId")

		// Try Redis cache first
		if cs.redis != nil {
			cachedResult, err := cs.redis.Get(context.Background(), fmt.Sprintf("cuda:result:%s", jobID)).Result()
			if err == nil {
				var response CudaResponse
				if json.Unmarshal([]byte(cachedResult), &response) == nil {
					c.JSON(200, response)
					return
				}
			}
		}

		// Fall back to database
		if cs.db != nil {
			var job EmbeddingJob
			if err := cs.db.Where("job_id = ?", jobID).First(&job).Error; err == nil {
				c.JSON(200, gin.H{
					"jobId":     job.JobID,
					"type":      job.Type,
					"status":    job.Status,
					"result":    job.Result,
					"createdAt": job.CreatedAt,
					"updatedAt": job.UpdatedAt,
				})
				return
			}
		}

		c.JSON(404, gin.H{"error": "Job not found"})
	}
}

func main() {
	// Initialize service
	service := NewCudaService()
	globalCudaService = service
	service.startWorkerSampler()
	service.startProfilingSampler() // start periodic profiling snapshot persistence (stub or real depending on build tag)

	initProm()
	initNVML()

	// Setup Gin router
	r := gin.Default()

	// CORS middleware
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Metrics auth middleware
	metricsAuth := metricsAuthMiddleware()

	// Protected metrics endpoints
	r.GET("/metrics", metricsAuth, gin.WrapH(promhttp.Handler()))
	r.GET("/metrics/enhanced", metricsAuth, service.enhancedMetricsHandler)
	r.GET("/metrics/history", metricsAuth, service.metricsHistoryHandler)
	r.GET("/metrics/alerts", metricsAuth, service.alertsHandler)
	r.GET("/metrics/anomalies", metricsAuth, service.anomaliesHandler)
	r.GET("/metrics/workers", metricsAuth, service.workerStatsHandler)
	r.GET("/metrics/gpu/engines", metricsAuth, service.gpuEnginesHandler)
	r.GET("/metrics/profiling/summary", metricsAuth, service.profilingSummaryHandler)
	r.GET("/metrics/profiling/history", metricsAuth, service.profilingHistoryHandler)
	r.GET("/metrics/wasm", metricsAuth, service.wasmMetricsHandler)

	// Public routes
	r.GET("/health", service.healthCheck())
	r.GET("/gpu/runtime", service.gpuRuntimeHandler)
	r.GET("/gpu/series", service.gpuSeriesHandler)
	r.POST("/vectorize", service.processEmbedding())
	r.POST("/embedding", service.processEmbedding()) // Alias
	r.GET("/job/:jobId", service.getJobStatus())

	// Service info
	r.GET("/info", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"service":     "cuda-service",
			"version":     "1.0.0",
			"description": "CUDA GPU processing microservice for Legal AI",
			"endpoints": map[string]string{
				"POST /vectorize":       "Process embeddings with CUDA",
				"POST /embedding":       "Alias for /vectorize",
				"GET /job/:jobId":       "Get job status and results",
				"GET /health":           "Service health check",
				"GET /gpu/runtime":      "Latest GPU/CPU/cache sample",
				"GET /gpu/series":       "Ring buffer GPU/CPU/cache series",
				"GET /metrics/enhanced": "Enhanced system metrics (CPU per-core, cache, memory)",
				"GET /info":             "Service information",
				"GET /metrics/history": "Historical enhanced metrics snapshots (newest first limited)",
				"GET /metrics/alerts":  "Recent server-side evaluated alerts",
			},
		})
	})

	// Start server
	port := os.Getenv("CUDA_SERVICE_PORT")
	if port == "" {
		port = "8096"
	}

	log.Printf("🚀 CUDA Service starting on port %s", port)
	log.Printf("📊 Health check: http://localhost:%s/health", port)
	log.Printf("📈 Prometheus metrics: http://localhost:%s/metrics", port)
	log.Printf("🔧 CUDA Worker: %s", service.workerPath)

	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

// background sampler for worker process stats
func (cs *CudaService) startWorkerSampler() {
	if len(cs.workerNames) == 0 { return }
	interval := 5 * time.Second
	if v := os.Getenv("WORKER_SCAN_INTERVAL_SEC"); v != "" { if n, err := strconv.Atoi(v); err==nil && n>0 { interval = time.Duration(n)*time.Second } }
	go func() {
		for {
			cs.sampleWorkers()
			time.Sleep(interval)
		}
	}()
}

func (cs *CudaService) sampleWorkers() {
	procs, err := process.Processes(); if err != nil { return }
	want := map[string]bool{}
	for _, n := range cs.workerNames { want[strings.ToLower(n)] = true }
	stats := make(map[int32]ProcStat)
	for _, p := range procs {
		name, err := p.Name(); if err != nil { continue }
		if !want[strings.ToLower(name)] { continue }
		cpuPct, _ := p.CPUPercent()
		memInfo, _ := p.MemoryInfo()
		threads, _ := p.NumThreads()
		ct, _ := p.CreateTime()
		stats[p.Pid] = ProcStat{PID: p.Pid, Name: name, CPUPercent: cpuPct, RSSBytes: memInfo.RSS, VMSBytes: memInfo.VMS, NumThreads: threads, CreateTime: ct}
	}
	cs.bufMu.Lock(); cs.workerStats = stats; cs.bufMu.Unlock()
}

func (cs *CudaService) workerStatsHandler(c *gin.Context) {
	cs.bufMu.RLock(); defer cs.bufMu.RUnlock()
	list := make([]ProcStat,0,len(cs.workerStats))
	for _, st := range cs.workerStats { list = append(list, st) }
	c.JSON(200, gin.H{"workers": list, "count": len(list)})
}

// --------------------------------------------------------------------------------
// Prometheus metrics & GPU utilization helpers
// --------------------------------------------------------------------------------

var (
	reqCounter = prometheus.NewCounter(prometheus.CounterOpts{Name: "cuda_service_requests_total", Help: "Total embedding requests"})
	errorCounter = prometheus.NewCounter(prometheus.CounterOpts{Name: "cuda_service_errors_total", Help: "Failed embedding requests"})
	jobDuration = prometheus.NewHistogram(prometheus.HistogramOpts{Name: "cuda_service_job_duration_seconds", Help: "Embedding job latency", Buckets: prometheus.DefBuckets})
	gpuUtilGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_gpu_utilization_percent", Help: "GPU utilization percent (device 0)"})
	gpuMemUsedGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_gpu_memory_used_bytes", Help: "GPU memory used bytes (device 0)"})
	gpuMemTotalGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_gpu_memory_total_bytes", Help: "GPU total memory bytes (device 0)"})
	queueLenGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_service_queue_length", Help: "Offline queue length"})
	cacheSizeGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_service_cache_size", Help: "In-memory cache size"})
	uptimeGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_service_uptime_seconds", Help: "Service uptime in seconds"})
	alertWarnGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_alerts_warn_total", Help: "Current WARN alerts stored (server-side)"})
	alertCritGauge = prometheus.NewGauge(prometheus.GaugeOpts{Name: "cuda_alerts_crit_total", Help: "Current CRIT alerts stored (server-side)"})
	alertTripsCounter = prometheus.NewCounterVec(prometheus.CounterOpts{Name: "cuda_alert_trips_total", Help: "Total alert trips by type/level/anomaly"}, []string{"type","level","anomaly"})
	nvmlInitialized atomic.Bool
)

var globalCudaService *CudaService

func initProm() {
	prometheus.MustRegister(reqCounter, errorCounter, jobDuration, gpuUtilGauge, gpuMemUsedGauge, gpuMemTotalGauge, queueLenGauge, cacheSizeGauge, uptimeGauge, alertWarnGauge, alertCritGauge, alertTripsCounter)
	// Background sampler for queue/cache/uptime
	go func(svc *CudaService) {
		ticker := time.NewTicker(5 * time.Second)
		for range ticker.C {
			cacheSizeGauge.Set(float64(getCacheSize(svc)))
			uptimeGauge.Set(time.Since(svc.startTime).Seconds())
			updateGPUMetrics()
			svc.captureLocalSamples()
			// update alert gauges
			if svc != nil {
				svc.bufMu.RLock()
				warnCount := 0; critCount := 0
				for _, a := range svc.alerts { if a.Level == "warn" { warnCount++ } else if a.Level == "crit" { critCount++ } }
				svc.bufMu.RUnlock()
				alertWarnGauge.Set(float64(warnCount))
				alertCritGauge.Set(float64(critCount))
			}
		}
	}(globalCudaService)
}

func initNVML() {
	if ret := nvml.Init(); ret == nvml.SUCCESS {
		nvmlInitialized.Store(true)
		log.Printf("🟢 NVML initialized for GPU metrics")
	} else {
		log.Printf("⚠️ NVML init failed: %v (GPU metrics will be best-effort)", nvml.ErrorString(ret))
	}
}

func updateGPUMetrics() {
	if !nvmlInitialized.Load() { return }
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS || count == 0 { return }
	dev, ret := nvml.DeviceGetHandleByIndex(0)
	if ret != nvml.SUCCESS { return }
	if util, ret := nvml.DeviceGetUtilizationRates(dev); ret == nvml.SUCCESS {
		gpuUtilGauge.Set(float64(util.Gpu))
	}
	if mem, ret := nvml.DeviceGetMemoryInfo(dev); ret == nvml.SUCCESS {
		gpuMemUsedGauge.Set(float64(mem.Used))
		gpuMemTotalGauge.Set(float64(mem.Total))
	}
}

// Simple cache size: count redis keys under prefix if redis disabled, else approximate job entries
func getCacheSize(svc *CudaService) int {
	if svc == nil { return 0 }
	// Basic heuristic: count embedding jobs completed in last minute if DB available
	if svc.db != nil {
		type res struct{ C int64 }
		var r res
		sqlDB, err := svc.db.DB(); if err==nil {
			// skip heavy queries; just return 0 if fail
			_ = sqlDB.QueryRow("SELECT COUNT(*) FROM embedding_jobs WHERE updated_at > NOW() - INTERVAL '1 minute'").Scan(&r.C)
			return int(r.C)
		}
	}
	return 0
}

// captureLocalSamples appends current GPU/CPU/cache values to ring buffers
func (cs *CudaService) captureLocalSamples() {
	now := time.Now().UnixMilli()
	var util float64
	var used, total uint64
	var tempC, powerMw, clockSM int
	if nvmlInitialized.Load() {
		if cnt, ret := nvml.DeviceGetCount(); ret == nvml.SUCCESS && cnt>0 {
			if dev, ret2 := nvml.DeviceGetHandleByIndex(0); ret2 == nvml.SUCCESS {
				if u, ret3 := nvml.DeviceGetUtilizationRates(dev); ret3 == nvml.SUCCESS { util = float64(u.Gpu) }
				if m, ret4 := nvml.DeviceGetMemoryInfo(dev); ret4 == nvml.SUCCESS { used = m.Used; total = m.Total }
				if t, ret5 := nvml.DeviceGetTemperature(dev, nvml.TEMPERATURE_GPU); ret5 == nvml.SUCCESS { tempC = int(t) }
				if p, ret6 := nvml.DeviceGetPowerUsage(dev); ret6 == nvml.SUCCESS { powerMw = int(p) }
				if clk, ret7 := nvml.DeviceGetClockInfo(dev, nvml.CLOCK_SM); ret7 == nvml.SUCCESS { clockSM = int(clk) }
			}
		}
	}
	g := GpuPoint{T: now, Util: util, MemUsed: used, MemTotal: total, TempC: tempC, PowerMilliW: powerMw, ClockSMMHz: clockSM}
	cpoint := CpuPoint{T: now, Goroutines: runtime.NumGoroutine(), GOMAXPROCS: runtime.GOMAXPROCS(0)}
	cacheSize := getCacheSize(cs)
	cachePoint := CachePoint{T: now, Size: cacheSize}
	cs.bufMu.Lock()
	cs.gpuSamples = append(cs.gpuSamples, g)
	cs.cpuSamples = append(cs.cpuSamples, cpoint)
	cs.cacheSamples = append(cs.cacheSamples, cachePoint)
	if len(cs.gpuSamples) > 720 { cs.gpuSamples = cs.gpuSamples[len(cs.gpuSamples)-720:] }
	if len(cs.cpuSamples) > 720 { cs.cpuSamples = cs.cpuSamples[len(cs.cpuSamples)-720:] }
	if len(cs.cacheSamples) > 720 { cs.cacheSamples = cs.cacheSamples[len(cs.cacheSamples)-720:] }
	cs.bufMu.Unlock()
}

// gpuRuntimeHandler returns latest snapshot of GPU + CPU + cache state
func (cs *CudaService) gpuRuntimeHandler(c *gin.Context) {
	cs.bufMu.RLock()
	var lastGpu *GpuPoint
	if n := len(cs.gpuSamples); n>0 { lg := cs.gpuSamples[n-1]; lastGpu = &lg }
	var lastCpu *CpuPoint
	if n := len(cs.cpuSamples); n>0 { lc := cs.cpuSamples[n-1]; lastCpu = &lc }
	var lastCache *CachePoint
	if n := len(cs.cacheSamples); n>0 { lca := cs.cacheSamples[n-1]; lastCache = &lca }
	cs.bufMu.RUnlock()
	c.JSON(200, gin.H{"gpu": lastGpu, "cpu": lastCpu, "cache": lastCache, "uptime_sec": time.Since(cs.startTime).Seconds()})
}

// gpuSeriesHandler returns ring buffer series for charts (compressed minimal)
func (cs *CudaService) gpuSeriesHandler(c *gin.Context) {
	cs.bufMu.RLock()
	resp := struct { Gpu []GpuPoint `json:"gpu"`; Cpu []CpuPoint `json:"cpu"`; Cache []CachePoint `json:"cache"` }{Gpu: cs.gpuSamples, Cpu: cs.cpuSamples, Cache: cs.cacheSamples}
	cs.bufMu.RUnlock()
	c.JSON(200, resp)
}

// enhancedMetricsHandler returns richer system metrics for frontend
func (cs *CudaService) enhancedMetricsHandler(c *gin.Context) {
	// Per-core CPU percentages (short interval)
	cpuPercents, _ := cpu.Percent(150*time.Millisecond, true)
	// Load averages
	loadAvg, _ := load.Avg()
	// Memory stats
	vm, _ := mem.VirtualMemory()
	// Current cache size (reuse helper)
	cacheSize := getCacheSize(cs)
	// Latest GPU sample
	var latestGpu *GpuPoint
	cs.bufMu.RLock()
	if n := len(cs.gpuSamples); n>0 { lg := cs.gpuSamples[n-1]; latestGpu = &lg }
	cs.bufMu.RUnlock()

	// Additional Redis cache stats
	redisKeys := 0
	redisMemory := int64(0)
	if cs.redis != nil {
		// Count keys with scan (limit to 10k for safety)
		ctx := context.Background()
		var cursor uint64
		for i := 0; i < 20; i++ { // up to 20 * 500 = 10k keys scanned
			keys, cur, err := cs.redis.Scan(ctx, cursor, "cuda:result:*", 500).Result()
			if err != nil { break }
			redisKeys += len(keys)
			cursor = cur
			if cursor == 0 { break }
		}
		if info, err := cs.redis.Info(ctx, "memory").Result(); err == nil {
			// parse used_memory:
			lines := strings.Split(info, "\n")
			for _, ln := range lines {
				if strings.HasPrefix(ln, "used_memory:") {
					fmt.Sscanf(strings.TrimSpace(ln), "used_memory:%d", &redisMemory)
					break
				}
			}
		}
	}

	// Job status breakdown (DB)
	statuses := map[string]int{}
	if cs.db != nil {
		type row struct{ Status string; C int }
		var rows []row
		// ignoring errors for lightweight optional stats
		cs.db.Raw("SELECT status, COUNT(*) as c FROM embedding_jobs GROUP BY status").Scan(&rows)
		for _, r := range rows { statuses[r.Status] = r.C }
	}

	// Build metrics object
	metrics := gin.H{
		"timestamp": time.Now().UnixMilli(),
		"cpu": gin.H{
			"per_core_percent": cpuPercents,
			"num_cores": runtime.NumCPU(),
			"gomaxprocs": runtime.GOMAXPROCS(0),
		},
		"load": gin.H{
			"load1": loadAvg.Load1,
			"load5": loadAvg.Load5,
			"load15": loadAvg.Load15,
		},
		"memory": gin.H{
			"total": vm.Total,
			"used": vm.Used,
			"used_percent": vm.UsedPercent,
			"available": vm.Available,
		},
		"cache": gin.H{
			"recent_embedding_jobs_minute": cacheSize,
			"redis_result_keys": redisKeys,
			"redis_used_memory_bytes": redisMemory,
			"job_status": statuses,
		},
		"gpu": latestGpu,
		"uptime_seconds": time.Since(cs.startTime).Seconds(),
	}
	// Evaluate alerts
	alertsNow := cs.evaluateAlerts(latestGpu, vm.UsedPercent, loadAvg.Load1, runtime.NumCPU(), redisMemory, cacheSize)
	if len(alertsNow) > 0 { metrics["alerts"] = alertsNow }
	// Persist to Redis history list if configured
	cs.persistEnhanced(metrics)
	c.JSON(200, metrics)
}

// evaluateAlerts computes alerts based on thresholds and returns newly generated ones.
func (cs *CudaService) evaluateAlerts(gpu *GpuPoint, memUsedPct float64, load1 float64, cores int, redisMemBytes int64, jobsPerMin int) []Alert {
	th := cs.thresholds
	now := time.Now().UnixMilli()
	var out []Alert
	webhook := os.Getenv("ALERT_WEBHOOK_URL")
	sendWebhook := func(a Alert) {
		if webhook == "" { return }
		payload, _ := json.Marshal(map[string]interface{}{
			"timestamp": a.Timestamp,
			"level": a.Level,
			"type": a.Type,
			"message": a.Message,
			"anomaly": a.Anomaly,
			"zscore": a.ZScore,
		})
		go func() {
			req, err := http.NewRequest("POST", webhook, bytes.NewReader(payload))
			if err != nil { return }
			req.Header.Set("Content-Type", "application/json")
			client := &http.Client{ Timeout: 3 * time.Second }
			resp, err := client.Do(req); if err == nil { resp.Body.Close() }
		}()
	}
	// helper to persist alerts (in-memory + Redis ZSET)
	persist := func(a Alert) {
		cs.bufMu.Lock()
		cs.alerts = append(cs.alerts, a)
		if len(cs.alerts) > 100 { cs.alerts = cs.alerts[len(cs.alerts)-100:] }
		cs.bufMu.Unlock()
		if cs.redis != nil {
			ctx := context.Background()
			key := "cuda:alerts:zset"
			payload, _ := json.Marshal(a)
			cs.redis.ZAdd(ctx, key, &redis.Z{Score: float64(a.Timestamp), Member: string(payload)})
			// prune by timestamp if retention configured
			if th.AlertRetentionSeconds > 0 {
				cutoff := float64(time.Now().Add(-time.Duration(th.AlertRetentionSeconds) * time.Second).UnixMilli())
				cs.redis.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%f", cutoff))
			}
		}
	}
	add := func(level, typ, msg string){
		key := fmt.Sprintf("%s|%s|false", typ, level)
		window := int64(th.SuppressWindowSec * 1000)
		if window > 0 {
			cs.bufMu.Lock(); last := cs.lastAlertEmitted[key]; if now-last < window { cs.bufMu.Unlock(); return }; cs.lastAlertEmitted[key]=now; cs.bufMu.Unlock()
		}
		a := Alert{Timestamp: now, Level: level, Type: typ, Message: msg}
		out = append(out, a)
		persist(a)
		alertTripsCounter.WithLabelValues(typ, level, "false").Inc()
		if level == "crit" || os.Getenv("ALERT_NOTIFY_ALL") == "1" { sendWebhook(a) }
	}
	if gpu != nil && gpu.TempC > 0 {
		if gpu.TempC >= th.GpuTempCrit {
			add("crit", "gpu_temp", fmt.Sprintf("GPU temperature %d°C", gpu.TempC))
		} else if gpu.TempC >= th.GpuTempWarn {
			add("warn", "gpu_temp", fmt.Sprintf("GPU temperature %d°C", gpu.TempC))
		}
	}
	if memUsedPct >= th.MemUsedCritPct {
		add("crit", "memory", fmt.Sprintf("Memory usage %.1f%%", memUsedPct))
	} else if memUsedPct >= th.MemUsedWarnPct {
		add("warn", "memory", fmt.Sprintf("Memory usage %.1f%%", memUsedPct))
	}
	if cores > 0 {
		if load1 >= float64(cores)*th.LoadCritMultiplier {
			add("crit", "load", fmt.Sprintf("Load1 %.2f > %.0f cores", load1, float64(cores)))
		} else if load1 >= float64(cores)*th.LoadWarnMultiplier {
			add("warn", "load", fmt.Sprintf("Load1 high %.2f", load1))
		}
	}
	redisMB := redisMemBytes / 1024 / 1024
	if redisMB >= th.RedisMemCritMB {
		add("crit", "redis_mem", fmt.Sprintf("Redis memory %dMB", redisMB))
	} else if redisMB >= th.RedisMemWarnMB {
		add("warn", "redis_mem", fmt.Sprintf("Redis memory %dMB", redisMB))
	}
	if jobsPerMin >= th.JobsPerMinCrit {
		add("crit", "jobs_rate", fmt.Sprintf("Embedding jobs %d/min", jobsPerMin))
	} else if jobsPerMin >= th.JobsPerMinWarn {
		add("warn", "jobs_rate", fmt.Sprintf("Embedding jobs %d/min", jobsPerMin))
	}
	// Anomaly detection (rolling z-score)
	if gpu != nil {
		cs.updateRollingStats(&cs.gpuCount, &cs.gpuMean, &cs.gpuM2, gpu.Util)
		if cs.gpuCount > 30 {
			z := cs.zScore(cs.gpuCount, cs.gpuMean, cs.gpuM2, gpu.Util)
			if math.Abs(z) >= th.GpuUtilAnomZ {
				key := fmt.Sprintf("%s|%s|%v", "gpu_util_anom", "warn", true)
				window := int64(th.SuppressWindowSec * 1000)
				if window > 0 {
					cs.bufMu.Lock(); last := cs.lastAlertEmitted[key]; if now-last < window { cs.bufMu.Unlock() } else { cs.lastAlertEmitted[key]=now; cs.bufMu.Unlock(); zc := z; a := Alert{Timestamp: now, Level: "warn", Type: "gpu_util_anom", Message: fmt.Sprintf("GPU util anomaly %.1f%% (z=%.2f)", gpu.Util, z), Anomaly: true, ZScore: &zc}; out = append(out, a); persist(a); alertTripsCounter.WithLabelValues("gpu_util_anom","warn","true").Inc(); if os.Getenv("ALERT_NOTIFY_ANOMALIES") == "1" { sendWebhook(a) } }
				} else { zc := z; a := Alert{Timestamp: now, Level: "warn", Type: "gpu_util_anom", Message: fmt.Sprintf("GPU util anomaly %.1f%% (z=%.2f)", gpu.Util, z), Anomaly: true, ZScore: &zc}; out = append(out, a); persist(a); alertTripsCounter.WithLabelValues("gpu_util_anom","warn","true").Inc(); if os.Getenv("ALERT_NOTIFY_ANOMALIES") == "1" { sendWebhook(a) } }
			}
		}
	}
	cs.updateRollingStats(&cs.jobsCount, &cs.jobsMean, &cs.jobsM2, float64(jobsPerMin))
	if cs.jobsCount > 30 {
		z := cs.zScore(cs.jobsCount, cs.jobsMean, cs.jobsM2, float64(jobsPerMin))
		if math.Abs(z) >= th.JobsRateAnomZ {
			key := fmt.Sprintf("%s|%s|%v", "jobs_rate_anom", "warn", true)
			window := int64(th.SuppressWindowSec * 1000)
			if window > 0 {
				cs.bufMu.Lock(); last := cs.lastAlertEmitted[key]; if now-last < window { cs.bufMu.Unlock() } else { cs.lastAlertEmitted[key]=now; cs.bufMu.Unlock(); zc := z; a := Alert{Timestamp: now, Level: "warn", Type: "jobs_rate_anom", Message: fmt.Sprintf("Jobs/min anomaly %d (z=%.2f)", jobsPerMin, z), Anomaly: true, ZScore: &zc}; out = append(out, a); persist(a); alertTripsCounter.WithLabelValues("jobs_rate_anom","warn","true").Inc(); if os.Getenv("ALERT_NOTIFY_ANOMALIES") == "1" { sendWebhook(a) } }
			} else { zc := z; a := Alert{Timestamp: now, Level: "warn", Type: "jobs_rate_anom", Message: fmt.Sprintf("Jobs/min anomaly %d (z=%.2f)", jobsPerMin, z), Anomaly: true, ZScore: &zc}; out = append(out, a); persist(a); alertTripsCounter.WithLabelValues("jobs_rate_anom","warn","true").Inc(); if os.Getenv("ALERT_NOTIFY_ANOMALIES") == "1" { sendWebhook(a) } }
		}
	}
	return out
}

// updateRollingStats updates Welford stats for anomaly detection
func (cs *CudaService) updateRollingStats(count *int64, mean *float64, m2 *float64, x float64) {
	*count = *count + 1
	delta := x - *mean
	*mean += delta / float64(*count)
	delta2 := x - *mean
	*m2 += delta * delta2
}

// zScore computes z for a sample given Welford aggregate
func (cs *CudaService) zScore(count int64, mean, m2 float64, x float64) float64 {
	if count < 2 { return 0 }
	variance := m2 / float64(count-1)
	if variance <= 0 { return 0 }
	std := math.Sqrt(variance)
	if std == 0 { return 0 }
	return (x - mean) / std
}

// persistEnhanced stores serialized metrics JSON into Redis list retaining max entries.
func (cs *CudaService) persistEnhanced(metrics gin.H) {
	if cs.redis == nil { return }
	b, err := json.Marshal(metrics); if err != nil { return }
	ctx := context.Background()
	const key = "cuda:metrics:enhanced:history"
	if err := cs.redis.LPush(ctx, key, string(b)).Err(); err != nil { return }
	cs.redis.LTrim(ctx, key, 0, int64(cs.thresholds.HistoryMaxEntries-1))
}

// metricsHistoryHandler returns last N persisted enhanced metric snapshots.
func (cs *CudaService) metricsHistoryHandler(c *gin.Context) {
	if cs.redis == nil { c.JSON(503, gin.H{"error":"redis not configured"}); return }
	limitStr := c.Query("limit")
	limit := 60
	if limitStr != "" { if v, err := strconv.Atoi(limitStr); err==nil && v>0 && v<=cs.thresholds.HistoryMaxEntries { limit = v } }
	ctx := context.Background()
	vals, err := cs.redis.LRange(ctx, "cuda:metrics:enhanced:history", 0, int64(limit-1)).Result()
	if err != nil { c.JSON(500, gin.H{"error": err.Error()}); return }
	// entries newest first; return in chronological order
	out := make([]map[string]interface{}, 0, len(vals))
	for i := len(vals)-1; i>=0; i-- {
		var m map[string]interface{}
		if json.Unmarshal([]byte(vals[i]), &m) == nil { out = append(out, m) }
	}
	c.JSON(200, gin.H{"history": out, "count": len(out)})
}

// --------------------------- Profiling Persistence ---------------------------
// startProfilingSampler periodically captures profiling snapshots (stub or real) and stores them
// in-memory (bounded) and in Redis list (if configured) for history/trend visualization.
func (cs *CudaService) startProfilingSampler() {
	interval := 15 * time.Second
	if v := os.Getenv("PROFILING_SNAPSHOT_INTERVAL_SEC"); v != "" { if n, err := strconv.Atoi(v); err==nil && n>0 { interval = time.Duration(n)*time.Second } }
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for range ticker.C {
			snap := getProfilingSnapshot()
			cs.bufMu.Lock()
			cs.profilingSnapshots = append(cs.profilingSnapshots, snap)
			if len(cs.profilingSnapshots) > 200 { cs.profilingSnapshots = cs.profilingSnapshots[len(cs.profilingSnapshots)-200:] }
			cs.bufMu.Unlock()
			cs.persistProfilingSnapshot(snap)
		}
	}()
}

// persistProfilingSnapshot pushes snapshot JSON into Redis list retaining max entries.
func (cs *CudaService) persistProfilingSnapshot(s profilingSnapshot) {
	if cs.redis == nil { return }
	b, err := json.Marshal(s); if err != nil { return }
	ctx := context.Background()
	key := "cuda:profiling:history"
	if err := cs.redis.LPush(ctx, key, string(b)).Err(); err != nil { return }
	// keep at most 500 entries (approx ~2 hours at 15s interval) unless env overrides
	maxEntries := 500
	if v := os.Getenv("PROFILING_HISTORY_MAX"); v != "" { if n, err := strconv.Atoi(v); err==nil && n>0 { maxEntries = n } }
	cs.redis.LTrim(ctx, key, 0, int64(maxEntries-1))
}

// profilingHistoryHandler exposes recent profiling snapshots (chronological)
func (cs *CudaService) profilingHistoryHandler(c *gin.Context) {
	limit := 100
	if v := c.Query("limit"); v != "" { if n, err := strconv.Atoi(v); err==nil && n>0 && n<=1000 { limit = n } }
	// Redis preferred for longer history
	var out []profilingSnapshot
	if cs.redis != nil {
		ctx := context.Background()
		vals, err := cs.redis.LRange(ctx, "cuda:profiling:history", 0, int64(limit-1)).Result()
		if err == nil {
			for i := len(vals)-1; i>=0; i-- { // reverse to chronological
				var s profilingSnapshot
				if json.Unmarshal([]byte(vals[i]), &s) == nil { out = append(out, s) }
			}
		}
	}
	// If Redis empty or not configured, fall back to in-memory ring
	if len(out) == 0 {
		cs.bufMu.RLock(); snaps := cs.profilingSnapshots; cs.bufMu.RUnlock()
		if len(snaps) > 0 {
			if len(snaps) > limit { snaps = snaps[len(snaps)-limit:] }
			out = append(out, snaps...)
		}
	}
	c.JSON(200, gin.H{"history": out, "count": len(out)})
}

// alertsHandler returns recent alerts.
func (cs *CudaService) alertsHandler(c *gin.Context) {
	cs.bufMu.RLock()
	defer cs.bufMu.RUnlock()
	// copy alerts
	list := make([]Alert, len(cs.alerts))
	copy(list, cs.alerts)
	counts := map[string]int{"warn":0, "crit":0}
	for _, a := range list { counts[a.Level]++ }
	c.JSON(200, gin.H{"alerts": list, "counts": counts})
}

// anomaliesHandler exposes current rolling statistics and last z-scores
func (cs *CudaService) anomaliesHandler(c *gin.Context) {
	std := func(count int64, m2 float64) float64 { if count < 2 { return 0 }; v := m2/float64(count-1); if v <= 0 { return 0 }; return math.Sqrt(v) }
	gpuStd := std(cs.gpuCount, cs.gpuM2)
	jobsStd := std(cs.jobsCount, cs.jobsM2)
	var lastGpu *GpuPoint
	cs.bufMu.RLock(); if n:=len(cs.gpuSamples); n>0 { lg:=cs.gpuSamples[n-1]; lastGpu=&lg }; cs.bufMu.RUnlock()
	lastJobs := getCacheSize(cs)
	var gpuZ float64; if lastGpu!=nil && gpuStd>0 { gpuZ = (lastGpu.Util - cs.gpuMean)/gpuStd }
	var jobsZ float64; if jobsStd>0 { jobsZ = (float64(lastJobs) - cs.jobsMean)/jobsStd }
	c.JSON(200, gin.H{
		"gpu_util": gin.H{"count": cs.gpuCount, "mean": cs.gpuMean, "std": gpuStd, "last_sample": lastGpu, "last_z": gpuZ, "threshold_z": cs.thresholds.GpuUtilAnomZ},
		"jobs_rate": gin.H{"count": cs.jobsCount, "mean": cs.jobsMean, "std": jobsStd, "last_jobs_per_min": lastJobs, "last_z": jobsZ, "threshold_z": cs.thresholds.JobsRateAnomZ},
	})
}

// metricsAuthMiddleware enforces optional API key for metrics endpoints
func metricsAuthMiddleware() gin.HandlerFunc {
	key := os.Getenv("METRICS_API_KEY")
	if key == "" { return func(c *gin.Context){ c.Next() } }
	return func(c *gin.Context){
		candidate := c.GetHeader("X-API-Key")
		auth := c.GetHeader("Authorization")
		if candidate == key || (strings.HasPrefix(strings.ToLower(auth), "bearer ") && strings.TrimSpace(auth[7:]) == key) {
			c.Next(); return
		}
		c.AbortWithStatusJSON(401, gin.H{"error":"unauthorized"})
	}
}

// gpuEnginesHandler returns best-effort per-engine utilization (graphics/compute) if NVML exposes it.
func (cs *CudaService) gpuEnginesHandler(c *gin.Context) {
	if !nvmlInitialized.Load() { c.JSON(503, gin.H{"error":"nvml not initialized"}); return }
	count, ret := nvml.DeviceGetCount(); if ret != nvml.SUCCESS || count==0 { c.JSON(500, gin.H{"error":"no devices"}); return }
	dev, ret := nvml.DeviceGetHandleByIndex(0); if ret != nvml.SUCCESS { c.JSON(500, gin.H{"error":"device handle"}); return }
	// NVML per-engine stats: using UtilizationRates already; stub extended fields.
	util, _ := nvml.DeviceGetUtilizationRates(dev)
	clocksGraphics, _ := nvml.DeviceGetClockInfo(dev, nvml.CLOCK_GRAPHICS)
	clocksSM, _ := nvml.DeviceGetClockInfo(dev, nvml.CLOCK_SM)
	clocksMem, _ := nvml.DeviceGetClockInfo(dev, nvml.CLOCK_MEM)
	power, _ := nvml.DeviceGetPowerUsage(dev)
	mem, _ := nvml.DeviceGetMemoryInfo(dev)

	// Per-process utilization (best-effort). Signature: DeviceGetProcessUtilization(dev, lastSeenTimestamp) ([]ProcessUtilizationSample, Return)
	var procUtil []map[string]interface{}
	if samples, ret2 := nvml.DeviceGetProcessUtilization(dev, 0); ret2 == nvml.SUCCESS {
		for _, pi := range samples {
			procUtil = append(procUtil, map[string]interface{}{
				"pid": pi.Pid,
				"timestamp": pi.TimeStamp,
				"sm_util": pi.SmUtil,
				"mem_util": pi.MemUtil,
				"enc_util": pi.EncUtil,
				"dec_util": pi.DecUtil,
			})
		}
	}
	c.JSON(200, gin.H{
		"device_index": 0,
		"engines": gin.H{
			"graphics_clock_mhz": clocksGraphics,
			"sm_clock_mhz": clocksSM,
			"mem_clock_mhz": clocksMem,
		},
		"utilization": gin.H{"gpu_percent": util.Gpu, "memory_percent": util.Memory},
		"power_watts": float64(power)/1000,
		"memory": gin.H{"used_bytes": mem.Used, "total_bytes": mem.Total},
		"process_utilization": procUtil,
	})
}

// profilingSummaryHandler stubs memory optimization / cublas profiling summary.
// Real implementation would integrate CUPTI or cublas API counters; here we provide placeholders.
// profilingSnapshot represents a point-in-time GPU profiling capture (stub unless built with cupti tag)
type profilingSnapshot struct {
	Timestamp int64 `json:"ts"`
	Enabled bool `json:"enabled"`
	KernelSamples int `json:"kernel_samples"`
	TensorCoreUtil float64 `json:"tensor_core_util"`
	DramThroughputGBs float64 `json:"dram_throughput_gbs"`
	OccupancyAvg float64 `json:"occupancy_avg"`
	Notes []string `json:"notes"`
}

// getProfilingSnapshot returns a stub unless built with real CUPTI integration (build tag controlled)
func getProfilingSnapshot() profilingSnapshot {
	// In real implementation (file with //go:build cupti) we'd gather CUPTI counters
	return profilingSnapshot{
		Timestamp: time.Now().UnixMilli(),
		Enabled: false,
		KernelSamples: 0,
		TensorCoreUtil: 0,
		DramThroughputGBs: 0,
		OccupancyAvg: 0,
		Notes: []string{"Build with -tags cupti to enable real profiling", "Add cupti_profiler.go implementing getProfilingSnapshot()"},
	}
}

func (cs *CudaService) profilingSummaryHandler(c *gin.Context) {
	snap := getProfilingSnapshot()
	c.JSON(200, gin.H{
		"snapshot": snap,
		"suggested_next_steps": []string{
			"Add cupti_profiler.go with //go:build cupti to collect kernel runtime stats",
			"Track per-kernel duration, grid/block sizes, achieved occupancy",
			"Derive tensor core utilization via HMMA instruction counts / perf metrics",
			"Compute DRAM throughput from bytes transferred / interval",
		},
	})
}

// wasmMetricsHandler exposes WebAssembly module memory stats if runtime provides them (stub now)
func (cs *CudaService) wasmMetricsHandler(c *gin.Context) {
	// If you embed Wasm (e.g., wazero, wasmtime), inject a collector to populate these.
	c.JSON(200, gin.H{
		"status": "stub",
		"modules": []interface{}{},
		"notes": []string{"Integrate with your Wasm runtime to report linear memory size, max pages, heap allocations"},
	})
}