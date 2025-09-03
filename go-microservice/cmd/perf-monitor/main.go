//go:build experimental
// +build experimental

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"runtime/metrics"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/prometheus/client_golang/prometheus"
	promhttp "github.com/prometheus/client_golang/prometheus/promhttp"
)

// MetricSnapshot represents a structured snapshot of runtime performance
type MetricSnapshot struct {
	Timestamp         time.Time       `json:"timestamp"`
	NumGoroutine      int             `json:"num_goroutine"`
	HeapAlloc         uint64          `json:"heap_alloc"`
	HeapInUse         uint64          `json:"heap_in_use"`
	NextGC            uint64          `json:"next_gc"`
	GCCount           uint32          `json:"gc_count"`
	LastPauseNs       uint64          `json:"last_pause_ns"`
	TotalPauseNs      uint64          `json:"total_pause_ns"`
	CPUPercent        float64         `json:"cpu_percent"`
	UptimeSeconds     float64         `json:"uptime_seconds"`
	RandomSample      float64         `json:"random_sample"`
	CustomCounters    map[string]int64 `json:"custom_counters"`
	GoVersion         string          `json:"go_version"`
	Process           string          `json:"process"`
}

var (
	startTime    = time.Now()
	reqCount     atomic.Int64
	errorCount   atomic.Int64
	customEvents atomic.Int64
	sigLimit     int64 = 5000

	// Signature frequency map: signature -> count
	sigMu       sync.RWMutex
	signatures  = make(map[string]int64)

	// Alert thresholds (MB / ms) via env
	heapWarnMB       = envInt("PERF_HEAP_WARN_MB", 800)
	heapCritMB       = envInt("PERF_HEAP_CRIT_MB", 1200)
	gcPauseWarnMs    = envInt("PERF_GC_PAUSE_WARN_MS", 200)
	gcPauseCritMs    = envInt("PERF_GC_PAUSE_CRIT_MS", 400)
	cpuWarnPercent   = envInt("PERF_CPU_WARN_PCT", 85)
	cpuCritPercent   = envInt("PERF_CPU_CRIT_PCT", 95)

	// Prometheus metrics
	promReg = prometheus.NewRegistry()
	mRequests = prometheus.NewCounter(prometheus.CounterOpts{Name: "perf_monitor_requests_total", Help: "Total /metrics/runtime requests"})
	mErrors = prometheus.NewCounter(prometheus.CounterOpts{Name: "perf_monitor_errors_total", Help: "Errors observed (manual increments)"})
	gGoroutines = prometheus.NewGauge(prometheus.GaugeOpts{Name: "perf_monitor_goroutines", Help: "Current goroutine count"})
	gHeapAlloc  = prometheus.NewGauge(prometheus.GaugeOpts{Name: "perf_monitor_heap_alloc_bytes", Help: "Heap alloc bytes"})
	gCPUPercent = prometheus.NewGauge(prometheus.GaugeOpts{Name: "perf_monitor_cpu_percent", Help: "Approx CPU percent"})
	gUptime     = prometheus.NewGauge(prometheus.GaugeOpts{Name: "perf_monitor_uptime_seconds", Help: "Uptime seconds"})

	// Postgres persistence
	pgDSN  = os.Getenv("PERF_PG_DSN")
	pgConn *sql.DB
)

// envInt helper
func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		var parsed int
		if _, err := fmt.Sscanf(v, "%d", &parsed); err == nil {
			return parsed
		}
	}
	return def
}

func captureSnapshot() MetricSnapshot {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	// CPU percent (simple approximation based on GOMAXPROCS and elapsed)
	cpuPercent := float64(runtime.NumGoroutine()) / float64(runtime.GOMAXPROCS(0)) * 10
	if cpuPercent > 100 {
		cpuPercent = 100
	}

	// Example reading from runtime/metrics (Go 1.20+)
	// Here we just demonstrate one metric if available
	// var lastPause uint64 // removed: unused placeholder; pause info captured via ms.PauseNs
	samples := []metrics.Sample{{Name: "/gc/pauses:seconds"}}
	metrics.Read(samples)
	if len(samples) > 0 && samples[0].Value.Kind() == metrics.KindFloat64Histogram {
		// Removed unused lastPause assignment; keeping histogram read conditional removed to avoid unused vars
	}

	return MetricSnapshot{
		Timestamp:     time.Now().UTC(),
		NumGoroutine:  runtime.NumGoroutine(),
		HeapAlloc:     ms.HeapAlloc,
		HeapInUse:     ms.HeapInuse,
		NextGC:        ms.NextGC,
		GCCount:       ms.NumGC,
		LastPauseNs:   ms.PauseNs[(ms.NumGC+255)%256],
		TotalPauseNs:  ms.PauseTotalNs,
		CPUPercent:    cpuPercent,
		UptimeSeconds: time.Since(startTime).Seconds(),
		RandomSample:  rand.Float64(),
		CustomCounters: map[string]int64{
			"requests": reqCount.Load(),
			"errors":   errorCount.Load(),
			"events":   customEvents.Load(),
		},
		GoVersion: runtime.Version(),
		Process:   os.Args[0],
	}
}

func metricsHandler(w http.ResponseWriter, r *http.Request) {
	reqCount.Add(1)
	w.Header().Set("Content-Type", "application/json")
	snap := captureSnapshot()
	json.NewEncoder(w).Encode(snap)
	// Update Prometheus gauges
	gGoroutines.Set(float64(snap.NumGoroutine))
	gHeapAlloc.Set(float64(snap.HeapAlloc))
	gCPUPercent.Set(snap.CPUPercent)
	gUptime.Set(snap.UptimeSeconds)
	mRequests.Inc()
	evaluateAlerts(snap)
	persistSnapshotAsync(snap)
}

func simulateWorkload() {
	for {
		// Simulate miscellaneous workload triggering counters
		customEvents.Add(1)
		time.Sleep(2 * time.Second)
	}
}

// Evaluate thresholds and log warnings
func evaluateAlerts(s MetricSnapshot) {
	heapMB := float64(s.HeapAlloc) / 1024.0 / 1024.0
	if int(heapMB) > heapCritMB {
		log.Printf("ALERT CRITICAL heap_alloc_mb=%.1f threshold=%d", heapMB, heapCritMB)
	} else if int(heapMB) > heapWarnMB {
		log.Printf("ALERT WARN heap_alloc_mb=%.1f threshold=%d", heapMB, heapWarnMB)
	}
	// Simple GC pause heuristic (last pause)
	lastPauseMs := float64(s.LastPauseNs) / 1e6
	if int(lastPauseMs) > gcPauseCritMs {
		log.Printf("ALERT CRITICAL gc_last_pause_ms=%.2f threshold=%d", lastPauseMs, gcPauseCritMs)
	} else if int(lastPauseMs) > gcPauseWarnMs {
		log.Printf("ALERT WARN gc_last_pause_ms=%.2f threshold=%d", lastPauseMs, gcPauseWarnMs)
	}
	if int(s.CPUPercent) > cpuCritPercent {
		log.Printf("ALERT CRITICAL cpu_percent=%.1f threshold=%d", s.CPUPercent, cpuCritPercent)
	} else if int(s.CPUPercent) > cpuWarnPercent {
		log.Printf("ALERT WARN cpu_percent=%.1f threshold=%d", s.CPUPercent, cpuWarnPercent)
	}
}

// Signature recording handler
func signaturePostHandler(w http.ResponseWriter, r *http.Request) {
	type payload struct {
		Fn       string `json:"fn"`
		ArgsHash string `json:"argsHash"`
		Signature string `json:"signature"`
	}
	var p payload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	sig := p.Signature
	if sig == "" {
		base := strings.TrimSpace(p.Fn)
		if base == "" { base = "unknown" }
		sig = base + ":" + p.ArgsHash
	}
	sigMu.Lock()
	if int64(len(signatures)) >= sigLimit {
		// Simple eviction: random deletion of 1 entry
		for k := range signatures { delete(signatures, k); break }
	}
	signatures[sig]++
	count := signatures[sig]
	sigMu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"signature": sig, "count": count})
}

func signatureListHandler(w http.ResponseWriter, r *http.Request) {
	limit := 25
	if v := r.URL.Query().Get("limit"); v != "" { fmt.Sscanf(v, "%d", &limit) }
	type pair struct { Sig string; Count int64 }
	sigMu.RLock()
	arr := make([]pair, 0, len(signatures))
	for k,v := range signatures { arr = append(arr, pair{k,v}) }
	sigMu.RUnlock()
	sort.Slice(arr, func(i,j int) bool { return arr[i].Count > arr[j].Count })
	if limit < len(arr) { arr = arr[:limit] }
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"top": arr, "totalDistinct": len(signatures)})
}

// Persistence
func initPostgres() {
	if pgDSN == "" { return }
	var err error
	pgConn, err = sql.Open("pgx", pgDSN)
	if err != nil { log.Printf("⚠️ Postgres connect failed: %v", err); return }
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err = pgConn.PingContext(ctx); err != nil { log.Printf("⚠️ Postgres ping failed: %v", err); return }
	_, err = pgConn.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS runtime_metrics (
		ts timestamptz,
		num_goroutine int,
		heap_alloc bigint,
		heap_in_use bigint,
		gc_count int,
		cpu_percent double precision,
		uptime_seconds double precision
	)`)
	if err != nil { log.Printf("⚠️ Create table failed: %v", err) } else { log.Println("🗄️ Postgres persistence enabled for runtime_metrics") }
}

func persistSnapshotAsync(s MetricSnapshot) {
	if pgConn == nil { return }
	go func(sn MetricSnapshot) {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_, err := pgConn.ExecContext(ctx, `INSERT INTO runtime_metrics(ts,num_goroutine,heap_alloc,heap_in_use,gc_count,cpu_percent,uptime_seconds) VALUES($1,$2,$3,$4,$5,$6,$7)`,
			sn.Timestamp, sn.NumGoroutine, sn.HeapAlloc, sn.HeapInUse, sn.GCCount, sn.CPUPercent, sn.UptimeSeconds)
		if err != nil { log.Printf("⚠️ Persist snapshot failed: %v", err) }
	}(s)
}

func main() {
	// Register Prometheus metrics
	promReg.MustRegister(mRequests, mErrors, gGoroutines, gHeapAlloc, gCPUPercent, gUptime)
	initPostgres()
	log.Println("🚀 Performance Monitor starting on :8098 (Prometheus /metrics, runtime /metrics/runtime)")
	go simulateWorkload()

	// Periodic log snapshots
	go func() {
		for range time.Tick(30 * time.Second) {
			s := captureSnapshot()
			b, _ := json.Marshal(s)
			log.Printf("METRIC %s\n", b)
		}
	}()

	http.HandleFunc("/metrics/runtime", metricsHandler)
	// Prometheus exposition
	http.Handle("/metrics", promhttp.HandlerFor(promReg, promhttp.HandlerOpts{}))
	// Signature frequency endpoints
	http.HandleFunc("/metrics/signature", signaturePostHandler)
	http.HandleFunc("/metrics/signatures", signatureListHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK); w.Write([]byte("OK")) })

	if err := http.ListenAndServe(":8098", nil); err != nil {
		log.Fatal(err)
	}
}
