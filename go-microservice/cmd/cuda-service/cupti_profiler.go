//go:build cupti

// NOTE: This file is only compiled when built with the `cupti` build tag:
//   go build -tags=cupti ./...
// or
//   go run -tags=cupti ./cmd/cuda-service
//
// It provides a best-effort GPU profiling snapshot implementation. A *real*
// integration should wire up CUPTI activity + event APIs to gather per-kernel
// metrics (durations, grid/block dims, achieved occupancy, tensor core (HMMA)
// utilization, DRAM throughput, etc.). This scaffold keeps the surface stable
// while you incrementally replace the internals with true CUPTI logic.
//
// QUICK INTEGRATION STEPS (incremental):
// 1. Install CUDA Toolkit (ensuring libcupti.so / cupti64.dll present).
// 2. Add cgo directives below (uncomment) pointing to CUDA include/lib paths.
// 3. Implement initCupti() to subscribe to Activity API (CUPTI_ACTIVITY_KIND_KERNEL, MEMCPY, MEMSET).
// 4. In capture loop, flush activity buffers periodically, aggregate per-kernel stats.
// 5. Optionally enable Event API counters for tensor core instructions & dram bytes.
// 6. Populate profilerState fields atomically. getProfilingSnapshot() will expose them.
//
// Example cgo directives (adjust for platform):
//   #cgo linux LDFLAGS: -lcupti -lcuda
//   #cgo linux CFLAGS: -I/usr/local/cuda/include
//   #cgo windows LDFLAGS: -LC:/Program\ Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64 -lcupti -lcuda
//   #include <stdlib.h>
//   // Placeholder: real CUPTI headers would be included here.
//
// For now this uses NVML sampling as a *fallback* so you immediately see value
// when enabling the tag even before full CUPTI integration.

package main

import (
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// profilingSnapshot mirrors the struct in main.go but this version will mark Enabled=true.
// (We redeclare to ensure any future field additions compile-time fail without update.)
// Keep fields identical to the non-cupti version for JSON stability.
// If you add fields, add them in both versions.
type profilingSnapshot struct {
    Timestamp          int64    `json:"ts"`
    Enabled            bool     `json:"enabled"`
    KernelSamples      int      `json:"kernel_samples"`
    TensorCoreUtil     float64  `json:"tensor_core_util"`
    DramThroughputGBs  float64  `json:"dram_throughput_gbs"`
    OccupancyAvg       float64  `json:"occupancy_avg"`
    Notes              []string `json:"notes"`
}

type kernelAgg struct {
    count int
    totalDuration time.Duration
}

// profilerState holds rolling aggregates. In a real impl, you would maintain maps keyed by kernel name.
var profilerState = struct {
    mu sync.RWMutex
    lastSample time.Time
    kernel kernelAgg
    tensorUtil float64
    dramGBs float64
    occupancy float64
    gpuUtilAvg float64
    gpuUtilSamples int
}{ }

func init() {
    // Start lightweight sampler. Replace with CUPTI activity buffer drain logic when ready.
    go backgroundNvmlSampler()
    log.Printf("🧪 CUPTI build tag active: using NVML fallback sampler (replace with real CUPTI integration)")
}

func backgroundNvmlSampler() {
    ticker := time.NewTicker(2 * time.Second)
    defer ticker.Stop()
    for range ticker.C {
        sampleOnce()
    }
}

func sampleOnce() {
    if ret := nvml.Init(); ret != nvml.SUCCESS { return } // ensure NVML ready (idempotent)
    defer nvml.Shutdown() // inexpensive here; real impl keep initialized globally
    count, ret := nvml.DeviceGetCount(); if ret != nvml.SUCCESS || count == 0 { return }
    dev, ret := nvml.DeviceGetHandleByIndex(0); if ret != nvml.SUCCESS { return }
    util, ret := nvml.DeviceGetUtilizationRates(dev); if ret != nvml.SUCCESS { return }

    profilerState.mu.Lock()
    // Fake a kernel duration aggregation (random jitter) until real CUPTI data is present
    profilerState.kernel.count += 3
    profilerState.kernel.totalDuration += time.Duration(50+rand.Intn(50))*time.Millisecond
    // Approximate tensor util heuristically from SM util (placeholder)
    profilerState.tensorUtil = float64(util.Gpu) * 0.65 // assume 65% of SM util corresponds to tensor core eligible kernels
    profilerState.dramGBs = float64(util.Memory) * 0.005 // arbitrary scaling placeholder
    profilerState.occupancy = 0.55 + rand.Float64()*0.15 // 55–70% placeholder
    profilerState.gpuUtilSamples++
    profilerState.gpuUtilAvg += (float64(util.Gpu) - profilerState.gpuUtilAvg)/float64(profilerState.gpuUtilSamples)
    profilerState.lastSample = time.Now()
    profilerState.mu.Unlock()
}

// getProfilingSnapshot overrides the stub version when built with -tags cupti.
func getProfilingSnapshot() profilingSnapshot {
    profilerState.mu.RLock()
    k := profilerState.kernel
    tensor := profilerState.tensorUtil
    dram := profilerState.dramGBs
    occ := profilerState.occupancy
    gpuAvg := profilerState.gpuUtilAvg
    profilerState.mu.RUnlock()

    var avgKernelMs float64
    if k.count > 0 && k.totalDuration > 0 {
        avgKernelMs = float64(k.totalDuration/time.Millisecond)/float64(k.count)
    }

    notes := []string{
        "CUPTI tag enabled (fallback NVML sampler)",
        "Replace sampler with CUPTI Activity + Event collection",
        "avg_kernel_ms is synthetic until real CUPTI instrumentation",
    }

    return profilingSnapshot{
        Timestamp: time.Now().UnixMilli(),
        Enabled: true,
        KernelSamples: k.count,
        TensorCoreUtil: tensor,
        DramThroughputGBs: dram,
        OccupancyAvg: occ,
        Notes: append(notes,
            // dynamic context lines
            // (Optionally include avgKernelMs when non-zero)
            func() string { if avgKernelMs > 0 { return "avg_kernel_ms=" + formatFloat(avgKernelMs, 2) } ; return "avg_kernel_ms=0" }(),
            "gpu_util_avg="+formatFloat(gpuAvg,1),
        ),
    }
}

func formatFloat(v float64, prec int) string {
    fmt := "%0."+string('0'+byte(prec))+"f" // simple builder without importing fmt inside hot path
    // Use Sprintf inside helper to avoid pulling fmt for each line above
    return sprintFmt(fmt, v)
}

// sprintFmt isolated to keep fmt usage localized
func sprintFmt(format string, v interface{}) string {
    return fmt.Sprintf(format, v)
}
