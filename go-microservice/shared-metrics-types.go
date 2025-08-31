// shared-metrics-types.go
// Shared metric type definitions for all monitoring services
// Prevents type conflicts between performance-monitor.go and gpu-health-monitor.go

package main

import "time"

// SystemMetrics tracks overall system performance - unified structure
type SystemMetrics struct {
	Timestamp time.Time                 `json:"timestamp"`
	CPU       CPUMetrics                `json:"cpu"`
	Memory    MemoryMetrics             `json:"memory"`
	Disk      DiskMetrics               `json:"disk"`
	Network   NetworkMetrics            `json:"network"`
	GPU       GPUHealthMetrics          `json:"gpu"`
	Runtime   RuntimeMetrics            `json:"runtime"`
	Services  map[string]ServiceMetrics `json:"services,omitempty"`
}

// CPUMetrics tracks CPU performance
type CPUMetrics struct {
	UsagePercent []float64 `json:"usage_percent"`
	LoadAverage  []float64 `json:"load_avg"`
	Cores        int       `json:"cores"`
}

// MemoryMetrics tracks memory usage
type MemoryMetrics struct {
	Total       uint64  `json:"total"`
	Available   uint64  `json:"available"`
	Used        uint64  `json:"used"`
	UsedPercent float64 `json:"used_percent"`
	Cached      uint64  `json:"cached"`
	Buffers     uint64  `json:"buffers"`
}

// DiskMetrics tracks disk usage
type DiskMetrics struct {
	Total       uint64  `json:"total"`
	Free        uint64  `json:"free"`
	Used        uint64  `json:"used"`
	UsedPercent float64 `json:"used_percent"`
}

// NetworkMetrics tracks network performance
type NetworkMetrics struct {
	BytesSent   uint64 `json:"bytes_sent"`
	BytesRecv   uint64 `json:"bytes_recv"`
	PacketsSent uint64 `json:"packets_sent"`
	PacketsRecv uint64 `json:"packets_recv"`
	ErrorsIn    uint64 `json:"errors_in"`
	ErrorsOut   uint64 `json:"errors_out"`
}

// GPUHealthMetrics tracks GPU performance and health
type GPUHealthMetrics struct {
	Available     bool    `json:"available"`
	Utilization   float64 `json:"utilization"`
	MemoryTotal   uint64  `json:"memory_total"`
	MemoryUsed    uint64  `json:"memory_used"`
	MemoryFree    uint64  `json:"memory_free"`
	Temperature   int     `json:"temperature"`
	PowerUsage    float64 `json:"power_usage"`
	ClockSpeed    int     `json:"clock_speed"`
	FanSpeed      int     `json:"fan_speed,omitempty"`
	ComputeMode   string  `json:"compute_mode,omitempty"`
	DriverVersion string  `json:"driver_version,omitempty"`
	CUDAVersion   string  `json:"cuda_version,omitempty"`
}

// RuntimeMetrics tracks Go runtime performance
type RuntimeMetrics struct {
	GoroutineCount int           `json:"goroutine_count"`
	GCPauseTime    time.Duration `json:"gc_pause_time"`
	HeapAllocMB    float64       `json:"heap_alloc_mb"`
	HeapSysMB      float64       `json:"heap_sys_mb"`
	NumGC          uint32        `json:"num_gc"`
	OpenFiles      int           `json:"open_files"`
	Connections    int           `json:"connections"`
}

// ServiceMetrics tracks individual service performance - unified fields from both monitors
type ServiceMetrics struct {
	Name            string        `json:"name"`
	Status          string        `json:"status"`
	Uptime          time.Duration `json:"uptime"`
	CPUPercent      float64       `json:"cpu_percent,omitempty"`
	CPUUsage        float64       `json:"cpu_usage,omitempty"` // Alternative field name
	MemoryMB        float64       `json:"memory_mb,omitempty"`
	MemoryUsage     uint64        `json:"memory_usage,omitempty"` // Alternative field name
	LastResponse    time.Duration `json:"last_response,omitempty"`
	ResponseTime    time.Duration `json:"response_time,omitempty"` // Alternative field name
	ErrorCount      int64         `json:"error_count,omitempty"`
	ErrorRate       float64       `json:"error_rate,omitempty"`
	RequestCount    int64         `json:"request_count,omitempty"`
	LastHealthCheck time.Time     `json:"last_health_check,omitempty"`
	Port            int           `json:"port,omitempty"`
	PID             int           `json:"pid,omitempty"`
}
