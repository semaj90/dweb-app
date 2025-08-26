//go:build !cuda || !cgo
// +build !cuda !cgo

// gpu-memory-manager_stub.go
// Non-CUDA stub implementation so normal builds succeed without NVIDIA toolchain.

package main

import (
	"fmt"
	"time"
)

type GPUMemoryManager struct{}
func NewGPUMemoryManager(deviceID int, memoryPoolMB int, maxBlocks int) (*GPUMemoryManager, error) { return &GPUMemoryManager{}, nil }
func (g *GPUMemoryManager) Close() error { return nil }
func (g *GPUMemoryManager) GetStats() *GPUMemoryStats { return &GPUMemoryStats{} }
func (g *GPUMemoryManager) GetPerformanceMetrics() *GPUPerformanceMetrics { return &GPUPerformanceMetrics{} }
func (g *GPUMemoryManager) CreateWorkerPool(id string, maxWorkers int, q int) (*GPUWorkerPool, error) { return nil, fmt.Errorf("cuda build tag required") }
func (g *GPUMemoryManager) SubmitJob(poolID string, job *GPUJob) error { return fmt.Errorf("cuda build tag required") }
func (g *GPUMemoryManager) GetJobResult(poolID string, timeout time.Duration) (*GPUJobResult, error) { return nil, fmt.Errorf("cuda build tag required") }

// Minimal dependent structs for interfaces used in integration tests
 type GPUWorkerPool struct{}
 type GPUJob struct{}
 type GPUJobResult struct{}
 type GPUMemoryStats struct{}
type GPUPerformanceMetrics struct{}
