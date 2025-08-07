// go-microservice/Dockerfile.simd
// Build: go build -ldflags="-s -w" -tags production -o simd-server-prod.exe simd-redis-vite-server.go

//go:build production
// +build production

package main

import (
	"os"
	"strconv"
)

func init() {
	// Production optimizations
	if workers := os.Getenv("SIMD_WORKERS"); workers != "" {
		if w, err := strconv.Atoi(workers); err == nil && w > 0 {
			workerPool = NewWorkerPool(w, metrics)
		}
	}
	
	// Production Redis config
	redisOptions.PoolSize = 50
	redisOptions.MinIdleConns = 10
	redisOptions.MaxRetries = 5
}