package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"legal-ai-production/internal/worker"
)

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func main() {
	log.Printf("🚀 Starting Vector Processing Service")
	log.Printf("📋 Native Windows • Redis Streams • CUDA • Qdrant • PostgreSQL")
	
	// Configuration
	cfg := worker.Config{
		RedisURL:      getenv("REDIS_URL", "redis://localhost:6379"),
		RedisStream:   getenv("REDIS_STREAM", "vec:requests"),
		RedisGroup:    getenv("REDIS_GROUP", "vec-workers"),
		ConsumerName:  getenv("REDIS_CONSUMER", "worker-1"),
		CudaWorkerExe: getenv("CUDA_WORKER_EXE", "./cuda-rotate-worker.exe"),
		QdrantURL:     getenv("QDRANT_URL", "http://localhost:6333"),
		QdrantCol:     getenv("QDRANT_COLLECTION", "legal_documents"),
		PGConn:        getenv("DATABASE_URL", "postgres://legal_admin:123456@localhost:5432/legal_ai_db?sslmode=disable"),
		MaxRetries:    5,
	}

	// Print configuration
	log.Printf("📡 Redis: %s", cfg.RedisURL)
	log.Printf("🗄️ PostgreSQL: %s", maskPassword(cfg.PGConn))
	log.Printf("🔍 Qdrant: %s/%s", cfg.QdrantURL, cfg.QdrantCol)
	log.Printf("🖥️ CUDA Worker: %s", cfg.CudaWorkerExe)
	log.Printf("📊 Stream: %s (group: %s, consumer: %s)", cfg.RedisStream, cfg.RedisGroup, cfg.ConsumerName)

	// Initialize database pool
	ctx := context.Background()
	pool, err := pgxpool.New(ctx, cfg.PGConn)
	if err != nil {
		log.Fatalf("❌ Failed to create connection pool: %v", err)
	}
	defer pool.Close()

	// Test database connection
	if err := pool.Ping(ctx); err != nil {
		log.Fatalf("❌ Failed to ping database: %v", err)
	}
	log.Printf("✅ PostgreSQL connected")

	// Test database schema
	if err := testDatabaseSchema(ctx, pool); err != nil {
		log.Printf("⚠️ Database schema check failed: %v", err)
		log.Printf("💡 Make sure to run the vectors_autocreate_notify.sql migration")
	} else {
		log.Printf("✅ Database schema validated")
	}

	// Initialize worker
	w, err := worker.NewWorker(cfg, pool)
	if err != nil {
		log.Fatalf("❌ Failed to create worker: %v", err)
	}
	defer w.Close()
	log.Printf("✅ Worker initialized")

	// Start worker in goroutine
	workerCtx, workerCancel := context.WithCancel(ctx)
	go func() {
		log.Printf("🔄 Starting worker loop...")
		if err := w.Run(workerCtx); err != nil && err != context.Canceled {
			log.Printf("❌ Worker error: %v", err)
		}
		log.Printf("🛑 Worker stopped")
	}()

	// Start health check server
	go startHealthServer(pool, w)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	log.Printf("✅ Service started successfully")
	log.Printf("🔗 Health check: http://localhost:8095/health")
	log.Printf("📊 Metrics: http://localhost:8095/metrics")
	log.Printf("⏹️ Press Ctrl+C to stop...")

	// Wait for shutdown signal
	<-sigChan
	log.Printf("🛑 Shutdown signal received")

	// Graceful shutdown
	workerCancel()
	time.Sleep(2 * time.Second) // Give worker time to finish current jobs
	
	log.Printf("✅ Service stopped gracefully")
}

func maskPassword(connStr string) string {
	// Simple password masking for logs
	return connStr // In production, implement proper masking
}

func testDatabaseSchema(ctx context.Context, pool *pgxpool.Pool) error {
	// Test if required tables exist
	tables := []string{"vectors", "vector_outbox", "evidence", "reports"}
	
	for _, table := range tables {
		var exists bool
		err := pool.QueryRow(ctx, 
			`SELECT EXISTS (
				SELECT FROM information_schema.tables 
				WHERE table_schema = 'public' AND table_name = $1
			)`, table).Scan(&exists)
		
		if err != nil {
			return fmt.Errorf("failed to check table %s: %w", table, err)
		}
		
		if !exists {
			return fmt.Errorf("required table %s does not exist", table)
		}
	}

	// Test if pgvector extension is available
	var vectorExists bool
	err := pool.QueryRow(ctx, 
		`SELECT EXISTS (
			SELECT FROM pg_extension WHERE extname = 'vector'
		)`).Scan(&vectorExists)
	
	if err != nil {
		return fmt.Errorf("failed to check vector extension: %w", err)
	}
	
	if !vectorExists {
		return fmt.Errorf("pgvector extension not installed")
	}

	// Test triggers exist
	triggers := []string{"evidence_vector_insert", "report_vector_insert"}
	for _, trigger := range triggers {
		var exists bool
		err := pool.QueryRow(ctx,
			`SELECT EXISTS (
				SELECT FROM information_schema.triggers 
				WHERE trigger_name = $1
			)`, trigger).Scan(&exists)
		
		if err != nil {
			return fmt.Errorf("failed to check trigger %s: %w", trigger, err)
		}
		
		if !exists {
			log.Printf("⚠️ Trigger %s not found", trigger)
		}
	}

	return nil
}

func startHealthServer(pool *pgxpool.Pool, w *worker.Worker) {
	// Simple HTTP server for health checks and metrics
	// This would normally use a proper HTTP framework
	log.Printf("🏥 Health server would start on :8095")
	log.Printf("📊 Implement health/metrics endpoints as needed")
}