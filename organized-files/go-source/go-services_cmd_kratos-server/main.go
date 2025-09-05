package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"legal-ai-services/internal/conf"
	"legal-ai-services/internal/data"
	"legal-ai-services/internal/biz"
	"legal-ai-services/internal/service"
	"legal-ai-services/internal/server"
)

// Version information
var (
	Name     = "legal-ai-kratos"
	Version  = "v1.0.0-placeholder"
	ID       = "legal-ai-001"
)

func main() {
	fmt.Printf("[%s] Starting Legal AI Kratos Server %s\n", Name, Version)
	
	// Load configuration
	config, err := conf.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}
	
	fmt.Printf("[%s] Configuration loaded for environment: %s\n", Name, config.Server.Environment)
	
	// Initialize data layer
	dataLayer := data.NewDataLayer(
		config.Database.PostgresURL,
		config.Vector.QdrantURL,
		config.Cache.RedisURL,
	)
	
	if err := dataLayer.Initialize(); err != nil {
		log.Fatalf("Failed to initialize data layer: %v", err)
	}
	
	// Initialize repositories
	docRepo := data.NewDocumentRepository(dataLayer)
	vectorRepo := data.NewVectorRepository(dataLayer)
	
	// Initialize business logic
	businessLogic := biz.NewBusinessLogic(docRepo, vectorRepo)
	
	// Initialize services
	legalService := service.NewLegalAIService()
	
	// Initialize server manager
	serverManager := server.NewServerManager()
	
	// Start servers (placeholder implementation)
	if err := serverManager.Start(); err != nil {
		log.Fatalf("Failed to start servers: %v", err)
	}
	
	fmt.Printf("[%s] Server started successfully\n", Name)
	fmt.Printf("[%s] HTTP server: http://%s:%d\n", Name, config.Server.Host, config.Server.HTTPPort)
	fmt.Printf("[%s] gRPC server: %s:%d\n", Name, config.Server.Host, config.Server.GRPCPort)
	fmt.Printf("[%s] QUIC server: %s:%d\n", Name, config.Server.Host, config.Server.QUICPort)
	
	// Wait for shutdown signal
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	
	select {
	case sig := <-c:
		fmt.Printf("[%s] Received signal: %v\n", Name, sig)
	case <-time.After(1 * time.Hour):
		fmt.Printf("[%s] Placeholder timeout reached\n", Name)
	}
	
	// Graceful shutdown
	fmt.Printf("[%s] Shutting down gracefully...\n", Name)
	
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	
	if err := serverManager.Stop(); err != nil {
		log.Printf("Error during shutdown: %v", err)
	}
	
	// Use the business logic to show it's wired (placeholder)
	_, _ = businessLogic.AnalyzeDocument(ctx, "placeholder document")
	_, _ = legalService.GetHealth(ctx)
	
	fmt.Printf("[%s] Server stopped\n", Name)
}
