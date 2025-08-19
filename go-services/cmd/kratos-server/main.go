package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	kratoslog "github.com/go-kratos/kratos/v2/log"

	"legal-ai-services/internal/conf"
	"legal-ai-services/internal/server"
	"legal-ai-services/internal/service"
)

func main() {
	// Initialize logger
	logger := kratoslog.NewStdLogger(os.Stdout)
	
	log.Printf("Starting Legal AI Services...")
	
	// Initialize configuration with defaults
	config := &conf.Server{
		Grpc: &conf.GRPC{
			Network: "tcp",
			Addr:    "0.0.0.0:8080",
			EnableTls: false,
		},
	}
	
	// Initialize services
	legalSvc := service.NewLegalService(logger)
	vectorSvc := service.NewVectorService(logger)
	
	// Create gRPC server
	grpcServer := server.NewGRPCServer(config, legalSvc, vectorSvc, logger)
	
	// Start server
	lis, err := net.Listen("tcp", config.Grpc.Addr)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	
	log.Printf("gRPC server starting on %s", config.Grpc.Addr)
	
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("Failed to serve gRPC: %v", err)
		}
	}()
	
	// Wait for shutdown signal
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	
	<-c
	log.Printf("Shutting down gracefully...")
	
	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	grpcServer.GracefulStop()
	
	select {
	case <-ctx.Done():
		log.Printf("Shutdown timeout exceeded")
	default:
		log.Printf("Server stopped")
	}
}