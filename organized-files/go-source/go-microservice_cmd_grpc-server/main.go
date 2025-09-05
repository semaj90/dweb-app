package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
)

// findAvailablePort finds an available port starting from the given port
func findAvailablePort(startPort int, maxAttempts int) (int, error) {
	for i := 0; i < maxAttempts; i++ {
		port := startPort + i
		addr := fmt.Sprintf(":%d", port)
		
		// Try to listen on the port
		lis, err := net.Listen("tcp", addr)
		if err == nil {
			lis.Close() // Close immediately since we just want to test availability
			return port, nil
		}
		
		log.Printf("Port %d is occupied, trying next...", port)
	}
	
	return 0, fmt.Errorf("no available port found starting from %d", startPort)
}

func main() {
	// Default port and intelligent discovery
	defaultPort := 8084
	if v := os.Getenv("GO_GRPC_PORT"); v != "" {
		if port, err := strconv.Atoi(v); err == nil {
			defaultPort = port
		}
	}

	// Find available port
	availablePort, err := findAvailablePort(defaultPort, 10)
	if err != nil {
		log.Fatalf("Failed to find available port: %v", err)
	}
	
	if availablePort != defaultPort {
		log.Printf("⚠️  Port %d was occupied, using port %d instead", defaultPort, availablePort)
	}

	addr := fmt.Sprintf(":%d", availablePort)
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("failed to listen on %s: %v", addr, err)
	}

	srv := grpc.NewServer()

	// Register standard gRPC Health service
	healthServer := health.NewServer()
	// Report SERVING by default for the overall server
	healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthpb.RegisterHealthServer(srv, healthServer)

	// Enable server reflection for debugging
	reflection.Register(srv)

	go func() {
		log.Printf("✅ gRPC server started on %s (health service enabled)", addr)
		log.Printf("🚀 Server running with intelligent port discovery on port %d", availablePort)
		if err := srv.Serve(lis); err != nil {
			log.Fatalf("gRPC server error: %v", err)
		}
	}()

	// Graceful shutdown on SIGINT/SIGTERM
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	<-sigCh
	log.Println("🛑 Shutting down gRPC server...")
	stopped := make(chan struct{})
	go func() {
		srv.GracefulStop()
		close(stopped)
	}()
	select {
	case <-stopped:
		log.Println("✅ gRPC server stopped gracefully")
	case <-time.After(5 * time.Second):
		log.Println("⏱️ Graceful stop timed out; forcing stop")
		srv.Stop()
	}
	// small delay to allow logs to flush
	_ = context.Background()
}
