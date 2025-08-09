#!/bin/bash

# Go SIMD Service Startup Script
# Enhanced with production-ready features

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="go-simd-service"
PORT=50051
GPU_SERVICE_PORT=50052
LOG_LEVEL=${LOG_LEVEL:-info}
WORKERS=${WORKERS:-$(nproc)}

echo -e "${BLUE}🚀 Starting Go SIMD Service${NC}"
echo -e "${BLUE}================================${NC}"

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $1 is already in use${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ Port $1 is available${NC}"
    return 0
}

# Function to wait for GPU service
wait_for_gpu_service() {
    echo -e "${YELLOW}⏳ Waiting for GPU service on port $GPU_SERVICE_PORT...${NC}"
    while ! nc -z localhost $GPU_SERVICE_PORT 2>/dev/null; do
        sleep 1
        echo -n "."
    done
    echo -e "\n${GREEN}✅ GPU service is ready${NC}"
}

# Function to build the service
build_service() {
    echo -e "${YELLOW}🔨 Building Go SIMD service...${NC}"
    
    # Initialize go module if not exists
    if [ ! -f go.mod ]; then
        go mod init github.com/legal-ai/go-simd-service
    fi
    
    # Download dependencies
    go mod tidy
    
    # Build with optimizations
    CGO_ENABLED=0 GOOS=linux go build \
        -ldflags="-w -s" \
        -o ${SERVICE_NAME} \
        main.go
        
    echo -e "${GREEN}✅ Build completed${NC}"
}

# Function to run the service
run_service() {
    echo -e "${YELLOW}🎯 Starting Go SIMD service on port $PORT...${NC}"
    
    # Set environment variables
    export GRPC_PORT=$PORT
    export GPU_SERVICE_URL="localhost:$GPU_SERVICE_PORT"
    export LOG_LEVEL=$LOG_LEVEL
    export WORKER_COUNT=$WORKERS
    export GOMAXPROCS=$WORKERS
    
    # Start the service
    ./${SERVICE_NAME} &
    SERVICE_PID=$!
    
    echo -e "${GREEN}✅ Service started with PID $SERVICE_PID${NC}"
    echo -e "${BLUE}📊 Service running with $WORKERS workers${NC}"
    
    # Save PID for cleanup
    echo $SERVICE_PID > ${SERVICE_NAME}.pid
}

# Function to monitor service health
monitor_service() {
    echo -e "${YELLOW}🔍 Monitoring service health...${NC}"
    
    sleep 2
    
    # Check if service is responding
    if curl -f -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service health check passed${NC}"
    else
        echo -e "${RED}❌ Service health check failed${NC}"
        return 1
    fi
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    
    if [ -f ${SERVICE_NAME}.pid ]; then
        PID=$(cat ${SERVICE_NAME}.pid)
        if kill -0 $PID 2>/dev/null; then
            echo -e "${YELLOW}Stopping service (PID $PID)...${NC}"
            kill $PID
            rm -f ${SERVICE_NAME}.pid
        fi
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
    exit 0
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-only    Build the service without running"
    echo "  --no-wait       Don't wait for GPU service"
    echo "  --workers N     Set number of workers (default: CPU cores)"
    echo "  --port N        Set service port (default: 50051)"
    echo "  --help          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LOG_LEVEL       Set log level (debug, info, warn, error)"
    echo "  WORKERS         Number of worker threads"
    echo ""
}

# Parse command line arguments
BUILD_ONLY=false
NO_WAIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --no-wait)
            NO_WAIT=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Main execution flow
main() {
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "${BLUE}  Port: $PORT${NC}"
    echo -e "${BLUE}  GPU Service Port: $GPU_SERVICE_PORT${NC}"
    echo -e "${BLUE}  Workers: $WORKERS${NC}"
    echo -e "${BLUE}  Log Level: $LOG_LEVEL${NC}"
    echo ""
    
    # Check prerequisites
    if ! command -v go &> /dev/null; then
        echo -e "${RED}❌ Go is not installed${NC}"
        exit 1
    fi
    
    if ! command -v nc &> /dev/null; then
        echo -e "${YELLOW}⚠️ netcat not available, skipping port checks${NC}"
    else
        check_port $PORT
    fi
    
    # Build the service
    build_service
    
    if [ "$BUILD_ONLY" = true ]; then
        echo -e "${GREEN}✅ Build completed (build-only mode)${NC}"
        exit 0
    fi
    
    # Wait for GPU service if requested
    if [ "$NO_WAIT" = false ]; then
        wait_for_gpu_service
    fi
    
    # Run the service
    run_service
    
    # Monitor health
    monitor_service
    
    echo -e "${GREEN}🎉 Go SIMD Service is running successfully!${NC}"
    echo -e "${BLUE}📝 Logs: tail -f /var/log/${SERVICE_NAME}.log${NC}"
    echo -e "${BLUE}🔍 Health: curl http://localhost:$PORT/health${NC}"
    echo -e "${BLUE}📊 Metrics: curl http://localhost:$PORT/metrics${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the service${NC}"
    
    # Keep the script running
    wait $SERVICE_PID
}

# Run main function
main "$@"