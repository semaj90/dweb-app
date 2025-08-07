# Makefile for Go AI Microservice with GPU Support

# Variables
BINARY_NAME=ai-microservice
MAIN_PATH=.
BUILD_DIR=build
VERSION=$(shell git describe --tags --always --dirty 2>/dev/null || echo "v2.0.0")
BUILD_TIME=$(shell date +%FT%T%z)
LDFLAGS=-ldflags "-X main.version=${VERSION} -X main.buildTime=${BUILD_TIME}"

# CUDA paths (Windows)
CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
CUDA_INCLUDE=$(CUDA_PATH)\include
CUDA_LIB=$(CUDA_PATH)\lib\x64

# Go build flags
GO=go
GOFLAGS=-v
CGO_CFLAGS=-I"$(CUDA_INCLUDE)"
CGO_LDFLAGS=-L"$(CUDA_LIB)" -lcudart -lcublas -lcublasLt -lcudnn

# Detect OS
ifeq ($(OS),Windows_NT)
	BINARY_EXT=.exe
	RM=del /Q
	MKDIR=mkdir
else
	BINARY_EXT=
	RM=rm -f
	MKDIR=mkdir -p
endif

# Default target
all: build

# Build the binary
build: deps
	@echo "Building $(BINARY_NAME)..."
	@$(MKDIR) $(BUILD_DIR) 2>nul || true
	CGO_ENABLED=1 CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" \
		$(GO) build $(GOFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)$(BINARY_EXT) $(MAIN_PATH)
	@echo "Build complete: $(BUILD_DIR)/$(BINARY_NAME)$(BINARY_EXT)"

# Build without GPU support
build-cpu:
	@echo "Building $(BINARY_NAME) (CPU only)..."
	@$(MKDIR) $(BUILD_DIR) 2>nul || true
	CGO_ENABLED=0 $(GO) build $(GOFLAGS) $(LDFLAGS) \
		-tags nogpu -o $(BUILD_DIR)/$(BINARY_NAME)-cpu$(BINARY_EXT) $(MAIN_PATH)
	@echo "Build complete: $(BUILD_DIR)/$(BINARY_NAME)-cpu$(BINARY_EXT)"

# Build with all optimizations
build-prod: deps
	@echo "Building production binary with optimizations..."
	@$(MKDIR) $(BUILD_DIR) 2>nul || true
	CGO_ENABLED=1 CGO_CFLAGS="$(CGO_CFLAGS) -O3 -march=native" \
		CGO_LDFLAGS="$(CGO_LDFLAGS)" GOOS=windows GOARCH=amd64 \
		$(GO) build -a -installsuffix cgo \
		-ldflags "-s -w $(LDFLAGS)" \
		-o $(BUILD_DIR)/$(BINARY_NAME)-prod$(BINARY_EXT) $(MAIN_PATH)
	@echo "Production build complete"

# Install dependencies
deps:
	@echo "Installing dependencies..."
	$(GO) mod download
	$(GO) mod tidy

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	$(GO) get -u ./...
	$(GO) mod tidy

# Run the application
run: build
	@echo "Starting $(BINARY_NAME)..."
	$(BUILD_DIR)/$(BINARY_NAME)$(BINARY_EXT)

# Run with environment variables
run-dev:
	@echo "Starting in development mode..."
	set PORT=8081 && \
	set REDIS_HOST=localhost:6379 && \
	set NEO4J_URI=neo4j://localhost:7687 && \
	set NEO4J_USER=neo4j && \
	set NEO4J_PASSWORD=password && \
	$(GO) run $(MAIN_PATH)

# Run tests
test:
	@echo "Running tests..."
	$(GO) test -v -cover ./...

# Run tests with race detector
test-race:
	@echo "Running tests with race detector..."
	CGO_ENABLED=1 $(GO) test -v -race ./...

# Benchmark tests
bench:
	@echo "Running benchmarks..."
	$(GO) test -bench=. -benchmem ./...

# Generate test coverage
coverage:
	@echo "Generating coverage report..."
	$(GO) test -v -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Lint the code
lint:
	@echo "Running linter..."
	golangci-lint run

# Format code
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...
	gofmt -s -w .

# Vet code
vet:
	@echo "Vetting code..."
	$(GO) vet ./...

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@$(RM) $(BUILD_DIR)\* 2>nul || true
	@$(RM) coverage.out coverage.html 2>nul || true
	@echo "Clean complete"

# Docker build
docker-build:
	@echo "Building Docker image..."
	docker build -t $(BINARY_NAME):$(VERSION) .
	docker tag $(BINARY_NAME):$(VERSION) $(BINARY_NAME):latest

# Docker run
docker-run:
	@echo "Running Docker container..."
	docker run -d \
		--name $(BINARY_NAME) \
		--gpus all \
		-p 8081:8081 \
		-e REDIS_HOST=host.docker.internal:6379 \
		-e NEO4J_URI=neo4j://host.docker.internal:7687 \
		$(BINARY_NAME):latest

# Generate API documentation
docs:
	@echo "Generating API documentation..."
	swag init -g main.go -o ./docs

# Install development tools
install-tools:
	@echo "Installing development tools..."
	go install github.com/swaggo/swag/cmd/swag@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install github.com/cosmtrek/air@latest

# Live reload for development
dev:
	@echo "Starting with live reload..."
	air -c .air.toml

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@nvidia-smi

# Profile CPU
profile-cpu:
	@echo "Running CPU profiling..."
	$(GO) test -cpuprofile=cpu.prof -bench=.
	$(GO) tool pprof cpu.prof

# Profile memory
profile-mem:
	@echo "Running memory profiling..."
	$(GO) test -memprofile=mem.prof -bench=.
	$(GO) tool pprof mem.prof

# Help target
help:
	@echo "Available targets:"
	@echo "  make build       - Build the binary with GPU support"
	@echo "  make build-cpu   - Build CPU-only binary"
	@echo "  make build-prod  - Build optimized production binary"
	@echo "  make run         - Build and run the application"
	@echo "  make run-dev     - Run in development mode"
	@echo "  make test        - Run tests"
	@echo "  make coverage    - Generate test coverage report"
	@echo "  make lint        - Run linter"
	@echo "  make fmt         - Format code"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run  - Run Docker container"
	@echo "  make deps        - Install dependencies"
	@echo "  make update-deps - Update dependencies"
	@echo "  make check-cuda  - Check CUDA installation"
	@echo "  make help        - Show this help message"

.PHONY: all build build-cpu build-prod deps update-deps run run-dev test test-race \
        bench coverage lint fmt vet clean docker-build docker-run docs install-tools \
        dev check-cuda profile-cpu profile-mem help
