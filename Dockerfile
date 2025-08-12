# Multi-stage build for optimized GPU-accelerated Legal AI Service
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git gcc musl-dev

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application with optimizations
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o legal-ai-service .

# Final stage - minimal runtime image with CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/legal-ai-service .
COPY --from=builder /app/go.mod .
COPY --from=builder /app/go.sum .

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8084

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8084/api/health || exit 1

# Set environment variables for GPU optimization
ENV CUDA_VISIBLE_DEVICES=0 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    CUDA_LAUNCH_BLOCKING=0 \
    CUDNN_BENCHMARK=1

# Run the service
ENTRYPOINT ["./legal-ai-service"]