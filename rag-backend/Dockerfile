# Enhanced RAG Backend Dockerfile
# Multi-stage build for production optimization

# Stage 1: Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    cairo-dev \
    jpeg-dev \
    pango-dev \
    musl-dev \
    giflib-dev \
    pixman-dev \
    pangomm-dev \
    libjpeg-turbo-dev \
    freetype-dev

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Stage 2: Production stage
FROM node:18-alpine AS production

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates \
    cairo \
    jpeg \
    pango \
    musl \
    giflib \
    pixman \
    pangomm \
    libjpeg-turbo \
    freetype \
    ttf-freefont

# Create app directory and user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S raguser -u 1001 -G nodejs

WORKDIR /app

# Copy built dependencies from builder stage
COPY --from=builder --chown=raguser:nodejs /app/node_modules ./node_modules

# Copy application code
COPY --chown=raguser:nodejs . .

# Create necessary directories
RUN mkdir -p uploads logs && \
    chown -R raguser:nodejs uploads logs

# Switch to non-root user
USER raguser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["node", "src/server.js"]

# Labels for metadata
LABEL maintainer="Legal AI Team" \
      version="1.0.0" \
      description="Enhanced RAG Backend with PostgreSQL, Ollama, and Multi-Agent Orchestration" \
      org.opencontainers.image.title="Enhanced RAG Backend" \
      org.opencontainers.image.description="Production-ready RAG system with local LLM integration" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.created="2025-01-31" \
      org.opencontainers.image.authors="Legal AI Team" \
      org.opencontainers.image.url="https://github.com/legal-ai/enhanced-rag-backend" \
      org.opencontainers.image.documentation="https://github.com/legal-ai/enhanced-rag-backend/blob/main/README.md" \
      org.opencontainers.image.source="https://github.com/legal-ai/enhanced-rag-backend" \
      org.opencontainers.image.licenses="MIT"
