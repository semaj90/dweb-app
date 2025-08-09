# Node.js GPU Service Dockerfile with WebGPU support

FROM node:20-bookworm

# Install system dependencies for WebGPU
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build TypeScript
RUN npm run build

# Create non-root user
RUN useradd -r -s /bin/false nodeuser && \
    chown -R nodeuser:nodeuser /app

# Switch to non-root user
USER nodeuser

# Expose port
EXPOSE 50052

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD node -e "const grpc = require('@grpc/grpc-js'); \
    const client = new grpc.Client('localhost:50052', grpc.credentials.createInsecure()); \
    client.waitForReady(Date.now() + 3000, (err) => process.exit(err ? 1 : 0));"

# Set environment variables
ENV NODE_ENV=production
ENV PORT=50052
ENV LOG_LEVEL=info
ENV ENABLE_METRICS=true

# Start the service
CMD ["node", "dist/index.js"]