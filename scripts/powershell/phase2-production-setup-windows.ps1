# Phase 2: Production Setup - Windows 10 + WSL + Docker Desktop
# Adapted for Windows environment

Write-Host "🚀 Phase 2: Production Setup (Windows)" -ForegroundColor Green
Write-Host "======================================="

# Check prerequisites
Write-Host "Checking prerequisites..."
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker Desktop not found. Please install Docker Desktop for Windows."
    exit 1
}

if (!(wsl --list --quiet | Select-String "docker-desktop")) {
    Write-Error "WSL Docker integration not detected. Enable WSL integration in Docker Desktop."
    exit 1
}

# Create production directories
$directories = @(
    "deployment\production",
    "deployment\nginx",
    "deployment\ssl",
    "logs\production",
    "backups\database"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
        Write-Host "✅ Created $dir"
    }
}

# Create production Docker Compose
@"
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    container_name: legal-ai-nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./deployment/ssl:/etc/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - frontend
    networks:
      - legal-ai-network
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg16
    container_name: legal-ai-postgres-prod
    environment:
      POSTGRES_DB: legal_ai_db
      POSTGRES_USER: legal_admin
      POSTGRES_PASSWORD: `${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_prod_data:/var/lib/postgresql/data
      - ./backups/database:/backups
      - ./database/migrations:/docker-entrypoint-initdb.d
    networks:
      - legal-ai-network
    restart: unless-stopped
    command: >
      postgres 
      -c shared_preload_libraries=vector 
      -c max_connections=200
      -c shared_buffers=1GB
      -c effective_cache_size=4GB
      -c log_statement=all
      -c log_destination=stderr

  redis:
    image: redis:7-alpine
    container_name: legal-ai-redis-prod
    ports:
      - "6379:6379"
    volumes:
      - redis_prod_data:/data
      - ./logs/redis:/var/log/redis
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    networks:
      - legal-ai-network
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: legal-ai-qdrant-prod
    ports:
      - "6333:6333"
    volumes:
      - qdrant_prod_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__LOG_LEVEL: INFO
    networks:
      - legal-ai-network
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: legal-ai-ollama-prod
    ports:
      - "11434:11434"
    volumes:
      - ollama_prod_data:/root/.ollama
      - ./models:/models
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_ORIGINS=*
      - OLLAMA_KEEP_ALIVE=30m
      - OLLAMA_NUM_PARALLEL=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - legal-ai-network
    restart: unless-stopped

  frontend:
    build:
      context: ./sveltekit-frontend
      dockerfile: Dockerfile.prod
    container_name: legal-ai-frontend-prod
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://legal_admin:`${POSTGRES_PASSWORD}@postgres:5432/legal_ai_db
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - postgres
      - redis
      - qdrant
      - ollama
    networks:
      - legal-ai-network
    restart: unless-stopped

volumes:
  postgres_prod_data:
  redis_prod_data:
  qdrant_prod_data:
  ollama_prod_data:

networks:
  legal-ai-network:
    driver: bridge
"@ | Out-File -FilePath "deployment\production\docker-compose.prod.yml" -Encoding UTF8

# Create Nginx configuration
@"
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name localhost;
        return 301 https://`$server_name`$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name localhost;

        ssl_certificate /etc/ssl/server.crt;
        ssl_certificate_key /etc/ssl/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;

        location / {
            proxy_pass http://frontend;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
            proxy_set_header X-Forwarded-For `$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto `$scheme;
        }

        location /api/ {
            proxy_pass http://frontend;
            proxy_set_header Host `$host;
            proxy_set_header X-Real-IP `$remote_addr;
        }

        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
"@ | Out-File -FilePath "deployment\nginx\nginx.conf" -Encoding UTF8

# Create production environment file
@"
# Production Environment Variables
NODE_ENV=production
POSTGRES_PASSWORD=LegalAI2024!Prod
DATABASE_URL=postgresql://legal_admin:LegalAI2024!Prod@localhost:5432/legal_ai_db
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434

# Security
SESSION_SECRET=your-super-secret-session-key-here
JWT_SECRET=your-jwt-secret-here

# Monitoring
LOG_LEVEL=info
METRICS_ENABLED=true
"@ | Out-File -FilePath ".env.production" -Encoding UTF8

# Create production Dockerfile for frontend
@"
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM node:18-alpine as runner
WORKDIR /app

COPY --from=builder /app/build ./build
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

EXPOSE 3000
CMD ["node", "build"]
"@ | Out-File -FilePath "sveltekit-frontend\Dockerfile.prod" -Encoding UTF8

# Create Windows production startup script
@"
@echo off
title Legal AI - Production Mode
color 0A

echo 🚀 Starting Legal AI in Production Mode
echo ========================================

set POSTGRES_PASSWORD=LegalAI2024!Prod

echo Checking Docker Desktop...
docker version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop not running
    echo Please start Docker Desktop first
    pause
    exit /b 1
)

echo Starting production services...
cd deployment\production
docker-compose -f docker-compose.prod.yml up -d

echo Waiting for services...
timeout /t 30 /nobreak >nul

echo Checking service health...
docker exec legal-ai-postgres-prod pg_isready -U legal_admin
docker exec legal-ai-redis-prod redis-cli ping
curl -s http://localhost:6333/health

echo ✅ Production environment ready!
echo.
echo 🌐 Access your application:
echo   HTTP:  http://localhost
echo   HTTPS: https://localhost (requires SSL setup)
echo.
echo 📊 Monitoring:
echo   Logs: docker-compose -f docker-compose.prod.yml logs -f
echo   Stop: docker-compose -f docker-compose.prod.yml down
echo.
pause
"@ | Out-File -FilePath "START-PRODUCTION.bat" -Encoding UTF8

Write-Host "✅ Phase 2 production setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Review .env.production and update secrets"
Write-Host "2. Add SSL certificates to deployment/ssl/"
Write-Host "3. Run START-PRODUCTION.bat to launch"
Write-Host "4. Access at https://localhost"
