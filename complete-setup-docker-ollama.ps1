# Complete Docker + Ollama + SvelteKit Setup Script
Write-Host "=== Complete Web App Setup with Docker & Ollama ===" -ForegroundColor Cyan

# Set error handling
$ErrorActionPreference = "Continue"

# Helper Functions
function Write-Step {
    param($message)
    Write-Host "`n$message" -ForegroundColor Yellow
}

function Write-Success {
    param($message)
    Write-Host "✓ $message" -ForegroundColor Green
}

function Write-Error {
    param($message)
    Write-Host "✗ $message" -ForegroundColor Red
}

function Test-DockerService {
    param($serviceName)
    $status = docker ps --filter "name=$serviceName" --format "{{.Status}}" 2>$null
    return $status -match "Up"
}

function Test-Port {
    param($port)
    $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
    return $connection.TcpTestSucceeded
}

# Step 1: Environment Setup
Write-Step "[1/8] Setting up environment..."

# Create .env if it doesn't exist
if (!(Test-Path ".\.env")) {
    @"
# PostgreSQL with pgvector configuration
DB_DIALECT=postgresql
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/prosecutor_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=prosecutor_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Application Environment
NODE_ENV=development
PUBLIC_BASE_URL=http://localhost:5173
PUBLIC_APP_NAME=Legal AI Assistant

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
AI_PROVIDER=ollama
ENABLE_AI_FEATURES=true

# Vector Database
QDRANT_URL=http://localhost:6333
EMBEDDING_DIMENSION=768

# Redis Cache
REDIS_URL=redis://localhost:6379

# Authentication
JWT_SECRET=development_jwt_secret_change_in_production_minimum_32_chars
AUTH_SECRET=development_auth_secret_change_in_production_minimum_32_chars
"@ | Set-Content -Path ".\.env" -Force
    Write-Success "Created .env file"
}

# Copy to frontend
Copy-Item -Path ".\.env" -Destination ".\sveltekit-frontend\.env" -Force
Write-Success "Environment files configured"

# Step 2: Docker Services
Write-Step "[2/8] Starting Docker services..."

# Create docker-compose.override.yml for better Ollama support
@"
version: '3.8'

services:
  ollama:
    environment:
      - OLLAMA_KEEP_ALIVE=24h
      - OLLAMA_NUM_PARALLEL=4
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
  
  postgres:
    command: >
      postgres
      -c shared_preload_libraries=pg_stat_statements,vector
      -c max_connections=200
      -c shared_buffers=256MB
"@ | Set-Content -Path ".\docker-compose.override.yml" -Force

# Start services
docker-compose down 2>$null
docker-compose up -d

Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Step 3: Verify Services
Write-Step "[3/8] Verifying services..."

$services = @{
    "PostgreSQL" = @{container="prosecutor_postgres"; port=5432}
    "Ollama" = @{container="prosecutor_ollama"; port=11434}
    "Qdrant" = @{container="prosecutor_qdrant"; port=6333}
    "Redis" = @{container="prosecutor_redis"; port=6379}
}

$allServicesUp = $true
foreach ($service in $services.Keys) {
    $info = $services[$service]
    if (Test-Port $info.port) {
        Write-Success "$service is running on port $($info.port)"
    } else {
        Write-Error "$service is not accessible on port $($info.port)"
        $allServicesUp = $false
    }
}

if (!$allServicesUp) {
    Write-Error "Some services failed to start. Checking Docker logs..."
    docker-compose logs --tail 50
    exit 1
}

# Step 4: Install Ollama Models
Write-Step "[4/8] Installing Ollama models..."

$models = @("gemma3", "gemma:2b", "nomic-embed-text")
foreach ($model in $models) {
    Write-Host "Pulling model: $model" -ForegroundColor White
    docker exec prosecutor_ollama ollama pull $model 2>&1 | Out-Null
    if ($?) {
        Write-Success "Installed $model"
    } else {
        Write-Error "Failed to install $model"
    }
}

# Verify models
$installedModels = docker exec prosecutor_ollama ollama list 2>$null
Write-Host "`nInstalled models:" -ForegroundColor Cyan
Write-Host $installedModels

# Step 5: Database Setup
Write-Step "[5/8] Setting up database..."

# Wait for PostgreSQL to be fully ready
$attempts = 0
while ($attempts -lt 30) {
    $pgReady = docker exec prosecutor_postgres pg_isready -U postgres 2>&1
    if ($pgReady -match "accepting connections") {
        Write-Success "PostgreSQL is ready"
        break
    }
    $attempts++
    Start-Sleep -Seconds 1
}

# Create database if it doesn't exist
docker exec prosecutor_postgres psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'prosecutor_db'" | Select-String -Pattern "1" -Quiet
if (!$?) {
    docker exec prosecutor_postgres createdb -U postgres prosecutor_db
    Write-Success "Created prosecutor_db database"
}

# Enable pgvector extension
docker exec prosecutor_postgres psql -U postgres -d prosecutor_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
Write-Success "Enabled pgvector extension"

# Step 6: Fix TypeScript Errors
Write-Step "[6/8] Fixing TypeScript errors..."

Set-Location ".\sveltekit-frontend"

# Run the TypeScript fix script
if (Test-Path "fix-all-typescript-errors.mjs") {
    node fix-all-typescript-errors.mjs
}

# Install dependencies
npm install --legacy-peer-deps

# Run database migrations
Write-Host "Running database migrations..." -ForegroundColor White
npm run db:migrate 2>&1 | Out-Null

Set-Location ..

# Step 7: Test Ollama Integration
Write-Step "[7/8] Testing Ollama integration..."

$ollamaTest = @"
{
    "model": "gemma3",
    "prompt": "Hello, this is a test. Respond with 'OK' if you receive this.",
    "stream": false
}
"@

try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
        -Method Post `
        -Body $ollamaTest `
        -ContentType "application/json" `
        -ErrorAction Stop
    
    if ($response.response) {
        Write-Success "Ollama is responding correctly"
        Write-Host "Response: $($response.response)" -ForegroundColor Gray
    }
} catch {
    Write-Error "Ollama test failed: $_"
}

# Step 8: Final Status
Write-Step "[8/8] Setup complete!"

Write-Host "`n=== Service Status ===" -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host "`n=== Quick Start Guide ===" -ForegroundColor Cyan
Write-Host "1. Start the development server:" -ForegroundColor White
Write-Host "   cd sveltekit-frontend && npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Access the application:" -ForegroundColor White
Write-Host "   http://localhost:5173" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Service URLs:" -ForegroundColor White
Write-Host "   Ollama API: http://localhost:11434" -ForegroundColor Gray
Write-Host "   Qdrant API: http://localhost:6333" -ForegroundColor Gray
Write-Host "   PostgreSQL: localhost:5432" -ForegroundColor Gray
Write-Host "   Redis: localhost:6379" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Test Ollama models:" -ForegroundColor White
Write-Host "   curl http://localhost:11434/api/tags" -ForegroundColor Gray

# Create a quick start script
@"
@echo off
cd sveltekit-frontend
npm run dev
"@ | Set-Content -Path ".\START-DEV.bat" -Force

Write-Success "Created START-DEV.bat for quick development start"
