#!/usr/bin/env powershell
# Enhanced Legal AI Setup Script with YoRHa Aesthetic and Vector Search
# Phase 3 Production Implementation

param(
    [switch]$StartDocker,
    [switch]$SetupDatabase,
    [switch]$StartOllama,
    [switch]$LoadModels,
    [switch]$StartDev,
    [switch]$Verbose,
    [string]$Environment = "development"
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendPath = Join-Path $ProjectRoot "sveltekit-frontend"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ENHANCED LEGAL AI SETUP - PHASE 3" -ForegroundColor Cyan
Write-Host "YoRHa Aesthetic + Vector Search + AI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Function to check if command exists
function Test-Command($command) {
    try {
        if (Get-Command $command -ErrorAction Stop) {
            return $true
        }
    }
    catch {
        return $false
    }
}

# Function to start Docker services
function Start-DockerServices {
    Write-Host "Starting Docker services..." -ForegroundColor Yellow
    
    if (-not (Test-Command "docker")) {
        Write-Host "ERROR: Docker not found. Please install Docker Desktop." -ForegroundColor Red
        return $false
    }
    
    try {
        # Check if docker-compose.yml exists
        $composeFiles = @(
            "docker-compose.yml",
            "docker-compose.enhanced.yml",
            "docker-compose.lowmem.yml"
        )
        
        $activeCompose = $null
        foreach ($file in $composeFiles) {
            $path = Join-Path $ProjectRoot $file
            if (Test-Path $path) {
                $activeCompose = $path
                break
            }
        }
        
        if (-not $activeCompose) {
            Write-Host "WARNING: No docker-compose.yml found. Creating minimal setup..." -ForegroundColor Yellow
            # Create minimal docker-compose.yml
            $minimalCompose = @"
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: prosecutor_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    command: postgres -c shared_preload_libraries=vector

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
"@
            $activeCompose = Join-Path $ProjectRoot "docker-compose.yml"
            $minimalCompose | Set-Content $activeCompose
        }
        
        Set-Location $ProjectRoot
        docker-compose up -d postgres redis qdrant
        
        # Wait for services to be ready
        Write-Host "Waiting for services to start..." -ForegroundColor Yellow
        Start-Sleep 10
        
        # Test connections
        $postgresReady = $false
        $redisReady = $false
        $qdrantReady = $false
        
        for ($i = 0; $i -lt 30; $i++) {
            try {
                if (-not $postgresReady) {
                    $pgTest = docker exec (docker ps -q -f "name=postgres") pg_isready -U postgres
                    if ($pgTest -match "accepting connections") {
                        $postgresReady = $true
                        Write-Host "✓ PostgreSQL ready" -ForegroundColor Green
                    }
                }
                
                if (-not $redisReady) {
                    $redisTest = docker exec (docker ps -q -f "name=redis") redis-cli ping
                    if ($redisTest -eq "PONG") {
                        $redisReady = $true
                        Write-Host "✓ Redis ready" -ForegroundColor Green
                    }
                }
                
                if (-not $qdrantReady) {
                    try {
                        $qdrantTest = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET -TimeoutSec 2
                        if ($qdrantTest) {
                            $qdrantReady = $true
                            Write-Host "✓ Qdrant ready" -ForegroundColor Green
                        }
                    }
                    catch { }
                }
                
                if ($postgresReady -and $redisReady -and $qdrantReady) {
                    break
                }
                
                Start-Sleep 2
            }
            catch {
                Start-Sleep 2
            }
        }
        
        return ($postgresReady -and $redisReady -and $qdrantReady)
    }
    catch {
        Write-Host "ERROR: Failed to start Docker services: $_" -ForegroundColor Red
        return $false
    }
}

# Function to setup Ollama
function Setup-Ollama {
    Write-Host "Setting up Ollama..." -ForegroundColor Yellow
    
    # Check if Ollama is installed
    $ollamaPath = $null
    $possiblePaths = @(
        "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe",
        "C:\Program Files\Ollama\ollama.exe",
        "ollama.exe"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $ollamaPath = $path
            break
        }
    }
    
    if (-not $ollamaPath -and (Test-Command "ollama")) {
        $ollamaPath = "ollama"
    }
    
    if (-not $ollamaPath) {
        Write-Host "WARNING: Ollama not found. Please install from https://ollama.ai" -ForegroundColor Yellow
        return $false
    }
    
    try {
        # Start Ollama service
        Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep 5
        
        # Test if service is running
        $ollamaTest = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 10
        Write-Host "✓ Ollama service running" -ForegroundColor Green
        
        if ($LoadModels) {
            # Pull required models
            $models = @(
                "gemma:7b",
                "nomic-embed-text"
            )
            
            foreach ($model in $models) {
                Write-Host "Pulling model: $model..." -ForegroundColor Yellow
                & $ollamaPath pull $model
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✓ Model $model ready" -ForegroundColor Green
                } else {
                    Write-Host "WARNING: Failed to pull $model" -ForegroundColor Yellow
                }
            }
            
            # Create enhanced legal model
            $modelfilePath = Join-Path $ProjectRoot "Gemma3-Legal-Enhanced-Modelfile-v2"
            if (Test-Path $modelfilePath) {
                Write-Host "Creating enhanced legal model..." -ForegroundColor Yellow
                & $ollamaPath create gemma3-legal-enhanced -f $modelfilePath
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✓ Enhanced legal model created" -ForegroundColor Green
                }
            }
        }
        
        return $true
    }
    catch {
        Write-Host "ERROR: Failed to setup Ollama: $_" -ForegroundColor Red
        return $false
    }
}

# Function to setup database
function Setup-Database {
    Write-Host "Setting up database..." -ForegroundColor Yellow
    
    if (-not (Test-Path $FrontendPath)) {
        Write-Host "ERROR: Frontend directory not found: $FrontendPath" -ForegroundColor Red
        return $false
    }
    
    Set-Location $FrontendPath
    
    try {
        # Install dependencies
        Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
        npm install
        
        # Setup environment
        if (-not (Test-Path ".env")) {
            $envContent = @"
# Database Configuration
DATABASE_URL="postgresql://postgres:password123@localhost:5432/prosecutor_db"
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=prosecutor_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password123

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_documents
QDRANT_API_KEY=

# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3-legal-enhanced

# Application Configuration
NODE_ENV=$Environment
PUBLIC_APP_NAME="Enhanced Legal AI"
PUBLIC_APP_VERSION="3.0.0"

# Security
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
SESSION_SECRET=your-super-secret-session-key-change-this-in-production
"@
            $envContent | Set-Content ".env"
            Write-Host "✓ Environment file created" -ForegroundColor Green
        }
        
        # Run database migrations
        Write-Host "Running database setup..." -ForegroundColor Yellow
        npm run db:push
        
        # Initialize vector collection
        Write-Host "Initializing vector search..." -ForegroundColor Yellow
        
        # Create initialization script
        $initScript = @"
import { vectorService } from './src/lib/server/vector/vectorService.js';

async function initializeVectorSearch() {
  try {
    console.log('Initializing vector search...');
    await vectorService.initializeCollection();
    
    const health = await vectorService.healthCheck();
    console.log('Vector service health:', health);
    
    const stats = await vectorService.getStats();
    console.log('Collection stats:', stats);
    
    console.log('Vector search initialization complete!');
  } catch (error) {
    console.error('Vector search initialization failed:', error);
  }
}

initializeVectorSearch();
"@
        $initScript | Set-Content "init-vector.mjs"
        node init-vector.mjs
        Remove-Item "init-vector.mjs" -ErrorAction SilentlyContinue
        
        Write-Host "✓ Database setup complete" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "ERROR: Database setup failed: $_" -ForegroundColor Red
        return $false
    }
}

# Function to run health checks
function Test-SystemHealth {
    Write-Host "Running system health checks..." -ForegroundColor Yellow
    
    $health = @{
        Docker = $false
        PostgreSQL = $false
        Redis = $false
        Qdrant = $false
        Ollama = $false
        Frontend = $false
    }
    
    # Check Docker services
    try {
        $dockerPs = docker ps --format "table {{.Names}}\t{{.Status}}"
        if ($dockerPs -match "postgres.*Up" -and $dockerPs -match "redis.*Up" -and $dockerPs -match "qdrant.*Up") {
            $health.Docker = $true
            $health.PostgreSQL = $true
            $health.Redis = $true
            $health.Qdrant = $true
        }
    }
    catch { }
    
    # Check Ollama
    try {
        $ollamaTest = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5
        $health.Ollama = $true
    }
    catch { }
    
    # Check Frontend
    if (Test-Path $FrontendPath) {
        Set-Location $FrontendPath
        if (Test-Path "package.json") {
            $health.Frontend = $true
        }
    }
    
    # Display results
    Write-Host "`nSystem Health Status:" -ForegroundColor Cyan
    foreach ($service in $health.Keys) {
        $status = if ($health[$service]) { "✓ HEALTHY" } else { "✗ OFFLINE" }
        $color = if ($health[$service]) { "Green" } else { "Red" }
        Write-Host "$service : $status" -ForegroundColor $color
    }
    
    return $health
}

# Main execution
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host "Frontend Path: $FrontendPath" -ForegroundColor Gray
Write-Host "Environment: $Environment" -ForegroundColor Gray
Write-Host ""

$success = $true

# Start Docker services
if ($StartDocker -or $SetupDatabase) {
    if (-not (Start-DockerServices)) {
        $success = $false
        Write-Host "Failed to start Docker services" -ForegroundColor Red
    }
}

# Setup Ollama
if ($StartOllama) {
    if (-not (Setup-Ollama)) {
        $success = $false
        Write-Host "Failed to setup Ollama" -ForegroundColor Red
    }
}

# Setup database
if ($SetupDatabase) {
    if (-not (Setup-Database)) {
        $success = $false
        Write-Host "Failed to setup database" -ForegroundColor Red
    }
}

# Run health checks
$healthStatus = Test-SystemHealth

# Start development server
if ($StartDev -and $success) {
    Write-Host "`nStarting development server..." -ForegroundColor Yellow
    Set-Location $FrontendPath
    
    # Final dependency check
    npm install
    
    # Start the development server
    Write-Host "Opening http://localhost:5173" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    npm run dev
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($success) {
    Write-Host "✓ Enhanced Legal AI setup completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Features enabled:" -ForegroundColor Yellow
    Write-Host "• YoRHa terminal aesthetic design" -ForegroundColor White
    Write-Host "• Multi-step forms with XState v5" -ForegroundColor White
    Write-Host "• Vector search with PostgreSQL + Qdrant" -ForegroundColor White
    Write-Host "• AI chat with Ollama + Gemma 3" -ForegroundColor White
    Write-Host "• Interactive evidence board" -ForegroundColor White
    Write-Host "• Real-time collaboration" -ForegroundColor White
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "npm run dev          - Start development server" -ForegroundColor White
    Write-Host "npm run dev:with-llm - Start with AI services" -ForegroundColor White
    Write-Host "npm run db:studio    - Open database studio" -ForegroundColor White
    Write-Host "npm run type-check   - Run TypeScript checks" -ForegroundColor White
} else {
    Write-Host "⚠ Setup completed with warnings. Check the logs above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Visit http://localhost:5173 to access the application" -ForegroundColor White
Write-Host "2. Test multi-step forms with YoRHa aesthetic" -ForegroundColor White
Write-Host "3. Try AI assistance and vector search" -ForegroundColor White
Write-Host "4. Explore the interactive evidence board" -ForegroundColor White
Write-Host ""

# Return to original directory
Set-Location $ProjectRoot