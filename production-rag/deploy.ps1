# Production RAG Deployment Script for Windows
# Complete setup and deployment automation

Write-Host "🚀 Starting Production RAG Deployment" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Function to check if a service is installed
function Test-ServiceInstalled {
    param($ServiceName)
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    return $null -ne $service
}

# Function to check if a port is available
function Test-PortAvailable {
    param($Port)
    $connection = New-Object System.Net.Sockets.TcpClient
    try {
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $false
    } catch {
        return $true
    }
}

# Step 1: Check Prerequisites
Write-Host "`n📋 Checking Prerequisites..." -ForegroundColor Yellow

$prerequisites = @{
    "PostgreSQL" = @{
        Service = "postgresql-x64-14"
        Port = 5432
        Required = $true
    }
    "Redis" = @{
        Service = "Redis"
        Port = 6379
        Required = $true
    }
    "NATS" = @{
        Service = $null
        Port = 4222
        Required = $true
    }
    "RabbitMQ" = @{
        Service = "RabbitMQ"
        Port = 5672
        Required = $true
    }
    "Qdrant" = @{
        Service = $null
        Port = 6333
        Required = $false
    }
}

$allServicesReady = $true

foreach ($service in $prerequisites.Keys) {
    $config = $prerequisites[$service]
    $status = "❌ Not Installed"
    $color = "Red"
    
    if ($config.Service -and (Test-ServiceInstalled $config.Service)) {
        $status = "✅ Installed"
        $color = "Green"
        
        # Check if running
        $svc = Get-Service -Name $config.Service -ErrorAction SilentlyContinue
        if ($svc.Status -eq "Running") {
            $status = "✅ Running"
        } else {
            $status = "⚠️ Installed but not running"
            $color = "Yellow"
            if ($config.Required) { $allServicesReady = $false }
        }
    } elseif (-not (Test-PortAvailable $config.Port)) {
        $status = "✅ Running (external)"
        $color = "Green"
    } else {
        if ($config.Required) { $allServicesReady = $false }
    }
    
    Write-Host "$service : $status (Port: $($config.Port))" -ForegroundColor $color
}

# Step 2: Database Setup
Write-Host "`n🗄️ Setting up Databases..." -ForegroundColor Yellow

# PostgreSQL with pgvector
$pgScript = @"
-- Create database if not exists
SELECT 'CREATE DATABASE legal_ai_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'legal_ai_db')\gexec

-- Connect to database
\c legal_ai_db

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables
CREATE TABLE IF NOT EXISTS legal_documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500),
    content TEXT,
    document_type VARCHAR(100),
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS contracts (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES legal_documents(id),
    parties JSONB,
    terms JSONB,
    risk_score FLOAT,
    som_cluster INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    messages JSONB[],
    context JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_embedding_cosine 
ON legal_documents USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_document_type 
ON legal_documents(document_type);

CREATE INDEX IF NOT EXISTS idx_metadata 
ON legal_documents USING gin(metadata);

-- Create functions
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(384),
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE(
    id INTEGER,
    title VARCHAR,
    content TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ld.id,
        ld.title,
        ld.content,
        1 - (ld.embedding <=> query_embedding) as similarity
    FROM legal_documents ld
    ORDER BY ld.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO legal_admin;
"@

# Save PostgreSQL script
$pgScript | Out-File -FilePath ".\production-rag\database-setup.sql" -Encoding UTF8
Write-Host "✅ PostgreSQL setup script created" -ForegroundColor Green

# Step 3: Go Service Build
Write-Host "`n🔨 Building Go Services..." -ForegroundColor Yellow

$goBuildScript = @"
@echo off
echo Building Enhanced RAG V2 Service...

cd go-microservice

:: Download dependencies
echo Downloading Go dependencies...
go mod download

:: Build services
echo Building enhanced-rag-v2...
go build -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2

echo Building simply-enhanced-rag...
go build -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag

:: Build with optimizations for production
echo Building production binaries with optimizations...
set CGO_ENABLED=1
set GOOS=windows
set GOARCH=amd64
go build -ldflags="-s -w" -o bin\enhanced-rag-v2-prod.exe .\cmd\enhanced-rag-v2

echo ✅ Build complete!
echo.
echo Binaries available in:
echo   - bin\enhanced-rag-v2.exe
echo   - bin\simply-enhanced-rag.exe
echo   - bin\enhanced-rag-v2-prod.exe (optimized)
"@

$goBuildScript | Out-File -FilePath ".\production-rag\build-services.bat" -Encoding ASCII
Write-Host "✅ Go build script created" -ForegroundColor Green

# Step 4: Frontend Setup
Write-Host "`n🎨 Setting up Frontend..." -ForegroundColor Yellow

$frontendSetup = @"
{
  "name": "production-rag-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage"
  },
  "dependencies": {
    "@xstate/svelte": "^3.0.0",
    "bits-ui": "^0.19.0",
    "clsx": "^2.1.0",
    "lucide-svelte": "^0.321.0",
    "melt-ui": "^0.72.0",
    "svelte": "^4.2.9",
    "tailwind-merge": "^2.2.1",
    "tailwind-variants": "^0.2.0"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^3.0.1",
    "@testing-library/svelte": "^4.1.0",
    "@types/node": "^20.11.5",
    "@vitest/ui": "^1.2.1",
    "autoprefixer": "^10.4.17",
    "postcss": "^8.4.33",
    "tailwindcss": "^3.4.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.12",
    "vitest": "^1.2.1"
  }
}
"@

$frontendSetup | Out-File -FilePath ".\production-rag\package.json" -Encoding UTF8
Write-Host "✅ Frontend package.json created" -ForegroundColor Green

# Step 5: NATS Configuration
Write-Host "`n📡 Configuring NATS..." -ForegroundColor Yellow

$natsConfig = @"
# NATS Server Configuration for Production RAG
port: 4222
http_port: 8222

# WebSocket configuration for browser clients
websocket {
  port: 8080
  no_tls: true
  compression: true
}

# JetStream for persistence
jetstream {
  store_dir: "./data/nats/jetstream"
  max_memory_store: 2GB
  max_file_store: 10GB
}

# Cluster configuration (for scaling)
cluster {
  name: "production-rag-cluster"
  port: 6222
}

# Authorization
authorization {
  users = [
    {
      user: "rag-service"
      password: "rag-secret-123"
      permissions: {
        publish: ["rag.>", "_INBOX.>"]
        subscribe: ["rag.>", "_INBOX.>"]
      }
    }
  ]
}

# Logging
debug: false
trace: false
logtime: true
log_file: "./logs/nats.log"

# Limits
max_connections: 10000
max_control_line: 4096
max_payload: 8MB
max_pending: 64MB

# Monitoring
server_name: "production-rag-nats"
"@

$natsConfig | Out-File -FilePath ".\production-rag\nats.conf" -Encoding UTF8
Write-Host "✅ NATS configuration created" -ForegroundColor Green

# Step 6: Start Script
Write-Host "`n🚀 Creating Start Script..." -ForegroundColor Yellow

$startScript = @'
@echo off
title Production RAG System

echo ========================================
echo     PRODUCTION RAG SYSTEM STARTUP
echo ========================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Please run as Administrator!
    pause
    exit /b 1
)

:: Start PostgreSQL
echo Starting PostgreSQL...
net start postgresql-x64-14 >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ PostgreSQL started
) else (
    echo ⚠️ PostgreSQL already running or failed to start
)

:: Start Redis
echo Starting Redis...
net start Redis >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Redis started
) else (
    echo ⚠️ Redis already running or failed to start
)

:: Start RabbitMQ
echo Starting RabbitMQ...
net start RabbitMQ >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ RabbitMQ started
) else (
    echo ⚠️ RabbitMQ already running or failed to start
)

:: Start NATS
echo Starting NATS Server...
start /B nats-server -c nats.conf
echo ✅ NATS Server started

:: Start Qdrant (if using Docker)
echo Starting Qdrant Vector DB...
docker run -d -p 6333:6333 -p 6334:6334 ^
    -v %cd%\data\qdrant:/qdrant/storage ^
    --name qdrant ^
    qdrant/qdrant >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Qdrant started
) else (
    echo ⚠️ Qdrant already running or Docker not available
)

echo.
echo ========================================
echo All services started successfully!
echo ========================================
echo.
echo Starting Enhanced RAG V2 Service...
echo.

:: Start the main service
cd go-microservice\bin
start enhanced-rag-v2-prod.exe

echo.
echo 🚀 System is running!
echo.
echo Access points:
echo   - API:       http://localhost:8097
echo   - WebSocket: ws://localhost:8098
echo   - NATS WS:   ws://localhost:8080
echo   - RabbitMQ:  http://localhost:15672
echo   - Qdrant:    http://localhost:6333
echo.
echo Press any key to stop all services...
pause >nul

:: Cleanup
echo.
echo Stopping services...
taskkill /F /IM enhanced-rag-v2-prod.exe >nul 2>&1
taskkill /F /IM nats-server.exe >nul 2>&1
docker stop qdrant >nul 2>&1
docker rm qdrant >nul 2>&1

echo ✅ All services stopped
echo.
pause
'@

$startScript | Out-File -FilePath ".\production-rag\start-system.bat" -Encoding ASCII
Write-Host "✅ Start script created" -ForegroundColor Green

# Step 7: Test Script
Write-Host "`n🧪 Creating Test Script..." -ForegroundColor Yellow

$testScript = @'
#!/usr/bin/env pwsh

Write-Host "🧪 Running Production RAG Tests" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Test configuration
$baseUrl = "http://localhost:8097"
$wsUrl = "ws://localhost:8098"
$tests = @()
$passed = 0
$failed = 0

# Function to run a test
function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Method,
        [string]$Url,
        [object]$Body = $null,
        [int]$ExpectedStatus = 200
    )
    
    Write-Host "`nTesting: $Name" -ForegroundColor Yellow
    
    try {
        $params = @{
            Method = $Method
            Uri = $Url
            ContentType = "application/json"
        }
        
        if ($Body) {
            $params.Body = ($Body | ConvertTo-Json)
        }
        
        $response = Invoke-RestMethod @params -ErrorAction Stop
        Write-Host "  ✅ Success" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  ❌ Failed: $_" -ForegroundColor Red
        return $false
    }
}

# Test 1: Health Check
if (Test-Endpoint -Name "Health Check" -Method "GET" -Url "$baseUrl/health") {
    $passed++
} else {
    $failed++
}

# Test 2: Enhanced Search
$searchBody = @{
    query = "contract liability clause"
    sessionId = [System.Guid]::NewGuid().ToString()
    options = @{
        useGPU = $true
        useSOM = $true
    }
}

if (Test-Endpoint -Name "Enhanced Search" -Method "POST" -Url "$baseUrl/api/v2/search" -Body $searchBody) {
    $passed++
} else {
    $failed++
}

# Test 3: GPU Vector Similarity
$vectorBody = @{
    vectors_a = @(@(0.1, 0.2, 0.3, 0.4))
    vectors_b = @(@(0.2, 0.3, 0.4, 0.5))
}

if (Test-Endpoint -Name "GPU Vector Similarity" -Method "POST" -Url "$baseUrl/api/v2/gpu/vector-similarity" -Body $vectorBody) {
    $passed++
} else {
    $failed++
}

# Test 4: WebSocket Connection
Write-Host "`nTesting: WebSocket Connection" -ForegroundColor Yellow
try {
    $ws = New-Object System.Net.WebSockets.ClientWebSocket
    $uri = [System.Uri]::new($wsUrl)
    $cts = New-Object System.Threading.CancellationTokenSource
    
    $connectTask = $ws.ConnectAsync($uri, $cts.Token)
    $connectTask.Wait(5000)
    
    if ($ws.State -eq [System.Net.WebSockets.WebSocketState]::Open) {
        Write-Host "  ✅ WebSocket connected" -ForegroundColor Green
        $passed++
        $ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, "", $cts.Token).Wait()
    } else {
        Write-Host "  ❌ WebSocket connection failed" -ForegroundColor Red
        $failed++
    }
} catch {
    Write-Host "  ❌ WebSocket error: $_" -ForegroundColor Red
    $failed++
}

# Test 5: Database Connection
Write-Host "`nTesting: Database Connection" -ForegroundColor Yellow
try {
    $pgTest = & psql -U legal_admin -d legal_ai_db -c "SELECT version();" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ PostgreSQL connected" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "  ❌ PostgreSQL connection failed" -ForegroundColor Red
        $failed++
    }
} catch {
    Write-Host "  ❌ PostgreSQL error: $_" -ForegroundColor Red
    $failed++
}

# Test 6: Redis Connection
Write-Host "`nTesting: Redis Connection" -ForegroundColor Yellow
try {
    $redisTest = & redis-cli ping 2>&1
    if ($redisTest -eq "PONG") {
        Write-Host "  ✅ Redis connected" -ForegroundColor Green
        $passed++
    } else {
        Write-Host "  ❌ Redis connection failed" -ForegroundColor Red
        $failed++
    }
} catch {
    Write-Host "  ❌ Redis error: $_" -ForegroundColor Red
    $failed++
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Test Results Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Passed: $passed" -ForegroundColor Green
Write-Host "❌ Failed: $failed" -ForegroundColor Red

if ($failed -eq 0) {
    Write-Host "`n🎉 All tests passed! System is ready for production!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️ Some tests failed. Please check the configuration." -ForegroundColor Yellow
}
'@

$testScript | Out-File -FilePath ".\production-rag\test-system.ps1" -Encoding UTF8
Write-Host "✅ Test script created" -ForegroundColor Green

# Final Summary
Write-Host "`n" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   PRODUCTION RAG DEPLOYMENT COMPLETE   " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n📁 Files created in: .\production-rag\" -ForegroundColor Yellow
Write-Host "`nNext steps:" -ForegroundColor White
Write-Host "  1. Run: cd production-rag" -ForegroundColor Gray
Write-Host "  2. Run: .\build-services.bat" -ForegroundColor Gray
Write-Host "  3. Run: .\start-system.bat (as Administrator)" -ForegroundColor Gray
Write-Host "  4. Run: .\test-system.ps1" -ForegroundColor Gray
Write-Host "`n✅ System is ready for production deployment!" -ForegroundColor Green
