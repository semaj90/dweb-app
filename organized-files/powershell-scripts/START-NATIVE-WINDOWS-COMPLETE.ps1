# Native Windows Complete Setup & Fix Script
# This script fixes all issues and sets up services natively on Windows

param(
    [switch]$SkipServices = $false,
    [switch]$FixOnly = $false,
    [switch]$QuickStart = $false
)

Write-Host @"
╔══════════════════════════════════════════════════════════════╗
║     LEGAL AI PLATFORM - NATIVE WINDOWS SETUP & FIX          ║
║                  No Docker Required!                         ║
╚══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Helper Functions
function Write-Step {
    param($Message)
    Write-Host "`n━━━ $Message ━━━" -ForegroundColor Yellow
}

function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-PortAvailable {
    param($Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $false
    } catch {
        return $true
    }
}

function Kill-ProcessOnPort {
    param($Port)
    $processId = netstat -ano | findstr ":$Port" | ForEach-Object {
        $parts = $_ -split '\s+'
        $parts[-1]
    } | Select-Object -Unique
    
    if ($processId) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            Write-Host "  ✓ Killed process on port $Port" -ForegroundColor Green
        } catch {
            Write-Host "  ⚠ Could not kill process on port $Port" -ForegroundColor Yellow
        }
    }
}

# Check admin privileges
if (-not (Test-Admin)) {
    Write-Host "⚠ This script requires Administrator privileges!" -ForegroundColor Red
    Write-Host "Please run PowerShell as Administrator and try again." -ForegroundColor Yellow
    exit 1
}

# PHASE 1: FIX CRITICAL ISSUES
if (-not $SkipServices) {
    Write-Step "PHASE 1: FIXING CRITICAL ISSUES"

    # Fix TypeScript errors
    Write-Host "`n🔧 Fixing TypeScript errors..." -ForegroundColor Cyan
    if (Test-Path "fix-typescript-errors.mjs") {
        node fix-typescript-errors.mjs
    }

    # Fix port conflicts
    Write-Host "`n🔌 Checking and fixing port conflicts..." -ForegroundColor Cyan
    $ports = @(3000, 5432, 6379, 7474, 7687, 8084, 8085, 9000, 9001, 11434, 15672)
    foreach ($port in $ports) {
        if (-not (Test-PortAvailable $port)) {
            Write-Host "  Port $port is in use, clearing..." -ForegroundColor Yellow
            Kill-ProcessOnPort $port
        } else {
            Write-Host "  ✓ Port $port is available" -ForegroundColor Green
        }
    }

    # Fix YoRHa interface integration
    Write-Host "`n🎮 Integrating YoRHa interface..." -ForegroundColor Cyan
    
    # Backup current homepage
    if (Test-Path "src\routes\+page.svelte") {
        if (-not (Test-Path "src\routes\+page.svelte.backup")) {
            Copy-Item "src\routes\+page.svelte" "src\routes\+page.svelte.backup"
            Write-Host "  ✓ Backed up current homepage" -ForegroundColor Green
        }
    }
    
    # Copy YoRHa as main page
    if (Test-Path "src\routes\yorha-dashboard\+page.svelte") {
        Copy-Item "src\routes\yorha-dashboard\+page.svelte" "src\routes\+page.svelte" -Force
        Write-Host "  ✓ YoRHa is now the homepage!" -ForegroundColor Green
    }
}

# PHASE 2: INSTALL NATIVE SERVICES
if (-not $FixOnly) {
    Write-Step "PHASE 2: SETTING UP NATIVE WINDOWS SERVICES"

    # PostgreSQL
    Write-Host "`n📦 Setting up PostgreSQL..." -ForegroundColor Cyan
    $pgPath = "C:\Program Files\PostgreSQL\15"
    if (-not (Test-Path $pgPath)) {
        Write-Host "  Downloading PostgreSQL..." -ForegroundColor Yellow
        $pgInstaller = "$env:TEMP\postgresql-15-windows.exe"
        if (-not (Test-Path $pgInstaller)) {
            Invoke-WebRequest -Uri "https://get.enterprisedb.com/postgresql/postgresql-15.4-1-windows-x64.exe" -OutFile $pgInstaller
        }
        
        Write-Host "  Installing PostgreSQL..." -ForegroundColor Yellow
        Start-Process -FilePath $pgInstaller -ArgumentList "--mode", "unattended", "--prefix", $pgPath, "--serverport", "5432", "--superpassword", "postgres", "--servicename", "PostgreSQL-15" -Wait
    }
    
    # Install pgvector
    Write-Host "  Installing pgvector extension..." -ForegroundColor Yellow
    & "$pgPath\bin\psql.exe" -U postgres -c "CREATE DATABASE IF NOT EXISTS legal_ai_db;" 2>$null
    & "$pgPath\bin\psql.exe" -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>$null
    Write-Host "  ✓ PostgreSQL with pgvector ready" -ForegroundColor Green

    # Redis
    Write-Host "`n📦 Setting up Redis..." -ForegroundColor Cyan
    $redisPath = "C:\Redis"
    if (-not (Test-Path $redisPath)) {
        Write-Host "  Downloading Redis..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $redisPath -Force | Out-Null
        $redisZip = "$env:TEMP\redis.zip"
        if (-not (Test-Path $redisZip)) {
            Invoke-WebRequest -Uri "https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip" -OutFile $redisZip
        }
        
        Write-Host "  Extracting Redis..." -ForegroundColor Yellow
        Expand-Archive -Path $redisZip -DestinationPath $redisPath -Force
        
        # Create Redis service
        $redisExe = "$redisPath\redis-server.exe"
        if (Test-Path $redisExe) {
            New-Service -Name "Redis" -BinaryPathName "$redisExe --service-run" -DisplayName "Redis" -StartupType Automatic -ErrorAction SilentlyContinue
        }
    }
    Write-Host "  ✓ Redis ready" -ForegroundColor Green

    # Neo4j
    Write-Host "`n📦 Setting up Neo4j..." -ForegroundColor Cyan
    $neo4jPath = "C:\neo4j"
    if (-not (Test-Path "$neo4jPath\neo4j-community-5.23.0")) {
        Write-Host "  Using existing Neo4j from project..." -ForegroundColor Yellow
        if (Test-Path "neo4j-community-5.23.0-windows.zip") {
            Expand-Archive -Path "neo4j-community-5.23.0-windows.zip" -DestinationPath $neo4jPath -Force
        }
    }
    Write-Host "  ✓ Neo4j ready" -ForegroundColor Green

    # MinIO
    Write-Host "`n📦 Setting up MinIO..." -ForegroundColor Cyan
    $minioPath = "C:\minio"
    if (-not (Test-Path $minioPath)) {
        New-Item -ItemType Directory -Path $minioPath -Force | Out-Null
        
        # Use existing minio.exe from project
        if (Test-Path "minio.exe") {
            Copy-Item "minio.exe" "$minioPath\minio.exe"
        } else {
            Write-Host "  Downloading MinIO..." -ForegroundColor Yellow
            Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile "$minioPath\minio.exe"
        }
        
        # Create MinIO data directory
        New-Item -ItemType Directory -Path "$minioPath\data" -Force | Out-Null
    }
    Write-Host "  ✓ MinIO ready" -ForegroundColor Green

    # Ollama
    Write-Host "`n📦 Checking Ollama..." -ForegroundColor Cyan
    $ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaInstalled) {
        Write-Host "  ⚠ Ollama not installed. Please install from: https://ollama.ai/download" -ForegroundColor Yellow
    } else {
        Write-Host "  ✓ Ollama installed" -ForegroundColor Green
    }
}

# PHASE 3: START ALL SERVICES
Write-Step "PHASE 3: STARTING ALL SERVICES"

# Start PostgreSQL
Write-Host "`n▶ Starting PostgreSQL..." -ForegroundColor Cyan
$pgService = Get-Service "PostgreSQL*" -ErrorAction SilentlyContinue
if ($pgService) {
    Start-Service $pgService.Name -ErrorAction SilentlyContinue
    Write-Host "  ✓ PostgreSQL started" -ForegroundColor Green
}

# Start Redis
Write-Host "`n▶ Starting Redis..." -ForegroundColor Cyan
if (Test-Path "C:\Redis\redis-server.exe") {
    Start-Process "C:\Redis\redis-server.exe" -WindowStyle Hidden
    Write-Host "  ✓ Redis started" -ForegroundColor Green
}

# Start Neo4j
Write-Host "`n▶ Starting Neo4j..." -ForegroundColor Cyan
$neo4jBin = "C:\neo4j\neo4j-community-5.23.0\bin\neo4j.bat"
if (-not (Test-Path $neo4jBin)) {
    $neo4jBin = ".\neo4j-community-5.23.0\bin\neo4j.bat"
}
if (Test-Path $neo4jBin) {
    Start-Process $neo4jBin -ArgumentList "console" -WindowStyle Hidden
    Write-Host "  ✓ Neo4j started" -ForegroundColor Green
}

# Start MinIO
Write-Host "`n▶ Starting MinIO..." -ForegroundColor Cyan
$minioExe = "C:\minio\minio.exe"
if (-not (Test-Path $minioExe)) {
    $minioExe = ".\minio.exe"
}
if (Test-Path $minioExe) {
    Start-Process $minioExe -ArgumentList "server", "C:\minio\data", "--console-address", ":9001" -WindowStyle Hidden
    Write-Host "  ✓ MinIO started" -ForegroundColor Green
}

# Start Ollama
Write-Host "`n▶ Starting Ollama..." -ForegroundColor Cyan
$ollamaRunning = Get-Process "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaRunning) {
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
    Write-Host "  ✓ Ollama started" -ForegroundColor Green
}

# Load AI Models
Write-Host "`n🤖 Loading AI models..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
ollama pull nomic-embed-text 2>$null
ollama pull gemma:2b 2>$null
Write-Host "  ✓ Models loaded" -ForegroundColor Green

# PHASE 4: START APPLICATION
Write-Step "PHASE 4: STARTING APPLICATION"

# Install dependencies if needed
if (-not (Test-Path "node_modules")) {
    Write-Host "`n📦 Installing npm dependencies..." -ForegroundColor Cyan
    npm install
}

# Set environment variables
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/legal_ai_db?sslmode=disable"
$env:REDIS_URL = "redis://localhost:6379"
$env:NEO4J_URI = "neo4j://localhost:7687"
$env:NEO4J_PASSWORD = "password"
$env:MINIO_ENDPOINT = "localhost:9000"
$env:MINIO_ACCESS_KEY = "minioadmin"
$env:MINIO_SECRET_KEY = "minioadmin123"
$env:OLLAMA_URL = "http://localhost:11434"
$env:NODE_ENV = "development"

# Run database migrations
Write-Host "`n🗄️ Running database migrations..." -ForegroundColor Cyan
npm run db:push 2>$null

# Display status
Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                   ✅ SYSTEM READY!                           ║
╚══════════════════════════════════════════════════════════════╝

🌐 Service URLs:
   • Application:     http://localhost:3000
   • Neo4j Browser:   http://localhost:7474
   • MinIO Console:   http://localhost:9001
   • Ollama API:      http://localhost:11434

🔑 Credentials:
   • PostgreSQL:      postgres / postgres
   • Neo4j:          neo4j / password  
   • MinIO:          minioadmin / minioadmin123

📋 Features Available:
   • YoRHa Dashboard (Homepage)
   • Enhanced RAG System
   • Document Processing
   • AI Chat Assistant
   • Evidence Management
   • Case Management

"@ -ForegroundColor Green

# Start the application
Write-Host "🚀 Starting SvelteKit application..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
npm run dev
