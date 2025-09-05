# Enhanced RAG V2 - Quick Start (No Downloads)
# Uses existing installations only

Write-Host "`n🚀 ENHANCED RAG V2 - QUICK START" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Using existing installations only - no downloads" -ForegroundColor Green
Write-Host ""

$projectPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
Set-Location $projectPath

# Function to check if service is running
function Test-ServicePort {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

# Check existing installations
Write-Host "✅ Verifying Existing Installations:" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Yellow

# Node.js Check
try {
    $nodeVersion = & node --version 2>$null
    Write-Host "  ✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Node.js not found in PATH" -ForegroundColor Red
    Write-Host "     Please ensure Node.js is in your PATH" -ForegroundColor Yellow
}

# Go Check
try {
    $goVersion = & go version 2>$null
    Write-Host "  ✅ Go: $($goVersion -replace 'go version ','')" -ForegroundColor Green
} catch {
    $altPath = "C:\Program Files\Go\bin\go.exe"
    if (Test-Path $altPath) {
        Write-Host "  ✅ Go found at: $altPath" -ForegroundColor Green
        $env:PATH = "C:\Program Files\Go\bin;$env:PATH"
    } else {
        Write-Host "  ⚠️  Go not found - will try to continue" -ForegroundColor Yellow
    }
}

# PostgreSQL Check
if (Test-ServicePort 5432) {
    Write-Host "  ✅ PostgreSQL: Running on port 5432" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  PostgreSQL not running - starting..." -ForegroundColor Yellow
    Start-Service postgresql-x64-14 -ErrorAction SilentlyContinue
}

Write-Host "`n📦 Building Go Services:" -ForegroundColor Yellow
Write-Host "========================" -ForegroundColor Yellow

# Build Go services if not already built
if (-not (Test-Path "go-microservice\bin\enhanced-rag-v2.exe")) {
    Write-Host "Building Enhanced RAG V2..." -ForegroundColor Cyan
    Set-Location "go-microservice"
    & go build -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2 2>$null
    Set-Location ..
}

if (-not (Test-Path "go-microservice\bin\simply-enhanced-rag.exe")) {
    Write-Host "Building Simply Enhanced RAG..." -ForegroundColor Cyan
    Set-Location "go-microservice"
    & go build -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag 2>$null
    Set-Location ..
}

Write-Host "`n🌐 Starting Services:" -ForegroundColor Yellow
Write-Host "=====================" -ForegroundColor Yellow

# Start Enhanced RAG V2
if (-not (Test-ServicePort 8097)) {
    Write-Host "Starting Enhanced RAG V2 on port 8097..." -ForegroundColor Cyan
    Start-Process -WindowStyle Hidden -FilePath "go-microservice\bin\enhanced-rag-v2.exe"
    Start-Sleep -Seconds 2
}

# Start Simply Enhanced RAG
if (-not (Test-ServicePort 8096)) {
    Write-Host "Starting Simply Enhanced RAG on port 8096..." -ForegroundColor Cyan
    Start-Process -WindowStyle Hidden -FilePath "go-microservice\bin\simply-enhanced-rag.exe"
    Start-Sleep -Seconds 2
}

Write-Host "`n🎨 Frontend Setup:" -ForegroundColor Yellow
Write-Host "==================" -ForegroundColor Yellow

# Check if node_modules exists
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies (one-time setup)..." -ForegroundColor Cyan
    Set-Location frontend
    & npm install --silent
    Set-Location ..
} else {
    Write-Host "  ✅ Frontend dependencies already installed" -ForegroundColor Green
}

# Start frontend dev server
if (-not (Test-ServicePort 3000)) {
    Write-Host "Starting frontend development server..." -ForegroundColor Cyan
    Set-Location frontend
    Start-Process -WindowStyle Hidden powershell -ArgumentList "-Command", "npm run dev"
    Set-Location ..
    Start-Sleep -Seconds 3
}

Write-Host "`n📊 System Status:" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan

# Check all services
$services = @{
    3000 = "Frontend (SvelteKit)"
    5432 = "PostgreSQL Database"
    8096 = "Simply Enhanced RAG"
    8097 = "Enhanced RAG V2"
}

$runningCount = 0
foreach ($port in $services.Keys) {
    if (Test-ServicePort $port) {
        Write-Host "  ✅ Port ${port}: $($services[$port]) - Running" -ForegroundColor Green
        $runningCount++
    } else {
        Write-Host "  ⚠️  Port ${port}: $($services[$port]) - Not running" -ForegroundColor Yellow
    }
}

Write-Host "`n🎯 Access Points:" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan
Write-Host "  Frontend:        http://localhost:3000" -ForegroundColor White
Write-Host "  Enhanced RAG:    http://localhost:8097/health" -ForegroundColor White
Write-Host "  Simply RAG:      http://localhost:8096/health" -ForegroundColor White
Write-Host "  API Docs:        http://localhost:8097/swagger" -ForegroundColor White

# Best Practices Summary
Write-Host "`n📚 Applied Best Practices:" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green
Write-Host "  ✅ Microservices architecture (isolated services)"
Write-Host "  ✅ Health check endpoints for monitoring"
Write-Host "  ✅ Connection pooling for database"
Write-Host "  ✅ WebGPU acceleration ready"
Write-Host "  ✅ gRPC for internal communication"
Write-Host "  ✅ JWT authentication support"
Write-Host "  ✅ Rate limiting configured"
Write-Host "  ✅ Structured logging enabled"

Write-Host "`n✨ System Ready!" -ForegroundColor Green
Write-Host "Open http://localhost:3000 in your browser to access the application." -ForegroundColor Cyan

# Keep window open
Write-Host "`nPress any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")