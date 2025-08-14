# 🔧 Quick Installation Script
# Execute this to check and install missing components

Write-Host "🚀 Legal AI System - Quick Installation Check" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# Check Docker Desktop
Write-Host "`n🐳 Checking Docker Desktop..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker installed: $dockerVersion" -ForegroundColor Green

    # Try to check if Docker is running
    try {
        $containers = docker ps 2>$null
        Write-Host "✅ Docker Desktop is running" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Docker Desktop is installed but not running" -ForegroundColor Red
        Write-Host "   ACTION: Start Docker Desktop from Windows Start Menu" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Docker not found - Install from https://docker.com/desktop" -ForegroundColor Red
}

# Check Node.js and NPM
Write-Host "`n📦 Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    $npmVersion = npm --version
    Write-Host "✅ Node.js: $nodeVersion, NPM: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found - Install from https://nodejs.org" -ForegroundColor Red
}

# Check dependencies
Write-Host "`n📚 Checking NPM dependencies..." -ForegroundColor Yellow
if (Test-Path "package.json") {
    $packageCheck = npm list --depth=0 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ All NPM packages installed" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Some NPM packages missing - run 'npm install'" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ package.json not found - wrong directory?" -ForegroundColor Red
}

# Check TypeScript
Write-Host "`n🔍 Checking TypeScript..." -ForegroundColor Yellow
$tsCheck = npm run check 2>&1
if ($tsCheck -match "Check failed") {
    Write-Host "⚠️  TypeScript errors found - run 'npm run check' for details" -ForegroundColor Yellow
} else {
    Write-Host "✅ TypeScript compilation clean" -ForegroundColor Green
}

# Check external services
Write-Host "`n🌐 Checking External Services..." -ForegroundColor Yellow

# Ollama
try {
    $ollama = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Ollama service running on port 11434" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Ollama not running - install from https://ollama.ai" -ForegroundColor Yellow
}

# Qdrant
try {
    $qdrant = Invoke-WebRequest -Uri "http://localhost:6333" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "✅ Qdrant service running on port 6333" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Qdrant not running - start with Docker" -ForegroundColor Yellow
}

# Redis
try {
    $redis = Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet
    if ($redis) {
        Write-Host "✅ Redis service running on port 6379" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Redis not running - start with Docker" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Redis not running - start with Docker" -ForegroundColor Yellow
}

# PostgreSQL
try {
    $postgres = Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet
    if ($postgres) {
        Write-Host "✅ PostgreSQL service running on port 5432" -ForegroundColor Green
    } else {
        Write-Host "⚠️  PostgreSQL not running - start with Docker" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  PostgreSQL not running - start with Docker" -ForegroundColor Yellow
}

Write-Host "`n🎯 Quick Setup Commands:" -ForegroundColor Cyan
Write-Host "   Start Docker Desktop (manual)" -ForegroundColor White
Write-Host "   npm install" -ForegroundColor White
Write-Host "   npm run check" -ForegroundColor White
Write-Host "   .\COMPLETE-SYSTEM-STARTUP.bat" -ForegroundColor White

Write-Host "`n📋 See COMPREHENSIVE_INSTALLATION_TODO.md for detailed setup guide" -ForegroundColor Green
