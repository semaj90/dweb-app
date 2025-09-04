# Enhanced RAG V2 - Complete Setup & Build Script
# Builds all services and installs all dependencies

Write-Host "`n🚀 ENHANCED RAG V2 - COMPLETE SETUP" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Building services and installing dependencies..." -ForegroundColor Green
Write-Host ""

$projectPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
Set-Location $projectPath

# Function to check if service is running
function Test-ServicePort {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

# 1. VERIFY INSTALLATIONS
Write-Host "📋 Step 1: Verifying Installations" -ForegroundColor Yellow
Write-Host "===================================" -ForegroundColor Yellow

# Node.js Check
$nodeOk = $false
try {
    $nodeVersion = & node --version 2>$null
    Write-Host "  ✅ Node.js: $nodeVersion" -ForegroundColor Green
    $nodeOk = $true
} catch {
    Write-Host "  ❌ Node.js not found - please install from nodejs.org" -ForegroundColor Red
}

# Go Check
$goOk = $false
try {
    $goVersion = & go version 2>$null
    Write-Host "  ✅ Go: $($goVersion -replace 'go version ','')" -ForegroundColor Green
    $goOk = $true
} catch {
    $altPath = "C:\Program Files\Go\bin\go.exe"
    if (Test-Path $altPath) {
        Write-Host "  ✅ Go found at: $altPath" -ForegroundColor Green
        $env:PATH = "C:\Program Files\Go\bin;$env:PATH"
        $goOk = $true
    } else {
        Write-Host "  ❌ Go not found - please install from go.dev" -ForegroundColor Red
    }
}

# PostgreSQL Check
if (Test-ServicePort 5432) {
    Write-Host "  ✅ PostgreSQL: Running on port 5432" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  PostgreSQL not running - attempting to start..." -ForegroundColor Yellow
    try {
        Start-Service postgresql-x64-14 -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        if (Test-ServicePort 5432) {
            Write-Host "  ✅ PostgreSQL started successfully" -ForegroundColor Green
        }
    } catch {
        Write-Host "  ⚠️  Could not start PostgreSQL - manual start may be needed" -ForegroundColor Yellow
    }
}

# 2. BUILD GO SERVICES
if ($goOk) {
    Write-Host "`n📦 Step 2: Building Go Services" -ForegroundColor Yellow
    Write-Host "================================" -ForegroundColor Yellow
    
    Set-Location "$projectPath\go-microservice"
    
    # Set environment for pure Go build (no CGO)
    $env:CGO_ENABLED = "0"
    $env:GOOS = "windows"
    $env:GOARCH = "amd64"
    
    # Create bin directory if it doesn't exist
    if (-not (Test-Path "bin")) {
        New-Item -ItemType Directory -Path "bin" | Out-Null
        Write-Host "  Created bin directory" -ForegroundColor Gray
    }
    
    # Build Enhanced RAG V2
    Write-Host "  Building Enhanced RAG V2..." -ForegroundColor Cyan
    try {
        & go build -ldflags="-s -w" -o bin\enhanced-rag-v2.exe .\cmd\enhanced-rag-v2\main.go 2>&1 | Out-Null
        if (Test-Path "bin\enhanced-rag-v2.exe") {
            $size = (Get-Item "bin\enhanced-rag-v2.exe").Length / 1MB
            Write-Host "  ✅ Enhanced RAG V2 built successfully ($('{0:N2}' -f $size) MB)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  Enhanced RAG V2 build completed but exe not found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ⚠️  Enhanced RAG V2 build failed: $_" -ForegroundColor Yellow
    }
    
    # Build Simply Enhanced RAG
    Write-Host "  Building Simply Enhanced RAG..." -ForegroundColor Cyan
    try {
        & go build -ldflags="-s -w" -o bin\simply-enhanced-rag.exe .\cmd\simply-enhanced-rag\main.go 2>&1 | Out-Null
        if (Test-Path "bin\simply-enhanced-rag.exe") {
            $size = (Get-Item "bin\simply-enhanced-rag.exe").Length / 1MB
            Write-Host "  ✅ Simply Enhanced RAG built successfully ($('{0:N2}' -f $size) MB)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  Simply Enhanced RAG build completed but exe not found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ⚠️  Simply Enhanced RAG build failed: $_" -ForegroundColor Yellow
    }
    
    # Build main service (port 8084)
    Write-Host "  Building Main Legal AI Service..." -ForegroundColor Cyan
    try {
        & go build -ldflags="-s -w" -o bin\legal-ai-service.exe main.go 2>&1 | Out-Null
        if (Test-Path "bin\legal-ai-service.exe") {
            $size = (Get-Item "bin\legal-ai-service.exe").Length / 1MB
            Write-Host "  ✅ Legal AI Service built successfully ($('{0:N2}' -f $size) MB)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  Legal AI Service build completed but exe not found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ⚠️  Legal AI Service build failed: $_" -ForegroundColor Yellow
    }
    
    Set-Location $projectPath
}

# 3. INSTALL FRONTEND DEPENDENCIES
if ($nodeOk) {
    Write-Host "`n📚 Step 3: Frontend Dependencies" -ForegroundColor Yellow
    Write-Host "=================================" -ForegroundColor Yellow
    
    if (-not (Test-Path "frontend\node_modules")) {
        Write-Host "  Installing frontend dependencies (this may take a few minutes)..." -ForegroundColor Cyan
        Set-Location "frontend"
        try {
            & npm install --silent 2>&1 | Out-Null
            Write-Host "  ✅ Frontend dependencies installed" -ForegroundColor Green
        } catch {
            Write-Host "  ⚠️  Some dependencies may have failed to install" -ForegroundColor Yellow
        }
        Set-Location $projectPath
    } else {
        Write-Host "  ✅ Frontend dependencies already installed" -ForegroundColor Green
        Write-Host "  Running npm update to ensure latest versions..." -ForegroundColor Cyan
        Set-Location "frontend"
        & npm update --silent 2>&1 | Out-Null
        Set-Location $projectPath
    }
}

# 4. START SERVICES
Write-Host "`n🌐 Step 4: Starting Services" -ForegroundColor Yellow
Write-Host "=============================" -ForegroundColor Yellow

# Start Legal AI Service (8084)
if (Test-Path "go-microservice\bin\legal-ai-service.exe") {
    if (-not (Test-ServicePort 8084)) {
        Write-Host "  Starting Legal AI Service on port 8084..." -ForegroundColor Cyan
        Start-Process -WindowStyle Hidden -FilePath "$projectPath\go-microservice\bin\legal-ai-service.exe"
        Start-Sleep -Seconds 2
        if (Test-ServicePort 8084) {
            Write-Host "  ✅ Legal AI Service started on port 8084" -ForegroundColor Green
        }
    } else {
        Write-Host "  ✅ Legal AI Service already running on port 8084" -ForegroundColor Green
    }
}

# Start Enhanced RAG V2 (8097)
if (Test-Path "go-microservice\bin\enhanced-rag-v2.exe") {
    if (-not (Test-ServicePort 8097)) {
        Write-Host "  Starting Enhanced RAG V2 on port 8097..." -ForegroundColor Cyan
        Start-Process -WindowStyle Hidden -FilePath "$projectPath\go-microservice\bin\enhanced-rag-v2.exe"
        Start-Sleep -Seconds 2
        if (Test-ServicePort 8097) {
            Write-Host "  ✅ Enhanced RAG V2 started on port 8097" -ForegroundColor Green
        }
    } else {
        Write-Host "  ✅ Enhanced RAG V2 already running on port 8097" -ForegroundColor Green
    }
}

# Start Simply Enhanced RAG (8096)
if (Test-Path "go-microservice\bin\simply-enhanced-rag.exe") {
    if (-not (Test-ServicePort 8096)) {
        Write-Host "  Starting Simply Enhanced RAG on port 8096..." -ForegroundColor Cyan
        Start-Process -WindowStyle Hidden -FilePath "$projectPath\go-microservice\bin\simply-enhanced-rag.exe"
        Start-Sleep -Seconds 2
        if (Test-ServicePort 8096) {
            Write-Host "  ✅ Simply Enhanced RAG started on port 8096" -ForegroundColor Green
        }
    } else {
        Write-Host "  ✅ Simply Enhanced RAG already running on port 8096" -ForegroundColor Green
    }
}

# Check Ollama
if (Test-ServicePort 11434) {
    Write-Host "  ✅ Ollama already running on port 11434" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Ollama not detected on port 11434" -ForegroundColor Yellow
    Write-Host "     Start Ollama manually with: ollama serve" -ForegroundColor Gray
}

# Start Frontend Dev Server
if ($nodeOk) {
    if (-not (Test-ServicePort 5173)) {
        Write-Host "  Starting Frontend Development Server..." -ForegroundColor Cyan
        Set-Location "frontend"
        Start-Process -WindowStyle Hidden powershell -ArgumentList "-Command", "npm run dev"
        Set-Location $projectPath
        Start-Sleep -Seconds 5
        if (Test-ServicePort 5173) {
            Write-Host "  ✅ Frontend started on port 5173" -ForegroundColor Green
        }
    } else {
        Write-Host "  ✅ Frontend already running on port 5173" -ForegroundColor Green
    }
}

# 5. SYSTEM STATUS
Write-Host "`n📊 Step 5: System Status Check" -ForegroundColor Yellow
Write-Host "===============================" -ForegroundColor Yellow

$services = @{
    5173 = "Frontend (SvelteKit)"
    5432 = "PostgreSQL Database"
    8084 = "Legal AI Service"
    8096 = "Simply Enhanced RAG"
    8097 = "Enhanced RAG V2"
    11434 = "Ollama LLM Service"
}

$runningCount = 0
$totalCount = $services.Count

foreach ($port in $services.Keys | Sort-Object) {
    if (Test-ServicePort $port) {
        Write-Host "  ✅ Port ${port}: $($services[$port])" -ForegroundColor Green
        $runningCount++
    } else {
        Write-Host "  ❌ Port ${port}: $($services[$port])" -ForegroundColor Red
    }
}

$percentage = [math]::Round(($runningCount / $totalCount) * 100)

Write-Host "`n📈 Service Availability: $runningCount/$totalCount ($percentage%)" -ForegroundColor $(if ($percentage -ge 80) { "Green" } elseif ($percentage -ge 50) { "Yellow" } else { "Red" })

# 6. ACCESS POINTS
Write-Host "`n🎯 Access Points" -ForegroundColor Cyan
Write-Host "=================" -ForegroundColor Cyan
Write-Host "  Frontend:          http://localhost:5173" -ForegroundColor White
Write-Host "  Legal AI API:      http://localhost:8084/api/health" -ForegroundColor White
Write-Host "  Enhanced RAG:      http://localhost:8097/health" -ForegroundColor White
Write-Host "  Simply RAG:        http://localhost:8096/health" -ForegroundColor White
Write-Host "  Ollama:           http://localhost:11434" -ForegroundColor White

# 7. TROUBLESHOOTING
if ($runningCount -lt $totalCount) {
    Write-Host "`n⚠️  Troubleshooting Tips" -ForegroundColor Yellow
    Write-Host "========================" -ForegroundColor Yellow
    
    if (-not (Test-ServicePort 11434)) {
        Write-Host "  • Start Ollama: ollama serve" -ForegroundColor Gray
    }
    if (-not (Test-ServicePort 5432)) {
        Write-Host "  • Start PostgreSQL from Services (services.msc)" -ForegroundColor Gray
    }
    if (-not (Test-ServicePort 5173)) {
        Write-Host "  • Check frontend: cd frontend && npm run dev" -ForegroundColor Gray
    }
    if (-not (Test-ServicePort 8084)) {
        Write-Host "  • Check Go build: cd go-microservice && go build main.go" -ForegroundColor Gray
    }
}

Write-Host "`n✨ Setup Complete!" -ForegroundColor Green
Write-Host "Press any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")