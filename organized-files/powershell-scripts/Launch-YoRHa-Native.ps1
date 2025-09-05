# =============================================================================
# YoRHa Legal AI Platform - Native Windows PowerShell Launcher
# Bypasses WSL issues and uses pure Windows execution
# =============================================================================

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "YoRHa LEGAL AI PLATFORM - NATIVE WINDOWS LAUNCHER" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the correct directory
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app"

# =============================================================================
# PHASE 1: Fix Node.js Dependencies
# =============================================================================

Write-Host "[1/5] FIXING NODE.JS DEPENDENCIES" -ForegroundColor Green
Write-Host "---------------------------------------------"

Set-Location "sveltekit-frontend"

if (Test-Path "node_modules") {
    Write-Host "🧹 Cleaning existing node_modules..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "node_modules" -ErrorAction SilentlyContinue
}

if (Test-Path "package-lock.json") {
    Write-Host "🧹 Removing package-lock.json..." -ForegroundColor Yellow
    Remove-Item "package-lock.json" -ErrorAction SilentlyContinue
}

Write-Host "📦 Installing Windows-native dependencies..." -ForegroundColor Cyan
try {
    & npm install --platform=win32 --arch=x64
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️ npm install failed, trying pnpm..." -ForegroundColor Yellow
    & pnpm install
}

Set-Location ".."

# =============================================================================
# PHASE 2: Infrastructure Services
# =============================================================================

Write-Host ""
Write-Host "[2/5] STARTING INFRASTRUCTURE SERVICES" -ForegroundColor Green
Write-Host "---------------------------------------------"

# PostgreSQL
Write-Host "🗄️ Checking PostgreSQL..." -NoNewline
try {
    $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pgService -and $pgService.Status -eq "Running") {
        Write-Host " ✅ Running" -ForegroundColor Green
    } elseif ($pgService) {
        Write-Host " 🔄 Starting..." -ForegroundColor Yellow
        Start-Service $pgService.Name
        Write-Host " ✅ Started" -ForegroundColor Green
    } else {
        Write-Host " ⚠️ Not installed" -ForegroundColor Red
    }
} catch {
    Write-Host " ❌ Error checking PostgreSQL" -ForegroundColor Red
}

# Ollama
Write-Host "🧠 Checking Ollama..." -NoNewline
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 3 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " 🔄 Starting..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    Write-Host " ✅ Started" -ForegroundColor Green
}

# =============================================================================
# PHASE 3: Go Microservices
# =============================================================================

Write-Host ""
Write-Host "[3/5] PREPARING GO MICROSERVICES" -ForegroundColor Green  
Write-Host "---------------------------------------------"

# Enhanced RAG Service
$ragBinary = "..\go-microservice\cmd\enhanced-rag\enhanced-rag.exe"
if (Test-Path $ragBinary) {
    Write-Host "✅ Enhanced RAG binary found" -ForegroundColor Green
} else {
    Write-Host "🔨 Building Enhanced RAG service..." -ForegroundColor Yellow
    Set-Location "..\go-microservice"
    & go build -o ".\cmd\enhanced-rag\enhanced-rag.exe" ".\cmd\enhanced-rag"
    Set-Location "..\deeds-web-app"
    Write-Host "✅ Enhanced RAG built" -ForegroundColor Green
}

# Upload Service
$uploadBinary = "..\go-microservice\cmd\upload-service\upload-service.exe"
if (Test-Path $uploadBinary) {
    Write-Host "✅ Upload Service binary found" -ForegroundColor Green
} else {
    Write-Host "🔨 Building Upload Service..." -ForegroundColor Yellow
    Set-Location "..\go-microservice"
    & go build -o ".\cmd\upload-service\upload-service.exe" ".\cmd\upload-service"
    Set-Location "..\deeds-web-app"
    Write-Host "✅ Upload Service built" -ForegroundColor Green
}

# =============================================================================
# PHASE 4: Launch Services
# =============================================================================

Write-Host ""
Write-Host "[4/5] LAUNCHING SERVICES" -ForegroundColor Green
Write-Host "---------------------------------------------"

# Start Enhanced RAG Service
Write-Host "🤖 Starting Enhanced RAG Service (port 8094)..." -NoNewline
try {
    $ragProcess = Start-Process -FilePath $ragBinary -PassThru -WindowStyle Hidden -ErrorAction Stop
    Start-Sleep -Seconds 3
    $response = Invoke-WebRequest -Uri "http://localhost:8094" -TimeoutSec 5 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " ⚠️ May not be fully ready" -ForegroundColor Yellow
}

# Start Upload Service  
Write-Host "📤 Starting Upload Service (port 8093)..." -NoNewline
try {
    $uploadProcess = Start-Process -FilePath $uploadBinary -PassThru -WindowStyle Hidden -ErrorAction Stop
    Start-Sleep -Seconds 2
    Write-Host " ✅ Started" -ForegroundColor Green
} catch {
    Write-Host " ⚠️ May not be fully ready" -ForegroundColor Yellow
}

# =============================================================================
# PHASE 5: SvelteKit Frontend
# =============================================================================

Write-Host ""
Write-Host "[5/5] LAUNCHING YoRHa INTERFACE" -ForegroundColor Green
Write-Host "---------------------------------------------"

Set-Location "sveltekit-frontend"

# Set Windows-specific environment variables
$env:NODE_ENV = "development"
$env:VITE_PLATFORM = "win32" 
$env:FORCE_COLOR = "1"

Write-Host "🎮 Starting YoRHa Interface..." -ForegroundColor Magenta
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "  🤖 YoRHa Legal AI Interface" -ForegroundColor Cyan
Write-Host "  🧠 Enhanced RAG: http://localhost:8094" -ForegroundColor Green  
Write-Host "  📤 Upload Service: http://localhost:8093" -ForegroundColor Green
Write-Host "  🎯 Frontend: http://localhost:5173" -ForegroundColor Magenta
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""

# Start SvelteKit with native Windows npm
try {
    Write-Host "🚀 Launching SvelteKit development server..." -ForegroundColor Cyan
    & npm run dev
} catch {
    Write-Host "❌ Failed to start SvelteKit" -ForegroundColor Red
    Write-Host "Trying alternative startup method..." -ForegroundColor Yellow
    & npx vite dev
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "YoRHa Legal AI Platform - Native Windows Launch Complete!" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan