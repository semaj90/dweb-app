# Enhanced RAG V2 - Node.js Quick Start (PowerShell)
# Uses Node.js only - no Go required

Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "   ENHANCED RAG V2 - NODE.JS QUICK START" -ForegroundColor Cyan
Write-Host "   No Go Required - Using Node.js Only" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

$projectPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
Set-Location $projectPath

# Add Node to PATH if needed
$env:PATH = "C:\Program Files\nodejs;$env:PATH"

# Step 1: Check Node.js
Write-Host "[STEP 1] Checking Node.js..." -ForegroundColor Yellow
try {
    $nodeVersion = & node --version 2>$null
    Write-Host "[OK] Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    if (Test-Path "C:\Program Files\nodejs\node.exe") {
        Write-Host "[OK] Node.js found at C:\Program Files\nodejs" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Node.js not found" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Install Frontend Dependencies
Write-Host "`n[STEP 2] Frontend Dependencies..." -ForegroundColor Yellow
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing packages..." -ForegroundColor Cyan
    Set-Location frontend
    & npm install 2>&1 | Out-Null
    Write-Host "[OK] Dependencies installed" -ForegroundColor Green
    Set-Location ..
} else {
    Write-Host "[OK] Dependencies already installed" -ForegroundColor Green
}

# Step 3: Start Services
Write-Host "`n[STEP 3] Starting Services..." -ForegroundColor Yellow

# Kill any process on port 8084
Get-Process | Where-Object {$_.Id -in (Get-NetTCPConnection -LocalPort 8084 -ErrorAction SilentlyContinue).OwningProcess} | Stop-Process -Force -ErrorAction SilentlyContinue

# Start API Server
Write-Host "Starting API server on port 8084..." -ForegroundColor Cyan
$apiProcess = Start-Process -PassThru -WindowStyle Hidden -FilePath "node" -ArgumentList "node-api-server.js" -WorkingDirectory $projectPath
Start-Sleep -Seconds 2

# Start Frontend
Write-Host "Starting frontend on port 5173..." -ForegroundColor Cyan
$frontendProcess = Start-Process -PassThru -WindowStyle Hidden -FilePath "powershell" -ArgumentList "-Command", "cd frontend; npm run dev" -WorkingDirectory $projectPath
Start-Sleep -Seconds 5

# Step 4: Status Check
Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "   SYSTEM STATUS" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

# Check services
$api8084 = Get-NetTCPConnection -LocalPort 8084 -ErrorAction SilentlyContinue
$frontend5173 = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue

if ($api8084) {
    Write-Host "[RUNNING] API Server:  http://localhost:8084/api/health" -ForegroundColor Green
} else {
    Write-Host "[STOPPED] API Server:  Not running" -ForegroundColor Red
}

if ($frontend5173) {
    Write-Host "[RUNNING] Frontend:    http://localhost:5173" -ForegroundColor Green
} else {
    Write-Host "[STOPPED] Frontend:    Not running" -ForegroundColor Red
}

Write-Host "`n===========================================" -ForegroundColor Cyan
Write-Host "   READY TO USE!" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Open http://localhost:5173 in your browser" -ForegroundColor White
Write-Host "2. API is available at http://localhost:8084" -ForegroundColor White
Write-Host ""
Write-Host "This is a temporary Node.js solution." -ForegroundColor Yellow
Write-Host "For full functionality, install Go from https://go.dev" -ForegroundColor Yellow
Write-Host ""
Write-Host "Process IDs:" -ForegroundColor Gray
if ($apiProcess) { Write-Host "  API Server PID: $($apiProcess.Id)" -ForegroundColor Gray }
if ($frontendProcess) { Write-Host "  Frontend PID: $($frontendProcess.Id)" -ForegroundColor Gray }
Write-Host ""
Write-Host "Press any key to stop services and exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Cleanup
Write-Host "Stopping services..." -ForegroundColor Yellow
if ($apiProcess -and !$apiProcess.HasExited) { Stop-Process -Id $apiProcess.Id -Force }
if ($frontendProcess -and !$frontendProcess.HasExited) { Stop-Process -Id $frontendProcess.Id -Force }
Write-Host "Services stopped." -ForegroundColor Green