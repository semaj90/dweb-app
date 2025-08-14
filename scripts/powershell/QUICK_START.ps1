# Quick Start Script for Legal GPU Processor
# Date: 2025-08-08
# Version: 2.0.0

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Legal GPU Processor v2.0.0 - Quick Start" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app"

# Step 1: Install dependencies
Write-Host "Step 1: Installing dependencies..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
# Step 2: Build the application
Write-Host "Step 2: Building the application..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Build failed, but continuing..." -ForegroundColor Yellow
}

Write-Host ""
# Step 3: Check services
Write-Host "Step 3: Checking required services..." -ForegroundColor Yellow
Write-Host "----------------------------------------"

# Check PostgreSQL
$pgRunning = Test-NetConnection -ComputerName localhost -Port 5432 -InformationLevel Quiet
if ($pgRunning) {
    Write-Host "✓ PostgreSQL is running (port 5432)" -ForegroundColor Green
} else {
    Write-Host "✗ PostgreSQL is not running (port 5432)" -ForegroundColor Red
}

# Check Redis
$redisRunning = Test-NetConnection -ComputerName localhost -Port 6379 -InformationLevel Quiet
if ($redisRunning) {
    Write-Host "✓ Redis is running (port 6379)" -ForegroundColor Green
} else {
    Write-Host "✗ Redis is not running (port 6379)" -ForegroundColor Red
}

# Check Ollama
$ollamaRunning = Test-NetConnection -ComputerName localhost -Port 11434 -InformationLevel Quiet
if ($ollamaRunning) {
    Write-Host "✓ Ollama is running (port 11434)" -ForegroundColor Green
} else {
    Write-Host "✗ Ollama is not running (port 11434)" -ForegroundColor Red
}

# Check Qdrant (optional)
$qdrantRunning = Test-NetConnection -ComputerName localhost -Port 6333 -InformationLevel Quiet
if ($qdrantRunning) {
    Write-Host "✓ Qdrant is running (port 6333)" -ForegroundColor Green
} else {
    Write-Host "○ Qdrant is not running (port 6333) - Optional" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Step 4: Starting development server..." -ForegroundColor Yellow
Write-Host "----------------------------------------"
Write-Host ""
Write-Host "The application will start at http://localhost:5173" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Start the development server
npm run dev