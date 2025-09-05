# Quick Start Script for Legal AI Indexing System
param(
    [string]$Path = "C:\Users\james\Desktop\deeds-web\deeds-web-app",
    [switch]$Monitor
)

Write-Host "🚀 Starting Legal AI Indexing System" -ForegroundColor Cyan

# Set working directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "📁 Working in: $ScriptDir" -ForegroundColor Blue

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
    Write-Host "📁 Created logs directory" -ForegroundColor Green
}

try {
    # Start services with PM2
    Write-Host "🚀 Starting PM2 services..." -ForegroundColor Yellow
    pm2 start pm2.config.js
    
    Write-Host "⏳ Waiting for services to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    
    # Show status
    Write-Host "📊 Service Status:" -ForegroundColor Cyan
    pm2 status
    
    Write-Host ""
    Write-Host "🌐 Access Points:" -ForegroundColor Cyan
    Write-Host "  • Monitor Dashboard: http://localhost:8084" -ForegroundColor Green
    Write-Host "  • Go Indexer API:    http://localhost:8081" -ForegroundColor Green
    Write-Host "  • GPU Clustering:    http://localhost:8085" -ForegroundColor Green
    Write-Host ""
    
    if ($Monitor) {
        Write-Host "👀 Running in monitor mode - press Ctrl+C to stop" -ForegroundColor Blue
        try {
            while ($true) {
                Start-Sleep -Seconds 30
                Write-Host "$(Get-Date): System running..." -ForegroundColor Gray
            }
        } catch {
            Write-Host "Stopping services..." -ForegroundColor Yellow
            pm2 stop all
        }
    }
    
    Write-Host "✅ System startup complete!" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Startup failed: $($_.Exception.Message)" -ForegroundColor Red
    
    # Cleanup on failure
    try {
        pm2 delete all 2>$null
    } catch {}
    
    exit 1
}