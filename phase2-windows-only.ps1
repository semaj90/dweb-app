# Phase 2: Production Setup - Windows Only (Skip WSL)
# Fixed for Windows 10 + Docker Desktop

Write-Host "🚀 Phase 2: Production Setup (Windows-Only)" -ForegroundColor Green

# Skip WSL check - use Windows Docker directly
Write-Host "Using Windows Docker Desktop directly..."

if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker Desktop not found"
    exit 1
}

# Test Docker connectivity
docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker not responding"
    exit 1
}

Write-Host "✅ Docker Desktop working"

# Create directories
$dirs = @("deployment\production", "logs\production")
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

# Windows production launcher
@"
@echo off
title Production Mode - Windows
docker-compose -f docker-compose.yml up -d
echo ✅ Production services started
echo Access at: http://localhost:5173
pause
"@ | Out-File -FilePath "START-PRODUCTION-WINDOWS.bat" -Encoding ASCII

Write-Host "✅ Phase 2 complete - Windows production ready"
Write-Host "Run: START-PRODUCTION-WINDOWS.bat"
