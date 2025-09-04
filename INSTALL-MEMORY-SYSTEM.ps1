#!/usr/bin/env pwsh
# Complete Memory System Installation Script
# Installs all components and starts the memory monitoring system

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   Legal AI Memory System - Complete Install" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$scriptPath = $PSScriptRoot
$projectRoot = Split-Path $scriptPath -Parent

# 1. Check Node.js installation
Write-Host "📦 Checking Node.js installation..." -ForegroundColor Yellow
$nodeVersion = node --version 2>$null
if ($nodeVersion) {
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Node.js not found. Please install Node.js first." -ForegroundColor Red
    Write-Host "   Download from: https://nodejs.org/" -ForegroundColor Gray
    exit 1
}

# 2. Install required npm packages
Write-Host "`n📦 Installing required npm packages..." -ForegroundColor Yellow
Set-Location $projectRoot

$packagesToInstall = @(
    "animejs",
    "express",
    "cors",
    "ws"
)

foreach ($package in $packagesToInstall) {
    Write-Host "  Installing $package..." -ForegroundColor Gray
    npm install $package --save 2>$null
}

Write-Host "✅ NPM packages installed" -ForegroundColor Green

# 3. Create directory structure
Write-Host "`n📁 Creating directory structure..." -ForegroundColor Yellow

$directories = @(
    "logs\memory",
    "logs\memory\crash-prevention",
    "cache\l1",
    "cache\l2"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $projectRoot $dir
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  ✅ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  ⏭️ Exists: $dir" -ForegroundColor Gray
    }
}

# 4. Create package.json updates
Write-Host "`n📝 Updating package.json scripts..." -ForegroundColor Yellow

$packageJsonPath = Join-Path $projectRoot "package.json"
if (Test-Path $packageJsonPath) {
    $packageJson = Get-Content $packageJsonPath -Raw | ConvertFrom-Json
    
    # Add new scripts if they don't exist
    if (-not $packageJson.scripts."memory:api") {
        $packageJson.scripts | Add-Member -Name "memory:api" -Value "node scripts/memory-monitor-api.js" -MemberType NoteProperty -Force
    }
    if (-not $packageJson.scripts."memory:monitor") {
        $packageJson.scripts | Add-Member -Name "memory:monitor" -Value "powershell -ExecutionPolicy Bypass -File scripts/memory-optimizer.ps1 -Monitor" -MemberType NoteProperty -Force
    }
    if (-not $packageJson.scripts."memory:optimize") {
        $packageJson.scripts | Add-Member -Name "memory:optimize" -Value "powershell -ExecutionPolicy Bypass -File scripts/memory-optimizer.ps1 -Optimize" -MemberType NoteProperty -Force
    }
    if (-not $packageJson.scripts."memory:dashboard") {
        $packageJson.scripts | Add-Member -Name "memory:dashboard" -Value "start admin-dashboard.html && node scripts/memory-monitor-api.js" -MemberType NoteProperty -Force
    }
    
    $packageJson | ConvertTo-Json -Depth 10 | Set-Content $packageJsonPath
    Write-Host "✅ Package.json updated with memory scripts" -ForegroundColor Green
} else {
    Write-Host "⚠️ Package.json not found" -ForegroundColor Yellow
}

# 5. Create VS Code task
Write-Host "`n📝 Creating VS Code task..." -ForegroundColor Yellow

$vscodePath = Join-Path $projectRoot ".vscode"
if (-not (Test-Path $vscodePath)) {
    New-Item -ItemType Directory -Path $vscodePath -Force | Out-Null
}

$tasksPath = Join-Path $vscodePath "tasks.json"
if (Test-Path $tasksPath) {
    $tasks = Get-Content $tasksPath -Raw | ConvertFrom-Json
    
    # Check if memory task already exists
    $memoryTaskExists = $tasks.tasks | Where-Object { $_.label -eq "🧠 Memory Monitor with Dashboard" }
    
    if (-not $memoryTaskExists) {
        $newTask = @{
            label = "🧠 Memory Monitor with Dashboard"
            type = "shell"
            command = "npm"
            args = @("run", "memory:dashboard")
            group = "build"
            presentation = @{
                echo = $true
                reveal = "always"
                focus = $false
                panel = "new"
            }
            problemMatcher = @()
            detail = "Start memory monitoring API and open dashboard"
        }
        
        $tasks.tasks += $newTask
        $tasks | ConvertTo-Json -Depth 10 | Set-Content $tasksPath
        Write-Host "✅ VS Code task added" -ForegroundColor Green
    } else {
        Write-Host "  ⏭️ VS Code task already exists" -ForegroundColor Gray
    }
}

# 6. Create startup script
Write-Host "`n📝 Creating startup script..." -ForegroundColor Yellow

$startupScript = @'
@echo off
REM Memory System Startup Script
cls
echo =====================================
echo    Starting Memory Monitor System
echo =====================================
echo.

REM Start Memory API
echo Starting Memory API...
start /B node scripts\memory-monitor-api.js

REM Wait for API to start
timeout /t 2 /nobreak >nul

REM Open Dashboard
echo Opening Dashboard...
start admin-dashboard.html

echo.
echo Memory system is running!
echo.
echo API: http://localhost:3456
echo Dashboard: admin-dashboard.html
echo.
pause
'@

$startupScript | Set-Content (Join-Path $projectRoot "START-MEMORY-SYSTEM.bat")
Write-Host "✅ Created START-MEMORY-SYSTEM.bat" -ForegroundColor Green

# 7. Test Memory API
Write-Host "`n🧪 Testing Memory API..." -ForegroundColor Yellow

# Start the API in background
$apiProcess = Start-Process node -ArgumentList "scripts\memory-monitor-api.js" -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 3

try {
    $response = Invoke-RestMethod -Uri "http://localhost:3456/health" -Method Get -TimeoutSec 5
    if ($response.status -eq "running") {
        Write-Host "✅ Memory API is running successfully" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Memory API test failed (this is normal on first run)" -ForegroundColor Yellow
} finally {
    # Stop the test process
    if ($apiProcess -and !$apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
}

# 8. Create desktop shortcuts (optional)
Write-Host "`n📌 Creating shortcuts..." -ForegroundColor Yellow

$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Legal AI Memory Monitor.url"

$shortcutContent = @"
[InternetShortcut]
URL=file:///$($projectRoot.Replace('\', '/'))/admin-dashboard.html
IconIndex=0
IconFile=$env:SystemRoot\System32\SHELL32.dll
"@

$shortcutContent | Set-Content $shortcutPath
Write-Host "✅ Desktop shortcut created" -ForegroundColor Green

# 9. Summary
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "   ✅ Installation Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Installed Components:" -ForegroundColor Yellow
Write-Host "  • Memory Monitor API (Port 3456)" -ForegroundColor White
Write-Host "  • Admin Dashboard (admin-dashboard.html)" -ForegroundColor White
Write-Host "  • Memory Optimizer Script" -ForegroundColor White
Write-Host "  • Crash Prevention System" -ForegroundColor White
Write-Host "  • VS Code Integration" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Quick Start Commands:" -ForegroundColor Yellow
Write-Host "  npm run memory:dashboard    # Start everything" -ForegroundColor White
Write-Host "  npm run memory:api          # API only" -ForegroundColor White
Write-Host "  npm run memory:optimize     # Run optimization" -ForegroundColor White
Write-Host "  .\START-MEMORY-SYSTEM.bat   # One-click start" -ForegroundColor White
Write-Host ""
Write-Host "📁 Log Locations:" -ForegroundColor Yellow
Write-Host "  Memory Logs: logs\memory\" -ForegroundColor Gray
Write-Host "  Crash Logs:  logs\memory\crash-prevention\" -ForegroundColor Gray
Write-Host ""

# 10. Ask to start now
$startNow = Read-Host "Would you like to start the Memory System now? (Y/N)"
if ($startNow -eq 'Y' -or $startNow -eq 'y') {
    Write-Host "`n🚀 Starting Memory System..." -ForegroundColor Cyan
    
    # Start API
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "node scripts\memory-monitor-api.js" -WindowStyle Normal
    
    Start-Sleep -Seconds 2
    
    # Open Dashboard
    Start-Process (Join-Path $projectRoot "admin-dashboard.html")
    
    Write-Host "✅ Memory System is now running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Dashboard opened in browser" -ForegroundColor Cyan
    Write-Host "API running at: http://localhost:3456" -ForegroundColor Cyan
} else {
    Write-Host "`n📌 To start later, run:" -ForegroundColor Yellow
    Write-Host "  .\START-MEMORY-SYSTEM.bat" -ForegroundColor White
}

Write-Host "`n✨ Installation complete! Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")