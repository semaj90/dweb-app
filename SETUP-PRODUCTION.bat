@echo off
REM Legal AI System - Production Startup Script
REM One-click production deployment with safety checks

echo ╔═══════════════════════════════════════════════════════╗
echo ║      Legal AI System - Production Deployment           ║
echo ║                  Version 1.0.0                         ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

REM Check administrator privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Administrator privileges required.
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Check PowerShell 7
pwsh -v >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PowerShell 7 required.
    echo Installing PowerShell 7...
    winget install Microsoft.PowerShell
    if %errorlevel% neq 0 (
        echo Failed to install PowerShell 7.
        echo Download from: https://github.com/PowerShell/PowerShell/releases
        pause
        exit /b 1
    )
)

echo [1/5] Running pre-flight validation...
pwsh -ExecutionPolicy Bypass -Command "& { if (Test-Path '.\validate.ps1') { .\validate.ps1 -Minimal } else { Write-Host 'Validation script not found' -ForegroundColor Red; exit 1 } }"
if %errorlevel% neq 0 (
    echo.
    echo Pre-flight validation failed. Running safe installation...
    echo.
    
    REM Use safe install script if available
    if exist ".\install-safe.ps1" (
        pwsh -ExecutionPolicy Bypass -File ".\install-safe.ps1"
    ) else if exist ".\install.ps1" (
        echo WARNING: Using original install script (no safety checks)
        pwsh -ExecutionPolicy Bypass -File ".\install.ps1"
    ) else (
        echo ERROR: No installation script found.
        pause
        exit /b 1
    )
    
    if %errorlevel% neq 0 (
        echo Installation failed. Check logs for details.
        pause
        exit /b 1
    )
)

echo.
echo [2/5] Ensuring Docker Desktop is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to initialize (30 seconds)...
    timeout /t 30 /nobreak >nul
    
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Docker Desktop failed to start.
        echo Please start Docker Desktop manually and retry.
        pause
        exit /b 1
    )
)

echo.
echo [3/5] Starting infrastructure services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Docker services.
    echo Running diagnostics...
    docker-compose logs --tail=50
    pause
    exit /b 1
)

echo.
echo [4/5] Waiting for services to stabilize...
timeout /t 15 /nobreak >nul

echo.
echo [5/5] Running comprehensive system validation...
pwsh -ExecutionPolicy Bypass -File ".\VALIDATE-SYSTEM.ps1" -Production
if %errorlevel% neq 0 (
    echo.
    echo WARNING: System validation reported issues.
    echo Check the validation report for details.
    echo.
    choice /C YN /M "Continue anyway?"
    if errorlevel 2 exit /b 1
)

echo.
echo ════════════════════════════════════════════════════════
echo ✅ Production deployment complete!
echo ════════════════════════════════════════════════════════
echo.
echo 📍 Application URL: http://localhost:5173
echo 📊 Health Status:  http://localhost:5173/api/health
echo 🔍 Qdrant UI:      http://localhost:6333/dashboard
echo 📈 Redis Stack:    http://localhost:8001
echo.
echo Starting application server...
echo Press Ctrl+C to stop
echo ════════════════════════════════════════════════════════
echo.

cd sveltekit-frontend
npm run dev

REM Cleanup on exit
echo.
echo Shutting down production services...
cd ..
docker-compose stop
echo Production services stopped.
