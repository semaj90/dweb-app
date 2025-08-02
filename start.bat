@echo off
REM Legal AI System - Quick Start Script
REM One-click startup for the entire system

echo ╔═══════════════════════════════════════════════════════╗
echo ║          Legal AI System - Quick Start                 ║
echo ║                  Version 1.0.0                         ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

REM Check if PowerShell 7 is available
pwsh -v >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PowerShell 7 is required but not found.
    echo Please install from: https://github.com/PowerShell/PowerShell/releases
    pause
    exit /b 1
)

REM Run validation first
echo [1/4] Running system validation...
pwsh -ExecutionPolicy Bypass -File ".\validate.ps1" -Minimal
if %errorlevel% neq 0 (
    echo.
    echo System validation failed. Running installer...
    pwsh -ExecutionPolicy Bypass -File ".\install.ps1"
    if %errorlevel% neq 0 (
        echo Installation failed. Please check the logs.
        pause
        exit /b 1
    )
)

REM Start Docker services
echo.
echo [2/4] Starting Docker services...
docker-compose up -d
if %errorlevel% neq 0 (
    echo Failed to start Docker services.
    echo Make sure Docker Desktop is running.
    pause
    exit /b 1
)

REM Wait for services to be ready
echo.
echo [3/4] Waiting for services to initialize...
timeout /t 10 /nobreak >nul

REM Start the development server
echo.
echo [4/4] Starting development server...
echo.
echo ════════════════════════════════════════════════════════
echo System starting up...
echo ════════════════════════════════════════════════════════
echo.
echo Legal AI System will be available at:
echo.
echo    http://localhost:5173
echo.
echo Press Ctrl+C to stop the server
echo ════════════════════════════════════════════════════════
echo.

cd sveltekit-frontend
npm run dev

REM Cleanup on exit
echo.
echo Shutting down services...
cd ..
docker-compose stop
