@echo off
title Legal AI Assistant - Docker Setup

echo.
echo =========================================================
echo   Legal AI Assistant - Complete Docker Setup
echo =========================================================
echo.
echo This script will start your full AI stack using Docker Desktop.
echo Make sure Docker Desktop is running and WSL integration is enabled.
echo.

REM Check if Docker Desktop is running
echo [1/4] Checking Docker Desktop...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo   [X] ERROR: Docker Desktop is not running.
    echo   Please start Docker Desktop and enable WSL integration:
    echo   Settings ^> Resources ^> WSL Integration ^> Enable integration
    echo.
    pause
    exit /b 1
)
echo   [+] SUCCESS: Docker Desktop is running.
echo.

REM Check if WSL is available
echo [2/4] Checking WSL availability...
wsl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [X] ERROR: WSL is not available.
    echo   Please install WSL2 and restart your computer.
    echo.
    pause
    exit /b 1
)
echo   [+] SUCCESS: WSL is available.
echo.

REM Change to project directory
echo [3/4] Setting up project directory...
cd /d "%~dp0"
echo   [+] SUCCESS: Working in project directory.
echo.

REM Run the WSL startup script
echo [4/4] Starting services via WSL...
echo   This may take a few minutes to pull images and start services...
echo.

wsl bash -c "chmod +x start-wsl.sh && ./start-wsl.sh"

if %errorlevel% neq 0 (
    echo.
    echo   [X] ERROR: Failed to start services.
    echo   Check the output above for details.
    echo.
    pause
    exit /b 1
)

echo.
echo =========================================================
echo  ✅ Setup Complete!
echo =========================================================
echo.
echo Your Legal AI Assistant is now running.
echo Open your browser to: http://localhost:5173
echo.
echo To stop services: npm run docker:down
echo To view logs: npm run docker:logs
echo.
pause
