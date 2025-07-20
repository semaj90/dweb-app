@echo off
REM Quick Start Script for Deeds Legal AI Assistant
REM This is a simple batch file for starting the development environment

echo ========================================
echo Deeds Legal AI Assistant - Quick Start
echo ========================================
echo.

REM Check if we're in the right directory
if not exist package.json (
    echo ERROR: Not in project directory!
    echo Please run from: C:\Users\james\Desktop\web-app
    pause
    exit /b 1
)

echo Starting Docker services...
docker-compose up -d postgres qdrant redis

if %errorlevel% neq 0 (
    echo ERROR: Failed to start Docker services
    echo Make sure Docker Desktop is running
    pause
    exit /b 1
)

echo.
echo Waiting for services to start...
timeout /t 5 /nobreak > nul

echo.
echo Installing dependencies...
call npm install

echo.
echo Starting development server...
echo.
echo ==========================================
echo Application URL: http://localhost:5173
echo Qdrant UI: http://localhost:6333/dashboard
echo ==========================================
echo.
echo Press Ctrl+C to stop the server
echo.

cd sveltekit-frontend
call npm install
call npm run dev
