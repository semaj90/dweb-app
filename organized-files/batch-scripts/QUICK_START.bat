@echo off
REM Quick Start Script for Legal GPU Processor
REM Date: 2025-08-08

echo ============================================
echo Legal GPU Processor v2.0.0 - Quick Start
echo ============================================
echo.

REM Navigate to project directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo Step 1: Installing dependencies...
echo ----------------------------------------
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 2: Building the application...
echo ----------------------------------------
call npm run build
if %errorlevel% neq 0 (
    echo WARNING: Build failed, but continuing...
)

echo.
echo Step 3: Starting development server...
echo ----------------------------------------
echo.
echo The application will start at http://localhost:5173
echo Press Ctrl+C to stop the server
echo.
echo IMPORTANT: Make sure these services are running:
echo - PostgreSQL (port 5432)
echo - Redis (port 6379)
echo - Ollama (port 11434)
echo - Qdrant (port 6333) [optional]
echo.
echo ============================================
echo.

REM Start the development server
call npm run dev

pause