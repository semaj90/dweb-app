@echo off
title Legal AI System - Complete Startup
color 0A

echo.
echo ========================================
echo  LEGAL AI SYSTEM - COMPLETE STARTUP
echo ========================================
echo  Version: 2.0.0-legal-ai
echo  Architecture: Go GPU Server + SvelteKit 2
echo  AI Models: gemma3-legal via Ollama
echo ========================================
echo.

:: Create logs directory
if not exist "logs" mkdir logs

:: Kill any existing processes on our ports
echo [1/6] Cleaning up existing processes...
taskkill /f /im legal-ai-server.exe /t 2>nul
taskkill /f /im node.exe /t 2>nul
timeout /t 2 >nul

:: Start PostgreSQL (if not running)
echo [2/6] Checking PostgreSQL...
sc query postgresql-x64-17 | find "RUNNING" >nul
if errorlevel 1 (
    echo Starting PostgreSQL...
    sc start postgresql-x64-17
    timeout /t 5 >nul
) else (
    echo PostgreSQL is already running.
)

:: Start Redis (if not running)
echo [3/6] Checking Redis...
tasklist /FI "IMAGENAME eq redis-server.exe" 2>nul | find /I "redis-server.exe" >nul
if errorlevel 1 (
    echo Starting Redis...
    start /B redis-windows\redis-server.exe redis-windows\redis.windows.conf
    timeout /t 3 >nul
) else (
    echo Redis is already running.
)

:: Start Ollama (if not running)
echo [4/6] Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if errorlevel 1 (
    echo Starting Ollama...
    start /B ollama serve
    timeout /t 5 >nul
) else (
    echo Ollama is already running.
)

:: Build and start Go GPU Server
echo [5/6] Starting Go GPU Server (Port 8081)...
cd go-microservice
if not exist "legal-ai-server.exe" (
    echo Building Go server...
    set CGO_ENABLED=0
    go build -o legal-ai-server.exe .
)
set PORT=8081
set OLLAMA_URL=http://localhost:11434
set DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db
start "Legal AI Go Server" legal-ai-server.exe
cd ..
timeout /t 3 >nul

:: Start SvelteKit with PM2
echo [6/6] Starting SvelteKit and Workers with PM2...
call npm install -g pm2 2>nul
call pm2 delete all 2>nul
call pm2 start ecosystem.config.js --env production

echo.
echo ====================================================================
echo   ALL SERVICES STARTED WITH LOCAL MODELS
echo ====================================================================
echo.
echo   Using Models:
echo     - Legal Analysis: %LEGAL_MODEL%
echo     - Embeddings: %EMBEDDING_MODEL%
echo.
echo   Service URLs:
echo     - Frontend:    http://localhost:5173
echo     - Go Server:   http://localhost:8081
echo     - Ollama:      http://localhost:11434
echo     - Redis:       localhost:6379
echo     - PostgreSQL:  localhost:5432
echo.
echo   Upload a document at: http://localhost:5173/upload
echo.
pause
