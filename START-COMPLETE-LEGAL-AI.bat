@echo off
title Legal AI System - Complete Startup
color 0A

echo ============================================
echo    LEGAL AI SYSTEM - COMPLETE STARTUP
echo ============================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running as Administrator
) else (
    echo ❌ Please run as Administrator for full functionality
    echo    Some services may not start properly
)
echo.

REM Set environment variables
echo 📝 Setting environment variables...
set OLLAMA_URL=http://localhost:11434
set DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db
set GO_SERVER_URL=http://localhost:8080
set REDIS_URL=redis://localhost:6379
set NODE_ENV=development

echo    - Ollama: %OLLAMA_URL%
echo    - Database: PostgreSQL + pgvector
echo    - Go Server: %GO_SERVER_URL%
echo    - Redis: %REDIS_URL%
echo.

REM Function to check if service is running
:CHECK_SERVICE
netstat -an | find ":%~1" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Service running on port %~1
) else (
    echo ❌ No service found on port %~1
)
goto :EOF

echo 🔍 Checking existing services...
call :CHECK_SERVICE 5432
call :CHECK_SERVICE 6379
call :CHECK_SERVICE 11434
echo.

echo 🚀 Starting services in order...
echo.

REM Step 1: Start PostgreSQL (if not running)
echo [1/6] PostgreSQL Database...
netstat -an | find ":5432" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL already running
) else (
    echo 🔄 Starting PostgreSQL...
    sc query postgresql-x64-17 | find "RUNNING" >nul 2>&1
    if %errorlevel% neq 0 (
        sc start postgresql-x64-17 >nul 2>&1
        timeout /t 3 /nobreak >nul
    )
)
echo.

REM Step 2: Start Redis (if not running)
echo [2/6] Redis Cache...
netstat -an | find ":6379" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Redis already running
) else (
    echo 🔄 Starting Redis...
    if exist "redis-windows\redis-server.exe" (
        start /B redis-windows\redis-server.exe redis-windows\redis.windows.conf
        timeout /t 2 /nobreak >nul
    ) else (
        echo ❌ Redis not found - install Redis for Windows
    )
)
echo.

REM Step 3: Start Ollama (if not running)
echo [3/6] Ollama LLM Server...
netstat -an | find ":11434" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama already running
) else (
    echo 🔄 Starting Ollama...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    
    REM Check if gemma3-legal model exists
    ollama list | find "gemma3-legal" >nul 2>&1
    if %errorlevel% neq 0 (
        echo 📥 Pulling gemma3-legal model...
        ollama create gemma3-legal -f Modelfile-gemma3-legal
    )
)
echo.

REM Step 4: Start Go Legal AI Server
echo [4/6] Go Legal AI Server...
cd /d "%~dp0go-microservice"
netstat -an | find ":8080" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Go server already running on port 8080
) else (
    echo 🔄 Building and starting Go server...
    if not exist legal-ai-server.exe (
        go mod tidy
        go build -o legal-ai-server.exe legal-ai-server.go
    )
    
    if exist legal-ai-server.exe (
        start /B legal-ai-server.exe
        timeout /t 3 /nobreak >nul
        echo ✅ Go Legal AI Server started
    ) else (
        echo ❌ Failed to build Go server
    )
)
cd /d "%~dp0"
echo.

REM Step 5: Start BullMQ Worker
echo [5/6] BullMQ Legal AI Worker...
cd /d "%~dp0sveltekit-frontend"
echo 🔄 Starting Legal AI Worker...
start /B node start-legal-ai-worker.mjs
cd /d "%~dp0"
echo ✅ Legal AI Worker started
echo.

REM Step 6: Start SvelteKit Frontend
echo [6/6] SvelteKit Frontend...
cd /d "%~dp0sveltekit-frontend"
netstat -an | find ":5173" >nul 2>&1
if %errorlevel` == 0 (
    echo ✅ SvelteKit already running on port 5173
) else (
    echo 🔄 Starting SvelteKit development server...
    start cmd /k "npm run dev"
    timeout /t 2 /nobreak >nul
)
cd /d "%~dp0"
echo.

echo ============================================
echo        🎉 LEGAL AI SYSTEM READY! 🎉
echo ============================================
echo.
echo 🌐 Frontend:      http://localhost:5173
echo 🔧 Go Server:     http://localhost:8080/health
echo 🧠 Ollama:        http://localhost:11434
echo 💾 PostgreSQL:    localhost:5432
echo 📦 Redis:         localhost:6379
echo.
echo 📊 System Status:
call :CHECK_SERVICE 5173
call :CHECK_SERVICE 8080
call :CHECK_SERVICE 11434
call :CHECK_SERVICE 5432
call :CHECK_SERVICE 6379
echo.

echo 🔗 Quick Links:
echo    - Frontend: http://localhost:5173
echo    - Go Server Health: http://localhost:8080/health
echo    - API Test: http://localhost:5173/api/legal-ai/process-document
echo.

echo ⚡ Testing Pipeline:
echo 🔄 Testing Go server connection...
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://localhost:8080/health' -TimeoutSec 5; Write-Host '✅ Go server: ' $response.status } catch { Write-Host '❌ Go server: Connection failed' }"

echo.
echo 📋 Next Steps:
echo    1. Visit http://localhost:5173 to access the frontend
echo    2. Upload a document to test the complete pipeline
echo    3. Check logs in the opened command windows
echo    4. Use Ctrl+C in each window to stop services
echo.

echo Press any key to open the frontend...
pause >nul
start http://localhost:5173

echo.
echo System is running! Keep this window open to monitor status.
echo Press any key to exit (services will continue running)...
pause >nul