@echo off
REM ================================================================
REM AI SYNTHESIS SYSTEM - WINDOWS NATIVE STARTUP
REM Legal AI System with AutoSolve Integration
REM ================================================================

echo.
echo ========================================================
echo    AI SYNTHESIS LEGAL SYSTEM - NATIVE WINDOWS
echo    Version: 4.0.0 - AutoSolve Enhanced
echo ========================================================
echo.

REM Check for Administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Not running as Administrator. Some features may be limited.
    echo.
)

REM Set environment variables for production system
set REDIS_HOST=localhost
set REDIS_PORT=6379
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=gemma3-legal
set ENHANCED_RAG_URL=http://localhost:8094
set PRODUCTION_UPLOAD_URL=http://localhost:5173/api/production-upload
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set QDRANT_URL=http://localhost:6333
set POSTGRES_PORT=5432
set SVELTEKIT_PORT=5173
set NODE_ENV=development
set GPU_ACCELERATION=true
set WASM_ENABLED=true

echo [1/7] Checking prerequisites...
echo ================================

REM Check Node.js
where node >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js first.
    pause
    exit /b 1
)
echo [OK] Node.js found

REM Check npm
where npm >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] npm not found. Please install npm first.
    pause
    exit /b 1
)
echo [OK] npm found

REM Check PowerShell
where powershell >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] PowerShell not found.
    pause
    exit /b 1
)
echo [OK] PowerShell found

echo.
echo [2/7] Starting Redis Cache Service...
echo =====================================

REM Check if Redis is installed
where redis-server >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Redis not installed. Attempting installation...
    
    REM Check for Chocolatey
    where choco >nul 2>&1
    if %errorLevel% neq 0 (
        echo [INFO] Installing Chocolatey...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
    )
    
    echo [INFO] Installing Redis via Chocolatey...
    choco install redis-64 -y
)

REM Start Redis in background
echo [INFO] Starting Redis server...
start /B redis-server --port 6379 --maxmemory 512mb --maxmemory-policy allkeys-lru >nul 2>&1
timeout /t 2 /nobreak >nul

REM Verify Redis is running
redis-cli ping >nul 2>&1
if %errorLevel% eq 0 (
    echo [OK] Redis is running on port 6379
) else (
    echo [WARNING] Redis may not be running properly
)

echo.
echo [3/7] Starting Ollama AI Service...
echo ====================================

REM Check if Ollama is installed
where ollama >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Ollama not found. Please install from https://ollama.ai
    echo [INFO] Continuing without Ollama...
) else (
    REM Check if Ollama is already running
    tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
    if %errorLevel% neq 0 (
        echo [INFO] Starting Ollama service...
        start /B ollama serve >nul 2>&1
        timeout /t 3 /nobreak >nul
    ) else (
        echo [OK] Ollama already running
    )
    
    REM Check for legal models
    ollama list 2>nul | find "llama2" >nul
    if %errorLevel% neq 0 (
        echo [INFO] Pulling llama2 model (this may take a while)...
        ollama pull llama2
    )
    echo [OK] Ollama ready on port 11434
)

echo.
echo [4/7] Checking existing Legal AI services...
echo ============================================

REM Check Enhanced RAG
powershell -Command "try { Invoke-RestMethod -Uri 'http://localhost:8094/health' -TimeoutSec 2 | Out-Null; Write-Host '[OK] Enhanced RAG connected (port 8094)' -ForegroundColor Green } catch { Write-Host '[WARNING] Enhanced RAG not available' -ForegroundColor Yellow }"

REM Check GPU Orchestrator
powershell -Command "try { Invoke-RestMethod -Uri 'http://localhost:8095/health' -TimeoutSec 2 | Out-Null; Write-Host '[OK] GPU Orchestrator connected (port 8095)' -ForegroundColor Green } catch { Write-Host '[WARNING] GPU Orchestrator not available' -ForegroundColor Yellow }"

REM Check Context7 MCP
powershell -Command "try { Invoke-RestMethod -Uri 'http://localhost:4000/health' -TimeoutSec 2 | Out-Null; Write-Host '[OK] Context7 MCP connected (port 4000)' -ForegroundColor Green } catch { Write-Host '[WARNING] Context7 MCP not available' -ForegroundColor Yellow }"

echo.
echo [5/7] Running AutoSolve for AI Synthesis components...
echo ======================================================

cd /d "%~dp0"
echo [INFO] Checking TypeScript errors...
call npm run autosolve:ai-synthesis 2>nul
if %errorLevel% eq 0 (
    echo [OK] AutoSolve completed successfully
) else (
    echo [INFO] Running manual AutoSolve...
    node scripts/autosolve-ai-synthesis.mjs
)

echo.
echo [6/7] Starting SvelteKit development server...
echo ==============================================

echo [INFO] Installing dependencies if needed...
call npm install --silent

echo [INFO] Starting development server...
start cmd /k "npm run dev"
timeout /t 5 /nobreak >nul

echo.
echo [7/7] Starting monitoring dashboard...
echo ======================================

REM Create monitoring script if it doesn't exist
if not exist "scripts\orchestration\monitor-ai-synthesis.ps1" (
    echo [INFO] Creating monitoring dashboard...
    powershell -Command "& '%~dp0scripts\orchestration\start-ai-synthesis.ps1'"
)

echo.
echo ========================================================
echo    AI SYNTHESIS SYSTEM STARTED SUCCESSFULLY!
echo ========================================================
echo.
echo Available Endpoints:
echo --------------------
echo   Production Upload:  http://localhost:5173/api/production-upload
echo   Health Check:       http://localhost:5173/api/health
echo   Legal AI Chat:      http://localhost:5173/api/ai/chat
echo   YorHa Legal Data:   http://localhost:5173/api/yorha/legal-data
echo.
echo Services Status:
echo ----------------
echo   [PORT]  [SERVICE]
echo   5173    SvelteKit Dev Server + Production Upload
echo   5432    PostgreSQL 17 + pgvector
echo   6333    Qdrant Vector Database  
echo   6379    Redis Cache
echo   11434   Ollama AI (gemma3-legal)
echo   8094    Enhanced RAG (if available)
echo.
echo Quick Test Commands:
echo --------------------
echo   System Health:    curl http://localhost:5173/api/health
echo   Upload Test:      Visit http://localhost:5173/upload-test
echo   Integration Check: RUN-INTEGRATION-CHECK.bat
echo.
echo Press any key to open the monitoring dashboard...
pause >nul

REM Open monitoring in new PowerShell window
start powershell -NoExit -Command "& '%~dp0scripts\orchestration\monitor-ai-synthesis.ps1'"

echo.
echo System is running. Press Ctrl+C to stop all services.
echo.

REM Keep the window open
:loop
timeout /t 60 /nobreak >nul
goto loop
