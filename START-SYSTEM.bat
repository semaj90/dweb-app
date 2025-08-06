@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo.
echo ================================================================
echo 🚀 Legal AI System Startup - Complete Stack
echo ================================================================
echo.

REM Set environment variables
set CURRENT_DIR=%~dp0
set DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db
set OLLAMA_URL=http://localhost:11434
set QDRANT_URL=http://localhost:6333
set REDIS_URL=redis://localhost:6379
set NODE_ENV=development

echo 📋 Starting services in sequence...
echo.

REM ==== STEP 1: Start PostgreSQL (if not running) ====
echo [1/7] 🐘 Checking PostgreSQL...
"C:\Program Files\PostgreSQL\17\bin\pg_isready.exe" -h localhost -p 5432
if %ERRORLEVEL% NEQ 0 (
    echo ❌ PostgreSQL not running - please start PostgreSQL service first
    echo    Run: net start postgresql-x64-17
    pause
    goto :error
) else (
    echo ✅ PostgreSQL is running
)

REM ==== STEP 2: Start Redis ====
echo [2/7] 🔴 Starting Redis server...
start /B "" "%CURRENT_DIR%redis-windows\redis-server.exe"
timeout /t 3 /nobreak >nul
echo ✅ Redis server started

REM ==== STEP 3: Start Qdrant ====
echo [3/7] 🔍 Starting Qdrant vector database...
start /B "" "%CURRENT_DIR%qdrant-windows\qdrant.exe"
timeout /t 3 /nobreak >nul
echo ✅ Qdrant started

REM ==== STEP 4: Start Ollama ====
echo [4/7] 🧠 Starting Ollama LLM service...
start /B "" ollama serve
timeout /t 5 /nobreak >nul
echo ✅ Ollama service started

REM ==== STEP 5: Start Go Microservice ====
echo [5/7] ⚡ Starting Go Legal AI server...
cd "%CURRENT_DIR%go-microservice"
start /B "" legal-ai-server.exe
cd "%CURRENT_DIR%"
timeout /t 3 /nobreak >nul
echo ✅ Go server started on port 8080

REM ==== STEP 6: Start Node.js services with PM2 ====
echo [6/7] 📦 Starting Node.js services...
REM Kill existing PM2 processes
pm2 kill >nul 2>&1

REM Start SvelteKit development server
echo    📄 Starting SvelteKit frontend...
cd "%CURRENT_DIR%sveltekit-frontend"
start /B "" npm run dev
cd "%CURRENT_DIR%"

REM Start workers and background services
echo    ⚙️  Starting document processor worker...
start /B "" node workers/document-processor.worker.js

timeout /t 5 /nobreak >nul
echo ✅ Node.js services started

REM ==== STEP 7: Verify all services ====
echo [7/7] ✅ Verifying services...
echo.

REM Check Redis
echo    🔴 Redis:
"%CURRENT_DIR%redis-windows\redis-cli.exe" ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo         ✅ Connected on port 6379) else (echo         ❌ Not responding)

REM Check Qdrant
echo    🔍 Qdrant:
curl -s http://localhost:6333/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo         ✅ Connected on port 6333) else (echo         ❌ Not responding)

REM Check Ollama
echo    🧠 Ollama:
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo         ✅ Connected on port 11434) else (echo         ❌ Not responding)

REM Check Go server
echo    ⚡ Go Server:
curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo         ✅ Connected on port 8080) else (echo         ❌ Not responding)

REM Check SvelteKit
echo    📄 SvelteKit:
curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% EQU 0 (echo         ✅ Connected on port 5173) else (echo         ❌ Not responding)

echo.
echo ================================================================
echo 🎉 Legal AI System Startup Complete!
echo ================================================================
echo.
echo 📊 Service Status:
echo     PostgreSQL:  localhost:5432
echo     Redis:       localhost:6379  
echo     Qdrant:      localhost:6333
echo     Ollama:      localhost:11434
echo     Go Server:   localhost:8080
echo     SvelteKit:   localhost:5173
echo.
echo 🌐 Web Interface: http://localhost:5173
echo 🔧 API Health:   http://localhost:8080/health
echo 📊 Metrics:      http://localhost:8080/metrics
echo.
echo Press any key to open the web interface...
pause >nul
start http://localhost:5173
goto :end

:error
echo.
echo ❌ System startup failed - please check the logs above
pause
exit /b 1

:end
echo.
echo 🔄 System is running. Close this window to keep services running.
echo    To stop all services, run: STOP-SYSTEM.bat
pause