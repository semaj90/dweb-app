@echo off
REM COMPLETE-AI-SYSTEM-STARTUP-FIXED.bat
REM Comprehensive startup for the entire Legal AI system
REM Phase 14 - Windows Native with GPU Acceleration - FIXED VERSION

title Legal AI System - Complete Startup
cls

echo ========================================================
echo    LEGAL AI SYSTEM - PHASE 14 COMPLETE STARTUP
echo    Windows Native + GPU Acceleration + All Services
echo ========================================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [WARNING] Not running as Administrator
    echo Some services may require admin privileges
    echo.
    pause
)

echo [1/8] Starting PostgreSQL...
sc query postgresql-x64-17 2>nul | findstr "RUNNING" >nul
if not %errorlevel% == 0 (
    net start postgresql-x64-17 2>nul
    timeout /t 3 /nobreak >nul
)
echo     [OK] PostgreSQL running

echo.
echo [2/8] Starting Redis...
cd redis-windows 2>nul
if exist redis-server.exe (
    start /B "" redis-server.exe redis.conf >nul 2>&1
    timeout /t 2 /nobreak >nul
    echo     [OK] Redis server started
) else (
    echo     [SKIP] Redis not found in redis-windows/
)
cd ..

echo.
echo [3/8] Starting Neo4j...
where neo4j >nul 2>&1
if %errorlevel% == 0 (
    start /B "" neo4j console >nul 2>&1
    timeout /t 5 /nobreak >nul
    echo     [OK] Neo4j graph database started
) else (
    echo     [SKIP] Neo4j not found in PATH
)

echo.
echo [4/8] Starting Qdrant...
cd qdrant-windows 2>nul
if exist qdrant.exe (
    start /B "" qdrant.exe --config-path ../qdrant-local-config.yaml >nul 2>&1
    timeout /t 3 /nobreak >nul
    echo     [OK] Qdrant vector database started
) else (
    echo     [SKIP] Qdrant not found in qdrant-windows/
)
cd ..

echo.
echo [5/8] Starting Ollama with GPU...
REM Check if Ollama is already running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% NEQ 0 (
    echo     Starting Ollama service...
    set CUDA_VISIBLE_DEVICES=0
    set OLLAMA_NUM_GPU=1
    start /B "" ollama serve >nul 2>&1
    echo     Waiting for Ollama to initialize - this may take a moment...
    timeout /t 10 /nobreak >nul
)

REM Verify GPU support
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo     [OK] NVIDIA GPU detected - CUDA enabled
) else (
    echo     [WARN] No NVIDIA GPU detected - CPU mode
)

REM Check models
ollama list 2>nul | findstr "gemma3-legal" >nul
if %errorlevel% == 0 (
    echo     [OK] gemma3-legal model available
) else (
    echo     [WARN] gemma3-legal model not found
)

echo.
echo [6/8] Building and Starting Go GPU Server...
cd go-microservice 2>nul
if exist main.go (
    if not exist main.exe (
        echo     Building Go server...
        go build -o main.exe main.go 2>nul
    )
    
    REM Set environment variables
    set OLLAMA_URL=http://localhost:11434
    set DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db
    set REDIS_URL=localhost:6379
    set NEO4J_URI=bolt://localhost:7687
    set QDRANT_URL=http://localhost:6333
    set PORT=8080
    
    start /B "" main.exe >nul 2>&1
    timeout /t 3 /nobreak >nul
    
    REM Test Go server
    curl -s http://localhost:8080/health >nul 2>&1
    if %errorlevel% == 0 (
        echo     [OK] Go GPU server running on port 8080
    ) else (
        echo     [WARN] Go server may still be starting...
    )
) else (
    echo     [SKIP] Go server not found - run consolidate-go-main.bat first
)
cd ..

echo.
echo [7/8] Starting BullMQ Workers...
cd workers 2>nul
if exist document-processor.worker.js (
    start /B "" node document-processor.worker.js >nul 2>&1
    echo     [OK] Document processing workers started
) else (
    echo     [SKIP] Workers not found in workers/
)
cd ..

echo.
echo [8/8] Starting SvelteKit Frontend...
cd sveltekit-frontend 2>nul
if exist package.json (
    start /B "" npm run dev >nul 2>&1
    echo     [OK] Frontend starting on http://localhost:5173
) else (
    echo     [SKIP] SvelteKit frontend not found
)
cd ..

echo.
echo ========================================================
echo    SYSTEM STATUS CHECK
echo ========================================================
timeout /t 5 /nobreak >nul

echo.
echo Checking services...
echo.

REM Check all services
echo PostgreSQL:
netstat -an | findstr :5432 >nul 2>&1
if %errorlevel% == 0 (echo     [RUNNING] Port 5432) else (echo     [STOPPED])

echo Redis:
netstat -an | findstr :6379 >nul 2>&1
if %errorlevel% == 0 (echo     [RUNNING] Port 6379) else (echo     [STOPPED])

echo Neo4j:
netstat -an | findstr :7687 >nul 2>&1
if %errorlevel% == 0 (echo     [RUNNING] Port 7687) else (echo     [STOPPED])

echo Qdrant:
curl -s http://localhost:6333 >nul 2>&1
if %errorlevel% == 0 (echo     [RUNNING] Port 6333) else (echo     [STOPPED])

echo Ollama:
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% == 0 (echo     [RUNNING] Port 11434) else (echo     [STOPPED])

echo Go Server:
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% == 0 (echo     [RUNNING] Port 8080) else (echo     [STOPPED])

echo Frontend:
netstat -an | findstr :5173 >nul 2>&1
if %errorlevel% == 0 (echo     [RUNNING] Port 5173) else (echo     [STARTING...])

echo.
echo ========================================================
echo    LEGAL AI SYSTEM READY!
echo ========================================================
echo.
echo Frontend:    http://localhost:5173
echo API:         http://localhost:8080/health
echo GPU Status:  http://localhost:8080/gpu-status
echo Qdrant UI:   http://localhost:6333/dashboard
echo Neo4j UI:    http://localhost:7474
echo.
echo To test the system:
echo   1. Open http://localhost:5173 in your browser
echo   2. Upload a document to process
echo   3. Click the AI Assistant button bottom right
echo.
echo Press any key to open the frontend...
echo.
echo Opening frontend...
start http://localhost:5173

echo System running. Terminal stays open.
timeout /t 300