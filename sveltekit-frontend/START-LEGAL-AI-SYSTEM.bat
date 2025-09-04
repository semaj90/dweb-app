@echo off
REM Legal AI System - Complete Startup with Health Check
REM Production-ready with error handling and service verification

cls
echo =====================================
echo   Legal AI System Startup
echo   With GPU Acceleration and Caching
echo =====================================
echo.

REM Check for admin rights (some services might need it)
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Not running as administrator
    echo Some services may require admin privileges
    echo.
)

REM Create necessary directories
echo [SETUP] Creating directories...
if not exist "uploads" mkdir uploads
if not exist "minio-data" mkdir minio-data
if not exist "logs" mkdir logs

REM Start PostgreSQL
echo.
echo [1/9] Starting PostgreSQL...
pg_ctl status -D "%PGDATA%" >nul 2>&1
if %errorlevel% neq 0 (
    pg_ctl start -D "%PGDATA%" >nul 2>&1
    timeout /t 3 /nobreak >nul
    echo       PostgreSQL started
) else (
    echo       PostgreSQL already running
)

REM Check database
psql -U postgres -lqt | findstr "legal_ai_db" >nul 2>&1
if %errorlevel% neq 0 (
    echo       Creating legal_ai_db...
    psql -U postgres -c "CREATE DATABASE legal_ai_db;" >nul 2>&1
    psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;" >nul 2>&1
    echo       Database created with pgvector
)

REM Start MinIO
echo [2/9] Starting MinIO...
tasklist /FI "IMAGENAME eq minio.exe" 2>NUL | find /I /N "minio.exe">NUL
if %errorlevel% neq 0 (
    if exist "minio.exe" (
        start /B minio.exe server ./minio-data --console-address :9001 >logs\minio.log 2>&1
        echo       MinIO started (Console: http://localhost:9001)
    ) else (
        echo       MinIO not found - using filesystem fallback
    )
) else (
    echo       MinIO already running
)

REM Start Ollama
echo [3/9] Starting Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %errorlevel% neq 0 (
    where ollama >nul 2>&1
    if %errorlevel% equ 0 (
        start /B ollama serve >logs\ollama.log 2>&1
        timeout /t 3 /nobreak >nul
        echo       Ollama started
        
        REM Check for gemma3:legal model
        ollama list | findstr "gemma" >nul 2>&1
        if %errorlevel% neq 0 (
            echo       Pulling gemma model...
            ollama pull gemma:2b >nul 2>&1
        )
    ) else (
        echo       Ollama not installed - AI features disabled
    )
) else (
    echo       Ollama already running
)

REM Start Qdrant
echo [4/9] Starting Qdrant...
tasklist /FI "IMAGENAME eq qdrant.exe" 2>NUL | find /I /N "qdrant.exe">NUL
if %errorlevel% neq 0 (
    if exist "qdrant.exe" (
        start /B qdrant.exe >logs\qdrant.log 2>&1
        echo       Qdrant started
    ) else if exist "qdrant-windows\qdrant.exe" (
        start /B qdrant-windows\qdrant.exe >logs\qdrant.log 2>&1
        echo       Qdrant started
    ) else (
        echo       Qdrant not found - vector search disabled
    )
) else (
    echo       Qdrant already running
)

REM Start Redis
echo [5/9] Starting Redis...
tasklist /FI "IMAGENAME eq redis-server.exe" 2>NUL | find /I /N "redis-server.exe">NUL
if %errorlevel% neq 0 (
    if exist "redis-server.exe" (
        start /B redis-server.exe >logs\redis.log 2>&1
        echo       Redis started
    ) else if exist "redis-windows\redis-server.exe" (
        start /B redis-windows\redis-server.exe >logs\redis.log 2>&1
        echo       Redis started
    ) else (
        echo       Redis not found - caching degraded
    )
) else (
    echo       Redis already running
)

REM Start Go microservices
echo [6/9] Starting Go Microservices...

REM Enhanced RAG
if exist "go-microservice\enhanced-rag-som-system.exe" (
    tasklist /FI "IMAGENAME eq enhanced-rag-som-system.exe" 2>NUL | find /I /N "enhanced-rag">NUL
    if %errorlevel% neq 0 (
        start /B go-microservice\enhanced-rag-som-system.exe >logs\enhanced-rag.log 2>&1
        echo       Enhanced RAG started
    ) else (
        echo       Enhanced RAG already running
    )
) else if exist "enhanced-rag-som-system.exe" (
    start /B enhanced-rag-som-system.exe >logs\enhanced-rag.log 2>&1
    echo       Enhanced RAG started
) else (
    echo       Enhanced RAG not found
)

REM GPU Orchestrator
if exist "go-microservice\gpu-orchestrator.exe" (
    tasklist /FI "IMAGENAME eq gpu-orchestrator.exe" 2>NUL | find /I /N "gpu-orchestrator">NUL
    if %errorlevel% neq 0 (
        start /B go-microservice\gpu-orchestrator.exe >logs\gpu-orchestrator.log 2>&1
        echo       GPU Orchestrator started
    ) else (
        echo       GPU Orchestrator already running
    )
) else if exist "gpu-orchestrator.exe" (
    start /B gpu-orchestrator.exe >logs\gpu-orchestrator.log 2>&1
    echo       GPU Orchestrator started
) else (
    echo       GPU Orchestrator not found - CPU fallback active
)

REM Install Node dependencies if needed
echo [7/9] Checking Node.js dependencies...
if not exist "node_modules" (
    echo       Installing dependencies...
    call npm install --silent
    if %errorlevel% neq 0 (
        echo       [ERROR] npm install failed
        echo       Please run: npm install
    )
) else (
    echo       Dependencies already installed
)

REM Build the project if needed
echo [8/9] Building project...
if not exist ".svelte-kit\output" (
    echo       Building SvelteKit...
    call npm run build --silent
)

REM Wait for services to stabilize
echo [9/9] Waiting for services to initialize...
timeout /t 5 /nobreak >nul

REM Health check
echo.
echo =====================================
echo   Service Health Check
echo =====================================
echo.

REM Check each service
echo Checking services...

REM PostgreSQL
psql -U postgres -d legal_ai_db -c "SELECT 1;" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]   PostgreSQL    - Connected to legal_ai_db
) else (
    echo [FAIL] PostgreSQL    - Not responding
)

REM MinIO
curl -s http://localhost:9000/minio/health/live >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]   MinIO         - Storage ready
) else (
    echo [WARN] MinIO         - Not available (using filesystem)
)

REM Ollama
curl -s http://localhost:11434 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]   Ollama        - AI service ready
) else (
    echo [WARN] Ollama        - Not available (AI disabled)
)

REM Qdrant
curl -s http://localhost:6333 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]   Qdrant        - Vector search ready
) else (
    echo [WARN] Qdrant        - Not available (vector search disabled)
)

REM Enhanced RAG
curl -s http://localhost:8094/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]   Enhanced RAG  - Document processing ready
) else (
    echo [WARN] Enhanced RAG  - Not available
)

REM GPU Orchestrator
curl -s http://localhost:8231/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]   GPU Service   - Acceleration ready
) else (
    echo [WARN] GPU Service   - Not available (CPU mode)
)

REM Redis
redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK]   Redis         - Cache ready
) else (
    echo [WARN] Redis         - Not available (degraded caching)
)

echo.
echo =====================================
echo   Starting Frontend Application
echo =====================================
echo.

REM Start the development server
echo Starting SvelteKit development server...
echo.
echo Application will be available at:
echo   http://localhost:5173
echo.
echo Service Dashboard:
echo   PostgreSQL: 5432
echo   MinIO Console: http://localhost:9001 (admin/minioadmin)
echo   Ollama: http://localhost:11434
echo   Qdrant: http://localhost:6333
echo   Enhanced RAG: http://localhost:8094
echo   GPU Service: http://localhost:8231
echo.
echo Press Ctrl+C to stop all services
echo.

REM Start the dev server
call npm run dev

REM Cleanup on exit
echo.
echo Shutting down services...
taskkill /F /IM minio.exe >nul 2>&1
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM qdrant.exe >nul 2>&1
taskkill /F /IM redis-server.exe >nul 2>&1
taskkill /F /IM enhanced-rag-som-system.exe >nul 2>&1
taskkill /F /IM gpu-orchestrator.exe >nul 2>&1
pg_ctl stop -D "%PGDATA%" >nul 2>&1

echo Services stopped.
pause
