@echo off
REM ============================================================================
REM COMPLETE LEGAL AI PLATFORM - PRODUCTION STARTUP WITH GPU CACHE
REM ============================================================================
REM YoRHa + NES.css + N64 UI + GPU Cache + All Services
REM Native Windows - No Docker Required
REM ============================================================================

setlocal enabledelayedexpansion
color 0A
title Legal AI Platform - Complete Production System

echo.
echo ============================================================
echo          LEGAL AI PLATFORM - PRODUCTION STARTUP
echo          YoRHa + NES.css + GPU Cache Integration
echo ============================================================
echo.

REM Set environment variables
set POSTGRES_PORT=5432
set REDIS_PORT=6379
set OLLAMA_PORT=11434
set ENHANCED_RAG_PORT=8094
set UPLOAD_SERVICE_PORT=8093
set VECTOR_SERVICE_PORT=8095
set GPU_ORCHESTRATOR_PORT=8231
set SVELTEKIT_PORT=5173
set NEO4J_PORT=7474
set QDRANT_PORT=6333
set MINIO_PORT=9000

REM Database configuration
set DB_USER=postgres
set DB_NAME=legal_ai_db
set DATABASE_URL=postgresql://%DB_USER%@localhost:%POSTGRES_PORT%/%DB_NAME%

REM GPU Configuration for RTX 3060 Ti
set GPU_LAYERS=35
set CONTEXT_SIZE=4096
set BATCH_SIZE=512
set VRAM_LIMIT=8192

echo [PHASE 1/10] Checking System Requirements...
echo ============================================

REM Check for Node.js
where /q node
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found. Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
) else (
    echo [OK] Node.js detected: 
    node --version
)

REM Check for PostgreSQL
echo.
echo [PHASE 2/10] Starting PostgreSQL with pgvector...
psql -U %DB_USER% -p %POSTGRES_PORT% -c "SELECT version();" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting PostgreSQL service...
    net start postgresql-x64-17 >nul 2>&1
    timeout /t 3 /nobreak >nul
)

REM Check pgvector extension
psql -U %DB_USER% -p %POSTGRES_PORT% -d %DB_NAME% -c "SELECT * FROM pg_extension WHERE extname = 'vector';" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Installing pgvector extension...
    psql -U %DB_USER% -p %POSTGRES_PORT% -d %DB_NAME% -c "CREATE EXTENSION IF NOT EXISTS vector;"
    psql -U %DB_USER% -p %POSTGRES_PORT% -d %DB_NAME% -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
    psql -U %DB_USER% -p %POSTGRES_PORT% -d %DB_NAME% -c "CREATE EXTENSION IF NOT EXISTS btree_gin;"
)
echo [OK] PostgreSQL running with pgvector on port %POSTGRES_PORT%

REM Check for Redis
echo.
echo [PHASE 3/10] Starting Redis Cache Service...
redis-cli -p %REDIS_PORT% ping >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting Redis server...
    start /B redis-server --port %REDIS_PORT% --maxmemory 2gb --maxmemory-policy allkeys-lru --save "" --appendonly no
    timeout /t 2 /nobreak >nul
)
echo [OK] Redis cache running on port %REDIS_PORT%

REM Check for Ollama
echo.
echo [PHASE 4/10] Starting Ollama AI Service...
curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting Ollama service...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
)

REM Pull gemma3-legal model if not exists
ollama list | findstr "gemma3-legal" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Pulling gemma3-legal model...
    ollama pull gemma2:9b
    
    REM Create custom legal model
    (
        echo FROM gemma2:9b
        echo SYSTEM You are a legal AI assistant specialized in case law, contracts, and legal analysis.
        echo PARAMETER temperature 0.1
        echo PARAMETER top_k 40
        echo PARAMETER top_p 0.9
        echo PARAMETER num_ctx %CONTEXT_SIZE%
        echo PARAMETER num_gpu %GPU_LAYERS%
    ) > Modelfile-legal
    
    ollama create gemma3-legal -f Modelfile-legal
    del Modelfile-legal
)
echo [OK] Ollama AI service running on port %OLLAMA_PORT%

REM Start MinIO if not running
echo.
echo [PHASE 5/10] Starting MinIO Object Storage...
curl -s http://localhost:%MINIO_PORT%/minio/health/live >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting MinIO server...
    if not exist "minio-data" mkdir minio-data
    start /B minio server minio-data --console-address ":9001"
    timeout /t 2 /nobreak >nul
)
echo [OK] MinIO running on port %MINIO_PORT%

REM Start Qdrant if available
echo.
echo [PHASE 6/10] Checking Qdrant Vector Database...
curl -s http://localhost:%QDRANT_PORT%/readiness >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Qdrant not running. Vector search may be limited.
    echo          To install: docker run -p 6333:6333 qdrant/qdrant
) else (
    echo [OK] Qdrant vector database running on port %QDRANT_PORT%
)

REM Navigate to frontend directory
echo.
echo [PHASE 7/10] Installing Frontend Dependencies...
cd /d "%~dp0sveltekit-frontend"

REM Install dependencies if needed
if not exist "node_modules" (
    echo [INFO] Installing npm packages...
    npm install --legacy-peer-deps
)

REM Check for NES.css
if not exist "node_modules\nes.css" (
    echo [INFO] Installing NES.css...
    npm install nes.css --save
)

REM Run database migrations
echo.
echo [PHASE 8/10] Running Database Migrations...
npm run db:generate 2>nul
npm run db:migrate 2>nul

REM Build the frontend if needed
echo.
echo [PHASE 9/10] Building Frontend Assets...
if not exist ".svelte-kit" (
    echo [INFO] Building SvelteKit application...
    npm run build
)

REM Kill any existing processes on our ports
echo.
echo [PHASE 10/10] Starting All Services...
echo ============================================

REM Kill existing processes to avoid port conflicts
taskkill /F /FI "WINDOWTITLE eq Enhanced RAG Service*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Upload Service*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Vector Service*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq GPU Orchestrator*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq SvelteKit Frontend*" >nul 2>&1

timeout /t 2 /nobreak >nul

REM Start Go microservices with unique window titles
echo [INFO] Starting Enhanced RAG Service...
start "Enhanced RAG Service - Port %ENHANCED_RAG_PORT%" /B cmd /c "cd go-microservice && set PORT=%ENHANCED_RAG_PORT% && go run cmd/enhanced-rag/main.go 2>&1"

echo [INFO] Starting Upload Service...
start "Upload Service - Port %UPLOAD_SERVICE_PORT%" /B cmd /c "cd go-microservice && set PORT=%UPLOAD_SERVICE_PORT% && go run cmd/upload-service/main.go 2>&1"

echo [INFO] Starting Vector Service...
start "Vector Service - Port %VECTOR_SERVICE_PORT%" /B cmd /c "cd go-microservice && set PORT=%VECTOR_SERVICE_PORT% && go run cmd/vector-service/main.go 2>&1"

echo [INFO] Starting GPU Orchestrator...
start "GPU Orchestrator - Port %GPU_ORCHESTRATOR_PORT%" /B cmd /c "cd go-microservice && set GPU_ORCHESTRATOR_PORT=%GPU_ORCHESTRATOR_PORT% && go run gpu-orchestrator.go 2>&1"

timeout /t 3 /nobreak >nul

REM Start SvelteKit frontend with all environment variables
echo [INFO] Starting SvelteKit Frontend...
start "SvelteKit Frontend - Port %SVELTEKIT_PORT%" cmd /c "npm run dev -- --host 0.0.0.0 --port %SVELTEKIT_PORT%"

timeout /t 5 /nobreak >nul

REM Verify all services
echo.
echo ============================================================
echo           SERVICE HEALTH CHECK
echo ============================================================
echo.

set ALL_OK=1

REM Check PostgreSQL
psql -U %DB_USER% -p %POSTGRES_PORT% -d %DB_NAME% -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] PostgreSQL:       postgresql://localhost:%POSTGRES_PORT%/%DB_NAME%
) else (
    echo [✗] PostgreSQL:       FAILED
    set ALL_OK=0
)

REM Check Redis
redis-cli -p %REDIS_PORT% ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Redis Cache:      redis://localhost:%REDIS_PORT%
) else (
    echo [✗] Redis Cache:      FAILED
    set ALL_OK=0
)

REM Check Ollama
curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Ollama AI:        http://localhost:%OLLAMA_PORT%
) else (
    echo [✗] Ollama AI:        FAILED
    set ALL_OK=0
)

REM Check MinIO
curl -s http://localhost:%MINIO_PORT%/minio/health/live >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] MinIO Storage:    http://localhost:%MINIO_PORT%
) else (
    echo [✗] MinIO Storage:    FAILED
    set ALL_OK=0
)

REM Check Enhanced RAG
timeout /t 2 /nobreak >nul
curl -s http://localhost:%ENHANCED_RAG_PORT%/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Enhanced RAG:     http://localhost:%ENHANCED_RAG_PORT%
) else (
    echo [⚠] Enhanced RAG:     Starting up...
)

REM Check Upload Service
curl -s http://localhost:%UPLOAD_SERVICE_PORT%/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Upload Service:   http://localhost:%UPLOAD_SERVICE_PORT%
) else (
    echo [⚠] Upload Service:   Starting up...
)

REM Check Vector Service
curl -s http://localhost:%VECTOR_SERVICE_PORT%/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Vector Service:   http://localhost:%VECTOR_SERVICE_PORT%
) else (
    echo [⚠] Vector Service:   Starting up...
)

REM Check GPU Orchestrator
curl -s http://localhost:%GPU_ORCHESTRATOR_PORT%/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] GPU Orchestrator: http://localhost:%GPU_ORCHESTRATOR_PORT%
) else (
    echo [⚠] GPU Orchestrator: Starting up...
)

REM Check SvelteKit
curl -s http://localhost:%SVELTEKIT_PORT% >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Frontend:         http://localhost:%SVELTEKIT_PORT%
) else (
    echo [⚠] Frontend:         Starting up...
)

echo.
echo ============================================================

if %ALL_OK% EQU 1 (
    echo           ✅ ALL CORE SERVICES RUNNING
    echo.
    echo   🚀 Legal AI Platform is ready!
    echo.
    echo   Main Application:  http://localhost:%SVELTEKIT_PORT%
    echo   YoRHa Command:     http://localhost:%SVELTEKIT_PORT%/yorha-command-center
    echo   Test Buttons:      http://localhost:%SVELTEKIT_PORT%/test-buttons
    echo   GPU Cache Test:    http://localhost:%SVELTEKIT_PORT%/test-gpu-cache
    echo   Evidence Upload:   http://localhost:%SVELTEKIT_PORT%/evidence/upload
    echo.
    echo   GPU Acceleration:  RTX 3060 Ti (%GPU_LAYERS% layers)
    echo   Context Size:      %CONTEXT_SIZE% tokens
    echo   VRAM Limit:        %VRAM_LIMIT% MB
    echo.
    echo   Default Credentials:
    echo   Username: admin@legal-ai.com
    echo   Password: admin123
) else (
    echo           ⚠️ SOME SERVICES NEED ATTENTION
    echo.
    echo   Please check the logs above for any failed services.
    echo   The application may still be functional with reduced features.
)

echo ============================================================
echo.
echo Press Ctrl+C to stop all services
echo.

REM Open the application in browser
timeout /t 3 /nobreak >nul
start http://localhost:%SVELTEKIT_PORT%

REM Keep the script running
:KEEPALIVE
timeout /t 30 /nobreak >nul
goto KEEPALIVE
