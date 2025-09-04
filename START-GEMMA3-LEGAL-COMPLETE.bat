@echo off
REM ============================================================================
REM GEMMA3 LEGAL MODEL - COMPLETE NATIVE WINDOWS INTEGRATION
REM ============================================================================
REM Integrates Gemma3 with your existing Legal AI Platform
REM Components: llama.cpp, WebAssembly, GPU acceleration, microservices
REM ============================================================================

setlocal enabledelayedexpansion
color 0A
title Gemma3 Legal Model Integration - Native Windows

echo.
echo ============================================================
echo          GEMMA3 LEGAL MODEL INTEGRATION
echo          Native Windows - No Docker Required
echo ============================================================
echo.

REM Set environment variables
set GEMMA3_MODEL_PATH=%~dp0local-models\gemma3-legal.gguf
set GEMMA3_PORT=8095
set OLLAMA_PORT=11434
set POSTGRES_PORT=5432
set REDIS_PORT=6379
set NATS_PORT=4222
set QDRANT_PORT=6333
set NEO4J_PORT=7474
set MINIO_PORT=9000
set GPU_LAYERS=35
set CONTEXT_SIZE=4096
set BATCH_SIZE=512

REM Check for required services
echo [1/10] Checking system requirements...
echo ----------------------------------------

REM Check for CUDA
where /q nvcc
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] CUDA not found in PATH. GPU acceleration may be limited.
    echo          Install CUDA Toolkit 12.8 for optimal performance.
) else (
    echo [OK] CUDA detected
    nvcc --version | findstr /C:"release"
)

REM Check for PostgreSQL
echo.
echo [2/10] Checking PostgreSQL with pgvector...
psql -U postgres -p %POSTGRES_PORT% -c "SELECT version();" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PostgreSQL not running on port %POSTGRES_PORT%
    echo Starting PostgreSQL...
    net start postgresql-x64-17 >nul 2>&1
    timeout /t 3 /nobreak >nul
) else (
    echo [OK] PostgreSQL running on port %POSTGRES_PORT%
)

REM Check pgvector extension
psql -U postgres -p %POSTGRES_PORT% -d legal_ai_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Installing pgvector extension...
    psql -U postgres -p %POSTGRES_PORT% -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
)

REM Check for Redis
echo.
echo [3/10] Checking Redis cache...
redis-cli -p %REDIS_PORT% ping >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting Redis...
    start /B redis-server --port %REDIS_PORT% --maxmemory 2gb --maxmemory-policy allkeys-lru
    timeout /t 2 /nobreak >nul
) else (
    echo [OK] Redis running on port %REDIS_PORT%
)

REM Check for NATS
echo.
echo [4/10] Checking NATS messaging...
curl -s http://localhost:8222/varz >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting NATS server...
    start /B nats-server -p %NATS_PORT% -m 8222 --store_dir ./nats-data
    timeout /t 2 /nobreak >nul
) else (
    echo [OK] NATS running
)

REM Check for Ollama
echo.
echo [5/10] Checking Ollama service...
curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting Ollama...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo [OK] Ollama running on port %OLLAMA_PORT%
)

REM Check for Gemma3 model
echo.
echo [6/10] Checking Gemma3 Legal model...
if not exist "%GEMMA3_MODEL_PATH%" (
    echo [ERROR] Gemma3 model not found at %GEMMA3_MODEL_PATH%
    echo.
    echo Please download the model first:
    echo   1. Download from Hugging Face or create with Ollama
    echo   2. Place in: %GEMMA3_MODEL_PATH%
    echo.
    echo Creating model with Ollama...
    
    REM Create Modelfile
    (
        echo FROM gemma2:9b-instruct-q4_K_M
        echo.
        echo TEMPLATE """{{ if .System }}^<start_of_turn^>system
        echo {{ .System }}^<end_of_turn^>
        echo {{ end }}^<start_of_turn^>user
        echo {{ .Prompt }}^<end_of_turn^>
        echo ^<start_of_turn^>model
        echo """
        echo.
        echo SYSTEM """You are a legal AI assistant trained on case law, statutes, and legal documents. Provide accurate, detailed legal analysis while noting this is not legal advice. Focus on jurisdiction, applicable laws, precedents, legal reasoning, and potential outcomes."""
        echo.
        echo PARAMETER temperature 0.1
        echo PARAMETER top_k 40
        echo PARAMETER top_p 0.9
        echo PARAMETER repeat_penalty 1.1
        echo PARAMETER num_ctx 4096
        echo PARAMETER num_batch 512
        echo PARAMETER num_gpu %GPU_LAYERS%
    ) > Modelfile-gemma3-legal
    
    ollama create gemma3-legal -f Modelfile-gemma3-legal
    
    echo Model created with Ollama
) else (
    echo [OK] Gemma3 model found
)

REM Build llama.cpp bridge if needed
echo.
echo [7/10] Building llama.cpp bridge...
if not exist "gemma3-legal-integration\gemma3-bridge.exe" (
    echo Compiling llama.cpp bridge with CUDA support...
    cd gemma3-legal-integration
    
    REM Check for Visual Studio compiler
    where /q cl
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Visual Studio compiler not found
        echo Using MinGW fallback...
        g++ -O3 -march=native -fopenmp -DGGML_USE_CUDA -I../llama.cpp -I../llama.cpp/ggml/include -L../llama.cpp/build -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" gemma3-llama-cpp-bridge.cpp -o gemma3-bridge.exe -lllama -lcudart -lcublas -lpthread -lws2_32
    ) else (
        cl /O2 /EHsc /I"../llama.cpp" /I"../llama.cpp/ggml/include" /DGGML_USE_CUDA /DNDEBUG gemma3-llama-cpp-bridge.cpp /link /LIBPATH:"../llama.cpp/build/Release" /LIBPATH:"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/lib/x64" llama.lib ggml.lib cudart.lib cublas.lib ws2_32.lib /OUT:gemma3-bridge.exe
    )
    
    cd ..
) else (
    echo [OK] llama.cpp bridge already built
)

REM Start Go microservices
echo.
echo [8/10] Starting Go microservices...

REM Enhanced RAG service
tasklist /FI "WINDOWTITLE eq Enhanced RAG Service" 2>nul | find /I /N "cmd.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Starting Enhanced RAG service...
    start "Enhanced RAG Service" /B cmd /c "cd go-microservice && go run cmd/enhanced-rag/main.go"
    timeout /t 2 /nobreak >nul
)

REM Upload service
tasklist /FI "WINDOWTITLE eq Upload Service" 2>nul | find /I /N "cmd.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Starting Upload service...
    start "Upload Service" /B cmd /c "cd go-microservice && go run cmd/upload-service/main.go"
    timeout /t 2 /nobreak >nul
)

REM Vector service
tasklist /FI "WINDOWTITLE eq Vector Service" 2>nul | find /I /N "cmd.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Starting Vector service...
    start "Vector Service" /B cmd /c "cd go-microservice && go run cmd/vector-service/main.go"
    timeout /t 2 /nobreak >nul
)

REM Start Gemma3 bridge
echo.
echo [9/10] Starting Gemma3 Legal bridge...
tasklist /FI "IMAGENAME eq gemma3-bridge.exe" 2>nul | find /I /N "gemma3-bridge.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    start "Gemma3 Legal Bridge" /B gemma3-legal-integration\gemma3-bridge.exe "%GEMMA3_MODEL_PATH%"
    timeout /t 3 /nobreak >nul
)

REM Start SvelteKit frontend
echo.
echo [10/10] Starting SvelteKit frontend...
tasklist /FI "WINDOWTITLE eq SvelteKit Frontend" 2>nul | find /I /N "node.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Starting SvelteKit with Gemma3 integration...
    start "SvelteKit Frontend" /B cmd /c "cd sveltekit-frontend && npm run dev"
    timeout /t 5 /nobreak >nul
)

REM Verify all services
echo.
echo ============================================================
echo           VERIFICATION & HEALTH CHECK
echo ============================================================
echo.

echo Checking service health...
echo.

set SERVICES_OK=1

REM Check PostgreSQL
psql -U postgres -p %POSTGRES_PORT% -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] PostgreSQL:         http://localhost:%POSTGRES_PORT%
) else (
    echo [✗] PostgreSQL:         FAILED
    set SERVICES_OK=0
)

REM Check Redis
redis-cli -p %REDIS_PORT% ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Redis:              redis://localhost:%REDIS_PORT%
) else (
    echo [✗] Redis:              FAILED
    set SERVICES_OK=0
)

REM Check NATS
curl -s http://localhost:8222/varz >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] NATS:               nats://localhost:%NATS_PORT%
) else (
    echo [✗] NATS:               FAILED
    set SERVICES_OK=0
)

REM Check Ollama
curl -s http://localhost:%OLLAMA_PORT%/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Ollama:             http://localhost:%OLLAMA_PORT%
) else (
    echo [✗] Ollama:             FAILED
    set SERVICES_OK=0
)

REM Check Gemma3 Bridge
curl -s http://localhost:%GEMMA3_PORT%/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Gemma3 Bridge:      http://localhost:%GEMMA3_PORT%
) else (
    echo [✗] Gemma3 Bridge:      FAILED
    set SERVICES_OK=0
)

REM Check Enhanced RAG
curl -s http://localhost:8094/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Enhanced RAG:       http://localhost:8094
) else (
    echo [✗] Enhanced RAG:       FAILED
    set SERVICES_OK=0
)

REM Check SvelteKit
curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] SvelteKit Frontend: http://localhost:5173
) else (
    echo [✗] SvelteKit Frontend: FAILED
    set SERVICES_OK=0
)

echo.
echo ============================================================

if %SERVICES_OK% EQU 1 (
    echo           ✓ ALL SERVICES RUNNING SUCCESSFULLY
    echo.
    echo   Gemma3 Legal AI Platform is ready!
    echo.
    echo   Access points:
    echo   - Frontend:        http://localhost:5173
    echo   - Gemma3 API:      http://localhost:%GEMMA3_PORT%
    echo   - Enhanced RAG:    http://localhost:8094
    echo   - Health Monitor:  http://localhost:8094/health
    echo.
    echo   GPU Acceleration:  %GPU_LAYERS% layers on RTX 3060 Ti
    echo   Context Size:      %CONTEXT_SIZE% tokens
    echo   Batch Size:        %BATCH_SIZE% tokens
) else (
    echo           ⚠ SOME SERVICES FAILED TO START
    echo.
    echo   Please check the logs and try again.
    echo   Run 'npm run dev:full' for detailed output.
)

echo ============================================================
echo.
echo Press any key to open the Legal AI Platform...
pause >nul

start http://localhost:5173

echo.
echo Platform launched in your browser.
echo Keep this window open to maintain services.
echo Press Ctrl+C to stop all services.
echo.

:KEEPALIVE
timeout /t 30 /nobreak >nul
goto KEEPALIVE
