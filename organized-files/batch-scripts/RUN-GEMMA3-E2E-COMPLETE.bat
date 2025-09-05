@echo off
REM ============================================================================
REM GEMMA3 LEGAL MODEL - COMPLETE END-TO-END NATIVE WINDOWS INTEGRATION
REM Now with full working pipeline and testing
REM ============================================================================

setlocal enabledelayedexpansion
color 0A
title Gemma3 Legal Model - Complete End-to-End Integration

echo.
echo ============================================================
echo          GEMMA3 LEGAL MODEL END-TO-END INTEGRATION
echo          Native Windows - Complete Working Pipeline
echo ============================================================
echo.

REM Kill any existing processes to avoid conflicts
echo [CLEANUP] Stopping any existing services...
taskkill /F /IM gemma3-bridge.exe >nul 2>&1
taskkill /F /IM node.exe /FI "WINDOWTITLE eq Gemma3*" >nul 2>&1
timeout /t 2 /nobreak >nul

REM Set environment variables
set GEMMA3_MODEL_PATH=%~dp0local-models\gemma3-legal.gguf
set GEMMA3_PORT=8095
set OLLAMA_PORT=11434
set POSTGRES_PORT=5432
set REDIS_PORT=6379
set NATS_PORT=4222

echo.
echo [STEP 1/10] Checking Prerequisites...
echo =====================================

REM Check Node.js
where /q node
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
) else (
    echo [OK] Node.js detected
)

REM Check for package.json in integration folder
if not exist "gemma3-legal-integration\package.json" (
    echo [INFO] Creating package.json for integration...
    (
        echo {
        echo   "name": "gemma3-legal-integration",
        echo   "version": "1.0.0",
        echo   "type": "module",
        echo   "scripts": {
        echo     "start": "node complete-integration.js",
        echo     "test": "node test-e2e.mjs"
        echo   },
        echo   "dependencies": {
        echo     "express": "^4.18.2",
        echo     "cors": "^2.8.5",
        echo     "ws": "^8.14.2",
        echo     "node-fetch": "^3.3.2"
        echo   }
        echo }
    ) > gemma3-legal-integration\package.json
)

echo.
echo [STEP 2/10] Installing Integration Dependencies...
echo ==================================================
cd gemma3-legal-integration
call npm install --silent
cd ..

echo.
echo [STEP 3/10] Starting Core Services...
echo =====================================

REM Start PostgreSQL
echo Starting PostgreSQL...
net start postgresql-x64-17 >nul 2>&1 || net start postgresql-x64-16 >nul 2>&1 || net start postgresql-x64-15 >nul 2>&1
timeout /t 2 /nobreak >nul

REM Start Redis
echo Starting Redis...
if not exist "redis-windows-latest\redis-server.exe" (
    echo [WARNING] Redis not found. Some features may be limited.
) else (
    start /B "Redis" redis-windows-latest\redis-server.exe --port %REDIS_PORT%
    timeout /t 2 /nobreak >nul
)

REM Start Ollama
echo Starting Ollama...
start /B "Ollama" ollama serve >nul 2>&1
timeout /t 3 /nobreak >nul

echo.
echo [STEP 4/10] Checking Gemma3 Model...
echo ====================================

REM Check if model exists in Ollama
ollama list | findstr /C:"gemma3-legal" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Creating Gemma3 Legal model...
    
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
        echo SYSTEM """You are an expert legal AI assistant trained on case law, statutes, and legal documents. 
        echo Provide accurate, detailed legal analysis while noting this is not legal advice. 
        echo Focus on: jurisdiction, applicable laws, precedents, legal reasoning, and potential outcomes.
        echo Be precise, cite relevant legal concepts, and structure your responses clearly."""
        echo.
        echo PARAMETER temperature 0.1
        echo PARAMETER top_k 40
        echo PARAMETER top_p 0.9
        echo PARAMETER repeat_penalty 1.1
        echo PARAMETER num_ctx 4096
        echo PARAMETER num_gpu 35
    ) > Modelfile-gemma3-legal
    
    ollama create gemma3-legal -f Modelfile-gemma3-legal
    echo [OK] Model created
) else (
    echo [OK] Gemma3 Legal model found
)

echo.
echo [STEP 5/10] Starting Gemma3 Integration Server...
echo =================================================

REM Start the complete integration server
start "Gemma3 Integration" /B cmd /c "cd gemma3-legal-integration && node complete-integration.js"
timeout /t 3 /nobreak >nul

echo.
echo [STEP 6/10] Starting Enhanced Services...
echo =========================================

REM Check if enhanced RAG is available
curl -s http://localhost:8094/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Starting Enhanced RAG service...
    if exist "go-microservice\bin\enhanced-rag.exe" (
        start "Enhanced RAG" /B go-microservice\bin\enhanced-rag.exe
        timeout /t 2 /nobreak >nul
    )
)

echo.
echo [STEP 7/10] Starting SvelteKit Frontend...
echo ==========================================

REM Check if SvelteKit is running
curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Starting SvelteKit development server...
    start "SvelteKit" /B cmd /c "cd sveltekit-frontend && npm run dev"
    timeout /t 5 /nobreak >nul
)

echo.
echo [STEP 8/10] Verifying Service Health...
echo ========================================

set SERVICES_OK=1

REM Check Gemma3 Integration
curl -s http://localhost:8095/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Gemma3 Integration: http://localhost:8095
) else (
    echo [✗] Gemma3 Integration: FAILED
    set SERVICES_OK=0
)

REM Check Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] Ollama Service: http://localhost:11434
) else (
    echo [✗] Ollama Service: FAILED
    set SERVICES_OK=0
)

REM Check SvelteKit
curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [✓] SvelteKit Frontend: http://localhost:5173
) else (
    echo [✗] SvelteKit Frontend: FAILED
)

echo.
echo [STEP 9/10] Running End-to-End Tests...
echo ========================================

if %SERVICES_OK% EQU 1 (
    echo Running comprehensive test suite...
    echo.
    cd gemma3-legal-integration
    call node test-e2e.mjs
    cd ..
) else (
    echo [WARNING] Skipping tests - some services are not running
)

echo.
echo [STEP 10/10] Final Integration Status
echo ======================================
echo.

if %SERVICES_OK% EQU 1 (
    echo ╔════════════════════════════════════════════════════════════╗
    echo ║     ✓ GEMMA3 LEGAL AI - END-TO-END INTEGRATION COMPLETE   ║
    echo ╚════════════════════════════════════════════════════════════╝
    echo.
    echo   Access Points:
    echo   ─────────────────────────────────────────────────────────
    echo   🌐 Frontend Application:    http://localhost:5173
    echo   🤖 Gemma3 API (OpenAI):    http://localhost:8095/v1/completions
    echo   💬 Chat API:               http://localhost:8095/v1/chat/completions
    echo   🔢 Embeddings API:         http://localhost:8095/v1/embeddings
    echo   ⚖️  Legal Analysis API:     http://localhost:8095/api/legal/analyze
    echo   📊 Health Monitor:         http://localhost:8095/health
    echo   🔌 WebSocket:              ws://localhost:8096
    echo.
    echo   Test the Integration:
    echo   ─────────────────────────────────────────────────────────
    echo   curl -X POST http://localhost:8095/v1/completions ^
    echo        -H "Content-Type: application/json" ^
    echo        -d "{\"prompt\": \"What is negligence in tort law?\", \"max_tokens\": 200}"
    echo.
) else (
    echo ╔════════════════════════════════════════════════════════════╗
    echo ║        ⚠ PARTIAL INTEGRATION - SOME SERVICES FAILED       ║
    echo ╚════════════════════════════════════════════════════════════╝
    echo.
    echo   Please check the logs and try again.
    echo   You can still access available services.
)

echo ════════════════════════════════════════════════════════════
echo.
echo Press any key to open the Legal AI Platform in your browser...
pause >nul

start http://localhost:5173

echo.
echo Platform launched. Keep this window open to maintain services.
echo Press Ctrl+C to stop all services.
echo.

:KEEPALIVE
timeout /t 30 /nobreak >nul
REM Quick health check
curl -s http://localhost:8095/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Gemma3 Integration may have stopped. Restarting...
    start "Gemma3 Integration" /B cmd /c "cd gemma3-legal-integration && node complete-integration.js"
)
goto KEEPALIVE
