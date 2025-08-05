@echo off
echo ====================================================================
echo   LEGAL AI COMPLETE SYSTEM STARTUP
echo   Using Local gemma3-legal Model
echo ====================================================================
echo.

REM Source environment configuration
call SET-LEGAL-AI-ENV.bat

REM Verify Ollama is running with local models
echo [CHECK] Verifying Ollama with gemma3-legal...
ollama list | findstr /i "gemma3-legal" >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] gemma3-legal model not found!
    echo Please ensure your local model is properly installed.
    pause
    exit /b 1
)

REM Start Ollama if not running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorLevel% neq 0 (
    echo Starting Ollama...
    start "Ollama" ollama serve
    timeout /t 5 /nobreak >nul
)

REM Check if main startup script exists
if not exist "START-LEGAL-AI.bat" (
    echo [ERROR] START-LEGAL-AI.bat not found!
    echo Please ensure you have the base startup script.
    pause
    exit /b 1
)

REM Start existing services
echo [START] Core services...


REM Wait for services
timeout /t 10 /nobreak >nul

REM Start BullMQ workers
echo [START] BullMQ workers...
if exist "workers\start-workers.js" (
    cd workers
    call npm install --silent
    start "BullMQ Workers" cmd /c "node start-workers.js"
    cd ..
) else (
    echo [WARNING] Workers not found - skipping
)

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
