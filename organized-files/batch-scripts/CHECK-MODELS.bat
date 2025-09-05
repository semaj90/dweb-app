@echo off
REM ================================================
REM QUICK MODEL VERIFICATION SCRIPT
REM Checks if Gemma3:legal and nomic-embed-text exist
REM ================================================

cls
echo ================================================
echo   LEGAL AI MODELS - VERIFICATION CHECK
echo ================================================
echo.

REM Check if Ollama is running
echo [1/3] Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo       [ERROR] Ollama not running
    echo       Start with: ollama serve
    echo.
    pause
    exit /b 1
) else (
    echo       [OK] Ollama service is running
)

echo.
echo [2/3] Checking installed models...
ollama list > models-list.tmp

REM Check for Gemma3:legal
findstr /C:"gemma3:legal" models-list.tmp >nul
if %ERRORLEVEL% EQU 0 (
    echo       [OK] gemma3:legal model found
) else (
    echo       [MISSING] gemma3:legal model
    echo       Install with: ollama pull gemma3:legal
    set MISSING_MODELS=1
)

REM Check for nomic-embed-text
findstr /C:"nomic-embed-text" models-list.tmp >nul
if %ERRORLEVEL% EQU 0 (
    echo       [OK] nomic-embed-text model found
) else (
    echo       [MISSING] nomic-embed-text model
    echo       Install with: ollama pull nomic-embed-text
    set MISSING_MODELS=1
)

REM Clean up
del models-list.tmp >nul 2>&1

echo.
echo [3/3] Model verification complete
echo.

if defined MISSING_MODELS (
    echo ================================================
    echo   ACTION REQUIRED - INSTALL MISSING MODELS
    echo ================================================
    echo.
    echo Run these commands:
    echo   ollama pull gemma3:legal
    echo   ollama pull nomic-embed-text
    echo.
    echo Then run this script again to verify.
) else (
    echo ================================================
    echo   ALL MODELS READY!
    echo ================================================
    echo.
    echo Your Legal AI system has all required models:
    echo   - gemma3:legal (Chat/Analysis)
    echo   - nomic-embed-text (Embeddings)
    echo.
    echo You can now start the system:
    echo   npm run start:production
)

echo.
pause
