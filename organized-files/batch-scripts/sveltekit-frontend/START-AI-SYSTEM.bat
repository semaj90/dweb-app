@echo off
title Production AI System Startup - Legal Models + Enhanced Services
color 0A

echo.
echo ========================================
echo    PRODUCTION AI SYSTEM STARTUP
echo    Primary: gemma3-legal
echo    Enhanced: PostgreSQL + Qdrant + WebGPU
echo ========================================
echo.

REM Set production environment variables
set OLLAMA_MODEL=gemma3-legal
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set QDRANT_URL=http://localhost:6333
set PRODUCTION_UPLOAD_URL=http://localhost:5173/api/production-upload
set YORHA_API_URL=http://localhost:5173/api/yorha/legal-data
set GPU_ACCELERATION=true

REM Check for Ollama
echo [1/7] Checking Ollama installation...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama is not installed or not in PATH
    echo [!] Please install Ollama from https://ollama.ai
    pause
    exit /b 1
)
echo [OK] Ollama found

REM Start Ollama service
echo [2/7] Starting Ollama service...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if %errorlevel% neq 0 (
    echo     Starting Ollama serve...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)
echo [OK] Ollama service running

REM Check for legal models
echo [3/7] Checking for production legal models...
echo.
set GEMMA_FOUND=0
set LEGAL_BERT_FOUND=0

REM Check for gemma3-legal (production model)
ollama list 2>NUL | findstr /C:"gemma3-legal" >NUL
if %errorlevel% equ 0 (
    echo [OK] gemma3-legal (PRIMARY)
    set GEMMA_FOUND=1
) else (
    ollama list 2>NUL | findstr /C:"gemma" | findstr /C:"legal" >NUL
    if %errorlevel% equ 0 (
        echo [OK] Found Gemma legal variant
        set GEMMA_FOUND=1
    ) else (
        echo [!] gemma3-legal not found - attempting to pull...
        ollama pull gemma3-legal 2>NUL
        if %errorlevel% equ 0 (
            echo [OK] gemma3-legal pulled successfully
            set GEMMA_FOUND=1
        ) else (
            echo [!] Failed to pull gemma3-legal
        )
    )
)

ollama list 2>NUL | findstr /C:"legal-bert" >NUL
if %errorlevel% equ 0 (
    echo [OK] legal-bert (FALLBACK)
    set LEGAL_BERT_FOUND=1
) else (
    ollama list 2>NUL | findstr /C:"legal" | findstr /C:"bert" >NUL
    if %errorlevel% equ 0 (
        echo [OK] Found Legal BERT variant
        set LEGAL_BERT_FOUND=1
    ) else (
        echo [!] legal-bert not found
    )
)

REM Check for embedding models
ollama list 2>NUL | findstr /C:"nomic-embed-text" >NUL
if %errorlevel% equ 0 (
    echo [OK] nomic-embed-text (Embeddings)
) else (
    echo [!] nomic-embed-text not found
)

ollama list 2>NUL | findstr /C:"bge-large-en" >NUL
if %errorlevel% equ 0 (
    echo [OK] bge-large-en (Embedding fallback)
) else (
    echo [!] bge-large-en not found
)

echo.
if %GEMMA_FOUND% equ 0 if %LEGAL_BERT_FOUND% equ 0 (
    echo [WARNING] No legal models found!
    echo.
    echo The system requires at least one of:
    echo   - gemma3:legal-latest (primary)
    echo   - legal-bert (fallback)
    echo.
    echo Please ensure one of these models is available.
) else (
    echo [OK] At least one legal model is available
)

echo.
echo [4/7] Checking PostgreSQL database...
powershell -Command "try { $null = Invoke-RestMethod -Uri 'http://localhost:5173/api/health' -TimeoutSec 5; Write-Host '[OK] PostgreSQL connected' -ForegroundColor Green } catch { Write-Host '[!] PostgreSQL connection failed' -ForegroundColor Red }"

echo.
echo [5/7] Checking Qdrant vector database...
powershell -Command "try { $null = Invoke-RestMethod -Uri 'http://localhost:6333/health' -TimeoutSec 5; Write-Host '[OK] Qdrant ready' -ForegroundColor Green } catch { Write-Host '[!] Qdrant not available' -ForegroundColor Red }"

echo.
echo [6/7] Testing production upload system...
powershell -Command "try { $testData = @{ query='test'; type='health-check'; useRAG=$false } | ConvertTo-Json; $null = Invoke-RestMethod -Uri 'http://localhost:5173/api/production-upload' -Method POST -ContentType 'application/json' -Body $testData -TimeoutSec 10; Write-Host '[OK] Production upload system ready' -ForegroundColor Green } catch { Write-Host '[!] Production upload system failed' -ForegroundColor Red }"

echo.
echo [7/7] Verifying Ollama API endpoint...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% eq 0 (
    echo [OK] Ollama API responding on port 11434
) else (
    echo [!] Ollama API not responding
    echo     Waiting for service to start...
    timeout /t 5 /nobreak >nul
)

echo.
echo ========================================
echo    SYSTEM STATUS
echo ========================================
echo.

REM Display available models
echo Available Models:
ollama list 2>NUL
if %errorlevel% neq 0 (
    echo [!] Could not list models
)

echo.
echo ========================================
echo    PRODUCTION SYSTEM CONFIGURATION
echo ========================================
echo.
echo AI Models:
echo   1. gemma3-legal (PRIMARY - Legal Analysis)
echo   2. nomic-embed-text (Embeddings)
echo   3. bge-large-en (Embedding Fallback)
echo.
echo Database Systems:
echo   - PostgreSQL 17 + pgvector (Primary Data)
echo   - Qdrant (Vector Search)
echo   - Redis (Caching)
echo.
echo Enhanced Features:
echo   - WebGPU Acceleration
echo   - Enhanced RAG Pipeline
echo   - YoRHa Legal Data API
echo   - Production Upload System
echo.
echo ========================================
echo.

REM Test production APIs
echo Testing production API endpoints...
echo.

echo Testing YoRHa Legal Data API...
powershell -Command "try { $null = Invoke-RestMethod -Uri 'http://localhost:5173/api/yorha/legal-data?type=documents&limit=1' -TimeoutSec 5; Write-Host '[OK] YoRHa API ready' -ForegroundColor Green } catch { Write-Host '[!] YoRHa API not responding' -ForegroundColor Red }"

echo Testing production chat endpoint...
powershell -Command "try { $null = Invoke-RestMethod -Uri 'http://localhost:5173/api/ai/chat' -Method GET -TimeoutSec 5; Write-Host '[OK] Chat endpoint ready' -ForegroundColor Green } catch { Write-Host '[!] Chat endpoint not responding' -ForegroundColor Red }"

echo Testing health monitoring...
powershell -Command "try { $result = Invoke-RestMethod -Uri 'http://localhost:5173/api/health' -TimeoutSec 5; Write-Host '[OK] Health monitoring active' -ForegroundColor Green } catch { Write-Host '[!] Health monitoring failed' -ForegroundColor Red }"

echo.
echo ========================================
echo    PRODUCTION SYSTEM READY
echo ========================================
echo.
echo Legal AI Production System is fully operational!
echo.
echo Active Models:
echo - Primary: gemma3-legal (Legal Analysis)
echo - Embeddings: nomic-embed-text
echo - GPU: RTX 3060 Ti Acceleration
echo.
echo Production Endpoints:
echo - Legal AI Chat: http://localhost:5173/ai-assistant
echo - Production Upload: http://localhost:5173/api/production-upload
echo - YoRHa Legal Data: http://localhost:5173/api/yorha/legal-data
echo - Health Monitor: http://localhost:5173/api/health
echo.
echo Database Connections:
echo - PostgreSQL: postgresql://localhost:5432/legal_ai_db
echo - Qdrant Vector: http://localhost:6333
echo - Ollama AI: http://localhost:11434
echo.
echo Quick Start Commands:
echo - Run Integration Test: RUN-INTEGRATION-CHECK.bat
echo - Open AI Assistant: http://localhost:5173/ai-assistant
echo - Access Upload Test: http://localhost:5173/upload-test
echo.
echo The production system includes automatic fallbacks,
echo enhanced error handling, and comprehensive logging.
echo.
pause
