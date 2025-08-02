@echo off
REM Legal AI System - Windows 10 Optimized Startup Script
REM Starts Docker services, MCP servers, and VS Code with proper configuration

echo ========================================
echo Legal AI System - Windows 10 Startup
echo ========================================
echo.

REM Check if Docker is running
echo [1/6] Checking Docker status...
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo Docker is running ✓
echo.

REM Start Docker services
echo [2/6] Starting Docker services...
docker-compose down --remove-orphans
docker-compose up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start Docker services
    pause
    exit /b 1
)
echo Docker services started ✓
echo.

REM Wait for services to be ready
echo [3/6] Waiting for services to initialize...
timeout /t 15 /nobreak >nul

REM Check service health
echo [4/6] Checking service health...
curl -f http://localhost:5432 >nul 2>&1 || echo PostgreSQL starting...
curl -f http://localhost:11434/api/version >nul 2>&1 || echo Ollama starting...
curl -f http://localhost:6333/collections >nul 2>&1 || echo Qdrant starting...
curl -f http://localhost:6379 >nul 2>&1 || echo Redis starting...
echo.

REM Setup local Gemma3 model
echo [5/6] Setting up local Gemma3 Legal AI model...
echo Creating legal-ai model from your local Gemma3 files...
docker exec legal_ai_ollama /tmp/setup-models.sh
echo.
echo Testing legal-ai model...
timeout /t 5 /nobreak >nul
docker exec legal_ai_ollama ollama ls | findstr "legal-ai"
if %errorlevel% equ 0 (
    echo ✅ Legal-AI model ready!
) else (
    echo ⚠️  Legal-AI model setup pending - check logs
)
echo.

REM Start VS Code with MCP configuration
echo [6/6] Starting VS Code with Legal AI configuration...
cd /d "%~dp0"
code . --profile "Legal AI"
echo.

echo ========================================
echo Legal AI System Started Successfully!
echo ========================================
echo.
echo Services running on:
echo - PostgreSQL: localhost:5432
echo - Ollama: localhost:11434  
echo - Qdrant: localhost:6333
echo - Redis: localhost:6379
echo - VS Code MCP: Auto-configured
echo.
echo Access your application at: http://localhost:5173
echo.
pause