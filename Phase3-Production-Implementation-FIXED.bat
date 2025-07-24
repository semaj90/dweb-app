@echo off
setlocal enabledelayedexpansion

echo ========================================
echo PHASE 3 PRODUCTION IMPLEMENTATION
echo YoRHa Aesthetic + AI Integration
echo ========================================

:: Set paths
set PROJECT_ROOT=%~dp0
set FRONTEND_PATH=%PROJECT_ROOT%sveltekit-frontend
set OLLAMA_PATH=%PROJECT_ROOT%Ollama

echo Current directory: %PROJECT_ROOT%
echo Frontend path: %FRONTEND_PATH%

:: Check if directories exist
if not exist "%FRONTEND_PATH%" (
    echo ERROR: Frontend directory not found!
    echo Expected: %FRONTEND_PATH%
    pause
    exit /b 1
)

:: Navigate to frontend
cd /d "%FRONTEND_PATH%"

echo.
echo === Phase 3 Implementation Steps ===
echo 1. Installing dependencies...
call npm install

echo.
echo 2. Setting up YoRHa design system...
if not exist "src\lib\styles" mkdir "src\lib\styles"

echo.
echo 3. Updating TypeScript configuration...
powershell -ExecutionPolicy Bypass -Command "(Get-Content tsconfig.json) -replace '\"strict\": false', '\"strict\": true' | Set-Content tsconfig.json"

echo.
echo 4. Setting up XState v5 integration...
call npm install xstate@5 @xstate/svelte@5

echo.
echo 5. Configuring Ollama integration...
call npm install ollama@latest

echo.
echo 6. Setting up vector search dependencies...
call npm install @qdrant/js-client-rest ioredis

echo.
echo 7. Checking database schema...
call npm run db:push

echo.
echo 8. Setting up Gemma 3 Legal Enhanced model...
cd /d "%OLLAMA_PATH%"
if exist "ollama.exe" (
    echo Starting Ollama service...
    start /b ollama.exe serve
    timeout /t 5 /nobreak > nul
    
    echo Pulling Gemma 3 model...
    ollama.exe pull gemma:7b
    
    echo Creating legal-enhanced model...
    if exist "..\Gemma3-Legal-Enhanced-Modelfile" (
        ollama.exe create gemma3-legal-enhanced -f "..\Gemma3-Legal-Enhanced-Modelfile"
    )
) else (
    echo WARNING: Ollama not found in expected location
    echo Please ensure Ollama is installed and accessible
)

:: Return to frontend
cd /d "%FRONTEND_PATH%"

echo.
echo 9. Starting development services...

:: Start containers if available
if exist "..\docker-compose.yml" (
    echo Starting Docker containers...
    cd /d "%PROJECT_ROOT%"
    docker-compose up -d postgres redis qdrant
    cd /d "%FRONTEND_PATH%"
)

echo.
echo 10. Running TypeScript checks...
call npm run check

echo.
echo 11. Testing Ollama integration...
powershell -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -Method GET } catch { Write-Host 'Ollama not responding' }"

echo.
echo === PHASE 3 IMPLEMENTATION COMPLETE ===
echo.
echo Next steps:
echo 1. Open http://localhost:5173 to access the application
echo 2. Test multi-step forms with YoRHa aesthetic
echo 3. Verify AI assistance functionality
echo 4. Test vector search integration
echo.
echo To start development server:
echo   npm run dev
echo.
echo To launch with AI services:
echo   npm run dev:with-llm
echo.

if "%1"=="--start-dev" (
    echo Starting development server...
    call npm run dev
) else (
    echo Press any key to start development server, or Ctrl+C to exit...
    pause > nul
    call npm run dev
)

endlocal