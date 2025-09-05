@echo off
REM SIMPLE-START.bat
REM Minimal startup that handles common failures

title Legal AI - Simple Start
cls

echo ========================================
echo    SIMPLE LEGAL AI STARTUP
echo    (Handles common issues)
echo ========================================
echo.

REM Step 1: Start only essential services
echo [1] Starting PostgreSQL (if not running)...
sc query postgresql-x64-17 >nul 2>&1
if %errorlevel% NEQ 0 (
    echo     PostgreSQL service not found, trying alternate names...
    net start postgresql* 2>nul || echo     [SKIP] PostgreSQL
) else (
    sc query postgresql-x64-17 | findstr RUNNING >nul || net start postgresql-x64-17 2>nul
)

echo.
echo [2] Starting Ollama...
tasklist | findstr ollama >nul 2>&1
if %errorlevel% NEQ 0 (
    start /B ollama serve
    echo     Waiting for Ollama to start...
    timeout /t 5 /nobreak >nul
) else (
    echo     Ollama already running
)

echo.
echo [3] Building Go Server (simplified version)...
cd go-microservice 2>nul
if exist enhanced-server-simple.go (
    go build -o simple-server.exe enhanced-server-simple.go 2>&1
    if exist simple-server.exe (
        echo     Starting Go server...
        start /B simple-server.exe
        timeout /t 2 /nobreak >nul
    ) else (
        echo     [ERROR] Go build failed - trying existing server
        if exist legal-ai-server.exe (
            start /B legal-ai-server.exe
        )
    )
) else (
    echo     [ERROR] No Go server found
)
cd .. 2>nul

echo.
echo [4] Starting Frontend...
cd sveltekit-frontend 2>nul
if exist package.json (
    if not exist node_modules (
        echo     Installing dependencies (this may take a minute)...
        call npm install
    )
    echo     Starting development server...
    start cmd /k "npm run dev"
) else (
    echo     [ERROR] Frontend not found
)
cd .. 2>nul

echo.
echo ========================================
echo    STARTUP COMPLETE
echo ========================================
echo.
echo Services that should be running:
echo   - Ollama: http://localhost:11434
echo   - Go API: http://localhost:8080/health
echo   - Frontend: http://localhost:5173
echo.
echo If something failed:
echo   1. Run DIAGNOSE-FAILURES.bat to see details
echo   2. Check error messages above
echo   3. Try starting services manually
echo Log files created. Terminal stays open.
timeout /t 300
