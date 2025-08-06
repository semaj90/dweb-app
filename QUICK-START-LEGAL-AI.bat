@echo off
REM QUICK-START-LEGAL-AI.bat
REM Simplified startup for daily use

title Legal AI System
cls

echo =========================================
echo      LEGAL AI SYSTEM - QUICK START
echo =========================================
echo.

REM Start PostgreSQL if not running
sc query postgresql-x64-17 2>nul | findstr "RUNNING" >nul
if not %errorlevel% == 0 (
    echo Starting PostgreSQL...
    net start postgresql-x64-17 2>nul
)

REM Start Redis
echo Starting Redis...
start /B "" "redis-windows\redis-server.exe" >nul 2>&1

REM Start Ollama
echo Starting Ollama...
start /B "" ollama serve >nul 2>&1

timeout /t 3 /nobreak >nul

REM Start Go Server
echo Starting Go Server...
cd go-microservice
if not exist legal-ai-server.exe (
    echo Building Go server...
    go build -o legal-ai-server.exe .
)
start /B "" legal-ai-server.exe >nul 2>&1
cd ..

REM Start Workers
echo Starting Workers...
cd workers
start /B "" node document-processor.worker.js >nul 2>&1
cd ..

REM Start Frontend
echo Starting Frontend...
cd sveltekit-frontend
start /B "" npm run dev >nul 2>&1
cd ..

echo.
echo =========================================
echo    System starting up...
echo =========================================
echo.
timeout /t 5 /nobreak >nul

echo Legal AI System is ready!
echo.
echo Frontend: http://localhost:5173
echo API: http://localhost:8080/health
echo.
echo Press any key to open the frontend...
pause >nul

start http://localhost:5173

echo.
echo System is running. Close this window to stop all services.
pause >nul
