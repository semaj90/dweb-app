@echo off
REM CHECK-LEGAL-AI-STATUS.bat
REM Quick status check for all Legal AI components

echo ================================================
echo    LEGAL AI SYSTEM - STATUS CHECK
echo ================================================
echo.

echo [CHECKING SERVICES]
echo ------------------

REM Check PostgreSQL
echo PostgreSQL:
sc query postgresql-x64-17 2>nul | findstr "RUNNING" >nul
if %errorlevel% == 0 (
    echo   [RUNNING] PostgreSQL Database
) else (
    echo   [STOPPED] PostgreSQL Database
)

REM Check Redis
echo.
echo Redis:
tasklist /FI "IMAGENAME eq redis-server.exe" 2>nul | find /I "redis-server.exe" >nul
if %errorlevel% == 0 (
    echo   [RUNNING] Redis Cache Server
) else (
    echo   [STOPPED] Redis Cache Server
)

REM Check Ollama
echo.
echo Ollama:
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% == 0 (
    echo   [RUNNING] Ollama AI Service
) else (
    echo   [STOPPED] Ollama AI Service
)

REM Check Go Server
echo.
echo Go Server:
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% == 0 (
    echo   [RUNNING] Legal AI Go Server
) else (
    echo   [STOPPED] Legal AI Go Server
)

REM Check Node processes
echo.
echo Node.js Services:
tasklist /FI "IMAGENAME eq node.exe" 2>nul | find /I "node.exe" >nul
if %errorlevel% == 0 (
    echo   [RUNNING] Node.js processes detected
) else (
    echo   [STOPPED] No Node.js processes
)

echo.
echo ================================================
echo.

REM Check ports
echo [PORT STATUS]
echo -------------
netstat -an | findstr :5432 >nul 2>&1
if %errorlevel% == 0 (
    echo   Port 5432: PostgreSQL [LISTENING]
) else (
    echo   Port 5432: PostgreSQL [NOT LISTENING]
)

netstat -an | findstr :6379 >nul 2>&1
if %errorlevel% == 0 (
    echo   Port 6379: Redis [LISTENING]
) else (
    echo   Port 6379: Redis [NOT LISTENING]
)

netstat -an | findstr :8080 >nul 2>&1
if %errorlevel% == 0 (
    echo   Port 8080: Go Server [LISTENING]
) else (
    echo   Port 8080: Go Server [NOT LISTENING]
)

netstat -an | findstr :11434 >nul 2>&1
if %errorlevel% == 0 (
    echo   Port 11434: Ollama [LISTENING]
) else (
    echo   Port 11434: Ollama [NOT LISTENING]
)

netstat -an | findstr :5173 >nul 2>&1
if %errorlevel% == 0 (
    echo   Port 5173: Frontend [LISTENING]
) else (
    echo   Port 5173: Frontend [NOT LISTENING]
)

echo.
echo ================================================
echo.
echo [QUICK ACTIONS]
echo 1. Start all: START-LEGAL-AI-SYSTEM.bat
echo 2. Test system: node test-legal-ai-system.mjs
echo 3. View logs: pm2 logs
echo 4. Stop all: pm2 stop all
echo.

pause
