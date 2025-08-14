@echo off
echo.
echo 🤖 YoRHa Legal AI - Starting Windows Services
echo.

REM Start PostgreSQL if not running
echo Checking PostgreSQL...
netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ PostgreSQL already running on port 5432
) else (
    echo ⚠ PostgreSQL not detected on port 5432
    echo   Please start PostgreSQL service manually
    echo   Command: net start postgresql-x64-17
)

REM Start Redis if not running  
echo.
echo Checking Redis...
netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Redis already running on port 6379
) else (
    echo ⚠ Starting Redis...
    if exist "redis-windows\redis-server.exe" (
        start /B "Redis Server" redis-windows\redis-server.exe
        timeout /t 3 >nul
        echo ✓ Redis started
    ) else (
        echo ✗ Redis not found in redis-windows directory
        echo   Please download Redis for Windows
    )
)

REM Start Ollama if not running
echo.
echo Checking Ollama...
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Ollama already running on port 11434
) else (
    echo ⚠ Starting Ollama...
    start /B "Ollama" ollama serve
    timeout /t 5 >nul
    echo ✓ Ollama started
)

echo.
echo 🚀 Core services check complete!
echo Run: npm run dev --frontend-only
echo.
pause