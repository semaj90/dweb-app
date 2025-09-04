@echo off
title Auto-Start System Services
color 0A

echo.
echo ========================================
echo    AUTO-START SYSTEM SERVICES
echo ========================================
echo.
echo This script will attempt to start all required services
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] This script requires Administrator privileges
    echo [!] Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

echo [*] Starting services...
echo.

REM 1. PostgreSQL
echo [1/4] Starting PostgreSQL...
sc query "postgresql-x64-15" >nul 2>&1
if %errorlevel%==0 (
    net start "postgresql-x64-15" >nul 2>&1
    if %errorlevel%==0 (
        echo [OK] PostgreSQL started successfully
    ) else (
        sc query "postgresql-x64-15" | findstr RUNNING >nul 2>&1
        if %errorlevel%==0 (
            echo [OK] PostgreSQL already running
        ) else (
            echo [X] Failed to start PostgreSQL
        )
    )
) else (
    echo [!] PostgreSQL service not found
    echo     Try: pg_ctl start -D "C:\Program Files\PostgreSQL\15\data"
)

REM 2. Redis
echo [2/4] Starting Redis...
sc query "Redis" >nul 2>&1
if %errorlevel%==0 (
    net start "Redis" >nul 2>&1
    if %errorlevel%==0 (
        echo [OK] Redis started successfully
    ) else (
        sc query "Redis" | findstr RUNNING >nul 2>&1
        if %errorlevel%==0 (
            echo [OK] Redis already running
        ) else (
            echo [X] Failed to start Redis
        )
    )
) else (
    echo [!] Redis service not found
    echo     Trying to start redis-server directly...
    start /B redis-server >nul 2>&1
    timeout /t 2 /nobreak >nul
    netstat -an | findstr ":6379" >nul 2>&1
    if %errorlevel%==0 (
        echo [OK] Redis started via redis-server
    ) else (
        echo [X] Could not start Redis
    )
)

REM 3. Ollama
echo [3/4] Starting Ollama...
tasklist | findstr /i "ollama.exe" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Ollama already running
) else (
    echo     Starting Ollama serve...
    start /B ollama serve >nul 2>&1
    timeout /t 3 /nobreak >nul
    netstat -an | findstr ":11434" >nul 2>&1
    if %errorlevel%==0 (
        echo [OK] Ollama started successfully
    ) else (
        echo [X] Could not start Ollama
        echo     Try: ollama serve
    )
)

REM 4. Dev Server
echo [4/4] Checking Dev Server...
netstat -an | findstr ":5173" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Dev Server already running
) else (
    echo [!] Dev Server not running
    echo     Run 'npm run dev' in the project directory
)

echo.
echo ========================================
echo    SERVICE STATUS SUMMARY
echo ========================================
echo.

REM Final status check
netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel%==0 (echo [✓] PostgreSQL) else (echo [✗] PostgreSQL)

netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel%==0 (echo [✓] Redis) else (echo [✗] Redis)

netstat -an | findstr ":11434" >nul 2>&1
if %errorlevel%==0 (echo [✓] Ollama) else (echo [✗] Ollama)

netstat -an | findstr ":5173" >nul 2>&1
if %errorlevel%==0 (echo [✓] Dev Server) else (echo [✗] Dev Server - Run: npm run dev)

echo.
echo [*] To run full verification: VERIFY-SYSTEM.bat
echo.
pause
