@echo off
cls
color 0B
title YoRHa Legal AI - Native Windows Status Check

echo ==============================================================
echo           YORHA LEGAL AI - NATIVE WINDOWS CHECK
echo                    System Status Inspector
echo ==============================================================
echo.

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with Administrator privileges
) else (
    echo [!] WARNING: Not running as Administrator
    echo     Some checks may fail. Run as Administrator for best results.
)
echo.

echo ==============================================================
echo CHECKING CORE TOOLS
echo ==============================================================

:: Node.js
where node >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Node.js is installed
    for /f "tokens=*" %%i in ('node --version 2^>nul') do echo      Version: %%i
) else (
    echo [X] Node.js is NOT installed
)

:: npm
where npm >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] npm is installed
    for /f "tokens=*" %%i in ('npm --version 2^>nul') do echo      Version: %%i
) else (
    echo [X] npm is NOT installed
)

:: Python
where python >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Python is installed
) else (
    echo [!] Python is NOT installed (optional for OCR)
)

:: Git
where git >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Git is installed
) else (
    echo [!] Git is NOT installed
)

echo.
echo ==============================================================
echo CHECKING DATABASE SERVICES
echo ==============================================================

:: PostgreSQL
netstat -an | findstr :5432 >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] PostgreSQL is running on port 5432
    
    :: Try to connect to database
    set PGPASSWORD=123456
    psql -U legal_admin -d legal_ai_db -h localhost -c "SELECT 'Connected'" >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Database connection successful
        for /f "tokens=*" %%i in ('psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT^(^*^) FROM cases;" 2^>nul') do (
            echo      Cases in database: %%i
        )
    ) else (
        echo [!] Database connection failed - check credentials
    )
) else (
    echo [X] PostgreSQL is NOT running
)

:: Redis
netstat -an | findstr :6379 >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Redis is running on port 6379
) else (
    echo [X] Redis is NOT running
    if exist "C:\Redis\redis-server.exe" (
        echo      Redis found at C:\Redis - run redis-server.exe to start
    )
)

:: Neo4j
netstat -an | findstr :7474 >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Neo4j Browser is accessible on port 7474
) else (
    echo [!] Neo4j is NOT running (optional)
)

echo.
echo ==============================================================
echo CHECKING AI SERVICES
echo ==============================================================

:: Ollama
where ollama >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Ollama is installed
    
    :: Check if Ollama is running
    netstat -an | findstr :11434 >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Ollama API is running on port 11434
        
        :: List models
        echo      Available models:
        for /f "skip=1 tokens=1" %%i in ('ollama list 2^>nul') do (
            if not "%%i"=="NAME" echo        - %%i
        )
    ) else (
        echo [!] Ollama is installed but not running
        echo      Run: ollama serve
    )
) else (
    echo [X] Ollama is NOT installed
    echo      Download from: https://ollama.ai/download
)

echo.
echo ==============================================================
echo CHECKING STORAGE SERVICES
echo ==============================================================

:: MinIO
netstat -an | findstr :9000 >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] MinIO API is running on port 9000
) else (
    echo [!] MinIO is NOT running (optional)
    if exist "minio.exe" (
        echo      MinIO found locally - run minio.exe to start
    )
)

echo.
echo ==============================================================
echo CHECKING APPLICATION
echo ==============================================================

:: Dev Server
netstat -an | findstr :5173 >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] Vite dev server is running on port 5173
    echo      Access YoRHa Dashboard at: http://localhost:5173
) else (
    netstat -an | findstr :3000 >nul 2>&1
    if %errorlevel% == 0 (
        echo [OK] Dev server is running on port 3000
        echo      Access YoRHa Dashboard at: http://localhost:3000
    ) else (
        echo [!] Development server is NOT running
        echo      Run: npm run dev
    )
)

:: Check project files
if exist "package.json" (
    echo [OK] Project files found
) else (
    echo [X] Not in project directory!
)

if exist "node_modules" (
    echo [OK] Dependencies installed
) else (
    echo [!] Dependencies not installed - run: npm install
)

if exist ".env" (
    echo [OK] Environment file exists
) else (
    echo [!] .env file missing - copy .env.example to .env
)

if exist "src\routes\yorha-dashboard" (
    echo [OK] YoRHa Dashboard components found
) else (
    echo [X] YoRHa Dashboard missing!
)

echo.
echo ==============================================================
echo SUMMARY
echo ==============================================================

:: Count issues
set /a issues=0

netstat -an | findstr :5432 >nul 2>&1
if %errorlevel% NEQ 0 set /a issues+=1

netstat -an | findstr :6379 >nul 2>&1
if %errorlevel% NEQ 0 set /a issues+=1

netstat -an | findstr :11434 >nul 2>&1
if %errorlevel% NEQ 0 set /a issues+=1

netstat -an | findstr :5173 >nul 2>&1
if %errorlevel% == 0 (
    set devserver=1
) else (
    netstat -an | findstr :3000 >nul 2>&1
    if %errorlevel% == 0 (
        set devserver=1
    ) else (
        set devserver=0
        set /a issues+=1
    )
)

if %issues% == 0 (
    echo.
    color 0A
    echo *************************************************************
    echo               SYSTEM FULLY OPERATIONAL!
    echo          All required services are running
    echo *************************************************************
    echo.
    echo Access your YoRHa Legal AI Dashboard at:
    echo http://localhost:5173 or http://localhost:3000
) else (
    echo.
    color 0C
    echo *************************************************************
    echo            ATTENTION: %issues% SERVICES NEED ATTENTION
    echo *************************************************************
    echo.
    echo To fix all issues, run as Administrator:
    echo   START-NATIVE-WINDOWS-COMPLETE.ps1
    echo.
    echo Or start services manually:
    if not exist "C:\Program Files\PostgreSQL" echo   - Install PostgreSQL
    if not exist "C:\Redis" echo   - Install Redis
    where ollama >nul 2>&1
    if %errorlevel% NEQ 0 echo   - Install Ollama
    if %devserver% == 0 echo   - Run: npm run dev
)

echo.
echo ==============================================================
echo Press any key to exit...
pause >nul
