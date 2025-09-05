@echo off
echo.
echo ========================================
echo    QUICK START - USE EXISTING BUILDS
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo Checking for existing builds...
echo.

REM Check what's already built
set FOUND_SERVICES=0

if exist "go-microservice\legal-ai-server.exe" (
    echo [FOUND] legal-ai-server.exe
    set FOUND_SERVICES=1
    set MAIN_SERVICE=go-microservice\legal-ai-server.exe
)

if exist "go-microservice\enhanced-legal-ai.exe" (
    echo [FOUND] enhanced-legal-ai.exe
    set FOUND_SERVICES=1
    set MAIN_SERVICE=go-microservice\enhanced-legal-ai.exe
)

if exist "go-microservice\simple-server.exe" (
    echo [FOUND] simple-server.exe
    set FOUND_SERVICES=1
    set SIMPLE_SERVICE=go-microservice\simple-server.exe
)

if exist "go-microservice\bin\legal-ai-service.exe" (
    echo [FOUND] bin\legal-ai-service.exe
    set FOUND_SERVICES=1
    set MAIN_SERVICE=go-microservice\bin\legal-ai-service.exe
)

if exist "go-microservice\gpu-legal-ai-8084.exe" (
    echo [FOUND] gpu-legal-ai-8084.exe
    set FOUND_SERVICES=1
    set MAIN_SERVICE=go-microservice\gpu-legal-ai-8084.exe
)

if %FOUND_SERVICES% equ 0 (
    echo.
    echo No pre-built services found!
    echo.
    echo Please run COMPLETE-SETUP.bat to build the services first.
    echo Or build manually with:
    echo   cd go-microservice
    echo   go build -o legal-ai-server.exe main.go
    echo.
    pause
    exit /b 1
)

echo.
echo Starting services...
echo.

REM Kill any running instances
taskkill /F /IM legal-ai-server.exe >nul 2>&1
taskkill /F /IM enhanced-legal-ai.exe >nul 2>&1
taskkill /F /IM gpu-legal-ai-8084.exe >nul 2>&1
taskkill /F /IM simple-server.exe >nul 2>&1

REM Start PostgreSQL if not running
netstat -an | findstr ":5432" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Starting PostgreSQL...
    net start postgresql-x64-14 >nul 2>&1
    timeout /t 2 >nul
)

REM Start Ollama if not running
netstat -an | findstr ":11434" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Starting Ollama...
    where ollama >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        start /min cmd /c "ollama serve"
        timeout /t 3 >nul
    )
)

REM Start main service
if defined MAIN_SERVICE (
    echo Starting main service: %MAIN_SERVICE%
    start /min "" "%MAIN_SERVICE%"
    timeout /t 2 >nul
)

REM Start simple service if exists
if defined SIMPLE_SERVICE (
    echo Starting simple service: %SIMPLE_SERVICE%
    start /min "" "%SIMPLE_SERVICE%"
    timeout /t 2 >nul
)

REM Start frontend if node_modules exists
if exist "frontend\node_modules" (
    echo Starting frontend...
    cd frontend
    start /min cmd /c "npm run dev"
    cd ..
    timeout /t 5 >nul
) else (
    echo.
    echo Frontend dependencies not installed.
    echo Run: cd frontend ^&^& npm install
    echo.
)

echo.
echo ========================================
echo    SERVICE STATUS
echo ========================================
echo.

netstat -an | findstr ":5173" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Frontend:     http://localhost:5173) else (echo [STOPPED] Frontend)

netstat -an | findstr ":8080" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Service 8080: http://localhost:8080) else (echo [STOPPED] Port 8080)

netstat -an | findstr ":8084" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Service 8084: http://localhost:8084) else (echo [STOPPED] Port 8084)

netstat -an | findstr ":11434" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] Ollama:       http://localhost:11434) else (echo [STOPPED] Ollama)

netstat -an | findstr ":5432" >nul 2>&1
if %ERRORLEVEL% equ 0 (echo [RUNNING] PostgreSQL:   Port 5432) else (echo [STOPPED] PostgreSQL)

echo.
echo Press any key to exit...
pause >nul