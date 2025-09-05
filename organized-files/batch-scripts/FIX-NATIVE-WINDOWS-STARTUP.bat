@echo off
REM =============================================================================
REM NATIVE WINDOWS STARTUP FIX - Resolves WSL/Linux compatibility issues
REM Fixes rollup native binary issues and ensures proper Windows execution
REM =============================================================================

echo ===============================================================================
echo NATIVE WINDOWS STARTUP FIX - LEGAL AI PLATFORM
echo ===============================================================================
echo [Issue]: WSL/Linux rollup binaries detected, need Windows-native execution
echo [Fix]: Native Windows dependency resolution and service startup
echo ===============================================================================

REM Force native Windows PowerShell (not WSL)
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo.
echo [1/4] === FIXING ROLLUP DEPENDENCY ISSUE ===
echo.

echo Cleaning problematic node_modules...
cd sveltekit-frontend
if exist node_modules (
    echo Removing existing node_modules directory...
    rmdir /s /q node_modules 2>nul
)

if exist package-lock.json (
    echo Removing package-lock.json...
    del package-lock.json 2>nul
)

echo Installing native Windows dependencies...
echo This will download Windows-specific rollup binaries...
call npm install --force
if %errorlevel% neq 0 (
    echo Trying with pnpm instead...
    call pnpm install --force
)

cd ..

echo.
echo [2/4] === STARTING INFRASTRUCTURE SERVICES ===
echo.

REM Check and start services in native Windows
echo Starting PostgreSQL...
sc query postgresql-x64-16 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL service found
    net start postgresql-x64-16 >nul 2>&1
) else (
    echo ⚠️ PostgreSQL service not installed - install PostgreSQL 16
)

REM Start Ollama if not running
echo Checking Ollama...
curl -s http://localhost:11434 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama already running
) else (
    echo Starting Ollama...
    start /B ollama serve
    timeout /t 5 >nul
)

REM Check for Go service binaries
echo.
echo [3/4] === PREPARING GO SERVICES ===
echo.

REM Enhanced RAG Service
if exist "..\go-microservice\cmd\enhanced-rag\enhanced-rag.exe" (
    echo ✅ Enhanced RAG binary found
) else (
    echo Building Enhanced RAG service...
    pushd ..\go-microservice
    go build -o .\cmd\enhanced-rag\enhanced-rag.exe .\cmd\enhanced-rag
    popd
)

REM Upload Service
if exist "..\go-microservice\cmd\upload-service\upload-service.exe" (
    echo ✅ Upload Service binary found  
) else (
    echo Building Upload Service...
    pushd ..\go-microservice
    go build -o .\cmd\upload-service\upload-service.exe .\cmd\upload-service
    popd
)

echo.
echo [4/4] === NATIVE WINDOWS STARTUP SEQUENCE ===
echo.

REM Set Windows environment variables
set NODE_ENV=development
set VITE_DEV=true
set NATIVE_WINDOWS=true

echo Starting Go services...
start /B "..\go-microservice\cmd\enhanced-rag\enhanced-rag.exe"
timeout /t 2 >nul
start /B "..\go-microservice\cmd\upload-service\upload-service.exe"
timeout /t 2 >nul

echo Starting SvelteKit in native Windows mode...
cd sveltekit-frontend

REM Use native Windows npm (not WSL)
call npm run dev

echo.
echo ===============================================================================
echo STARTUP COMPLETE - CHECK RESULTS
echo ===============================================================================

pause