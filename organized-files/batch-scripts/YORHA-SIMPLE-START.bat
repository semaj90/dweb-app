@echo off
REM =============================================================================
REM YoRHa Interface - Simple Direct Startup
REM Bypasses complex launch sequences and gets the interface running quickly
REM =============================================================================

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                                                              ║
echo ║  🎮 YoRHa LEGAL AI INTERFACE - DIRECT START                  ║
echo ║                                                              ║
echo ║  ▼ Bypassing WSL issues...                                   ║
echo ║  ▼ Native Windows execution...                               ║
echo ║  ▼ Glory to Mankind                                         ║
echo ║                                                              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Force native Windows execution
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo [1/3] FIXING ROLLUP DEPENDENCY ISSUE
echo =====================================
echo.
echo The issue is WSL trying to use Linux rollup binaries on Windows.
echo Fixing with native Windows dependency installation...
echo.

REM Clean and reinstall with Windows-specific options
if exist "node_modules" rmdir /s /q "node_modules" 2>nul
if exist "package-lock.json" del "package-lock.json" 2>nul

echo Installing Windows-native dependencies...
call npm install --platform=win32 --arch=x64 --ignore-engines
if %errorlevel% neq 0 (
    echo.
    echo Trying alternative installation method...
    call npm install --force --no-optional
)

echo.
echo [2/3] CHECKING REQUIRED SERVICES  
echo =====================================
echo.

REM Quick service check (non-blocking)
echo Checking Ollama AI service...
curl -s -m 3 http://localhost:11434 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama: Running
) else (
    echo ⚠️ Ollama: Not running ^(will use mock responses^)
    echo   To start: ollama serve
)

echo.
echo Checking Go microservices...
curl -s -m 3 http://localhost:8094 >nul 2>&1  
if %errorlevel% == 0 (
    echo ✅ Enhanced RAG: Running ^(port 8094^)
) else (
    echo ⚠️ Enhanced RAG: Not running ^(will use mock responses^)
)

curl -s -m 3 http://localhost:8093 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Upload Service: Running ^(port 8093^)  
) else (
    echo ⚠️ Upload Service: Not running ^(will use mock responses^)
)

echo.
echo [3/3] STARTING YoRHa INTERFACE
echo =====================================
echo.

REM Set environment for native Windows
set NODE_ENV=development
set VITE_HOST=localhost
set VITE_PORT=5173
set FORCE_COLOR=1

echo 🎮 Starting YoRHa Interface on Windows...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   🤖 YoRHa Legal AI Interface
echo   🌐 http://localhost:5173
echo   🎯 YoRHa Homepage: http://localhost:5173/yorha-home  
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

REM Use direct vite instead of complex npm scripts
echo Starting development server...
call npm run dev
if %errorlevel% neq 0 (
    echo.
    echo Trying direct Vite startup...
    call npx vite dev --host localhost --port 5173
)

echo.
echo ===============================================================================
echo YoRHa Interface startup complete!
echo.
echo 🎯 Next Steps:
echo   1. Open: http://localhost:5173/yorha-home
echo   2. Click the API integration buttons
echo   3. Test YoRHaTable and YoRHaCommandCenter components
echo.
echo Note: Backend services will use mock data if not running.
echo       Full integration available when Go services are started.
echo ===============================================================================
echo.

pause