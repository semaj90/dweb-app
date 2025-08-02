@echo off
cls
echo.
echo 🚀 AI AGENT STACK - PRODUCTION LAUNCHER
echo =====================================
echo.

REM Check if in correct directory
if not exist "package.json" (
    echo ❌ Error: package.json not found
    echo Please run this script from the sveltekit-frontend directory
    pause
    exit /b 1
)

echo 📋 Pre-flight Check...
echo.

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found. Please install Node.js 18+
    pause
    exit /b 1
) else (
    echo ✅ Node.js detected
)

REM Check npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm not found
    pause
    exit /b 1
) else (
    echo ✅ npm detected
)

echo.
echo 🤖 Starting AI Services...
echo.

REM Check if Ollama is running
powershell -Command "try { Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 3 | Out-Null; Write-Host '✅ Ollama is running' } catch { Write-Host '⚠️  Ollama not detected - please start with: ollama serve' }"

echo.
echo 📦 Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ❌ npm install failed
    pause
    exit /b 1
)

echo.
echo 🏗️  Building application...
call npm run build
if %errorlevel% neq 0 (
    echo ⚠️  Build issues detected, but continuing...
)

echo.
echo 🌐 Starting Development Server...
echo.
echo 📱 Your AI Agent will be available at:
echo    👉 Main Chat: http://localhost:5173
echo    👉 Test Page: http://localhost:5173/test
echo    👉 API Health: http://localhost:5173/api/ai/health
echo.
echo 💡 Tips:
echo    - Make sure Ollama is running: ollama serve
echo    - Pull models if needed: ollama pull gemma2:2b
echo    - Press Ctrl+C to stop the server
echo.

REM Start the development server
call npm run dev

echo.
echo 👋 Server stopped. Press any key to exit...
pause >nul
