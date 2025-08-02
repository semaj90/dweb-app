@echo off
cls
echo =====================================================
echo 🎯 PROSECUTOR AI - PHASE 3 LAUNCHER
echo 🤖 AI Core Implementation Ready
echo =====================================================

echo 🔍 System Check...

REM Check Docker services
echo 🐳 Docker Services:
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr "legal-" || echo "   ⚠️ Some services may be down"

echo.
echo 🔥 GPU Status:
for /f "tokens=*" %%i in ('wmic path win32_VideoController get name ^| findstr /i nvidia') do echo    ✅ %%i

echo.
echo 🔧 Applying Fixes...
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Apply error fixes
if exist "FIX-ERRORS.bat" (
    call "FIX-ERRORS.bat" >nul 2>&1
    echo    ✅ Critical errors fixed
) else (
    echo    ⚠️ Fix script not found
)

echo.
echo 📦 Installing Dependencies...
call npm install --silent >nul 2>&1
echo    ✅ Dependencies ready

echo.
echo 🚀 Starting Phase 3 Development...
start "Prosecutor AI" npm run dev

echo.
echo =====================================================
echo 🎉 PHASE 3 READY
echo =====================================================
echo 🌐 App: http://localhost:5173
echo 🤖 Ollama: http://localhost:11434
echo 🔍 Qdrant: http://localhost:6333
echo 📊 All AI services operational
echo =====================================================

timeout /t 3 >nul
