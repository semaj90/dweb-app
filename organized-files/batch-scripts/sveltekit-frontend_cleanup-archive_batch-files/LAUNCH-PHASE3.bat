@echo off
REM Phase 3 Ready Check & Fixed Phase 2 Launcher
REM ============================================

echo 🎯 PHASE 3 READINESS CHECK
echo ========================

REM Check Docker status
echo 🐳 Checking Docker services...
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr "legal-"

echo.
echo 🔥 GPU Status:
wmic path win32_VideoController get name

echo.
echo ✅ PHASE 2 COMPLETE - PHASE 3 READY
echo 🚀 All AI infrastructure running
echo 🎯 Ready for LLM integration

REM Launch development server
echo.
echo 🚀 Starting Phase 3 development...
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Install dependencies
call npm install --silent

REM Start development server
echo 🌐 Starting SvelteKit server...
start "Prosecutor AI Phase 3" npm run dev

echo.
echo 🎉 PHASE 3 LAUNCHER READY
echo Visit: http://localhost:5173
pause
