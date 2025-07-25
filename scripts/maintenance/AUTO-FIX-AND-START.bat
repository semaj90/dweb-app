@echo off
echo.
echo 🔧 Enhanced Legal AI - Auto Fix and Start
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\web-app"

echo 🛠️ Step 1: Running comprehensive error fixer...
echo.
powershell -ExecutionPolicy Bypass -File "fix-all-errors.ps1"

echo.
echo 🚀 Step 2: Starting Enhanced Legal AI System...
echo.
powershell -ExecutionPolicy Bypass -File "start-lowmem-legal-ai.ps1"

echo.
echo ✅ Complete! Check above for any errors.
pause
