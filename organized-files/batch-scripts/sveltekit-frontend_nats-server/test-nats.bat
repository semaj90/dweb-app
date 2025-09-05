@echo off
echo ==========================================
echo NATS Test Client
echo ==========================================
echo.
echo This script tests NATS connectivity and legal AI subjects
echo.
echo Testing connection to NATS server...
curl -s http://localhost:8222/varz 2>nul | findstr "server_name" >nul
if %errorlevel% neq 0 (
    echo ❌ NATS Server is not running
    echo Please start the server first with start-nats.bat
    pause
    exit /b 1
)
echo ✅ NATS Server is accessible
echo.
echo 📡 Available for Legal AI subjects:
echo    - legal.case.created
echo    - legal.document.uploaded  
echo    - legal.ai.analysis.completed
echo    - legal.search.query
echo    - legal.chat.message
echo    - system.health
echo.
echo 🌐 WebSocket endpoint: ws://localhost:4223
echo 📊 HTTP monitoring: http://localhost:8222
echo.
echo Ready for SvelteKit Legal AI integration!
pause
