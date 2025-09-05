@echo off
echo 🚀 Smart Legal AI System Startup (Optimized for Current Setup)
echo ================================================================

:: Set environment variables
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set OLLAMA_URL=http://localhost:11434
set REDIS_URL=redis://localhost:6379
set PORT=5173
set NODE_OPTIONS=--max-old-space-size=8192

echo 📊 Current System Status:
echo ========================

:: Check PostgreSQL
echo PostgreSQL:
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();" --quiet >nul 2>&1
if %ERRORLEVEL%==0 (
    echo ✅ PostgreSQL: Connected
) else (
    echo ❌ PostgreSQL: Not connected - please start PostgreSQL service
    pause
    exit /b 1
)

:: Check Ollama
echo Ollama:
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL%==0 (
    echo ✅ Ollama: Running
) else (
    echo ❌ Ollama: Not running - starting...
    start /b ollama serve
    timeout /t 3 >nul
)

:: Check Redis
echo Redis:
redis-cli ping >nul 2>&1
if %ERRORLEVEL%==0 (
    echo ✅ Redis: Running
    set REDIS_AVAILABLE=true
) else (
    echo ⚠️ Redis: Not running - system will work without caching
    set REDIS_AVAILABLE=false
)

echo.
echo 🔧 Starting Services:
echo ====================

:: Start XState Manager (Real-time LLM Training)
echo 🧠 Starting XState Manager...
cd go-microservice
start /b cmd /c "xstate-manager.exe > xstate-manager.log 2>&1"
cd ..
timeout /t 2 >nul

:: Start Enhanced RAG Service (if available)
if exist "go-microservice\enhanced-rag.exe" (
    echo 🤖 Starting Enhanced RAG Service...
    cd go-microservice
    start /b cmd /c "enhanced-rag.exe > enhanced-rag.log 2>&1"
    cd ..
    timeout /t 2 >nul
)

:: Start Node.js development server
echo 🌐 Starting SvelteKit Development Server...
start /b cmd /c "npm run dev > sveltekit.log 2>&1"
timeout /t 3 >nul

echo.
echo 🎯 System Ready!
echo ===============
echo 🌐 Frontend: http://localhost:5173
echo 🧠 XState Manager: http://localhost:8095/health
echo 🤖 Enhanced RAG: http://localhost:8080/health (if available)
echo 📊 Analytics: http://localhost:8095/api/learning-analytics
echo 🔗 WebSocket: ws://localhost:8095/ws?userId=demo

if "%REDIS_AVAILABLE%"=="false" (
    echo.
    echo 💡 Optimization Tip: Install Redis for better performance
    echo    Download: https://github.com/tporadowski/redis/releases
    echo    Or run: winget install Redis.Redis
)

echo.
echo 📝 Logs available in:
echo   - go-microservice\xstate-manager.log
echo   - go-microservice\enhanced-rag.log (if available)
echo   - sveltekit.log
echo.

:: Wait for user input
echo Press any key to open system dashboard...
pause >nul

:: Open browser tabs
start http://localhost:5173
start http://localhost:8095/health
start http://localhost:8095/api/learning-analytics

echo ✅ Legal AI System is running optimally!