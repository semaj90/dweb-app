@echo off
echo 🔄 Restarting Go Legal AI Server with Database Connection
echo =========================================================

echo 🛑 Stopping existing Go service...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8080') do (
    if not "%%p"=="0" (
        taskkill /f /pid %%p >nul 2>&1
        echo ✅ Killed process %%p
    )
)

timeout /t 2 >nul

echo 🚀 Starting Go Legal AI Server with database...
cd go-microservice

set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set PGPASSWORD=123456
set PORT=8080
set OLLAMA_URL=http://localhost:11434

echo 📊 Environment:
echo   DATABASE_URL: %DATABASE_URL%
echo   PORT: %PORT%
echo   OLLAMA_URL: %OLLAMA_URL%
echo.

start /B cmd /c "go run legal-ai-server.go > ../logs/go-service.log 2>&1"

timeout /t 3 >nul

echo 🧪 Testing service...
netstat -an | findstr :8080 >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Go service started on port 8080
    curl -s http://localhost:8080/health | findstr "database"
    echo.
    curl -s http://localhost:8080/database-status
) else (
    echo ❌ Go service failed to start
)

echo.
pause