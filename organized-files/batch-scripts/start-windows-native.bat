@echo off
echo 🚀 Starting Legal AI System - Windows Native Development
echo =======================================================
echo.

echo 🔍 Checking Windows native services...
echo.

REM Check PostgreSQL
echo 📊 PostgreSQL Status:
netstat -an | findstr :5432 >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ PostgreSQL is running on port 5432
) else (
    echo ❌ PostgreSQL not detected - Please start PostgreSQL service
    echo    Run: net start postgresql-x64-17
)

REM Check Ollama
echo.
echo 🧠 Ollama Status:
netstat -an | findstr :11434 >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Ollama is running on port 11434
) else (
    echo ❌ Ollama not detected - Please start Ollama
    echo    Run: ollama serve
)

REM Check for Go microservice
echo.
echo 🔧 Go Microservice Status:
netstat -an | findstr :8080 >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Go microservice is running on port 8080
) else (
    echo ⚠️  Go microservice not running - Will start it
)

echo.
echo 🎯 Starting Legal AI Components...
echo.

REM Start Go microservice in background
echo Starting Go Legal AI Server...
start /B cmd /c "cd go-microservice && set PORT=8080 && set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db && go run legal-ai-server.go"

timeout /t 3 >nul

REM Check if Go service started
netstat -an | findstr :8080 >nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Go microservice started successfully
) else (
    echo ⚠️  Go microservice may still be starting...
)

echo.
echo 🌐 Starting SvelteKit Development Server...
echo.

REM Start SvelteKit dev server
cd sveltekit-frontend
start cmd /k "npm run dev"

echo.
echo 🎉 Windows Native Legal AI System Started!
echo.
echo 📋 Available Services:
echo ┌─────────────────────────────────────────┐
echo │ Service          │ URL                  │
echo ├─────────────────────────────────────────┤
echo │ SvelteKit Frontend│ http://localhost:5173│
echo │ Go Legal AI API  │ http://localhost:8080│  
echo │ Ollama AI        │ http://localhost:11434│
echo │ PostgreSQL DB    │ localhost:5432       │
echo └─────────────────────────────────────────┘
echo.
echo 🧪 Test Commands:
echo curl http://localhost:8080/health
echo curl http://localhost:5173
echo curl http://localhost:11434/api/tags
echo.
echo 📝 Logs available in respective terminal windows
echo Press any key to return to command prompt...
pause >nul