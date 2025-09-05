@echo off
echo Starting Native Windows Legal AI System...
echo.

echo [1/6] Starting Windows Services...
echo   Starting PostgreSQL...
net start legal-ai-postgres >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ PostgreSQL started
) else (
    echo   ⚠️ PostgreSQL already running or failed to start
)

echo   Starting MinIO...
net start MinIO >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ MinIO started
) else (
    echo   ⚠️ MinIO already running or failed to start
)

echo   Starting Qdrant...
net start Qdrant >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ Qdrant started
) else (
    echo   ⚠️ Qdrant already running or failed to start
)

echo   Starting Redis...
net start Redis >nul 2>&1
if %errorlevel% equ 0 (
    echo   ✅ Redis started
) else (
    echo   ⚠️ Redis already running or failed to start
)

echo.
echo [2/6] Starting Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo   ✅ Ollama already running
) else (
    start "Ollama Server" /MIN ollama serve
    echo   ✅ Ollama started
)

echo.
echo [3/6] Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo.
echo [4/6] Checking service health...
echo   Checking PostgreSQL (5432)...
netstat -an | find "5432" >nul && echo   ✅ PostgreSQL ready || echo   ❌ PostgreSQL not ready

echo   Checking MinIO (9000)...
netstat -an | find "9000" >nul && echo   ✅ MinIO ready || echo   ❌ MinIO not ready

echo   Checking Qdrant (6333)...
netstat -an | find "6333" >nul && echo   ✅ Qdrant ready || echo   ❌ Qdrant not ready

echo   Checking Redis (6379)...
netstat -an | find "6379" >nul && echo   ✅ Redis ready || echo   ❌ Redis not ready

echo   Checking Ollama (11434)...
netstat -an | find "11434" >nul && echo   ✅ Ollama ready || echo   ❌ Ollama not ready

echo.
echo [5/6] Starting application services...

echo   Starting Go backend (port 8084)...
start "Legal AI Backend" cmd /k "title Legal AI Go Backend && go run main.go"

echo   Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo   Starting SvelteKit frontend (port 5173)...
start "Legal AI Frontend" cmd /k "title Legal AI Frontend && cd sveltekit-frontend && npm run dev"

echo.
echo [6/6] Opening browser...
timeout /t 8 /nobreak >nul
start http://localhost:5173

echo.
echo ✅ Native Windows Legal AI System Started!
echo.
echo 📋 Service URLs:
echo   • Frontend:        http://localhost:5173
echo   • Go Backend:      http://localhost:8084
echo   • MinIO Console:   http://localhost:9001
echo   • Qdrant API:      http://localhost:6333
echo   • PostgreSQL:      localhost:5432
echo.
echo 💡 Credentials:
echo   • PostgreSQL:      postgres / legal_ai_password_123
echo   • MinIO:           minioadmin / minioadmin123
echo.
echo 🔧 Management:
echo   • Stop System:     stop-native-system.bat
echo   • Check Status:    check-system-status.bat
echo.
echo Check the opened windows for service logs.
echo Press any key to continue...
pause >nul