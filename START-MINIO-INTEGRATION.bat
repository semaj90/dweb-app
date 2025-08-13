@echo off
REM MinIO + SvelteKit 2 + RAG Integration Startup Script
REM PyTorch-style caching: only downloads what's missing

setlocal enabledelayedexpansion

echo ================================================================
echo       MinIO + SvelteKit 2 + RAG Integration Startup
echo       With Smart Caching (PyTorch-style)
echo ================================================================
echo.

REM Colors for output
for /f %%A in ('echo prompt $E ^| cmd') do set "ESC=%%A"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "RED=%ESC%[31m"
set "CYAN=%ESC%[36m"
set "MAGENTA=%ESC%[35m"
set "RESET=%ESC%[0m"

echo %CYAN%🚀 Starting with smart caching...%RESET%
echo.

REM Run cache manager first
echo %MAGENTA%📦 Running cache manager (PyTorch-style)...%RESET%
powershell -ExecutionPolicy Bypass -File "scripts\cache-manager.ps1"
if %errorlevel% neq 0 (
    echo %RED%❌ Cache manager failed!%RESET%
    pause
    exit /b 1
)

REM Load cache configuration
if exist "%USERPROFILE%\.legal-ai-cache\config.json" (
    echo %GREEN%✅ Cache configuration loaded%RESET%
    for /f "delims=" %%i in ('powershell -Command "(Get-Content '%USERPROFILE%\.legal-ai-cache\config.json' | ConvertFrom-Json).cache_dir"') do set CACHE_DIR=%%i
    for /f "delims=" %%i in ('powershell -Command "(Get-Content '%USERPROFILE%\.legal-ai-cache\config.json' | ConvertFrom-Json).minio_path"') do set MINIO_PATH=%%i
) else (
    echo %YELLOW%⚠️  Using default paths (cache not available)%RESET%
    set CACHE_DIR=%USERPROFILE%\.legal-ai-cache
    set MINIO_PATH=%CACHE_DIR%\binaries\minio.exe
)

echo %CYAN%📁 Cache directory: %CACHE_DIR%%RESET%
echo.

REM Check if MinIO is available (from cache or system)
echo %YELLOW%📦 Checking MinIO installation...%RESET%
if exist "%MINIO_PATH%" (
    echo %GREEN%✅ MinIO found in cache: %MINIO_PATH%%RESET%
) else (
    where minio >nul 2>&1
    if %errorlevel% neq 0 (
        echo %RED%❌ MinIO not found! Cache manager should have downloaded it.%RESET%
        echo %YELLOW%💡 Try running the cache manager again%RESET%
        echo.
        pause
        exit /b 1
    )
    echo %GREEN%✅ MinIO found in system PATH%RESET%
    set MINIO_PATH=minio
)

REM Check if Go is installed
echo %YELLOW%📦 Checking Go installation...%RESET%
where go >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Go not found! Please install Go first.%RESET%
    echo %YELLOW%💡 Download from: https://golang.org/dl/%RESET%
    echo.
    pause
    exit /b 1
)
echo %GREEN%✅ Go found%RESET%

REM Check if Node.js is installed
echo %YELLOW%📦 Checking Node.js installation...%RESET%
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Node.js not found! Please install Node.js first.%RESET%
    echo %YELLOW%💡 Download from: https://nodejs.org/%RESET%
    echo.
    pause
    exit /b 1
)
echo %GREEN%✅ Node.js found%RESET%

REM Check if PostgreSQL is running
echo %YELLOW%📦 Checking PostgreSQL...%RESET%
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT version();" >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ PostgreSQL not accessible! Please start PostgreSQL first.%RESET%
    echo %YELLOW%💡 Check if PostgreSQL service is running%RESET%
    echo.
    pause
    exit /b 1
)
echo %GREEN%✅ PostgreSQL accessible%RESET%

echo.
echo %CYAN%🏁 Starting services in parallel...%RESET%
echo.

REM Set environment variables (including cache paths)
set MINIO_ROOT_USER=minioadmin
set MINIO_ROOT_PASSWORD=minioadmin
set DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db
set PG_CONN_STRING=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db
set MINIO_ENDPOINT=localhost:9000
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin
set MINIO_BUCKET=legal-documents
set MINIO_USE_SSL=false
set UPLOAD_SERVICE_PORT=8094
set RAG_QUIC_PORT=8092
set RAG_HTTP_PORT=8093
set OLLAMA_BASE_URL=http://localhost:11434
set EMBED_MODEL=nomic-embed-text

REM Cache environment variables (set by cache manager)
REM GOMODCACHE already set by cache manager
REM npm cache already configured by cache manager

REM Start MinIO Server (using cached or system binary)
echo %YELLOW%1️⃣ Starting MinIO Server on port 9000...%RESET%
if not exist "C:\minio-data" mkdir "C:\minio-data"
start /B "MinIO Server" cmd /c ""%MINIO_PATH%" server C:\minio-data --console-address :9001 > minio.log 2>&1"
timeout /t 3 >nul

REM Start Ollama (if not running)
echo %YELLOW%2️⃣ Starting Ollama service...%RESET%
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    start /B "Ollama" cmd /c "ollama serve > ollama.log 2>&1"
    timeout /t 3 >nul
    echo %GREEN%✅ Ollama started%RESET%
) else (
    echo %GREEN%✅ Ollama already running%RESET%
)

REM Pull embedding model if needed
echo %YELLOW%3️⃣ Checking embedding model...%RESET%
ollama list | findstr "nomic-embed-text" >nul
if %errorlevel% neq 0 (
    echo %YELLOW%📥 Pulling nomic-embed-text model...%RESET%
    ollama pull nomic-embed-text
    echo %GREEN%✅ Model downloaded%RESET%
) else (
    echo %GREEN%✅ Embedding model available%RESET%
)

REM Install Go dependencies and start RAG service (using cached modules)
echo %YELLOW%4️⃣ Starting RAG Kratos service on port 8093...%RESET%
cd go-microservice
echo %CYAN%Using Go module cache: %GOMODCACHE%%RESET%
go mod tidy >nul 2>&1
start /B "RAG Service" cmd /c "go run cmd/rag-kratos/main.go > ../rag-service.log 2>&1"
cd ..
timeout /t 2 >nul

REM Start Upload service
echo %YELLOW%5️⃣ Starting Upload service on port 8094...%RESET%
cd go-microservice
start /B "Upload Service" cmd /c "go run cmd/upload-service/main.go > ../upload-service.log 2>&1"
cd ..
timeout /t 2 >nul

REM Install npm dependencies and start SvelteKit (using cached packages)
echo %YELLOW%6️⃣ Starting SvelteKit frontend on port 5173...%RESET%
cd sveltekit-frontend
if not exist node_modules (
    echo %YELLOW%📦 Installing npm dependencies (using cache)...%RESET%
    npm install
) else (
    echo %GREEN%✅ Node modules already exist%RESET%
)
start /B "SvelteKit" cmd /c "npm run dev > ../sveltekit.log 2>&1"
cd ..

echo.
echo %GREEN%🎉 All services started successfully!%RESET%
echo.
echo %CYAN%📊 Service Status:%RESET%
echo   • MinIO Server:      http://localhost:9000 (Console: http://localhost:9001)
echo   • MinIO Credentials: minioadmin / minioadmin
echo   • RAG Service:       http://localhost:8093 (/embed, /rag, /health)
echo   • Upload Service:    http://localhost:8094
echo   • SvelteKit App:     http://localhost:5173
echo   • Ollama:           http://localhost:11434
echo   • PostgreSQL:       localhost:5432 (legal_ai_db)
echo.
echo %YELLOW%📋 Next Steps:%RESET%
echo   1. Open http://localhost:5173/upload to test file uploads
echo   2. Check MinIO console at http://localhost:9001
echo   3. Monitor logs in: minio.log, rag-service.log, upload-service.log, sveltekit.log
echo.
echo %CYAN%🔍 Health Checks:%RESET%
timeout /t 5 >nul

REM Health check function
:healthcheck
echo %YELLOW%Checking service health...%RESET%

REM Check MinIO
curl -s http://localhost:9000/minio/health/live >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ MinIO: Healthy%RESET%
) else (
    echo %RED%❌ MinIO: Not responding%RESET%
)

REM Check RAG Service
curl -s http://localhost:8092/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ RAG Service: Healthy%RESET%
) else (
    echo %RED%❌ RAG Service: Not responding%RESET%
)

REM Check Upload Service
curl -s http://localhost:8094/health >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ Upload Service: Healthy%RESET%
) else (
    echo %RED%❌ Upload Service: Not responding%RESET%
)

REM Check SvelteKit
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo %GREEN%✅ SvelteKit: Healthy%RESET%
) else (
    echo %RED%❌ SvelteKit: Not responding%RESET%
)

echo.
echo %GREEN%🌟 Integration ready! Happy coding! 🌟%RESET%
echo.
echo %YELLOW%💡 Pro Tips:%RESET%
echo   • Use Ctrl+C to stop services
echo   • Check logs if services fail to start
echo   • Ensure PostgreSQL pgvector extension is installed
echo   • Visit http://localhost:5173/upload to test the integration
echo.

REM Keep the window open
echo %CYAN%Press any key to open the application in browser...%RESET%
pause >nul
start http://localhost:5173/upload

REM Wait for user input to close
echo.
echo %YELLOW%Press any key to stop all services and exit...%RESET%
pause >nul

REM Cleanup
echo %YELLOW%🧹 Stopping all services...%RESET%
taskkill /F /IM minio.exe >nul 2>&1
taskkill /F /IM ollama.exe >nul 2>&1
taskkill /F /IM go.exe >nul 2>&1
taskkill /F /IM node.exe >nul 2>&1

echo %GREEN%✅ All services stopped%RESET%
echo %CYAN%👋 Goodbye!%RESET%

endlocal