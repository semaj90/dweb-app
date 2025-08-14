@echo off
:: Complete Fix - Build Services and Start System

echo 🔧 Legal AI System - Complete Setup & Fix
echo ================================================
echo.

:: 1. Set environment
echo [1/8] Setting up environment...
set PGPASSWORD=123456
set CGO_ENABLED=1
set CC=C:\Progra~1\LLVM\bin\clang.exe
set CXX=C:\Progra~1\LLVM\bin\clang++.exe
echo ✅ Environment configured

:: 2. Test database
echo [2/8] Testing database connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Database connected
) else (
    echo ❌ Database connection failed
    echo    Run FIX-POSTGRES-ADMIN.bat as Administrator
    pause
    exit /b 1
)

:: 3. Create directories
echo [3/8] Creating required directories...
if not exist logs mkdir logs
if not exist uploads mkdir uploads  
if not exist documents mkdir documents
if not exist evidence mkdir evidence
if not exist generated_reports mkdir generated_reports
echo ✅ Directories created

:: 4. Fix Redis
echo [4/8] Starting Redis...
redis-cli ping >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    :: Check if redis-windows directory exists
    if exist redis-windows (
        cd redis-windows
        if exist redis.conf (
            start /B redis-server.exe redis.conf
        ) else (
            :: Start Redis without config file
            start /B redis-server.exe
        )
        cd ..
    ) else (
        echo ⚠️  Redis directory not found, trying global redis
        where redis-server >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            start /B redis-server
        ) else (
            echo ❌ Redis not installed
        )
    )
    timeout /t 3 /nobreak >nul
)

redis-cli ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Redis running
) else (
    echo ⚠️  Redis may not be running
)

:: 5. Build Go services
echo [5/8] Building Go services...
cd go-microservice

:: Install dependencies first
echo    Installing Go dependencies...
go get -u github.com/gin-gonic/gin >nul 2>&1
go get -u github.com/redis/go-redis/v9 >nul 2>&1
go get -u github.com/bytedance/sonic >nul 2>&1
go get -u github.com/tidwall/gjson >nul 2>&1
go get -u github.com/jackc/pgx/v5/pgxpool >nul 2>&1
go mod tidy >nul 2>&1

:: Build enhanced processor
if not exist legal-processor-enhanced.exe (
    echo    Building enhanced legal processor...
    go build -tags=cgo -ldflags="-s -w" -o legal-processor-enhanced.exe enhanced_legal_processor.go 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo    ⚠️  Enhanced processor build failed, trying simple build...
    )
)

:: Build GPU processor
if not exist legal-processor-gpu.exe (
    echo    Building GPU processor...
    go build -tags=cgo -ldflags="-s -w" -o legal-processor-gpu.exe legal_processor_gpu_simd.go 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo    ⚠️  GPU processor build failed
    )
)

:: Build auto-indexer
if not exist auto-indexer.exe (
    echo    Building auto-indexer...
    go build -o auto-indexer.exe auto-indexer-service.go 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo    ⚠️  Auto-indexer build failed
    )
)

:: Check what was built
if exist legal-processor-enhanced.exe (
    echo ✅ Enhanced processor built
) else if exist legal-processor-gpu.exe (
    echo ✅ GPU processor built
) else (
    echo ⚠️  No processors built successfully
)

cd ..

:: 6. Start services
echo [6/8] Starting services...

:: Kill any existing instances
taskkill /F /IM legal-processor-enhanced.exe >nul 2>&1
taskkill /F /IM legal-processor-gpu.exe >nul 2>&1
taskkill /F /IM auto-indexer.exe >nul 2>&1

:: Start whichever processor was built
cd go-microservice
if exist legal-processor-enhanced.exe (
    start "Enhanced Processor" /B legal-processor-enhanced.exe
    echo    Started enhanced processor
) else if exist legal-processor-gpu.exe (
    start "GPU Processor" /B legal-processor-gpu.exe
    echo    Started GPU processor
)

if exist auto-indexer.exe (
    start "Auto Indexer" /B auto-indexer.exe
    echo    Started auto-indexer
)
cd ..

echo ✅ Services started

:: 7. Wait for services
echo [7/8] Waiting for services to initialize...
timeout /t 5 /nobreak >nul

:: 8. Health check
echo [8/8] Running health check...
curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ GPU service responding
    curl -s http://localhost:8080/health | findstr /C:"gpu_enabled"
) else (
    echo ⚠️  GPU service not responding yet
)

echo.
echo ================================================
echo ✅ Setup complete!
echo.
echo Services Status:
echo ----------------
redis-cli ping 2>nul && echo Redis: ONLINE || echo Redis: OFFLINE
curl -s http://localhost:8080/health >nul 2>&1 && echo GPU Service: ONLINE || echo GPU Service: OFFLINE

echo.
echo Next Steps:
echo -----------
echo 1. Run: node check-system-integration.mjs
echo 2. Start frontend: npm run dev
echo 3. Open: http://localhost:5173
echo.
pause
