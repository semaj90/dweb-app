@echo off
:: Complete System Fix with Proper Build Steps

echo =====================================
echo 🚀 LEGAL AI SYSTEM - COMPLETE FIX
echo =====================================
echo.

:: Step 1: Database Check
echo [Step 1/6] Checking Database...
echo ---------------------------------
set PGPASSWORD=123456
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -c "SELECT 'Database OK' as status;" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Database: Connected
) else (
    echo ❌ Database: Failed
    echo    Fix: Run FIX-POSTGRES-ADMIN.bat as Administrator
    pause
    exit /b 1
)

:: Step 2: Redis Setup
echo.
echo [Step 2/6] Setting up Redis...
echo ---------------------------------
call SETUP-REDIS.bat
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Redis setup needs attention
)

:: Step 3: Create Directories
echo.
echo [Step 3/6] Creating Directories...
echo ---------------------------------
for %%D in (logs uploads documents evidence generated_reports) do (
    if not exist %%D (
        mkdir %%D
        echo ✅ Created: %%D
    )
)

:: Step 4: Build Go Services
echo.
echo [Step 4/6] Building Go Services...
echo ---------------------------------
cd go-microservice

:: Set Go environment
set GO111MODULE=on
set CGO_ENABLED=1

:: Install dependencies
echo Installing dependencies...
go get -u github.com/gin-gonic/gin >nul 2>&1
go get -u github.com/redis/go-redis/v9 >nul 2>&1
go get -u github.com/jackc/pgx/v5/pgxpool >nul 2>&1
go get -u github.com/bytedance/sonic >nul 2>&1
go get -u github.com/tidwall/gjson >nul 2>&1
go mod tidy >nul 2>&1

:: Kill existing processes
taskkill /F /IM legal-processor-simple.exe >nul 2>&1
taskkill /F /IM legal-processor-enhanced.exe >nul 2>&1
taskkill /F /IM legal-processor-gpu.exe >nul 2>&1

:: Build simple processor (always works)
echo Building simple processor (fallback)...
go build -o legal-processor-simple.exe legal-processor-simple.go
if %ERRORLEVEL% EQU 0 (
    echo ✅ Simple processor built
    set PROCESSOR=legal-processor-simple.exe
) else (
    echo ❌ Simple processor build failed
)

:: Try to build enhanced processor
echo Building enhanced processor (with GPU)...
set CC=C:\Progra~1\LLVM\bin\clang.exe
set CXX=C:\Progra~1\LLVM\bin\clang++.exe
go build -tags=cgo -o legal-processor-enhanced.exe enhanced_legal_processor.go 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✅ Enhanced processor built
    set PROCESSOR=legal-processor-enhanced.exe
) else (
    echo ⚠️  Enhanced processor build failed (using simple)
)

cd ..

:: Step 5: Start Services
echo.
echo [Step 5/6] Starting Services...
echo ---------------------------------

cd go-microservice
if defined PROCESSOR (
    echo Starting %PROCESSOR%...
    start "Legal Processor" /B %PROCESSOR%
    echo ✅ Processor started
) else (
    echo ❌ No processor available
)
cd ..

:: Wait for services
timeout /t 5 /nobreak >nul

:: Step 6: Health Check
echo.
echo [Step 6/6] System Health Check...
echo ---------------------------------

:: Database
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Database: ONLINE
) else (
    echo ❌ Database: OFFLINE
)

:: Redis
redis-cli ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Redis: ONLINE
) else (
    if exist redis-windows\redis-cli.exe (
        redis-windows\redis-cli.exe ping >nul 2>&1
        if %ERRORLEVEL% EQU 0 (
            echo ✅ Redis: ONLINE
        ) else (
            echo ❌ Redis: OFFLINE
        )
    ) else (
        echo ❌ Redis: OFFLINE
    )
)

:: API Service
curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ API Service: ONLINE
    echo.
    echo Service Details:
    curl -s http://localhost:8080/health 2>nul | findstr /C:"status" /C:"gpu_enabled" /C:"redis_connected" /C:"db_connected"
) else (
    echo ❌ API Service: OFFLINE
)

echo.
echo =====================================
echo 📊 SYSTEM STATUS COMPLETE
echo =====================================
echo.
echo Next Steps:
echo -----------
echo 1. Run: node check-system-integration.mjs
echo 2. Start frontend: npm run dev
echo 3. Access: http://localhost:5173
echo.
echo API Endpoints:
echo -------------
echo Health: http://localhost:8080/health
echo Metrics: http://localhost:8080/metrics
echo.
pause
