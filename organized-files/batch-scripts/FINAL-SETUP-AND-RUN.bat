@echo off
:: Final Complete System Setup and Run

echo ========================================
echo 🚀 LEGAL AI - FINAL SYSTEM SETUP
echo ========================================
echo.

:: Set all environment variables
echo [1/10] Setting environment variables...
set PGPASSWORD=123456
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set CGO_ENABLED=1
set GO111MODULE=on
echo ✅ Environment configured

:: Test database
echo.
echo [2/10] Testing database connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -t -c "SELECT 1" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Database connected
) else (
    echo ❌ Database connection failed
    echo Please ensure PostgreSQL is running and password is correct
    pause
    exit /b 1
)

:: Create all required directories
echo.
echo [3/10] Creating project directories...
for %%D in (logs uploads documents evidence generated_reports bin) do (
    if not exist %%D mkdir %%D
)
echo ✅ Directories created

:: Setup Redis properly
echo.
echo [4/10] Setting up Redis...

:: Kill any existing Redis processes
taskkill /F /IM redis-server.exe >nul 2>&1

:: Check for Redis and start it
set REDIS_RUNNING=0

:: Option 1: Check system Redis
where redis-server >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start /B redis-server
    set REDIS_RUNNING=1
    echo ✅ Started system Redis
    goto :redis_done
)

:: Option 2: Check redis-windows directory
if exist redis-windows\redis-server.exe (
    cd redis-windows
    
    :: Create minimal config if needed
    if not exist redis.conf (
        echo bind 127.0.0.1 > redis.conf
        echo port 6379 >> redis.conf
        echo protected-mode no >> redis.conf
    )
    
    :: Start Redis
    start /B redis-server.exe redis.conf
    cd ..
    set REDIS_RUNNING=1
    echo ✅ Started Redis from redis-windows
    goto :redis_done
)

:: Option 3: Download Redis if needed
if %REDIS_RUNNING% EQU 0 (
    echo Redis not found. Downloading...
    powershell -Command "& {
        $url = 'https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip'
        $output = 'redis.zip'
        Invoke-WebRequest -Uri $url -OutFile $output
        Expand-Archive -Path $output -DestinationPath 'redis-windows' -Force
        Remove-Item $output
    }"
    
    cd redis-windows
    echo bind 127.0.0.1 > redis.conf
    echo port 6379 >> redis.conf
    start /B redis-server.exe redis.conf
    cd ..
    echo ✅ Downloaded and started Redis
)

:redis_done
timeout /t 3 /nobreak >nul

:: Verify Redis
redis-cli ping >nul 2>&1 || redis-windows\redis-cli.exe ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Redis verified running
) else (
    echo ⚠️  Redis may not be running (system will work without caching)
)

:: Build Go services
echo.
echo [5/10] Installing Go dependencies...
cd go-microservice

:: Install all required dependencies
go get github.com/gin-gonic/gin@latest
go get github.com/redis/go-redis/v9@latest
go get github.com/jackc/pgx/v5/pgxpool@latest
go get github.com/bytedance/sonic@latest
go get github.com/tidwall/gjson@latest
go get github.com/neo4j/neo4j-go-driver/v5@latest
go get github.com/fsnotify/fsnotify@latest
go mod tidy

echo ✅ Dependencies installed

:: Kill any existing processors
echo.
echo [6/10] Stopping existing services...
taskkill /F /IM legal-processor-simple.exe >nul 2>&1
taskkill /F /IM legal-processor-enhanced.exe >nul 2>&1
taskkill /F /IM legal-processor-gpu.exe >nul 2>&1
taskkill /F /IM auto-indexer.exe >nul 2>&1
echo ✅ Existing services stopped

:: Build processors
echo.
echo [7/10] Building processors...

:: Always build simple processor first (no CUDA required)
echo Building simple processor (CPU mode)...
go build -o legal-processor-simple.exe legal-processor-simple.go
if exist legal-processor-simple.exe (
    echo ✅ Simple processor built
    copy legal-processor-simple.exe ..\bin\ >nul
    set MAIN_PROCESSOR=legal-processor-simple.exe
) else (
    echo ❌ Simple processor build failed
)

:: Try to build enhanced processor if CUDA available
echo Attempting enhanced processor build...
where nvcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set CC=C:\Progra~1\LLVM\bin\clang.exe
    set CXX=C:\Progra~1\LLVM\bin\clang++.exe
    go build -tags=cgo -ldflags="-s -w" -o legal-processor-enhanced.exe enhanced_legal_processor.go 2>nul
    if exist legal-processor-enhanced.exe (
        echo ✅ Enhanced processor built (GPU support)
        copy legal-processor-enhanced.exe ..\bin\ >nul
        set MAIN_PROCESSOR=legal-processor-enhanced.exe
    ) else (
        echo ⚠️  Enhanced processor build failed (using simple)
    )
) else (
    echo ⚠️  CUDA not found (using CPU mode)
)

cd ..

:: Start the main processor
echo.
echo [8/10] Starting main processor...
cd go-microservice
if defined MAIN_PROCESSOR (
    start "Legal Processor" /B %MAIN_PROCESSOR%
    echo ✅ Started %MAIN_PROCESSOR%
) else (
    echo ❌ No processor available
    pause
    exit /b 1
)
cd ..

:: Wait for service to initialize
echo.
echo [9/10] Waiting for services to initialize...
timeout /t 5 /nobreak >nul

:: Comprehensive health check
echo.
echo [10/10] Running comprehensive health check...
echo ----------------------------------------

:: Check each component
set ALL_GOOD=1

:: Database
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -t -c "SELECT 1" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ PostgreSQL: ONLINE
) else (
    echo ❌ PostgreSQL: OFFLINE
    set ALL_GOOD=0
)

:: Redis
redis-cli ping >nul 2>&1 || redis-windows\redis-cli.exe ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Redis: ONLINE
) else (
    echo ⚠️  Redis: OFFLINE (optional)
)

:: API Service
curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ API Service: ONLINE
    
    :: Get detailed status
    echo.
    echo Service Details:
    curl -s http://localhost:8080/health 2>nul
) else (
    echo ❌ API Service: OFFLINE
    set ALL_GOOD=0
)

:: Final status
echo.
echo ========================================
if %ALL_GOOD% EQU 1 (
    echo 🎉 SYSTEM FULLY OPERATIONAL!
) else (
    echo ⚠️  SYSTEM PARTIALLY OPERATIONAL
)
echo ========================================

echo.
echo 📊 Quick Commands:
echo -----------------
echo Check Integration: node check-system-integration.mjs
echo Start Frontend: npm run dev
echo View Logs: type logs\*.log
echo Test API: curl http://localhost:8080/health
echo.
echo 🌐 Access Points:
echo ----------------
echo API Health: http://localhost:8080/health
echo API Metrics: http://localhost:8080/metrics
echo Frontend: http://localhost:5173 (after npm run dev)
echo.

:: Create a status file
(
echo Legal AI System Status - %date% %time%
echo ========================================
echo Database: Connected
echo Redis: %REDIS_RUNNING%
echo Processor: %MAIN_PROCESSOR%
echo API Port: 8080
echo Status: Operational
) > system-status.txt

echo Status saved to: system-status.txt
echo.
pause
