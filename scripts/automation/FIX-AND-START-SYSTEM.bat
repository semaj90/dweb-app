@echo off
:: Load Environment Variables and Fix System Integration Issues

echo 🔧 Loading Environment Configuration...
echo ================================================

:: Load .env file variables
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" (
        set "%%a=%%b"
    )
)

echo ✅ Environment variables loaded from .env

:: Set PostgreSQL password for commands
set PGPASSWORD=123456

:: Test database connection
echo.
echo 🔍 Testing database connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -c "SELECT version();" >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo ✅ Database connection successful
) else (
    echo ❌ Database connection failed
    echo.
    echo Attempting to fix authentication...
    
    :: Try to set password using postgres user
    set PGPASSWORD=postgres
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "ALTER USER legal_admin PASSWORD '123456';" >nul 2>&1
    
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Password reset successful
        set PGPASSWORD=123456
    ) else (
        echo ❌ Could not reset password. Run FIX-POSTGRES-ADMIN.bat as Administrator
        pause
        exit /b 1
    )
)

:: Apply database migration
echo.
echo 📊 Applying database migrations...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -f database\gpu-schema-migration.sql >nul 2>&1

if %ERRORLEVEL% EQU 0 (
    echo ✅ Database migrations applied
) else (
    echo ⚠️  Migration may already be applied or partially failed
)

:: Start Redis if not running
echo.
echo 🔧 Starting Redis...
redis-cli ping >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    start /B redis-windows\redis-server.exe redis-windows\redis.conf
    timeout /t 3 /nobreak >nul
    echo ✅ Redis started
) else (
    echo ✅ Redis already running
)

:: Build and start Go services with proper environment
echo.
echo 🚀 Starting GPU services...

cd go-microservice

:: Ensure Go modules are up to date
go mod tidy >nul 2>&1

:: Build GPU processor if not exists
if not exist legal-processor-gpu.exe (
    echo    Building GPU processor...
    go build -tags=cgo -ldflags="-s -w" -o legal-processor-gpu.exe legal_processor_gpu_simd.go
)

:: Build enhanced processor if not exists
if not exist legal-processor-enhanced.exe (
    echo    Building enhanced processor...
    go build -tags=cgo -ldflags="-s -w" -o legal-processor-enhanced.exe enhanced_legal_processor.go
)

:: Build auto-indexer if not exists
if not exist auto-indexer.exe (
    echo    Building auto-indexer...
    go build -tags=cgo -o auto-indexer.exe auto-indexer-service.go
)

:: Start services if not running
tasklist /FI "IMAGENAME eq legal-processor-gpu.exe" 2>nul | find /I /N "legal-processor-gpu.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    start "GPU Processor" /B legal-processor-enhanced.exe
    echo ✅ GPU processor started
)

tasklist /FI "IMAGENAME eq auto-indexer.exe" 2>nul | find /I /N "auto-indexer.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    start "Auto Indexer" /B auto-indexer.exe
    echo ✅ Auto-indexer started
)

cd ..

:: Start BullMQ producer
echo.
echo 🔧 Starting BullMQ producer...
start "BullMQ" /B node backend\bullmq-producer.mjs

:: Wait for services to initialize
timeout /t 5 /nobreak >nul

:: Run system check
echo.
echo 📊 Running system integration check...
echo ================================================
node check-system-integration.mjs

echo.
echo 🎯 System Status:
echo ================================================
echo Environment: Loaded from .env
echo Database: legal_ai_db (legal_admin/123456)
echo GPU Service: http://localhost:8080
echo Auto-Indexer: http://localhost:8081
echo Redis: localhost:6379
echo.
echo Next: npm run dev (to start frontend)
echo.
pause
