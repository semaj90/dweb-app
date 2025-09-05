@echo off
REM Enterprise Vector Consumer Service v2.0 Startup Script - Native Windows
REM Production-ready service launcher with health monitoring

title Enterprise Vector Consumer Service v2.0 - Native Windows

echo ====================================
echo Enterprise Vector Consumer Service v2.0
echo Native Windows Deployment
echo Starting Production Services...
echo ====================================

REM Set default configuration
set SERVICE_PORT=8095
set DB_URL=postgres://legal_admin:123456@localhost:5432/legal_ai_db?sslmode=disable
set REDIS_URL=localhost:6379
set KRATOS_URL=http://localhost:4433
set CUDA_ENABLED=true
set LOG_LEVEL=info
set MAX_CONCURRENCY=1000

REM Check if executable exists
if not exist "bin\vector-consumer-v2.exe" (
    echo ERROR: Service executable not found!
    echo Please run build-vector-consumer-v2.bat first
    pause
    exit /b 1
)

REM Check PostgreSQL connection
echo Checking PostgreSQL connection...
psql %DB_URL% -c "SELECT 1;" > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: PostgreSQL connection failed
    echo Make sure PostgreSQL is running on localhost:5432
    echo Continuing anyway...
)

REM Check Redis connection
echo Checking Redis connection...
redis-cli -h localhost -p 6379 ping > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: Redis connection failed
    echo Make sure Redis is running on localhost:6379
    echo Continuing anyway...
)

REM Check CUDA availability
echo Checking CUDA availability...
nvidia-smi > nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo CUDA driver detected - GPU acceleration available
    set CUDA_ENABLED=true
) else (
    echo CUDA driver not found - falling back to CPU only
    set CUDA_ENABLED=false
)

echo ====================================
echo Service Configuration:
echo   Port: %SERVICE_PORT%
echo   Database: PostgreSQL (localhost:5432)
echo   Cache: Redis (localhost:6379)
echo   Auth: Kratos (localhost:4433)
echo   CUDA: %CUDA_ENABLED%
echo   Log Level: %LOG_LEVEL%
echo   Max Concurrency: %MAX_CONCURRENCY%
echo ====================================

REM Start the service with all enterprise features
echo Starting Enterprise Vector Consumer Service v2.0...
echo Press Ctrl+C to stop the service

bin\vector-consumer-v2.exe ^
    --port=%SERVICE_PORT% ^
    --db-url="%DB_URL%" ^
    --redis-url=%REDIS_URL% ^
    --kratos-url=%KRATOS_URL% ^
    --cuda=%CUDA_ENABLED% ^
    --log-level=%LOG_LEVEL% ^
    --max-concurrency=%MAX_CONCURRENCY%

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Service failed to start or exited with error
    echo Check the logs above for details
    pause
    exit /b 1
)

echo.
echo Service stopped gracefully
pause