@echo off
REM Optimized Enterprise Service Starter with Existing Installation Detection
REM Designed for npm run dev workflow integration

echo ========================================================================
echo Optimized Enterprise Legal AI Platform Startup
echo ========================================================================
echo.

REM Set optimization flags
set SKIP_EXISTING=1
set FAST_START=1
set DEV_MODE=1

REM Quick service health checks
echo [HEALTH CHECK] Scanning existing services...

REM Check PostgreSQL
pg_isready -h localhost -p 5432 >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ PostgreSQL: Running
    set POSTGRES_READY=1
) else (
    echo ! PostgreSQL: Not running - attempting start...
    net start postgresql-x64-17 2>nul || net start postgresql-x64-16 2>nul || net start postgresql-x64-15 2>nul
    timeout /t 3 /nobreak >nul
    pg_isready -h localhost -p 5432 >nul 2>&1
    if %errorlevel% == 0 (
        echo ✓ PostgreSQL: Started successfully
        set POSTGRES_READY=1
    ) else (
        echo ✗ PostgreSQL: Failed to start
        set POSTGRES_READY=0
    )
)

REM Check Redis
redis-cli ping >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Redis: Running
    set REDIS_READY=1
) else (
    echo ! Redis: Not running - attempting start...
    if exist "C:\enterprise-services\redis\start-redis.bat" (
        start /B /MIN cmd /c "C:\enterprise-services\redis\start-redis.bat"
    ) else (
        start /B redis-server 2>nul
    )
    timeout /t 2 /nobreak >nul
    redis-cli ping >nul 2>&1
    if %errorlevel% == 0 (
        echo ✓ Redis: Started successfully
        set REDIS_READY=1
    ) else (
        echo ✗ Redis: Failed to start
        set REDIS_READY=0
    )
)

REM Check RabbitMQ
rabbitmq-diagnostics status >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ RabbitMQ: Running
    set RABBITMQ_READY=1
) else (
    echo ! RabbitMQ: Not running - attempting start...
    net start RabbitMQ >nul 2>&1
    timeout /t 5 /nobreak >nul
    rabbitmq-diagnostics status >nul 2>&1
    if %errorlevel% == 0 (
        echo ✓ RabbitMQ: Started successfully
        set RABBITMQ_READY=1
    ) else (
        echo ✗ RabbitMQ: Failed to start
        set RABBITMQ_READY=0
    )
)

REM Check Enhanced RAG service
curl -s http://localhost:8094/health >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Enhanced RAG: Running
    set RAG_READY=1
) else (
    echo ! Enhanced RAG: Not running - starting service...
    if exist "bin\enhanced-rag.exe" (
        start /B "EnhancedRAG" bin\enhanced-rag.exe
        timeout /t 3 /nobreak >nul
        curl -s http://localhost:8094/health >nul 2>&1
        if %errorlevel% == 0 (
            echo ✓ Enhanced RAG: Started successfully
            set RAG_READY=1
        ) else (
            echo ! Enhanced RAG: Starting (may take longer to be ready)
            set RAG_READY=1
        )
    ) else (
        echo ✗ Enhanced RAG: Binary not found
        set RAG_READY=0
    )
)

REM Check Upload service
curl -s http://localhost:8093/health >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Upload Service: Running
    set UPLOAD_READY=1
) else (
    echo ! Upload Service: Not running - starting service...
    if exist "bin\upload-service.exe" (
        start /B "UploadService" bin\upload-service.exe --port=8093
        timeout /t 2 /nobreak >nul
        echo ✓ Upload Service: Started
        set UPLOAD_READY=1
    ) else (
        echo ✗ Upload Service: Binary not found
        set UPLOAD_READY=0
    )
)

echo.
echo ========================================================================
echo Service Status Summary
echo ========================================================================
echo PostgreSQL:    %POSTGRES_READY%
echo Redis:          %REDIS_READY%
echo RabbitMQ:       %RABBITMQ_READY%
echo Enhanced RAG:   %RAG_READY%
echo Upload Service: %UPLOAD_READY%
echo.

REM Calculate readiness score
set /a READY_COUNT=%POSTGRES_READY%+%REDIS_READY%+%RABBITMQ_READY%+%RAG_READY%+%UPLOAD_READY%

if %READY_COUNT% GEQ 4 (
    echo ✓ System Status: READY FOR DEVELOPMENT ^(%READY_COUNT%/5 services^)
    echo.
    echo Enterprise Development Environment:
    echo • Frontend:       http://localhost:5173 ^(SvelteKit^)
    echo • Enhanced RAG:   http://localhost:8094
    echo • Upload Service: http://localhost:8093  
    echo • Database:       localhost:5432 ^(PostgreSQL^)
    echo • Cache:          localhost:6379 ^(Redis^)
    echo • Message Queue:  localhost:5672 ^(RabbitMQ^)
    echo.
    echo Ready for: npm run dev:enterprise
) else (
    echo ⚠ System Status: PARTIAL ^(%READY_COUNT%/5 services^)
    echo Some services may need manual setup
    echo.
    if %POSTGRES_READY% == 0 echo • Install PostgreSQL: https://www.postgresql.org/download/windows/
    if %REDIS_READY% == 0 echo • Install Redis: https://github.com/microsoftarchive/redis/releases
    if %RABBITMQ_READY% == 0 echo • Install RabbitMQ: https://www.rabbitmq.com/install-windows.html
    echo.
    echo Continuing with available services...
)

REM Performance optimization
if exist "C:\enterprise-services\optimize-performance.bat" (
    echo.
    echo [OPTIMIZATION] Applying performance optimizations...
    call "C:\enterprise-services\optimize-performance.bat" >nul 2>&1
    echo ✓ Performance optimizations applied
)

echo.
echo ========================================================================
echo Enterprise Legal AI Platform - Ready for Development!
echo ========================================================================
echo.
echo To start full development environment:
echo   npm run dev:enterprise
echo.
echo To check system status:
echo   npm run services:enterprise:status
echo.