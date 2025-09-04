@echo off
REM Enhanced Enterprise Setup with Optimization and Existing Installation Checks
REM Integrates with npm run dev workflow

echo ========================================================================
echo Enhanced Enterprise Vector Consumer Service Setup v2.0
echo ========================================================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Administrator privileges recommended for full setup
    echo Some features may not work without admin rights
    echo.
)

REM Set environment variables for optimization
set SETUP_SKIP_EXISTING=1
set SETUP_VERBOSE=1
set SETUP_OPTIMIZE=1

echo [PHASE 1] Checking existing installations...
echo ============================================

REM Check PostgreSQL
echo Checking PostgreSQL installation...
where psql >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ PostgreSQL found - checking version...
    psql --version 2>nul | findstr "psql"
    pg_isready -h localhost -p 5432 >nul 2>&1
    if %errorlevel% == 0 (
        echo ✓ PostgreSQL service is running
        set POSTGRES_READY=1
    ) else (
        echo ! PostgreSQL installed but not running
        set POSTGRES_READY=0
    )
) else (
    echo ✗ PostgreSQL not found
    set POSTGRES_READY=0
)

REM Check Redis
echo Checking Redis installation...
where redis-server >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Redis found - checking version...
    redis-server --version 2>nul
    redis-cli ping >nul 2>&1
    if %errorlevel% == 0 (
        echo ✓ Redis service is running
        set REDIS_READY=1
    ) else (
        echo ! Redis installed but not running
        set REDIS_READY=0
    )
) else (
    echo ✗ Redis not found
    set REDIS_READY=0
)

REM Check RabbitMQ
echo Checking RabbitMQ installation...
where rabbitmq-server >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ RabbitMQ found - checking version...
    rabbitmq-diagnostics status >nul 2>&1
    if %errorlevel% == 0 (
        echo ✓ RabbitMQ service is running
        set RABBITMQ_READY=1
    ) else (
        echo ! RabbitMQ installed but not running
        set RABBITMQ_READY=0
    )
) else (
    echo ✗ RabbitMQ not found
    set RABBITMQ_READY=0
)

REM Check Go installation
echo Checking Go installation...
where go >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Go found - version:
    go version
    set GO_READY=1
) else (
    echo ✗ Go not found
    set GO_READY=0
)

REM Check Node.js/npm
echo Checking Node.js installation...
where node >nul 2>&1
if %errorlevel% == 0 (
    echo ✓ Node.js found - version:
    node --version
    npm --version
    set NODE_READY=1
) else (
    echo ✗ Node.js not found
    set NODE_READY=0
)

echo.
echo [PHASE 2] Creating optimized directory structure...
echo ===============================================

REM Create enterprise directories with optimization
if not exist "C:\enterprise-services" (
    echo Creating C:\enterprise-services...
    mkdir "C:\enterprise-services"
)

for %%d in (logs data config services redis rabbitmq postgres) do (
    if not exist "C:\enterprise-services\%%d" (
        echo Creating C:\enterprise-services\%%d...
        mkdir "C:\enterprise-services\%%d"
    )
)

echo.
echo [PHASE 3] Smart service configuration...
echo ======================================

REM PostgreSQL setup (only if needed)
if "%POSTGRES_READY%"=="0" (
    echo Configuring PostgreSQL setup...
    call setup-pgvector.bat
) else (
    echo PostgreSQL already configured and running - skipping
)

REM Redis setup (only if needed)
if "%REDIS_READY%"=="0" (
    echo Configuring Redis setup...
    call setup-redis.bat
) else (
    echo Redis already configured and running - skipping
)

REM RabbitMQ setup (only if needed)
if "%RABBITMQ_READY%"=="0" (
    echo Configuring RabbitMQ setup...
    call setup-rabbitmq.bat
) else (
    echo RabbitMQ already configured and running - skipping
)

echo.
echo [PHASE 4] Building enhanced vector consumer service...
echo ==================================================

if "%GO_READY%"=="1" (
    echo Building enterprise binary...
    if exist "build-enterprise.bat" (
        call build-enterprise.bat
    ) else (
        echo Building vector consumer manually...
        go build -o bin/vector-consumer-enterprise.exe -ldflags="-s -w" vector-consumer-service-v2.go
    )
) else (
    echo Warning: Go not found, skipping build step
)

echo.
echo [PHASE 5] Windows services integration...
echo =======================================

REM Create Windows services (if admin)
net session >nul 2>&1
if %errorlevel% == 0 (
    echo Installing Windows services...
    if exist "create-windows-services.bat" (
        call create-windows-services.bat
    )
) else (
    echo Skipping Windows service installation (requires admin)
)

echo.
echo [PHASE 6] NPM integration setup...
echo ================================

REM Create npm integration script
(
echo @echo off
echo REM Enhanced NPM Dev Integration
echo echo Starting Enhanced Enterprise Legal AI Platform...
echo.
echo REM Start PostgreSQL if not running
echo pg_isready -h localhost -p 5432 ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo Starting PostgreSQL...
echo     net start postgresql-x64-17 2^>nul ^|^| net start postgresql-x64-16 2^>nul ^|^| net start postgresql-x64-15
echo ^)
echo.
echo REM Start Redis if not running
echo redis-cli ping ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo Starting Redis...
echo     if exist "C:\enterprise-services\redis\start-redis.bat" ^(
echo         start /B /MIN cmd /c "C:\enterprise-services\redis\start-redis.bat"
echo     ^)
echo ^)
echo.
echo REM Start RabbitMQ if not running
echo rabbitmq-diagnostics status ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo Starting RabbitMQ...
echo     net start RabbitMQ 2^>nul
echo ^)
echo.
echo REM Start vector consumer service
echo if exist "bin\vector-consumer-enterprise.exe" ^(
echo     echo Starting Vector Consumer Enterprise...
echo     start /B "VectorConsumer" bin\vector-consumer-enterprise.exe
echo ^)
echo.
echo echo All enterprise services started!
echo echo Ready for npm run dev
) > "C:\enterprise-services\start-for-npm-dev.bat"

echo.
echo [PHASE 7] Creating status dashboard...
echo ====================================

REM Create enhanced status script
(
echo @echo off
echo echo ========================================================================
echo echo Enhanced Enterprise Legal AI Platform Status
echo echo ========================================================================
echo.
echo echo [DATABASE SERVICES]
echo echo PostgreSQL:
echo pg_isready -h localhost -p 5432 2^>nul ^&^& echo   ✓ Running ^& psql --version 2^>nul ^| findstr "psql" ^|^| echo   ✗ Not running
echo.
echo echo Redis:
echo redis-cli ping 2^>nul ^&^& echo   ✓ Running ^& redis-server --version 2^>nul ^|^| echo   ✗ Not running
echo.
echo echo RabbitMQ:
echo rabbitmq-diagnostics status 2^>nul ^&^& echo   ✓ Running ^|^| echo   ✗ Not running
echo.
echo echo [MICROSERVICES]
echo echo Vector Consumer Enterprise:
echo tasklist /FI "IMAGENAME eq vector-consumer-enterprise.exe" 2^>NUL ^| find /I /N "vector-consumer-enterprise.exe" ^>NUL ^&^& echo   ✓ Running ^|^| echo   ✗ Not running
echo.
echo echo [DEVELOPMENT SERVICES]
echo echo Node.js/NPM:
echo where node ^>nul 2^>^&1 ^&^& node --version ^& npm --version ^|^| echo   ✗ Not installed
echo.
echo echo Go:
echo where go ^>nul 2^>^&1 ^&^& go version ^|^| echo   ✗ Not installed
echo.
echo echo [SERVICE ENDPOINTS]
echo echo ==================
echo echo PostgreSQL: localhost:5432
echo echo Redis: localhost:6379
echo echo RabbitMQ: localhost:5672 ^(Management: localhost:15672^)
echo echo Vector Consumer gRPC: localhost:8080
echo echo Vector Consumer HTTP: localhost:8081
echo.
echo pause
) > "C:\enterprise-services\enhanced-status.bat"

echo.
echo [PHASE 8] Performance optimization...
echo ===================================

REM Create performance tuning script
(
echo @echo off
echo REM Performance optimization for Legal AI Platform
echo echo Applying performance optimizations...
echo.
echo REM Windows performance settings
echo echo Optimizing Windows settings for AI workloads...
echo powershell -Command "Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\PriorityControl' -Name 'Win32PrioritySeparation' -Value 38"
echo.
echo REM Network optimizations
echo echo Optimizing network settings...
echo netsh int tcp set global autotuninglevel=normal
echo netsh int tcp set global rss=enabled
echo netsh int tcp set global netdma=enabled
echo.
echo echo Performance optimization complete!
) > "C:\enterprise-services\optimize-performance.bat"

echo.
echo ========================================================================
echo Enhanced Enterprise Setup Complete!
echo ========================================================================
echo.
echo Status Summary:
echo PostgreSQL: %POSTGRES_READY%
echo Redis: %REDIS_READY%
echo RabbitMQ: %RABBITMQ_READY%
echo Go: %GO_READY%
echo Node.js: %NODE_READY%
echo.
echo Quick Commands:
echo ===============
echo Start for NPM Dev: C:\enterprise-services\start-for-npm-dev.bat
echo Status Dashboard:   C:\enterprise-services\enhanced-status.bat
echo Optimize Performance: C:\enterprise-services\optimize-performance.bat
echo.
echo Integration complete! Ready for npm run dev workflow.
echo.
pause