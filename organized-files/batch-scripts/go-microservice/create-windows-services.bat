@echo off
REM Windows Service Installation for Vector Consumer Service v2.0
REM Creates and manages Windows services for enterprise deployment

echo Creating Windows Services for Vector Consumer Enterprise Deployment...
echo.

REM Check if running as Administrator
echo [1/5] Checking Administrator privileges...
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: This script must be run as Administrator
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

echo Running with Administrator privileges
echo.

REM Create service installation directory
echo [2/5] Creating service directories...
if not exist "C:\enterprise-services\services" mkdir "C:\enterprise-services\services"
if not exist "C:\enterprise-services\services\vector-consumer" mkdir "C:\enterprise-services\services\vector-consumer"

REM Copy vector consumer binary
if exist "bin\vector-consumer-enterprise.exe" (
    echo Copying vector-consumer-enterprise.exe to services directory...
    copy "bin\vector-consumer-enterprise.exe" "C:\enterprise-services\services\vector-consumer\"
) else (
    echo Warning: vector-consumer-enterprise.exe not found in bin directory
    echo Build the service first using: build-enterprise.bat
)

REM Create service wrapper script
echo [3/5] Creating Windows service wrapper...
(
echo @echo off
echo REM Vector Consumer Service v2.0 - Windows Service Wrapper
echo REM This script is executed by the Windows Service Manager
echo.
echo REM Set environment variables
echo set SERVICE_NAME=VectorConsumerEnterprise
echo set SERVICE_VERSION=2.0.0
echo set GRPC_PORT=8080
echo set HTTP_PORT=8081
echo.
echo REM Database Configuration
echo set POSTGRESQL_URL=postgres://postgres:postgres@localhost:5432/vector_db?sslmode=disable
echo set MIGRATIONS_PATH=file://db/migrations
echo.
echo REM Caching Configuration
echo set REDIS_URL=redis://localhost:6379
echo set RISTRETTO_MAX_COST=104857600
echo set CACHE_TIMEOUT_MINUTES=60
echo.
echo REM GPU Configuration
echo set CUDA_WORKER_PATH=C:\Users\james\Desktop\deeds-web\deeds-web-app\cuda-worker\cuda-worker.exe
echo set USE_CUBLAS=true
echo set MAX_GPU_BATCH_SIZE=32
echo.
echo REM Message Queue Configuration
echo set RABBITMQ_URL=amqp://vector_admin:vector_2024_secure@localhost:5672/vector_processing
echo set QUEUE_NAME=vector_processing_v2
echo set MAX_WORKERS=4
echo.
echo REM Observability Configuration
echo set LOG_LEVEL=info
echo set METRICS_ENABLED=true
echo set TRACING_ENABLED=false
echo set LOG_FILE=C:\enterprise-services\logs\vector-service.log
echo.
echo REM Performance Configuration
echo set PROCESS_TIMEOUT_SECONDS=30
echo set MAX_CONCURRENT_JOBS=10
echo set POSTGRES_MAX_CONNECTIONS=25
echo set POSTGRES_MIN_CONNECTIONS=5
echo.
echo REM Change to service directory
echo cd /d "C:\enterprise-services\services\vector-consumer"
echo.
echo REM Start the service
echo echo Starting Vector Consumer Enterprise Service...
echo vector-consumer-enterprise.exe
) > "C:\enterprise-services\services\vector-consumer\service-wrapper.bat"

REM Install Vector Consumer Service
echo [4/5] Installing Vector Consumer Windows Service...

REM Remove existing service if it exists
sc query "VectorConsumerEnterprise" >nul 2>&1
if %errorlevel% == 0 (
    echo Removing existing VectorConsumerEnterprise service...
    sc stop "VectorConsumerEnterprise" >nul 2>&1
    timeout /t 5 /nobreak >nul
    sc delete "VectorConsumerEnterprise"
    timeout /t 3 /nobreak >nul
)

REM Create new service
echo Creating VectorConsumerEnterprise Windows service...
sc create "VectorConsumerEnterprise" ^
    binPath= "C:\enterprise-services\services\vector-consumer\service-wrapper.bat" ^
    DisplayName= "Vector Consumer Enterprise v2.0" ^
    Description= "Enterprise-grade vector processing service with gRPC, cuBLAS optimization, and multi-layer caching" ^
    start= auto ^
    type= own

if %errorlevel% == 0 (
    echo ✓ VectorConsumerEnterprise service created successfully
) else (
    echo ✗ Failed to create VectorConsumerEnterprise service
)

REM Configure service recovery options
echo Configuring service recovery options...
sc failure "VectorConsumerEnterprise" reset= 86400 actions= restart/30000/restart/60000/restart/120000

REM Create service management scripts
echo [5/5] Creating service management scripts...

REM Start all services script
(
echo @echo off
echo echo Starting All Enterprise Services for Vector Consumer v2.0...
echo echo ========================================================
echo.
echo echo [1/4] Starting PostgreSQL service...
echo net start postgresql-x64-17 2^>nul ^|^| net start postgresql-x64-16 2^>nul ^|^| net start postgresql-x64-15
echo if errorlevel 1 echo Warning: PostgreSQL service may not be installed as a Windows service
echo.
echo echo [2/4] Starting Redis...
echo if exist "C:\enterprise-services\redis\start-redis.bat" ^(
echo     call "C:\enterprise-services\redis\start-redis.bat"
echo ^) else ^(
echo     echo Warning: Redis startup script not found
echo ^)
echo.
echo echo [3/4] Starting RabbitMQ service...
echo net start RabbitMQ
echo if errorlevel 1 echo Warning: RabbitMQ service may not be installed
echo.
echo echo [4/4] Starting Vector Consumer Enterprise service...
echo net start VectorConsumerEnterprise
echo if errorlevel 1 ^(
echo     echo Error: Failed to start VectorConsumerEnterprise service
echo     echo Check the Event Logs for details
echo ^) else ^(
echo     echo ✓ All services started successfully!
echo ^)
echo.
echo echo Service Status Summary:
echo echo =====================
echo echo PostgreSQL: ^(checking...^)
echo sc query postgresql-x64-17 ^| findstr "RUNNING" ^>nul 2^>^&1 ^|^| sc query postgresql-x64-16 ^| findstr "RUNNING" ^>nul 2^>^&1 ^|^| sc query postgresql-x64-15 ^| findstr "RUNNING" ^>nul 2^>^&1
echo if errorlevel 1 ^(echo   Status: Not running as service^) else ^(echo   Status: Running^)
echo.
echo echo Redis: ^(checking...^)
echo tasklist /FI "IMAGENAME eq redis-server.exe" 2^>NUL ^| find /I /N "redis-server.exe" ^>NUL
echo if errorlevel 1 ^(echo   Status: Not running^) else ^(echo   Status: Running^)
echo.
echo echo RabbitMQ: ^(checking...^)
echo sc query RabbitMQ ^| findstr "RUNNING" ^>nul 2^>^&1
echo if errorlevel 1 ^(echo   Status: Not running^) else ^(echo   Status: Running^)
echo.
echo echo Vector Consumer Enterprise: ^(checking...^)
echo sc query VectorConsumerEnterprise ^| findstr "RUNNING" ^>nul 2^>^&1
echo if errorlevel 1 ^(echo   Status: Not running^) else ^(echo   Status: Running^)
echo.
echo echo Service URLs:
echo echo ============
echo echo Vector Service gRPC: localhost:8080
echo echo Vector Service HTTP: localhost:8081  
echo echo PostgreSQL: localhost:5432
echo echo Redis: localhost:6379
echo echo RabbitMQ AMQP: localhost:5672
echo echo RabbitMQ Management: http://localhost:15672
echo.
echo pause
) > "C:\enterprise-services\start-all-services.bat"

REM Stop all services script
(
echo @echo off
echo echo Stopping All Enterprise Services for Vector Consumer v2.0...
echo echo ========================================================
echo.
echo echo [1/4] Stopping Vector Consumer Enterprise service...
echo net stop VectorConsumerEnterprise
echo.
echo echo [2/4] Stopping RabbitMQ service...
echo net stop RabbitMQ
echo.
echo echo [3/4] Stopping Redis...
echo taskkill /F /IM redis-server.exe 2^>nul
echo if errorlevel 1 echo Redis was not running
echo.
echo echo [4/4] Stopping PostgreSQL service...
echo net stop postgresql-x64-17 2^>nul ^|^| net stop postgresql-x64-16 2^>nul ^|^| net stop postgresql-x64-15 2^>nul
echo if errorlevel 1 echo PostgreSQL service may not be installed as a Windows service
echo.
echo echo All services stopped.
echo pause
) > "C:\enterprise-services\stop-all-services.bat"

REM Service status script
(
echo @echo off
echo echo Enterprise Services Status for Vector Consumer v2.0
echo echo ==================================================
echo.
echo echo PostgreSQL Status:
echo sc query postgresql-x64-17 ^| findstr "STATE" 2^>nul ^|^| sc query postgresql-x64-16 ^| findstr "STATE" 2^>nul ^|^| sc query postgresql-x64-15 ^| findstr "STATE" 2^>nul ^|^| echo   Not installed as Windows service
echo.
echo echo Redis Status:
echo tasklist /FI "IMAGENAME eq redis-server.exe" 2^>NUL ^| find /I /N "redis-server.exe" ^>NUL
echo if errorlevel 1 ^(echo   Status: Not running^) else ^(echo   Status: Running^)
echo.
echo echo RabbitMQ Status:
echo sc query RabbitMQ ^| findstr "STATE" 2^>nul ^|^| echo   Not installed as Windows service
echo.
echo echo Vector Consumer Enterprise Status:
echo sc query VectorConsumerEnterprise ^| findstr "STATE" 2^>nul ^|^| echo   Not installed as Windows service
echo.
echo echo Detailed Service Information:
echo echo ============================
echo sc query VectorConsumerEnterprise
echo.
echo pause
) > "C:\enterprise-services\service-status.bat"

REM Service logs viewer script
(
echo @echo off
echo echo Vector Consumer Enterprise Service Logs
echo echo =====================================
echo.
echo echo Recent Windows Event Log entries:
echo wevtutil qe System /f:text /c:10 /q:"*[System[Provider[@Name='Service Control Manager'] and EventID=7036 and TimeCreated[timediff(@SystemTime) ^<= 3600000]]]"
echo.
echo echo Vector Consumer Service Log:
echo if exist "C:\enterprise-services\logs\vector-service.log" ^(
echo     echo Last 50 lines:
echo     powershell "Get-Content 'C:\enterprise-services\logs\vector-service.log' -Tail 50"
echo ^) else ^(
echo     echo Log file not found: C:\enterprise-services\logs\vector-service.log
echo ^)
echo.
echo pause
) > "C:\enterprise-services\view-service-logs.bat"

echo.
echo ===================================================
echo Windows Services Setup Complete!
echo ===================================================
echo.
echo Services Created:
echo - VectorConsumerEnterprise (Auto-start)
echo.
echo Management Scripts Created:
echo - C:\enterprise-services\start-all-services.bat
echo - C:\enterprise-services\stop-all-services.bat  
echo - C:\enterprise-services\service-status.bat
echo - C:\enterprise-services\view-service-logs.bat
echo.
echo Service Configuration:
echo - Display Name: Vector Consumer Enterprise v2.0
echo - Service Type: Own process, Auto-start
echo - Recovery: Auto-restart on failure
echo.
echo To manage the service:
echo   Start All:  C:\enterprise-services\start-all-services.bat
echo   Stop All:   C:\enterprise-services\stop-all-services.bat
echo   Status:     C:\enterprise-services\service-status.bat
echo   Logs:       C:\enterprise-services\view-service-logs.bat
echo.
echo Manual service commands:
echo   net start VectorConsumerEnterprise
echo   net stop VectorConsumerEnterprise
echo   sc query VectorConsumerEnterprise
echo.
echo Next Steps:
echo 1. Build the vector-consumer-enterprise.exe binary
echo 2. Run C:\enterprise-services\start-all-services.bat
echo 3. Verify services with C:\enterprise-services\service-status.bat
echo 4. Test endpoints: http://localhost:8081/health
echo.
pause