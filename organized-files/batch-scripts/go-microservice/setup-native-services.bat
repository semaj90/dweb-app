@echo off
REM Native Windows Enterprise Services Setup
REM Sets up PostgreSQL, Redis, RabbitMQ, and Vector Service natively on Windows

echo Setting up Native Windows Enterprise Services...
echo.

REM Create directories
if not exist "C:\enterprise-services" mkdir "C:\enterprise-services"
if not exist "C:\enterprise-services\logs" mkdir "C:\enterprise-services\logs"
if not exist "C:\enterprise-services\data" mkdir "C:\enterprise-services\data"
if not exist "C:\enterprise-services\config" mkdir "C:\enterprise-services\config"

echo [1/6] Setting up PostgreSQL with pgvector...

REM Check if PostgreSQL is already installed
where psql >nul 2>&1
if %errorlevel% == 0 (
    echo PostgreSQL already installed
) else (
    echo Installing PostgreSQL 15 with pgvector...
    echo Please install from: https://www.postgresql.org/download/windows/
    echo After installation, run: setup-pgvector.bat
    pause
)

echo [2/6] Setting up Redis (Memurai alternative)...

REM Check if Redis is available
where redis-server >nul 2>&1
if %errorlevel% == 0 (
    echo Redis already installed
) else (
    echo Installing Redis for Windows...
    echo Please install from: https://github.com/microsoftarchive/redis/releases
    echo Or use WSL2 Redis for better performance
    pause
)

echo [3/6] Setting up RabbitMQ...

REM Check if RabbitMQ is available  
where rabbitmq-server >nul 2>&1
if %errorlevel% == 0 (
    echo RabbitMQ already installed
) else (
    echo Installing RabbitMQ...
    echo Please install from: https://www.rabbitmq.com/install-windows.html
    echo Also install Erlang: https://www.erlang.org/downloads
    pause
)

echo [4/6] Configuring services...

REM Create PostgreSQL configuration
echo Creating PostgreSQL configuration...
(
echo # PostgreSQL Configuration for Vector Service
echo port = 5432
echo max_connections = 200
echo shared_buffers = 256MB
echo effective_cache_size = 1GB
echo work_mem = 16MB
echo maintenance_work_mem = 256MB
echo checkpoint_completion_target = 0.9
echo wal_buffers = 16MB
echo default_statistics_target = 100
echo random_page_cost = 1.1
echo effective_io_concurrency = 200
echo min_wal_size = 1GB
echo max_wal_size = 4GB
) > "C:\enterprise-services\config\postgresql.conf"

REM Create Redis configuration
echo Creating Redis configuration...
(
echo # Redis Configuration for Vector Service
echo port 6379
echo bind 127.0.0.1
echo maxmemory 2gb
echo maxmemory-policy allkeys-lru
echo save 900 1
echo save 300 10
echo save 60 10000
echo rdbcompression yes
echo rdbchecksum yes
echo dir C:\enterprise-services\data\redis
) > "C:\enterprise-services\config\redis.conf"

REM Create RabbitMQ configuration
echo Creating RabbitMQ configuration...
(
echo # RabbitMQ Configuration
echo listeners.tcp.default = 5672
echo management.listener.port = 15672
echo management.listener.ssl = false
echo default_user = vector_admin
echo default_pass = vector_2024
echo vm_memory_high_watermark.relative = 0.6
echo disk_free_limit.relative = 2.0
) > "C:\enterprise-services\config\rabbitmq.conf"

echo [5/6] Creating environment configuration...

REM Create environment file for Vector Service
(
echo # Vector Consumer Service v2.0 - Native Windows Configuration
echo SERVICE_NAME=vector-consumer-enterprise
echo SERVICE_VERSION=2.0.0
echo GRPC_PORT=8080
echo HTTP_PORT=8081
echo.
echo # Database Configuration
echo POSTGRESQL_URL=postgres://postgres:postgres@localhost:5432/vector_db?sslmode=disable
echo MIGRATIONS_PATH=file://db/migrations
echo.
echo # Caching Configuration  
echo REDIS_URL=redis://localhost:6379
echo RISTRETTO_MAX_COST=104857600
echo CACHE_TIMEOUT_MINUTES=60
echo.
echo # GPU Configuration ^(Native RTX 3060 Ti^)
echo CUDA_WORKER_PATH=C:\Users\james\Desktop\deeds-web\deeds-web-app\cuda-worker\cuda-worker.exe
echo USE_CUBLAS=true
echo MAX_GPU_BATCH_SIZE=32
echo.
echo # Message Queue Configuration
echo RABBITMQ_URL=amqp://vector_admin:vector_2024@localhost:5672/
echo QUEUE_NAME=vector_processing_v2
echo MAX_WORKERS=4
echo.
echo # Observability Configuration
echo LOG_LEVEL=info
echo METRICS_ENABLED=true
echo TRACING_ENABLED=false
echo LOG_FILE=C:\enterprise-services\logs\vector-service.log
echo.
echo # Performance Configuration
echo PROCESS_TIMEOUT_SECONDS=30
echo MAX_CONCURRENT_JOBS=10
echo POSTGRES_MAX_CONNECTIONS=25
echo POSTGRES_MIN_CONNECTIONS=5
) > "C:\enterprise-services\config\.env"

echo [6/6] Creating startup scripts...

REM Create database initialization script
(
echo @echo off
echo echo Initializing Vector Database...
echo psql -U postgres -c "CREATE DATABASE IF NOT EXISTS vector_db;"
echo psql -U postgres -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo psql -U postgres -d vector_db -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
echo echo Database initialized successfully!
) > "C:\enterprise-services\init-database.bat"

REM Create service startup script
(
echo @echo off
echo echo Starting Native Windows Enterprise Services...
echo.
echo echo Starting PostgreSQL...
echo net start postgresql-x64-15
echo.
echo echo Starting Redis...
echo start /B redis-server "C:\enterprise-services\config\redis.conf"
echo.
echo echo Starting RabbitMQ...
echo net start RabbitMQ
echo.
echo echo Initializing database...
echo call "C:\enterprise-services\init-database.bat"
echo.
echo echo Starting Vector Consumer Service...
echo cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice"
echo set /p=^<"C:\enterprise-services\config\.env"
echo start /B bin\vector-consumer-enterprise.exe
echo.
echo echo All services started successfully!
echo echo.
echo echo Service URLs:
echo echo - Vector Service gRPC: localhost:8080
echo echo - Vector Service HTTP: localhost:8081
echo echo - PostgreSQL: localhost:5432
echo echo - Redis: localhost:6379  
echo echo - RabbitMQ Management: http://localhost:15672
echo echo.
echo pause
) > "C:\enterprise-services\start-services.bat"

REM Create service shutdown script
(
echo @echo off
echo echo Stopping Native Windows Enterprise Services...
echo.
echo echo Stopping Vector Consumer Service...
echo taskkill /F /IM vector-consumer-enterprise.exe 2^>nul
echo.
echo echo Stopping RabbitMQ...
echo net stop RabbitMQ
echo.
echo echo Stopping Redis...
echo taskkill /F /IM redis-server.exe 2^>nul
echo.
echo echo Stopping PostgreSQL...
echo net stop postgresql-x64-15
echo.
echo echo All services stopped.
) > "C:\enterprise-services\stop-services.bat"

echo.
echo ===================================================
echo Native Windows Enterprise Services Setup Complete!
echo ===================================================
echo.
echo Next Steps:
echo 1. Install PostgreSQL, Redis, and RabbitMQ if not already installed
echo 2. Run: C:\enterprise-services\start-services.bat
echo 3. Build and run your Vector Consumer Service
echo.
echo Configuration files created:
echo - C:\enterprise-services\config\.env
echo - C:\enterprise-services\config\postgresql.conf  
echo - C:\enterprise-services\config\redis.conf
echo - C:\enterprise-services\config\rabbitmq.conf
echo.
echo Management scripts created:
echo - C:\enterprise-services\start-services.bat
echo - C:\enterprise-services\stop-services.bat
echo - C:\enterprise-services\init-database.bat
echo.
pause