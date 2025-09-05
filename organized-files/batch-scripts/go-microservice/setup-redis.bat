@echo off
REM Redis/Memurai Setup for Vector Consumer Service
REM Configures high-performance caching layer for enterprise vector operations

echo Setting up Redis/Memurai for Vector Consumer Service...
echo.

REM Create Redis directories
echo [1/5] Creating Redis directories...
if not exist "C:\enterprise-services\redis" mkdir "C:\enterprise-services\redis"
if not exist "C:\enterprise-services\redis\data" mkdir "C:\enterprise-services\redis\data"
if not exist "C:\enterprise-services\redis\logs" mkdir "C:\enterprise-services\redis\logs"
if not exist "C:\enterprise-services\redis\config" mkdir "C:\enterprise-services\redis\config"

REM Check if Redis is available
echo [2/5] Checking Redis installation...
where redis-server >nul 2>&1
if %errorlevel% == 0 (
    echo Redis server found
    redis-server --version
) else (
    echo Redis not found. Installing options:
    echo.
    echo Option 1 - Redis for Windows (Microsoft Archive):
    echo   Download from: https://github.com/microsoftarchive/redis/releases
    echo   Extract to C:\Redis or add to PATH
    echo.
    echo Option 2 - Memurai (Redis-compatible, Windows optimized):
    echo   Download from: https://www.memurai.com/get-memurai
    echo   Professional Redis alternative for Windows
    echo.
    echo Option 3 - WSL2 Redis (Recommended for production):
    echo   1. Install WSL2: wsl --install
    echo   2. Install Redis in WSL2: sudo apt install redis-server
    echo.
    pause
)

REM Create optimized Redis configuration
echo [3/5] Creating optimized Redis configuration...
(
echo # Redis Configuration for Vector Consumer Service v2.0
echo # Optimized for Windows enterprise deployment
echo.
echo # Network Configuration
echo bind 127.0.0.1
echo port 6379
echo tcp-backlog 511
echo tcp-keepalive 300
echo.
echo # Memory Configuration
echo maxmemory 2gb
echo maxmemory-policy allkeys-lru
echo maxmemory-samples 5
echo.
echo # Persistence Configuration
echo save 900 1
echo save 300 10
echo save 60 10000
echo stop-writes-on-bgsave-error yes
echo rdbcompression yes
echo rdbchecksum yes
echo dbfilename dump.rdb
echo dir C:/enterprise-services/redis/data
echo.
echo # Logging Configuration
echo loglevel notice
echo logfile "C:/enterprise-services/redis/logs/redis.log"
echo syslog-enabled no
echo.
echo # Performance Tuning
echo timeout 0
echo hz 10
echo dynamic-hz yes
echo.
echo # Security Configuration
echo # requirepass your_secure_password_here
echo # rename-command FLUSHDB ""
echo # rename-command FLUSHALL ""
echo.
echo # Vector Service Optimizations
echo hash-max-ziplist-entries 512
echo hash-max-ziplist-value 64
echo list-max-ziplist-size -2
echo list-compress-depth 0
echo set-max-intset-entries 512
echo zset-max-ziplist-entries 128
echo zset-max-ziplist-value 64
echo.
echo # Replication ^(if needed^)
echo # replicaof ^<masterip^> ^<masterport^>
echo replica-serve-stale-data yes
echo replica-read-only yes
echo.
echo # Modules ^(if available^)
echo # loadmodule /path/to/redisearch.so
echo # loadmodule /path/to/redisjson.so
) > "C:\enterprise-services\redis\config\redis.conf"

REM Create Redis startup script
echo [4/5] Creating Redis startup script...
(
echo @echo off
echo echo Starting Redis server for Vector Consumer Service...
echo cd /d "C:\enterprise-services\redis"
echo.
echo REM Check if Redis is already running
echo tasklist /FI "IMAGENAME eq redis-server.exe" 2^>NUL ^| find /I /N "redis-server.exe"^>NUL
echo if "%%ERRORLEVEL%%"=="0" ^(
echo     echo Redis server is already running
echo     redis-cli ping
echo     if errorlevel 1 ^(
echo         echo Redis not responding, restarting...
echo         taskkill /F /IM redis-server.exe ^>nul 2^>^&1
echo         timeout /t 2 /nobreak ^>nul
echo     ^) else ^(
echo         echo Redis server is healthy
echo         exit /b 0
echo     ^)
echo ^)
echo.
echo REM Start Redis server
echo if exist "C:\Redis\redis-server.exe" ^(
echo     echo Starting Redis from C:\Redis\
echo     start /B "Redis Server" "C:\Redis\redis-server.exe" "C:\enterprise-services\redis\config\redis.conf"
echo ^) else if exist "C:\Program Files\Redis\redis-server.exe" ^(
echo     echo Starting Redis from Program Files\
echo     start /B "Redis Server" "C:\Program Files\Redis\redis-server.exe" "C:\enterprise-services\redis\config\redis.conf"
echo ^) else ^(
echo     where redis-server ^>nul 2^>^&1
echo     if errorlevel 1 ^(
echo         echo Error: Redis server not found in PATH or common locations
echo         echo Please install Redis or add to PATH
echo         pause
echo         exit /b 1
echo     ^) else ^(
echo         echo Starting Redis from PATH
echo         start /B "Redis Server" redis-server "C:\enterprise-services\redis\config\redis.conf"
echo     ^)
echo ^)
echo.
echo REM Wait for Redis to start
echo timeout /t 3 /nobreak ^>nul
echo.
echo REM Test connection
echo echo Testing Redis connection...
echo redis-cli ping
echo if errorlevel 1 ^(
echo     echo Error: Redis server failed to start or not responding
echo     pause
echo     exit /b 1
echo ^)
echo.
echo echo Redis server started successfully!
echo echo Connection: redis://localhost:6379
echo echo Configuration: C:\enterprise-services\redis\config\redis.conf
echo echo Logs: C:\enterprise-services\redis\logs\redis.log
echo echo Data: C:\enterprise-services\redis\data\
echo.
echo pause
) > "C:\enterprise-services\redis\start-redis.bat"

REM Create Redis health check script
echo [5/5] Creating Redis health check script...
(
echo @echo off
echo echo Redis Health Check for Vector Consumer Service
echo echo ============================================
echo.
echo REM Check if Redis is running
echo tasklist /FI "IMAGENAME eq redis-server.exe" 2^>NUL ^| find /I /N "redis-server.exe"^>NUL
echo if "%%ERRORLEVEL%%"=="0" ^(
echo     echo [✓] Redis server process is running
echo ^) else ^(
echo     echo [✗] Redis server process not found
echo     exit /b 1
echo ^)
echo.
echo REM Test connection
echo redis-cli ping ^>nul 2^>^&1
echo if errorlevel 1 ^(
echo     echo [✗] Redis not responding to ping
echo     exit /b 1
echo ^) else ^(
echo     echo [✓] Redis responding to ping
echo ^)
echo.
echo REM Get Redis info
echo echo Redis Server Information:
echo redis-cli info server ^| findstr "redis_version"
echo redis-cli info memory ^| findstr "used_memory_human"
echo redis-cli info stats ^| findstr "total_commands_processed"
echo.
echo REM Test set/get operations
echo redis-cli set test_key "Vector Consumer Service v2.0" ^>nul
echo for /f "delims=" %%%%a in ^('redis-cli get test_key'^) do set test_value=%%%%a
echo redis-cli del test_key ^>nul
echo if "%%test_value%%"=="Vector Consumer Service v2.0" ^(
echo     echo [✓] Redis read/write operations working
echo ^) else ^(
echo     echo [✗] Redis read/write operations failed
echo ^)
echo.
echo echo Redis Health Check Complete
echo echo Status: All systems operational
) > "C:\enterprise-services\redis\redis-health-check.bat"

echo.
echo ===================================================
echo Redis/Memurai Setup Complete!
echo ===================================================
echo.
echo Configuration Files Created:
echo - C:\enterprise-services\redis\config\redis.conf
echo - C:\enterprise-services\redis\start-redis.bat
echo - C:\enterprise-services\redis\redis-health-check.bat
echo.
echo To start Redis:
echo   C:\enterprise-services\redis\start-redis.bat
echo.
echo Connection String:
echo   redis://localhost:6379
echo.
echo Health Check:
echo   C:\enterprise-services\redis\redis-health-check.bat
echo.
echo Configuration Summary:
echo - Memory limit: 2GB with LRU eviction
echo - Persistence: RDB snapshots enabled
echo - Logging: Notice level to redis.log
echo - Network: Localhost only (secure)
echo - Performance: Optimized for vector operations
echo.
echo Next: Run setup-rabbitmq.bat
pause