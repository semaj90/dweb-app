@echo off
:: Redis Setup for Windows

echo 🔧 Setting up Redis for Windows
echo ================================
echo.

:: Check if Redis is already installed as a service
sc query Redis >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Redis service exists, starting it...
    net start Redis 2>nul
    goto :check_redis
)

:: Check if redis-cli exists in PATH
where redis-cli >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Redis found in PATH
    goto :check_redis
)

:: Check local redis-windows directory
if exist redis-windows\redis-server.exe (
    echo Found Redis in redis-windows directory
    cd redis-windows
    
    :: Create a basic config if it doesn't exist
    if not exist redis.conf (
        echo Creating basic Redis configuration...
        (
            echo port 6379
            echo bind 127.0.0.1
            echo protected-mode no
            echo daemonize no
            echo dir ./
            echo logfile redis.log
        ) > redis.conf
    )
    
    :: Start Redis
    echo Starting Redis server...
    start "Redis Server" /B redis-server.exe redis.conf
    cd ..
    timeout /t 3 /nobreak >nul
    goto :check_redis
)

:: Download Redis if not found
echo Redis not found. Would you like to download it? (Y/N)
choice /C YN /N
if %ERRORLEVEL% EQU 1 (
    echo Downloading Redis for Windows...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip' -OutFile 'redis-windows.zip'"
    
    echo Extracting Redis...
    powershell -Command "Expand-Archive -Path 'redis-windows.zip' -DestinationPath 'redis-windows' -Force"
    del redis-windows.zip
    
    cd redis-windows
    echo Creating Redis configuration...
    (
        echo port 6379
        echo bind 127.0.0.1
        echo protected-mode no
        echo daemonize no
        echo dir ./
        echo logfile redis.log
    ) > redis.conf
    
    echo Starting Redis...
    start "Redis Server" /B redis-server.exe redis.conf
    cd ..
    timeout /t 3 /nobreak >nul
)

:check_redis
:: Verify Redis is running
echo.
echo Checking Redis status...

:: Try different methods to check Redis
redis-cli ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ Redis is running successfully!
    redis-cli INFO server | findstr redis_version
    exit /b 0
)

if exist redis-windows\redis-cli.exe (
    redis-windows\redis-cli.exe ping >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Redis is running successfully!
        redis-windows\redis-cli.exe INFO server | findstr redis_version
        exit /b 0
    )
)

echo ❌ Redis is not responding
echo.
echo Troubleshooting:
echo 1. Check if port 6379 is in use: netstat -an | findstr :6379
echo 2. Check Windows Firewall settings
echo 3. Try running Redis manually: redis-windows\redis-server.exe

exit /b 1
