@echo off
REM Native Windows Deployment Script for Modular Clustering Service
echo ===== Starting Modular Clustering Service (Native Windows) =====

REM Set environment variables
set CONFIG_PATH=config.yaml
set LOG_LEVEL=info
set GOOS=windows
set GOARCH=amd64

echo.
echo [1/5] Checking prerequisites...

REM Check if Go is installed
go version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Go is not installed or not in PATH
    echo Please install Go from https://golang.org/dl/
    pause
    exit /b 1
)

echo ✅ Go is installed

REM Check if config file exists
if not exist "%CONFIG_PATH%" (
    echo ERROR: Configuration file %CONFIG_PATH% not found
    pause
    exit /b 1
)

echo ✅ Configuration file found

echo.
echo [2/5] Building production service...

REM Build the service
go build -o modular-cluster-service-production.exe modular-cluster-service-production.go
if %errorlevel% neq 0 (
    echo ERROR: Failed to build service
    pause
    exit /b 1
)

echo ✅ Service built successfully

echo.
echo [3/5] Running tests...

REM Run tests
go test -v
if %errorlevel% neq 0 (
    echo WARNING: Some tests failed, but continuing...
) else (
    echo ✅ All tests passed
)

echo.
echo [4/5] Starting monitoring services (optional)...

REM Check if Prometheus is available (optional)
where prometheus >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting Prometheus on port 9090...
    start /B prometheus --config.file=prometheus.yml --storage.tsdb.path=prometheus-data
    timeout /t 2 >nul
    echo ✅ Prometheus started
) else (
    echo ⚠️  Prometheus not found - metrics collection disabled
    echo Install from: https://prometheus.io/download/
)

REM Check if Redis is available (optional)
where redis-server >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting Redis on port 6379...
    start /B redis-server
    timeout /t 2 >nul
    echo ✅ Redis started
) else (
    echo ⚠️  Redis not found - caching disabled
    echo Install from: https://redis.io/download
)

echo.
echo [5/5] Starting Modular Clustering Service...

echo.
echo ====================================
echo Service Configuration:
echo - HTTP Port: 8085
echo - gRPC Port: 50051
echo - Config: %CONFIG_PATH%
echo - Log Level: %LOG_LEVEL%
echo ====================================
echo.

echo Starting service... (Press Ctrl+C to stop)
echo.

REM Start the main service
modular-cluster-service-production.exe

echo.
echo Service stopped.
pause