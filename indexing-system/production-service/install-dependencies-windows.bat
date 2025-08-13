@echo off
REM Windows Native Dependencies Installation Script
echo ===== Installing Dependencies for Native Windows Deployment =====

echo.
echo This script will help you install the required dependencies for the
echo Modular Clustering Service on Windows (without Docker).
echo.

echo [1/6] Checking existing installations...

REM Check Go
go version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Go is already installed
    go version
) else (
    echo ❌ Go is not installed
    echo.
    echo Please install Go from: https://golang.org/dl/
    echo Download: go1.21.windows-amd64.msi
    echo After installation, restart this script.
    pause
    exit /b 1
)

REM Check Git
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Git is already installed
) else (
    echo ❌ Git is not installed
    echo.
    echo Please install Git from: https://git-scm.com/download/win
    echo This is required for Go module dependencies.
    pause
    exit /b 1
)

echo.
echo [2/6] Optional: Prometheus (for metrics monitoring)

where prometheus >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Prometheus is already installed
) else (
    echo ❌ Prometheus is not installed
    echo.
    echo To install Prometheus:
    echo 1. Download from: https://prometheus.io/download/
    echo 2. Extract prometheus-2.x.x.windows-amd64.zip
    echo 3. Add prometheus.exe to your PATH
    echo.
    set /p install_prometheus="Install Prometheus now? (y/n): "
    if /i "%install_prometheus%"=="y" (
        echo.
        echo Please download and install Prometheus manually from:
        echo https://prometheus.io/download/
        echo.
        echo Steps:
        echo 1. Download prometheus-2.x.x.windows-amd64.zip
        echo 2. Extract to C:\prometheus\
        echo 3. Add C:\prometheus\ to your PATH environment variable
        echo 4. Restart command prompt
        echo.
        pause
    )
)

echo.
echo [3/6] Optional: Redis (for caching)

where redis-server >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Redis is already installed
) else (
    echo ❌ Redis is not installed
    echo.
    echo To install Redis on Windows:
    echo 1. Download from: https://github.com/microsoftarchive/redis/releases
    echo 2. Install Redis-x64-3.x.x.msi
    echo 3. Start Redis service
    echo.
    set /p install_redis="Install Redis now? (y/n): "
    if /i "%install_redis%"=="y" (
        echo.
        echo Please download and install Redis manually from:
        echo https://github.com/microsoftarchive/redis/releases
        echo.
        echo Steps:
        echo 1. Download Redis-x64-3.x.x.msi
        echo 2. Run installer as Administrator
        echo 3. Redis will start as a Windows service
        echo.
        pause
    )
)

echo.
echo [4/6] Optional: Grafana (for dashboards)

where grafana-server >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Grafana is already installed
) else (
    echo ❌ Grafana is not installed
    echo.
    echo To install Grafana:
    echo 1. Download from: https://grafana.com/grafana/download?platform=windows
    echo 2. Extract and run grafana-server.exe
    echo 3. Access at http://localhost:3000
    echo.
    set /p install_grafana="Install Grafana now? (y/n): "
    if /i "%install_grafana%"=="y" (
        echo.
        echo Please download and install Grafana manually from:
        echo https://grafana.com/grafana/download?platform=windows
        echo.
        echo Steps:
        echo 1. Download grafana-x.x.x.windows-amd64.zip
        echo 2. Extract to C:\grafana\
        echo 3. Run: C:\grafana\bin\grafana-server.exe
        echo 4. Access: http://localhost:3000 (admin/admin)
        echo.
        pause
    )
)

echo.
echo [5/6] Installing Go dependencies...

go mod tidy
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Go dependencies
    pause
    exit /b 1
)

echo ✅ Go dependencies installed successfully

echo.
echo [6/6] Creating Windows service (optional)...

set /p create_service="Create Windows service for auto-start? (y/n): "
if /i "%create_service%"=="y" (
    echo.
    echo Creating Windows service...
    
    REM Create service wrapper script
    echo @echo off > service-wrapper.bat
    echo cd /d "%~dp0" >> service-wrapper.bat
    echo modular-cluster-service-production.exe >> service-wrapper.bat
    
    REM Create service using sc command
    sc create "ModularClusteringService" binPath= "%CD%\service-wrapper.bat" start= auto DisplayName= "Modular Clustering Service"
    if %errorlevel% equ 0 (
        echo ✅ Windows service created successfully
        echo.
        echo Service management commands:
        echo - Start: sc start ModularClusteringService
        echo - Stop:  sc stop ModularClusteringService
        echo - Delete: sc delete ModularClusteringService
    ) else (
        echo ❌ Failed to create Windows service
        echo Note: You may need to run as Administrator
    )
)

echo.
echo ================================================
echo Installation Summary:
echo ================================================
echo.
echo Required (for basic functionality):
if exist "%GOPATH%\bin\go.exe" (echo ✅ Go) else (echo ✅ Go)
echo.
echo Optional (for full monitoring):
where prometheus >nul 2>&1 && echo ✅ Prometheus || echo ❌ Prometheus
where redis-server >nul 2>&1 && echo ✅ Redis || echo ❌ Redis  
where grafana-server >nul 2>&1 && echo ✅ Grafana || echo ❌ Grafana
echo.
echo ================================================
echo Next Steps:
echo ================================================
echo.
echo 1. Run: start-native-windows.bat
echo 2. Access service: http://localhost:8085
echo 3. View metrics: http://localhost:9090 (if Prometheus installed)
echo 4. View dashboards: http://localhost:3000 (if Grafana installed)
echo.
echo For advanced configuration, edit config.yaml
echo.
pause