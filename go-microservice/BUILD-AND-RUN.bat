@echo off
echo.
echo ========================================
echo    SIMPLE BUILD AND RUN
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice"

echo Setting up Go environment...
set CGO_ENABLED=0
set PATH=C:\Program Files\Go\bin;%PATH%

echo.
echo Building main service...
go build -o legal-ai-server.exe main.go

if exist "legal-ai-server.exe" (
    echo [SUCCESS] Build complete!
    echo.
    echo Starting service...
    taskkill /F /IM legal-ai-server.exe >nul 2>&1
    start /B legal-ai-server.exe
    echo.
    echo Service started. Check http://localhost:8084/api/health
) else (
    echo [ERROR] Build failed!
    echo.
    echo Please ensure Go is installed:
    echo 1. Download from https://go.dev/dl/
    echo 2. Install to C:\Program Files\Go
    echo 3. Run this script again
)

echo.
echo Press any key to exit...
pause >nul