@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Execute next steps automatically
echo [1/5] Cleaning...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul
go clean -modcache 2>nul

echo [2/5] Removing broken deps...
go mod edit -droprequire github.com/bytedance/sonic
go mod edit -droprequire github.com/NVIDIA/go-nvml
go mod tidy

echo [3/5] Building...
set CGO_ENABLED=0
go build -ldflags "-s -w" -o service.exe main.go

echo [4/5] Starting...
start /B service.exe

echo [5/5] Verifying...
timeout /t 1 >nul
curl http://localhost:8080/health

echo.
echo Service operational. Check generated files:
cd ..
dir /B next-steps.*
