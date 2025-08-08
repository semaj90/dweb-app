@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Nuclear option - complete reset
rd /s /q "%GOPATH%\pkg\mod\github.com\bytedance" 2>nul
rd /s /q "%GOPATH%\pkg\mod\github.com\NVIDIA\go-nvml" 2>nul
del go.mod go.sum 2>nul

REM Fresh module with working deps only
(
echo module microservice
echo.
echo go 1.23
echo.
echo require ^(
echo     github.com/gin-gonic/gin v1.10.1
echo     github.com/valyala/fastjson v1.6.4
echo     github.com/minio/simdjson-go v0.4.5
echo     github.com/jackc/pgx/v5 v5.7.2
echo     github.com/go-redis/redis/v8 v8.11.5
echo     github.com/gorilla/websocket v1.5.3
echo ^)
) > go.mod

go mod tidy

REM Pure Go build - no CGO complications
set CGO_ENABLED=0
set GOOS=windows
set GOARCH=amd64
go build -ldflags "-s -w" -o service.exe main.go

REM Force kill any stragglers
taskkill /F /IM service.exe 2>nul
taskkill /F /IM ai-microservice.exe 2>nul
taskkill /F /IM legal-ai-server.exe 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

REM Launch
start /B service.exe
timeout /t 1 >nul

REM Validate
echo.
echo Service Check:
curl -s http://localhost:8080/health || echo Service failed to start
echo.
echo Port Status:
netstat -an | findstr :8080
