@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

echo Removing broken dependencies...
go mod edit -droprequire github.com/bytedance/sonic
go mod edit -dropreplace github.com/bytedance/sonic

echo Building clean version...
set CGO_ENABLED=0
go build -o ai-microservice.exe main_simple.go

if exist ai-microservice.exe (
    start ai-microservice.exe
    timeout /t 2 >nul
    curl http://localhost:8080/health
) else (
    echo Direct fallback...
    go run main_simple.go
)
