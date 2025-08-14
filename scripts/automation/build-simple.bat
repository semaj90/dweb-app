@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo Removing problematic dependencies...
go mod init microservice 2>nul
go get github.com/gin-gonic/gin
go get github.com/valyala/fastjson
go get github.com/jackc/pgx/v5

echo Building simplified version...
set CGO_ENABLED=0
go build -o ai-microservice.exe main_simple.go

if exist ai-microservice.exe (
    taskkill /F /IM ai-microservice.exe 2>nul
    start ai-microservice.exe
    timeout /t 2 >nul
    curl http://localhost:8080/health
) else (
    echo Build failed
)
