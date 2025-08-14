@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Direct fallback - skip CUDA entirely for now
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

echo Building operational version...
set CGO_ENABLED=0
go build -tags nocuda -o ai-microservice.exe main_simple.go

if exist ai-microservice.exe (
    start ai-microservice.exe
    timeout /t 2 >nul
    curl http://localhost:8080/health
) else (
    echo Fallback failed. Running doc_processor directly...
    go run doc_processor.go
)
