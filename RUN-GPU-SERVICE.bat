@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

set CGO_ENABLED=0
go build -ldflags "-s -w" -o gpu_service.exe gpu_service.go

taskkill /F /IM gpu_service.exe 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

gpu_service.exe
