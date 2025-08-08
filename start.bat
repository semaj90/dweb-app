@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

go build -o ai-microservice.exe main.go
start ai-microservice.exe
timeout /t 1 >nul
curl http://localhost:8080/health
