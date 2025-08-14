@echo off
echo 🚀 Starting Go Service in Background
cd go-microservice
set PORT=8080
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set PGPASSWORD=123456
set OLLAMA_URL=http://localhost:11434

start /B legal-ai-server.exe > ..\logs\go-service.log 2>&1
timeout /t 3 >nul
netstat -an | findstr :8080
if %ERRORLEVEL% EQU 0 (
    echo ✅ Go service started successfully on port 8080
) else (
    echo ❌ Go service failed to start
)