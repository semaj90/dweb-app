@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo [1/5] Removing broken dependencies...
go mod edit -droprequire github.com/bytedance/sonic
go mod edit -droprequire github.com/NVIDIA/go-nvml
go mod edit -droprequire github.com/bytedance/sonic/loader
go mod tidy -e

echo [2/5] Building service...
set CGO_ENABLED=0
go build -ldflags "-s -w" -o service.exe main.go

echo [3/5] Clearing ports...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081') do taskkill /F /PID %%a 2>nul

echo [4/5] Starting on 8081...
set PORT=8081
start /B service.exe

echo [5/5] PostgreSQL trust auth...
echo host all all 127.0.0.1/32 trust > "%PROGRAMDATA%\PostgreSQL\16\pg_hba.conf.new"
echo host all all ::1/128 trust >> "%PROGRAMDATA%\PostgreSQL\16\pg_hba.conf.new"
move /Y "%PROGRAMDATA%\PostgreSQL\16\pg_hba.conf.new" "%PROGRAMDATA%\PostgreSQL\16\pg_hba.conf" 2>nul
pg_ctl reload -D "C:\Program Files\PostgreSQL\16\data" 2>nul

timeout /t 2 >nul

echo.
echo Service Status:
curl -s http://localhost:8081/health || curl -s http://localhost:8080/health
echo.
echo Components:
curl -s http://localhost:11434/api/tags >nul && echo [OK] Ollama GPU Ready || echo [FAIL] Ollama
redis-cli ping >nul 2>&1 && echo [OK] Redis || echo [FAIL] Redis
psql -U postgres -c "SELECT 1" >nul 2>&1 && echo [OK] PostgreSQL || echo [FAIL] PostgreSQL

echo.
echo Access: http://localhost:8081
