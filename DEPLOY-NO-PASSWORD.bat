@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app

REM Skip PostgreSQL password prompts - use trust authentication
echo Configuring PostgreSQL for passwordless local access...
echo local   all   all                trust > "%PROGRAMDATA%\PostgreSQL\16\pg_hba.conf"
echo host    all   all   127.0.0.1/32 trust >> "%PROGRAMDATA%\PostgreSQL\16\pg_hba.conf"
echo host    all   all   ::1/128      trust >> "%PROGRAMDATA%\PostgreSQL\16\pg_hba.conf"

net stop postgresql-x64-16
net start postgresql-x64-16

REM Create database without password prompt
psql -U postgres -c "CREATE DATABASE IF NOT EXISTS docs_db;"
psql -U postgres -d docs_db -f schema.sql

REM Start services
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul
start /B "C:\Program Files\Memurai\memurai.exe"
start /B ollama serve

cd go-microservice
set CGO_ENABLED=0
go build -o ai-microservice.exe main_simple.go
start ai-microservice.exe
cd ..

timeout /t 3 >nul
curl http://localhost:8080/health
