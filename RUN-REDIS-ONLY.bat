@echo off
echo Starting AI System (No PostgreSQL)

REM Kill port 8080
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

REM Redis only
start /B "C:\Program Files\Memurai\memurai.exe"

REM Ollama
start /B ollama serve
timeout /t 2 >nul

REM Go service - Redis only mode
cd go-microservice
set CGO_ENABLED=0
go run main_simple.go
