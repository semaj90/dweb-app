@echo off
REM Simple launcher - No Neo4j required

echo Starting AI System (No Neo4j)...

REM 1. PostgreSQL
net start postgresql-x64-16 2>nul
pg_isready >nul 2>&1 || echo PostgreSQL not running

REM 2. Redis/Memurai
start /B "C:\Program Files\Memurai\memurai.exe" 2>nul

REM 3. Ollama
start /B ollama serve
timeout /t 2 >nul
ollama pull nomic-embed-text 2>nul

REM 4. Go Service (Port 8080 is already running based on your logs)
cd go-microservice
if not exist ai-microservice.exe (
    go build -tags nogpu -o ai-microservice.exe .
)
start /B ai-microservice.exe

echo.
echo Services:
echo   AI Service: http://localhost:8080 (Already Running)
echo   Ollama: http://localhost:11434
echo.
pause
