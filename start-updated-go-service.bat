@echo off
echo 🚀 Starting Updated Go Legal AI Server
echo ======================================

cd go-microservice

set PORT=8080
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set PGPASSWORD=123456
set OLLAMA_URL=http://localhost:11434

echo 📊 Starting with environment:
echo   PORT: %PORT%
echo   DATABASE_URL: %DATABASE_URL%
echo.

legal-ai-server.exe

pause