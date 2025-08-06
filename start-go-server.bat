@echo off
echo ========================================
echo Starting Legal AI Go GPU Server
echo ========================================

:: Set environment variables
set PORT=8081
set OLLAMA_URL=http://localhost:11434
set DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db

:: Change to go microservice directory
cd /d "%~dp0go-microservice"

:: Check if executable exists
if not exist "legal-ai-server.exe" (
    echo Building Go server...
    set CGO_ENABLED=0
    go build -o legal-ai-server.exe .
)

:: Start the server
echo Starting Legal AI GPU Server on port %PORT%...
echo Ollama URL: %OLLAMA_URL%
echo Database: %DATABASE_URL%
echo.

legal-ai-server.exe