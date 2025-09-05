@echo off
echo Starting Legal AI Go Server...
echo.

REM Set environment variables
set OLLAMA_URL=http://localhost:11434
set DATABASE_URL=postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db
set PORT=8080
set CUDA_AVAILABLE=true
set CUDA_DEVICE_COUNT=1

echo Environment Configuration:
echo - Ollama URL: %OLLAMA_URL%
echo - Database: %DATABASE_URL%
echo - Port: %PORT%
echo - CUDA Available: %CUDA_AVAILABLE%
echo.

REM Build the Go server if needed
if not exist legal-ai-server.exe (
    echo Building Go server...
    go build -o legal-ai-server.exe legal-ai-server.go
    if errorlevel 1 (
        echo Failed to build Go server
        pause
        exit /b 1
    )
)

REM Start the server
echo Starting Legal AI Server on port %PORT%...
legal-ai-server.exe

pause