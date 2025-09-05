@echo off
REM Test local Gemma3 Legal AI model setup

echo ========================================
echo Testing Gemma3 Legal AI Model
echo ========================================
echo.

echo [1/4] Checking if Ollama is running...
curl -f http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ollama is not running. Starting Docker services...
    docker-compose up -d ollama
    echo Waiting for Ollama to start...
    timeout /t 30 /nobreak >nul
)
echo Ollama is running ✓
echo.

echo [2/4] Checking available models...
curl -s http://localhost:11434/api/tags | findstr "legal-ai"
if %errorlevel% neq 0 (
    echo Setting up legal-ai model from local Gemma3...
    docker exec legal_ai_ollama /tmp/setup-models.sh
) else (
    echo legal-ai model found ✓
)
echo.

echo [3/4] Testing legal-ai model...
echo Testing with legal question...
curl -X POST http://localhost:11434/api/generate ^
    -H "Content-Type: application/json" ^
    -d "{\"model\":\"legal-ai\",\"prompt\":\"Explain the difference between civil and criminal law in simple terms.\",\"stream\":false}" ^
    --max-time 60
echo.

echo [4/4] Testing case analysis...
echo Testing case analysis capability...
curl -X POST http://localhost:11434/api/generate ^
    -H "Content-Type: application/json" ^
    -d "{\"model\":\"legal-ai\",\"prompt\":\"As a prosecutor, how would you analyze a case with digital evidence of financial fraud?\",\"stream\":false}" ^
    --max-time 60
echo.

echo ========================================
echo Gemma3 Legal AI Test Complete!
echo ========================================
echo.
echo If you see JSON responses above, your model is working correctly.
echo You can now use the legal-ai model in your application.
echo.
pause