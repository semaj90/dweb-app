@echo off
rem =============================================================================
rem VERIFY LOCAL OLLAMA MODELS
rem Quick verification that local models are ready
rem =============================================================================

echo.
echo Checking Ollama status...
echo.

rem Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Ollama not running. Starting...
    start "Ollama" ollama serve
    timeout /t 5 /nobreak >nul
)

rem List available models
echo Available local models:
echo ====================================
ollama list
echo ====================================
echo.

rem Quick test of gemma3-legal
echo Testing gemma3-legal model...
echo.
echo "Summarize: This is a test legal document." | ollama run gemma3-legal --verbose

echo.
echo ====================================
echo.
echo If gemma3-legal is listed above and responded correctly,
echo your system is ready to process legal documents!
echo.
echo The Go server expects:
echo   - Model name: gemma3-legal
echo   - Endpoint: http://localhost:11434
echo.
pause
