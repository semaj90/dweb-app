@echo off
REM Ollama GPU Launcher for Windows - Corrected Version
echo.
echo ========================================
echo    OLLAMA GPU ACCELERATED LAUNCHER
echo ========================================
echo.

REM Set only the necessary environment variable
echo Setting GPU environment...
set CUDA_VISIBLE_DEVICES=0

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: NVIDIA GPU not detected or drivers not installed
    pause
    exit /b 1
)

echo GPU Configuration:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo.

REM Kill existing Ollama process
echo Stopping existing Ollama instances...
taskkill /F /IM ollama.exe >nul 2>&1
timeout /t 2 >nul

REM Start Ollama
echo Starting Ollama server...
start /B ollama serve

REM Wait for Ollama to start
echo Waiting for Ollama to initialize...
timeout /t 5 >nul

REM Verify Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Ollama failed to start
    echo Check logs at: %%LOCALAPPDATA%%\Ollama\logs\server.log
    pause
    exit /b 1
)

echo.
echo ✓ Ollama is running with GPU acceleration ready!
echo.
echo IMPORTANT: GPU settings are controlled in the Modelfile
echo.
echo Next steps:
echo 1. Create models (if not already created):
echo    ollama create gemma3-legal -f Modelfile.gemma3-legal
echo    ollama create gemma3-quick -f Modelfile.gemma3-quick
echo.
echo 2. Test the models:
echo    ollama run gemma3-legal "Your legal question"
echo    ollama run gemma3-quick "Quick legal query"
echo.
echo 3. Monitor GPU usage (in another terminal):
echo    nvidia-smi -l 1
echo.
echo Press any key to keep Ollama running in background...
pause >nul
