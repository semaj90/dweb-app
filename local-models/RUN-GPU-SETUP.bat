@echo off
cls
echo.
echo ============================================
echo      OLLAMA GPU QUICK START
echo ============================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\local-models"

echo Checking GPU...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if errorlevel 1 (
    echo ERROR: No NVIDIA GPU detected!
    pause
    exit /b 1
)

echo.
echo Setting GPU environment...
set OLLAMA_GPU_DRIVER=cuda
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_NUM_GPU=1
set OLLAMA_GPU_LAYERS=999

echo.
echo Starting Ollama with GPU support...
start /B ollama serve

timeout /t 3 >nul

echo.
echo Running setup and test script...
powershell -ExecutionPolicy Bypass -File "setup-and-test-gpu.ps1"

echo.
echo ============================================
echo.
echo Quick commands:
echo   ollama run gemma3-legal "your legal question"
echo   ollama run gemma3-quick "quick legal query"
echo   nvidia-smi -l 1  (monitor GPU in another window)
echo.
pause
