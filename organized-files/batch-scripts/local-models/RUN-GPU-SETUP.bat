@echo off
cls
echo.
echo ============================================
echo      OLLAMA GPU QUICK START
echo ============================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\local-models"

echo Checking GPU...
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo ERROR: NVIDIA tools not found (nvidia-smi missing); please install NVIDIA drivers.
    pause
    exit /b 1
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
if errorlevel 1 (
    echo ERROR: No NVIDIA GPU detected or driver not ready!
    pause
    exit /b 1
)

echo.
echo.
echo Starting Ollama with GPU support...
where ollama >nul 2>nul
if errorlevel 1 (
    echo ERROR: 'ollama' not found in PATH. Install Ollama and try again.
    pause
echo.
echo Running GPU setup/validation script...
if exist "setup-and-test-gpu.ps1" (
    powershell -ExecutionPolicy Bypass -File "setup-and-test-gpu.ps1"
) else (
    if exist "fix-ollama-gpu.ps1" (
        powershell -ExecutionPolicy Bypass -File "fix-ollama-gpu.ps1"
    ) else (
        echo INFO: No setup script found; skipping GPU validation.
    )
)

echo.
echo Waiting for Ollama server to become ready...
set /a __tries=0
:wait_ollama
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:11434/api/tags' -UseBasicParsing -TimeoutSec 1 >$null; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 (
    set /a __tries+=1
    if %__tries% GEQ 30 (
        echo ERROR: Ollama server did not become ready in time.
        pause
        exit /b 1
    )
    timeout /t 1 >nul
    goto :wait_ollama
)

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
