@echo off
REM Ultimate Gemma3 Model Setup and Test
REM This script sets up and tests the custom Gemma3 model with multiple approaches

echo ==========================================
echo    Ultimate Gemma3 Model Setup
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if model file exists
set MODEL_PATH=C:\Users\james\Desktop\deeds-web\deeds-web-app\gemma3Q4_K_M\mo16.gguf
if not exist "%MODEL_PATH%" (
    echo ❌ Model file not found: %MODEL_PATH%
    echo Please ensure the GGUF model file is present.
    pause
    exit /b 1
)

echo ✅ Python found
echo ✅ Model file found: %MODEL_PATH%
echo.

REM Install all required packages
echo 🔧 Installing dependencies...
echo.

echo 📦 Installing core packages...
pip install fastapi uvicorn requests pydantic

echo 📦 Installing model backends...
REM Try vLLM first (best performance)
pip install vllm
if %errorlevel% neq 0 (
    echo ⚠️  vLLM installation failed, trying CPU version...
    pip install vllm-cpu
)

REM Install llama-cpp-python as fallback
echo 📦 Installing llama-cpp-python (fallback)...
pip install llama-cpp-python

echo.
echo 🎯 Choose your preferred approach:
echo.
echo 1. vLLM Server (Recommended - Best Performance)
echo 2. Direct llama-cpp-python Server
echo 3. Test All Approaches
echo 4. Exit
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto vllm_server
if "%choice%"=="2" goto direct_server
if "%choice%"=="3" goto test_all
if "%choice%"=="4" goto end

:vllm_server
echo.
echo 🚀 Starting vLLM Server...
echo.
START-GEMMA3-VLLM.bat
goto end

:direct_server
echo.
echo 🚀 Starting Direct Server...
echo.
GEMMA3-DIRECT-LOADER.bat
goto end

:test_all
echo.
echo 🧪 Testing All Approaches...
echo.

echo ----------------------------------------
echo Testing vLLM Server
echo ----------------------------------------
start /B START-GEMMA3-VLLM.bat

REM Wait a bit for server to start
timeout /t 10 /nobreak >nul

echo Running vLLM tests...
python test-gemma3-vllm.py

echo.
echo ----------------------------------------
echo Testing Direct Server
echo ----------------------------------------

REM Stop vLLM server (if running)
taskkill /f /im python.exe >nul 2>&1

REM Start direct server
start /B GEMMA3-DIRECT-LOADER.bat

REM Wait a bit for server to start
timeout /t 10 /nobreak >nul

echo Running direct server tests...
python test-gemma3-direct.py

echo.
echo 📊 All tests completed!
goto end

:end
echo.
echo ✅ Setup completed!
echo.
echo 🚀 Available scripts:
echo   • START-GEMMA3-VLLM.bat     - Start vLLM server (recommended)
echo   • GEMMA3-DIRECT-LOADER.bat  - Start direct llama-cpp server
echo   • TEST-GEMMA3-VLLM.bat      - Test vLLM server
echo   • test-gemma3-direct.py     - Test direct server
echo.
echo 🌐 Server endpoints (when running):
echo   • http://localhost:8001/v1/chat/completions (OpenAI compatible)
echo   • http://localhost:8001/v1/completions
echo   • http://localhost:8001/health
echo   • http://localhost:8001/docs (API documentation)
echo.
pause
