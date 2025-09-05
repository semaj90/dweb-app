@echo off
REM Gemma3 vLLM Direct Server Launcher
REM This script starts the Gemma3 GGUF model server using vLLM

echo ======================================
echo    Gemma3 vLLM Direct Server
echo ======================================
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

REM Install required packages if not present
echo 🔧 Checking dependencies...
python -c "import vllm" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing vLLM...
    pip install vllm
    if %errorlevel% neq 0 (
        echo ⚠️  vLLM installation failed, falling back to llama-cpp-python
        pip install llama-cpp-python
    )
)

python -c "import fastapi" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing FastAPI...
    pip install fastapi uvicorn
)

python -c "import requests" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing requests...
    pip install requests
)

echo.
echo 🚀 Starting Gemma3 vLLM Server...
echo 🌐 Server will be available at: http://localhost:8001
echo 📚 API documentation: http://localhost:8001/docs
echo 🏥 Health check: http://localhost:8001/health
echo.
echo ⏹️  Press Ctrl+C to stop the server
echo.

REM Start the server
python direct-gemma3-vllm-server.py

REM If we get here, the server stopped
echo.
echo 🔄 Server stopped.
pause
