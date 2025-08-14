@echo off
REM Test Gemma3 vLLM Server
REM This script tests the vLLM server endpoints

echo ======================================
echo    Test Gemma3 vLLM Server
echo ======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install requests if not present
python -c "import requests" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing requests...
    pip install requests
)

echo 🧪 Running Gemma3 vLLM Server Tests...
echo 🌐 Testing server at: http://localhost:8001
echo.

REM Run the test script
python test-gemma3-vllm.py

echo.
echo 📊 Test completed!
pause
