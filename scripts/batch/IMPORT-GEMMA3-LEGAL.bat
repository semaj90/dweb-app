@echo off
echo ==========================================
echo   Importing Gemma3 Legal AI Model
echo ==========================================
echo.

echo 📁 Checking model file...
if not exist "gemma3Q4_K_M\mohf16-Q4_K_M.gguf" (
    echo ❌ Model file not found: gemma3Q4_K_M\mohf16-Q4_K_M.gguf
    echo Please ensure the model file is in the correct location.
    pause
    exit /b 1
)

echo ✅ Model file found
echo.

echo 📝 Creating Ollama model from Modelfile...
ollama create gemma3-legal -f Modelfile-gemma3-legal

if %errorlevel% equ 0 (
    echo.
    echo ✅ Successfully created gemma3-legal model!
    echo.
    echo 📝 Available models:
    ollama list
    echo.
    echo 🗺️ Test the model:
    echo ollama run gemma3-legal "What are the key elements of a contract?"
) else (
    echo.
    echo ❌ Failed to create model. Check the error messages above.
    echo.
    echo 🔍 Troubleshooting:
    echo 1. Ensure Ollama is running
    echo 2. Check model file path and permissions
    echo 3. Verify Modelfile syntax
)

echo.
echo Press any key to continue...
pause >nul