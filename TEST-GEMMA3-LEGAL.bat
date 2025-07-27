@echo off
echo ==========================================
echo   Testing Gemma3 Legal AI Model
echo ==========================================
echo.

echo 🔍 Checking if model exists...
ollama list | findstr "gemma3-legal" >nul
if %errorlevel% neq 0 (
    echo ❌ gemma3-legal model not found
    echo Run IMPORT-GEMMA3-LEGAL.bat first to create the model
    pause
    exit /b 1
)

echo ✅ Model found!
echo.

echo 📝 Testing basic legal query...
echo Query: "What are the key elements of a valid contract?"
echo.
ollama run gemma3-legal "What are the key elements of a valid contract?"

echo.
echo ==========================================
echo 📝 Testing legal analysis query...
echo Query: "Analyze the enforceability of a non-compete clause"
echo.
ollama run gemma3-legal "Analyze the enforceability of a non-compete clause in an employment contract. What factors should be considered?"

echo.
echo ==========================================
echo ✅ Testing complete!
echo.
echo The model is ready for use in your SvelteKit application.
echo.
pause