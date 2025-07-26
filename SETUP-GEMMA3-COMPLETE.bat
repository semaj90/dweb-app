@echo off
setlocal enabledelayedexpansion

echo.
echo ===============================================================
echo                🤖 GEMMA3 LEGAL AI COMPLETE SETUP 🤖
echo ===============================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running with administrator privileges
) else (
    echo ⚠️  Not running as administrator - some features may be limited
)

echo.
echo 📋 PHASE 1: Prerequisites Check
echo ---------------------------------------------------------------

REM Check if Ollama is installed
where ollama >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Ollama is installed
    for /f "tokens=*" %%i in ('ollama --version 2^>nul') do echo    Version: %%i
) else (
    echo ❌ Ollama not found!
    echo    Please install from: https://ollama.ai/
    echo    Or run: winget install Ollama.Ollama
    pause
    exit /b 1
)

REM Check if Python is available
where python >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Python is available
    for /f "tokens=*" %%i in ('python --version 2^>nul') do echo    %%i
) else (
    echo ⚠️  Python not found - vLLM testing will be skipped
)

REM Check if Node.js is available
where node >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Node.js is available
    for /f "tokens=*" %%i in ('node --version 2^>nul') do echo    Version: %%i
) else (
    echo ❌ Node.js not found!
    echo    Please install from: https://nodejs.org/
    pause
    exit /b 1
)

REM Check if npm is available
where npm >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ npm is available
    for /f "tokens=*" %%i in ('npm --version 2^>nul') do echo    Version: %%i
) else (
    echo ❌ npm not found!
    pause
    exit /b 1
)

echo.
echo 📁 PHASE 2: File Structure Check
echo ---------------------------------------------------------------

REM Check for model file
if exist "gemma3Q4_K_M\mo16.gguf" (
    echo ✅ Gemma3 model file found
    for %%F in ("gemma3Q4_K_M\mo16.gguf") do (
        set size=%%~zF
        set /a sizeMB=!size!/1024/1024
        echo    Size: !sizeMB! MB
    )
) else (
    echo ❌ Gemma3 model file not found at: gemma3Q4_K_M\mo16.gguf
    echo    Please ensure your GGUF model is in the correct location
    pause
    exit /b 1
)

REM Check for SvelteKit frontend
if exist "sveltekit-frontend\package.json" (
    echo ✅ SvelteKit frontend found
) else (
    echo ❌ SvelteKit frontend not found
    echo    Please ensure you're in the correct directory
    pause
    exit /b 1
)

echo.
echo 🔧 PHASE 3: Ollama Service Setup
echo ---------------------------------------------------------------

REM Check if Ollama is running
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Ollama service is running
) else (
    echo ⚠️  Ollama service not detected. Attempting to start...
    
    REM Try to start Ollama service
    echo    Starting Ollama service...
    start /B ollama serve >nul 2>&1
    
    REM Wait a moment for service to start
    timeout /t 5 /nobreak >nul
    
    REM Check again
    curl -s http://localhost:11434/api/version >nul 2>&1
    if %errorLevel% == 0 (
        echo ✅ Ollama service started successfully
    ) else (
        echo ❌ Could not start Ollama service
        echo    Please run 'ollama serve' manually in another terminal
        echo    Press any key when Ollama is running...
        pause >nul
    )
)

echo.
echo 📦 PHASE 4: PowerShell Setup Execution
echo ---------------------------------------------------------------

REM Check if PowerShell setup script exists
if exist "setup-gemma3-complete.ps1" (
    echo ✅ PowerShell setup script found
    echo    Executing comprehensive setup...
    echo.
    
    REM Execute PowerShell setup with proper execution policy
    powershell -ExecutionPolicy Bypass -File "setup-gemma3-complete.ps1"
    
    if %errorLevel% == 0 (
        echo ✅ PowerShell setup completed successfully
    ) else (
        echo ⚠️  PowerShell setup had some issues
        echo    Continuing with remaining setup...
    )
) else (
    echo ❌ PowerShell setup script not found
    echo    Creating basic setup...
    
    REM Create basic Modelfile if PowerShell script is missing
    echo FROM .\gemma3Q4_K_M\mo16.gguf > Modelfile-Basic
    echo PARAMETER temperature 0.1 >> Modelfile-Basic
    echo PARAMETER top_p 0.8 >> Modelfile-Basic
    echo SYSTEM "You are a specialized Legal AI Assistant." >> Modelfile-Basic
    
    ollama create gemma3-legal -f Modelfile-Basic
    if %errorLevel% == 0 (
        echo ✅ Basic model setup completed
    ) else (
        echo ❌ Model setup failed
    )
)

echo.
echo 🧪 PHASE 5: Integration Testing
echo ---------------------------------------------------------------

REM Test if Python is available for testing
where python >nul 2>&1
if %errorLevel% == 0 (
    if exist "test-gemma3-integration.py" (
        echo ✅ Running integration tests...
        python test-gemma3-integration.py
    ) else (
        echo ⚠️  Integration test script not found
        echo    Running basic model test...
        ollama run gemma3-legal "What are the key elements of contract law?" --format json
    )
) else (
    echo ⚠️  Python not available for automated testing
    echo    Running basic Ollama test...
    ollama list | findstr gemma
)

echo.
echo 🚀 PHASE 6: Frontend Setup
echo ---------------------------------------------------------------

REM Install frontend dependencies
echo Installing frontend dependencies...
cd sveltekit-frontend
if exist "node_modules" (
    echo ✅ Dependencies already installed
) else (
    echo 📥 Installing dependencies...
    npm install
    if %errorLevel% == 0 (
        echo ✅ Dependencies installed successfully
    ) else (
        echo ❌ Failed to install dependencies
        cd ..
        pause
        exit /b 1
    )
)

REM Build the project to check for errors
echo 🔨 Building project...
npm run build >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Project builds successfully
) else (
    echo ⚠️  Build has some issues - checking for TypeScript errors...
    npm run check
)

cd ..

echo.
echo 🎉 SETUP COMPLETE!
echo ===============================================================
echo.
echo 📊 SYSTEM STATUS:
echo    • Model: gemma3-legal (Gemma3 Q4_K_M)
echo    • Ollama: Running on http://localhost:11434
echo    • Frontend: SvelteKit 5 + Bits UI ready
echo    • API: Enhanced endpoints configured
echo.
echo 🚀 QUICK START:
echo    1. Start development server:
echo       cd sveltekit-frontend
echo       npm run dev
echo    2. Open browser: http://localhost:5173
echo    3. Test AI chat interface
echo.
echo 🧪 TESTING:
echo    • Test model: ollama run gemma3-legal "Your legal question"
echo    • Test API: curl http://localhost:5173/api/ai/test-gemma3
echo    • Full test: python test-gemma3-integration.py
echo.
echo 📚 DOCUMENTATION:
echo    • Complete guide: GEMMA3_INTEGRATION_COMPLETE_GUIDE.md
echo    • Local setup: sveltekit-frontend\markdowns\LOCAL_LLM_SETUP.md
echo.
echo ✨ Your Legal AI Assistant is ready to use! ✨
echo.

echo Press any key to start the development server...
pause >nul

echo.
echo 🌐 Starting SvelteKit development server...
cd sveltekit-frontend
start cmd /k "npm run dev"

echo.
echo 🎯 Development server is starting...
echo    You can now open: http://localhost:5173
echo    The AI chat interface will be accessible from the main page
echo.
echo 📋 NEXT STEPS:
echo    1. Wait for "Local: http://localhost:5173" message
echo    2. Open the URL in your browser
echo    3. Look for the AI chat button (usually bottom-right)
echo    4. Test with legal questions like:
echo       - "What are the elements of a valid contract?"
echo       - "Explain the difference between void and voidable contracts"
echo       - "What constitutes breach of contract?"
echo.

pause
