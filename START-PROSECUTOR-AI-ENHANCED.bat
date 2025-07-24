@echo off
color 0A
echo =====================================================
echo 🎯 PROSECUTOR AI - LEGAL CASE MANAGEMENT SYSTEM
echo =====================================================
echo.
echo 🏛️  Welcome to Prosecutor AI - Your AI-Powered Legal Assistant
echo 📁 Advanced Evidence Management & Case Analysis Platform
echo.
echo =====================================================
echo 📋 EVIDENCE SYSTEM FEATURES
echo =====================================================
echo.
echo The evidence system supports:
echo 1. 📂 **Drag files** from your computer onto the evidence board
echo 2. ➕ **Click "ADD EVIDENCE"** to open the upload dialog
echo 3. 📄 **Multiple file types:** PDF, images (JPG, PNG, GIF), videos (MP4, MOV, AVI), documents
echo 4. 🏷️  **File metadata:** Automatic file size, type, and thumbnail generation
echo 5. 📊 **Evidence organization:** Categorize and prioritize uploaded evidence
echo.
echo =====================================================
echo 🚀 STARTING PROSECUTOR AI
echo =====================================================
echo.

REM Navigate to the correct directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo 📍 Current Directory: %CD%
echo.

REM Check if the PowerShell script exists
if not exist "FIX-AND-START-PROSECUTOR-AI.ps1" (
    echo ❌ PowerShell fix script not found!
    echo 📍 Please ensure you're in the correct directory.
    echo.
    goto MANUAL_START
)

echo 🔓 Running comprehensive PowerShell fix script...
echo 🔧 This will check dependencies, fix configuration issues, and start the server.
echo.

REM Run the PowerShell script with execution policy bypass
powershell.exe -ExecutionPolicy Bypass -File "FIX-AND-START-PROSECUTOR-AI.ps1"

if %errorlevel% neq 0 (
    echo.
    echo ⚠️  PowerShell script encountered issues. Trying manual startup...
    goto MANUAL_START
) else (
    goto SUCCESS
)

:MANUAL_START
echo =====================================================
echo 🔧 MANUAL STARTUP PROCEDURE
echo =====================================================
echo.

REM Navigate to frontend directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo 📍 Switching to frontend directory: %CD%
echo.

echo 🔍 Checking Node.js installation...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found! 
    echo 📥 Please install Node.js from: https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo ✅ Node.js found: 
node --version
echo.

echo 🔍 Checking npm installation...
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm not found! Please install npm first.
    pause
    exit /b 1
)

echo ✅ npm found:
npm --version
echo.

echo 📦 Checking dependencies...
if not exist "node_modules" (
    echo 📥 Installing dependencies...
    npm install
    if %errorlevel% neq 0 (
        echo ❌ npm install failed!
        echo 🔧 Try running: npm cache clean --force
        pause
        exit /b 1
    )
) else (
    echo ✅ Dependencies already installed
)

echo.
echo 🚀 Starting Prosecutor AI Development Server...
echo.
echo =====================================================
echo 📱 ACCESS YOUR APPLICATION
echo =====================================================
echo 🌐 URL: http://localhost:5173
echo 🛑 Press Ctrl+C to stop the server
echo 📋 Use the evidence features described above once loaded
echo =====================================================
echo.

REM Try multiple startup methods
echo 🔥 Attempting to start development server...

npm run dev
if %errorlevel% equ 0 goto SUCCESS

echo.
echo ⚠️  Standard startup failed. Trying clean startup...
npm run clean
npm run dev:clean
if %errorlevel% equ 0 goto SUCCESS

echo.
echo ⚠️  Clean startup failed. Trying safe startup...
npm run dev:safe
if %errorlevel% equ 0 goto SUCCESS

echo.
echo ❌ All automatic startup methods failed.
echo.
echo 🔧 TROUBLESHOOTING STEPS:
echo    1. npm run fix:all
echo    2. npm run clean
echo    3. npm install
echo    4. npm run dev
echo.
echo 📝 Check the error messages above for specific issues.
goto END

:SUCCESS
echo.
echo =====================================================
echo 🎉 PROSECUTOR AI STARTED SUCCESSFULLY!
echo =====================================================
echo.
echo 🌐 Your application is running at: http://localhost:5173
echo.
echo 📋 QUICK START GUIDE:
echo • Navigate to the Evidence section
echo • Drag files onto the evidence board OR click "ADD EVIDENCE"
echo • Supported formats: PDF, JPG, PNG, GIF, MP4, MOV, AVI, DOC, DOCX
echo • Files will automatically generate metadata and thumbnails
echo • Use the organization tools to categorize your evidence
echo.
echo 🎯 FEATURES AVAILABLE:
echo • Case Management
echo • Evidence Upload & Organization  
echo • AI-Powered Legal Analysis
echo • Document Processing
echo • Interactive Evidence Canvas
echo • Real-time Collaboration
echo.

:END
echo.
echo 📝 Session completed. Safe to close this window.
pause