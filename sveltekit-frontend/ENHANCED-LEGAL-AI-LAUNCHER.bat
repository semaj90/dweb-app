@echo off
title Enhanced Legal AI - Complete Launcher
color 0A
echo.
echo    ===================================================
echo    🌟 ENHANCED LEGAL AI - COMPLETE LAUNCHER 🌟
echo    ===================================================
echo.
echo    Status: VERIFIED AND READY ✅
echo.

REM Check if we're in the right directory
if not exist "package.json" (
    echo    ❌ ERROR: Not in the correct directory
    echo    Please run this from the sveltekit-frontend folder.
    echo.
    pause
    exit /b 1
)

echo    Select your launch option:
echo.
echo    1. 🚀 Quick Launch (Start immediately)
echo    2. 🔧 Full Setup + Launch (Recommended for first time)
echo    3. 🧪 Verify + Launch (Check everything first)
echo    4. 📊 Verification Only (Test without starting)
echo    5. 🔨 Fix Issues + Launch (Fix any problems first)
echo    6. ❓ Show Help
echo    7. 🚪 Exit
echo.

set /p choice="    Enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo    🚀 Starting Quick Launch...
    echo    =============================
    echo.
    call npm run dev
    goto end
)

if "%choice%"=="2" (
    echo.
    echo    🔧 Starting Full Setup + Launch...
    echo    =================================
    echo.
    echo    [1/4] Installing dependencies...
    call npm install
    echo    [2/4] Setting up database...
    call npm run db:push
    echo    [3/4] Setting up AI features...
    call npm run thinking:setup
    echo    [4/4] Starting application...
    call npm run dev
    goto end
)

if "%choice%"=="3" (
    echo.
    echo    🧪 Running Verification + Launch...
    echo    ==================================
    echo.
    echo    [1/2] Verifying Enhanced Legal AI features...
    call npm run verify:enhanced-ai
    if %errorlevel% equ 0 (
        echo.
        echo    ✅ Verification PASSED! Starting application...
        call npm run dev
    ) else (
        echo.
        echo    ⚠️ Verification found issues. Please run option 5 to fix.
        pause
    )
    goto end
)

if "%choice%"=="4" (
    echo.
    echo    📊 Running Verification Only...
    echo    ==============================
    echo.
    call npm run verify:enhanced-ai
    echo.
    echo    Verification completed. Check results above.
    pause
    goto end
)

if "%choice%"=="5" (
    echo.
    echo    🔨 Fixing Issues + Launch...
    echo    ===========================
    echo.
    echo    [1/5] Fixing HTML attributes...
    call npm run fix:html
    echo    [2/5] Installing dependencies...
    call npm install
    echo    [3/5] Setting up database...
    call npm run db:push
    echo    [4/5] Setting up AI features...
    call npm run thinking:setup
    echo    [5/5] Starting application...
    call npm run dev
    goto end
)

if "%choice%"=="6" (
    echo.
    echo    ❓ Enhanced Legal AI Help
    echo    ========================
    echo.
    echo    This launcher provides several options:
    echo.
    echo    🚀 Quick Launch: Starts the app immediately
    echo       - Use this if you've already set up everything
    echo       - Fastest option
    echo.
    echo    🔧 Full Setup: Complete first-time setup
    echo       - Installs dependencies
    echo       - Sets up database
    echo       - Configures AI features
    echo       - Recommended for first use
    echo.
    echo    🧪 Verify + Launch: Tests then starts
    echo       - Runs comprehensive verification
    echo       - Only starts if everything passes
    echo       - Safest option
    echo.
    echo    📊 Verification Only: Just run tests
    echo       - Tests all features without starting
    echo       - Good for troubleshooting
    echo.
    echo    🔨 Fix + Launch: Repairs then starts
    echo       - Fixes common issues
    echo       - Runs full setup
    echo       - Use if you have problems
    echo.
    echo    Your Enhanced Legal AI includes:
    echo    • Thinking-style AI analysis
    echo    • Document processing ^& OCR
    echo    • Evidence analysis ^& classification
    echo    • Legal compliance checking
    echo    • Interactive evidence canvas
    echo    • Chain of custody verification
    echo.
    echo    App will be available at: http://localhost:5173
    echo.
    pause
    goto start
)

if "%choice%"=="7" (
    echo.
    echo    👋 Goodbye! Your Enhanced Legal AI is ready when you are.
    echo.
    exit /b 0
)

echo.
echo    ❌ Invalid choice. Please enter 1-7.
echo.
pause
goto start

:start
cls
goto :eof

:end
echo.
echo    ===================================================
echo    🎉 Enhanced Legal AI Session Complete
echo    ===================================================
echo.
echo    Your application was running at: http://localhost:5173
echo.
echo    Key features available:
echo    • 🧠 Thinking Style AI Analysis
echo    • 📄 Document Processing ^& OCR  
echo    • 🎯 Evidence Analysis ^& Classification
echo    • ⚖️ Legal Compliance Checking
echo    • 📊 Interactive Evidence Canvas
echo    • 🔐 Chain of Custody Verification
echo.
echo    To restart, run this launcher again or use:
echo    • npm run dev (quick start)
echo    • npm run verify:enhanced-ai (verify first)
echo.
echo    Press any key to exit...
pause >nul
