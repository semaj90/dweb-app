@echo off
echo 🚀 Running Complete Gemma3 Integration Fix...
echo.

REM Check if PowerShell is available
powershell -Command "exit" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ PowerShell is required but not available
    pause
    exit /b 1
)

REM Run the PowerShell fix script
echo 📋 Executing COMPLETE_GEMMA3_FIX.ps1...
powershell -ExecutionPolicy Bypass -File "COMPLETE_GEMMA3_FIX.ps1"

if %errorlevel% equ 0 (
    echo.
    echo ✅ Gemma3 Integration Fix Completed Successfully!
    echo.
    echo 📋 Next Steps:
    echo    1. Run: npm run dev
    echo    2. Navigate to: http://localhost:5173/test-gemma3
    echo    3. Test the Gemma3 integration
    echo.
) else (
    echo.
    echo ❌ Fix script encountered errors
    echo    Check the output above for details
    echo.
)

pause
