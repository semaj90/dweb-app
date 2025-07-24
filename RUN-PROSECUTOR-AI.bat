@echo off
echo =====================================================
echo 🎯 PROSECUTOR AI - RUNNING POWERSHELL FIX SCRIPT
echo =====================================================
echo.

REM Navigate to the correct directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo 📍 Current Directory: %CD%
echo.

echo 🔓 Running PowerShell script with bypass execution policy...
echo.

REM Run the PowerShell script with execution policy bypass
powershell.exe -ExecutionPolicy Bypass -File "FIX-AND-START-PROSECUTOR-AI.ps1"

if %errorlevel% neq 0 (
    echo.
    echo ❌ PowerShell script encountered an error.
    echo 🔧 You can also try running these commands manually in PowerShell:
    echo    Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
    echo    npm install
    echo    npm run dev
    echo.
)

echo.
echo 📝 Script execution completed.
pause