@echo off
title Enhanced Legal AI Setup and Launch
echo Enhanced Legal AI Quick Launcher
echo ===================================
echo.

REM Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell check'" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell not available. Running basic launcher...
    echo [INFO] Starting development server...
    npm run dev
    pause
    exit /b
)

echo Select an option:
echo 1. Quick Start (assumes setup is done)
echo 2. Full Setup (first time or after updates)
echo 3. Run Tests Only
echo 4. Show Help
echo 5. Manual NPM Dev Start
echo.

set /p choice="Enter choice (1-5): "

if "%choice%"=="1" (
    echo [INFO] Starting Quick Launch...
    powershell -ExecutionPolicy Bypass -File "launch-enhanced-legal-ai.ps1" -QuickStart
) else if "%choice%"=="2" (
    echo [INFO] Starting Full Setup...
    powershell -ExecutionPolicy Bypass -File "launch-enhanced-legal-ai.ps1" -FullSetup
) else if "%choice%"=="3" (
    echo [INFO] Running Tests...
    powershell -ExecutionPolicy Bypass -File "launch-enhanced-legal-ai.ps1" -TestOnly
) else if "%choice%"=="4" (
    powershell -ExecutionPolicy Bypass -File "launch-enhanced-legal-ai.ps1" -Help
) else if "%choice%"=="5" (
    echo [INFO] Starting manual NPM development server...
    npm install
    npm run dev
) else (
    echo [WARNING] Invalid choice. Running Quick Start...
    powershell -ExecutionPolicy Bypass -File "launch-enhanced-legal-ai.ps1" -QuickStart
)

echo.
echo Press any key to close this window...
pause >nul
