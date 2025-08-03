@echo off
REM 🚀 Legal AI System - One-Click Launcher
REM Windows 10 Batch Wrapper

title Legal AI System Launcher

echo.
echo 🏛️ LEGAL AI SYSTEM - ONE-CLICK LAUNCHER
echo =====================================
echo.

REM Check for parameters
set SETUP_MODE=
set GPU_MODE=
set QUICK_MODE=
set RESET_MODE=

:check_params
if "%1"=="--setup" set SETUP_MODE=-Setup
if "%1"=="--gpu" set GPU_MODE=-GPU
if "%1"=="--quick" set QUICK_MODE=-Quick
if "%1"=="--reset" set RESET_MODE=-Reset
if "%1"=="--help" goto show_help
shift
if not "%1"=="" goto check_params

REM Execute PowerShell script
echo 🔄 Launching PowerShell script...
powershell.exe -ExecutionPolicy Bypass -File "one-click-legal-ai-launcher.ps1" %SETUP_MODE% %GPU_MODE% %QUICK_MODE% %RESET_MODE%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Launch completed successfully!
    echo 🌐 Web Interface: http://localhost:5173
    echo.
    echo Press any key to exit...
    pause >nul
) else (
    echo.
    echo ❌ Launch failed. Check the output above for errors.
    echo.
    echo Press any key to exit...
    pause >nul
)

goto end

:show_help
echo.
echo 📋 USAGE:
echo    legal-ai-launcher.bat [options]
echo.
echo 🔧 OPTIONS:
echo    --setup     First-time setup (installs and configures everything)
echo    --gpu       Enable GPU acceleration for AI models
echo    --quick     Quick launch (skip health checks)
echo    --reset     Reset all databases and configurations
echo    --help      Show this help message
echo.
echo 💡 EXAMPLES:
echo    legal-ai-launcher.bat --setup --gpu     (First-time setup with GPU)
echo    legal-ai-launcher.bat --quick           (Quick launch)
echo    legal-ai-launcher.bat                   (Normal launch)
echo    legal-ai-launcher.bat --reset           (Reset everything)
echo.
pause

:end
