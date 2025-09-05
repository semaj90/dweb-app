@echo off
echo ===========================================
echo Fixing Web App Docker and Ollama Errors
echo ===========================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running with administrator privileges...
) else (
    echo This script requires administrator privileges for Docker operations.
    echo Please run as administrator.
    pause
    exit /b 1
)

:: Run the PowerShell fix script
echo Running comprehensive fix script...
powershell -ExecutionPolicy Bypass -File "%~dp0fix-all-docker-ollama-errors.ps1"

echo.
echo ===========================================
echo Fix process completed!
echo ===========================================
echo.
echo To start development:
echo   npm run dev
echo.
pause
