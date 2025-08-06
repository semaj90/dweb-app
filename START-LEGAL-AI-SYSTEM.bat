@echo off
REM START-LEGAL-AI-SYSTEM.bat
REM Quick launcher for the Legal AI System

echo ===============================================
echo    LEGAL AI SYSTEM - WINDOWS NATIVE EDITION
echo ===============================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [WARNING] Not running as Administrator
    echo Some features may not work properly
    echo.
)

REM Run the PowerShell startup script
echo Starting Legal AI System...
powershell -ExecutionPolicy Bypass -File "%~dp0Complete-Legal-AI-Startup.ps1"

pause
