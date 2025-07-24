@echo off
title Prosecutor AI - Legal Case Management System
color 0A

echo =====================================================
echo 🎯 PROSECUTOR AI - LEGAL CASE MANAGEMENT SYSTEM
echo =====================================================
echo.
echo 🏛️  Welcome to Prosecutor AI
echo 📁 Advanced Evidence Management Platform
echo.

REM Navigate to the correct directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo 📍 Current Directory: %CD%
echo.

REM Check if the enhanced PowerShell script exists
if exist "LAUNCH-PROSECUTOR-AI.ps1" (
    echo 🚀 Running enhanced PowerShell launcher...
    echo 📋 This includes evidence system features and comprehensive setup.
    echo.
    powershell.exe -ExecutionPolicy Bypass -File "LAUNCH-PROSECUTOR-AI.ps1"
) else if exist "FIX-AND-START-PROSECUTOR-AI.ps1" (
    echo 🔧 Running PowerShell fix script...
    powershell.exe -ExecutionPolicy Bypass -File "FIX-AND-START-PROSECUTOR-AI.ps1"
) else (
    echo ❌ PowerShell scripts not found!
    echo 📍 Please ensure you're in the correct directory.
    echo.
    echo 🔧 Trying manual startup...
    cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
    
    if exist "package.json" (
        echo 📦 Installing dependencies...
        npm install
        echo.
        echo 🚀 Starting development server...
        npm run dev
    ) else (
        echo ❌ package.json not found! Please check your project structure.
    )
)

echo.
echo 📝 Launcher session completed.
pause