@echo off
title Simple Legal AI Launcher
echo ========================================
echo SIMPLE LEGAL AI LAUNCHER
echo ========================================
echo.
cd /d "%~dp0sveltekit-frontend"
if not exist "package.json" (
    echo ❌ package.json not found
    pause
    exit /b 1
)
echo Installing dependencies...
npm install
echo Starting development server...
npm run dev
pause
