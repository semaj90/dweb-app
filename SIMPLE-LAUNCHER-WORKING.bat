@echo off
title Simple Legal AI Launcher - Working Version
echo ========================================
echo SIMPLE LEGAL AI LAUNCHER
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%sveltekit-frontend"

if not exist "package.json" (
    echo ❌ package.json not found
    echo Current path: %CD%
    pause
    exit /b 1
)

echo ✅ Found package.json
echo Installing dependencies...
npm install
echo Starting development server...
npm run dev
pause
