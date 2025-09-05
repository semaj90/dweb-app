@echo off
setlocal enabledelayedexpansion
title Legal AI - Error Free Launcher
color 0A

echo ========================================
echo LEGAL AI - ERROR FREE LAUNCHER
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "FRONTEND_PATH=%PROJECT_ROOT%sveltekit-frontend"

echo Checking environment...

echo [1/4] Checking Node.js...
node --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Node.js not found
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
echo ✅ Node.js available

echo [2/4] Checking project structure...
if not exist "%FRONTEND_PATH%\package.json" (
    echo ❌ Frontend package.json missing
    echo Path: %FRONTEND_PATH%
    pause
    exit /b 1
)
echo ✅ Project structure valid

echo [3/4] Installing dependencies...
cd /d "%FRONTEND_PATH%"
npm install
if !errorlevel! neq 0 (
    echo ❌ npm install failed
    pause
    exit /b 1
)
echo ✅ Dependencies installed

echo [4/4] Starting development server...
echo.
echo 📱 Server will be at: http://localhost:5173
echo 🛑 Press Ctrl+C to stop
echo.
start http://localhost:5173
npm run dev

echo Development server stopped.
pause
