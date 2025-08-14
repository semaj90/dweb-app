@echo off
title Legal AI - Simple Starter
echo ========================================
echo Legal AI - Simple Starter
echo ========================================
echo.

cd /d "%~dp0sveltekit-frontend"

echo Checking if we're in the right directory...
if not exist "package.json" (
    echo ❌ package.json not found
    echo Current directory: %CD%
    echo Please run this from the deeds-web-app root directory
    pause
    exit /b 1
)

echo ✅ Found package.json
echo.

echo Checking Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found
    echo Please install Node.js from https://nodejs.org/
    echo Then restart this script
    pause
    exit /b 1
)

echo ✅ Node.js found:
node --version

echo.
echo Installing dependencies...
npm install
if %errorlevel% neq 0 (
    echo ❌ npm install failed
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed
echo.
echo Starting development server...
echo 📱 Server will be at: http://localhost:5173
echo 🛑 Press Ctrl+C to stop
echo.

start http://localhost:5173
npm run dev

pause