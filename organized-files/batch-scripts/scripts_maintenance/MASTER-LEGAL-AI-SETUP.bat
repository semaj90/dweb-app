@echo off
setlocal enabledelayedexpansion
title Legal AI - Master Setup
color 0A

set "PROJECT_ROOT=%~dp0"
set "FRONTEND_PATH=%PROJECT_ROOT%sveltekit-frontend"

:MAIN_MENU
cls
echo ========================================
echo 🎯 LEGAL AI - MASTER CONTROL PANEL
echo ========================================
echo.
echo 1. Quick Start (Recommended)
echo 2. Health Check
echo 3. Start Development Only
echo 4. Exit
echo.
set /p "choice=Choose (1-4): "

if "%choice%"=="1" goto QUICK_START
if "%choice%"=="2" goto HEALTH_CHECK
if "%choice%"=="3" goto DEV_ONLY
if "%choice%"=="4" exit /b 0

echo Invalid choice. Try again...
timeout /t 2 > nul
goto MAIN_MENU

:QUICK_START
echo.
echo === QUICK START ===
echo.
cd /d "%FRONTEND_PATH%"
if not exist "package.json" (
    echo ❌ package.json not found in %FRONTEND_PATH%
    pause
    goto MAIN_MENU
)
echo ✅ Found project directory
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
)
echo Starting development server...
start http://localhost:5173
npm run dev
goto END

:HEALTH_CHECK
echo.
echo === HEALTH CHECK ===
echo.
node --version >nul 2>&1 && echo ✅ Node.js: && node --version || echo ❌ Node.js not found
npm --version >nul 2>&1 && echo ✅ npm: && npm --version || echo ❌ npm not found
docker --version >nul 2>&1 && echo ✅ Docker: && docker --version || echo ❌ Docker not found
echo.
if exist "%FRONTEND_PATH%\package.json" (
    echo ✅ Frontend project found
) else (
    echo ❌ Frontend project not found
)
echo.
pause
goto MAIN_MENU

:DEV_ONLY
echo.
echo === DEVELOPMENT SERVER ===
echo.
cd /d "%FRONTEND_PATH%"
npm run dev
goto END

:END
echo.
echo Press any key to return to menu...
pause > nul
goto MAIN_MENU