@echo off
setlocal enabledelayedexpansion
title Legal AI - Enhanced Control Panel
color 0A

set "LOG_FILE=%~dp0HEALTH_CHECK_LOG_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log"
set "TODO_FILE=%~dp0TODO_GENERATED_%date:~-4,4%%date:~-10,2%%date:~-7,2%.md"

echo ========================================
echo LEGAL AI - ENHANCED CONTROL PANEL
echo ========================================
echo.

:: Check if AI stack is installed
if exist "sveltekit-frontend\src\lib\services\embedding-service.ts" (
    set "AI_INSTALLED=true"
) else (
    set "AI_INSTALLED=false"
)

echo AI Stack Status: !AI_INSTALLED!
echo.

:MENU
echo SYSTEM OPTIONS:
echo [1] Health Check + TypeScript Validation
echo [2] Setup Local AI Stack (LangChain + pgvector)
echo [3] Load Local Models (Gemma3 + Nomic)
echo [4] Fix TypeScript Issues
echo [5] Start Development Server
echo [6] View Generated TODOs
echo [0] Exit
echo.
set /p choice="Select option: "

if "%choice%"=="1" goto HEALTH_CHECK
if "%choice%"=="2" goto SETUP_AI
if "%choice%"=="3" goto LOAD_MODELS
if "%choice%"=="4" goto FIX_TS
if "%choice%"=="5" goto START_DEV
if "%choice%"=="6" goto VIEW_TODO
if "%choice%"=="0" goto EXIT
goto MENU

:HEALTH_CHECK
echo Running health check...
call AI-HEALTH-CHECK.bat
goto MENU

:SETUP_AI
echo Setting up AI stack...
powershell -ExecutionPolicy Bypass -File setup-local-ai-stack.ps1
goto MENU

:LOAD_MODELS
echo Loading models...
call LOAD-LOCAL-GEMMA3.bat
goto MENU

:FIX_TS
echo Fixing TypeScript...
call FIX-TYPESCRIPT-ISSUES.bat
goto MENU

:START_DEV
echo Starting development...
cd sveltekit-frontend
npm run dev
cd ..
goto MENU

:VIEW_TODO
if exist "%TODO_FILE%" (
    type "%TODO_FILE%"
) else (
    echo No TODO file found
)
pause
goto MENU

:EXIT
exit
