@echo off
title Legal AI Development Environment
color 0A

echo ================================================
echo        LEGAL AI DEVELOPMENT LAUNCHER
echo            Native Windows Mode
echo ================================================
echo.

:menu
echo Select startup mode:
echo.
echo [1] Full Stack (All services + monitoring)
echo [2] Enhanced (Frontend + Go API)
echo [3] Basic (Frontend only)
echo [4] GPU Mode (With GPU acceleration)
echo [5] Health Check
echo [6] Error Check
echo [7] Setup Environment
echo [8] Exit
echo.

set /p choice="Enter choice (1-8): "

if "%choice%"=="1" goto full_stack
if "%choice%"=="2" goto enhanced
if "%choice%"=="3" goto basic
if "%choice%"=="4" goto gpu_mode
if "%choice%"=="5" goto health_check
if "%choice%"=="6" goto error_check
if "%choice%"=="7" goto setup
if "%choice%"=="8" goto exit

:full_stack
echo.
echo Starting Full Stack Development Environment...
echo.
npm run dev:full
goto end

:enhanced
echo.
echo Starting Enhanced Development (Frontend + Go API)...
echo.
npm run dev:enhanced
goto end

:basic
echo.
echo Starting Basic Development (Frontend only)...
echo.
npm run dev
goto end

:gpu_mode
echo.
echo Starting GPU-Accelerated Development...
echo.
set ENABLE_GPU=true
powershell -ExecutionPolicy Bypass -File scripts\start-dev-windows.ps1 -GPUMode
goto end

:health_check
echo.
echo Running Health Check...
echo.
npm run test:health
pause
goto menu

:error_check
echo.
echo Running Error Check...
echo.
npm run check:errors
pause
goto menu

:setup
echo.
echo Setting up environment...
echo.
npm run setup
pause
goto menu

:exit
exit

:end
pause