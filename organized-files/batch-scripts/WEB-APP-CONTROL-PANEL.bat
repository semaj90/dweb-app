@echo off
cls
echo =========================================
echo     Web App Setup & Error Fix Tool
echo =========================================
echo.

:: Display menu
echo Please select an option:
echo.
echo [1] Complete Setup (Docker + Ollama + SvelteKit)
echo [2] Fix All TypeScript Errors
echo [3] Quick Fix (All errors)
echo [4] Start Development Server
echo [5] Check Service Status
echo [6] Reset Everything (Clean install)
echo [7] Exit
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto complete_setup
if "%choice%"=="2" goto fix_typescript
if "%choice%"=="3" goto quick_fix
if "%choice%"=="4" goto start_dev
if "%choice%"=="5" goto check_status
if "%choice%"=="6" goto reset_all
if "%choice%"=="7" goto end

:complete_setup
echo.
echo Running complete setup...
powershell -ExecutionPolicy Bypass -File "%~dp0complete-setup-docker-ollama.ps1"
pause
goto menu

:fix_typescript
echo.
echo Fixing TypeScript errors...
cd sveltekit-frontend
node fix-all-typescript-errors.mjs
cd ..
pause
goto menu

:quick_fix
echo.
echo Running quick fix for all errors...
powershell -ExecutionPolicy Bypass -File "%~dp0fix-all-docker-ollama-errors.ps1"
pause
goto menu

:start_dev
echo.
echo Starting development server...
cd sveltekit-frontend
start cmd /k "npm run dev"
cd ..
echo Development server started in new window.
pause
goto menu

:check_status
echo.
echo Checking service status...
echo.
echo === Docker Services ===
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.
echo === Ollama Models ===
docker exec prosecutor_ollama ollama list 2>nul
echo.
echo === Port Status ===
powershell -Command "Test-NetConnection localhost -Port 5173 | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded"
powershell -Command "Test-NetConnection localhost -Port 11434 | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded"
powershell -Command "Test-NetConnection localhost -Port 5432 | Select-Object -Property ComputerName,RemotePort,TcpTestSucceeded"
pause
goto menu

:reset_all
echo.
echo WARNING: This will delete all data and start fresh!
set /p confirm="Are you sure? (y/N): "
if /i "%confirm%"=="y" (
    echo Resetting everything...
    docker-compose down -v
    rmdir /s /q node_modules 2>nul
    rmdir /s /q sveltekit-frontend\node_modules 2>nul
    rmdir /s /q sveltekit-frontend\.svelte-kit 2>nul
    del /q sveltekit-frontend\dev.db* 2>nul
    echo Reset complete. Run option 1 to set up again.
) else (
    echo Reset cancelled.
)
pause
goto menu

:menu
cls
echo =========================================
echo     Web App Setup & Error Fix Tool
echo =========================================
echo.
echo Please select an option:
echo.
echo [1] Complete Setup (Docker + Ollama + SvelteKit)
echo [2] Fix All TypeScript Errors
echo [3] Quick Fix (All errors)
echo [4] Start Development Server
echo [5] Check Service Status
echo [6] Reset Everything (Clean install)
echo [7] Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto complete_setup
if "%choice%"=="2" goto fix_typescript
if "%choice%"=="3" goto quick_fix
if "%choice%"=="4" goto start_dev
if "%choice%"=="5" goto check_status
if "%choice%"=="6" goto reset_all
if "%choice%"=="7" goto end

echo Invalid choice. Please try again.
pause
goto menu

:end
echo.
echo Thank you for using the Web App Setup Tool!
echo.
pause
