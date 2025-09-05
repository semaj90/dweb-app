@echo off
title System Verification Tool
color 0E

echo.
echo =====================================
echo    DEEDS-WEB SYSTEM VERIFICATION
echo =====================================
echo.
echo Choose verification method:
echo.
echo [1] Run PowerShell version (Recommended for Windows)
echo [2] Run Node.js version (Requires Node.js)
echo [3] Quick status check
echo [4] Exit
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto powershell
if "%choice%"=="2" goto nodejs
if "%choice%"=="3" goto quick
if "%choice%"=="4" goto exit

:powershell
echo.
echo Running PowerShell verification...
echo.
powershell -ExecutionPolicy Bypass -File "scripts\verify-system.ps1"
goto end

:nodejs
echo.
echo Running Node.js verification...
echo.
node scripts\verify-system.mjs
goto end

:quick
echo.
echo Running quick status check...
echo.
echo Checking services...
echo ----------------------
netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] PostgreSQL - Port 5432
) else (
    echo [X] PostgreSQL - Port 5432 NOT LISTENING
)

netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Redis - Port 6379
) else (
    echo [X] Redis - Port 6379 NOT LISTENING
)

netstat -an | findstr ":11434" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Ollama - Port 11434
) else (
    echo [X] Ollama - Port 11434 NOT LISTENING
)

netstat -an | findstr ":5173" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Dev Server - Port 5173
) else (
    echo [X] Dev Server - Port 5173 NOT LISTENING
)

echo.
echo Testing API endpoint...
echo ----------------------
curl -s -o nul -w "Dev Server Response: %%{http_code}\n" http://localhost:5173/ 2>nul
echo.
goto end

:end
echo.
pause
goto :eof

:exit
exit /b 0
