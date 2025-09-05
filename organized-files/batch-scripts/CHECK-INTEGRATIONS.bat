@echo off
cls
title Legal AI Platform - Integration Status
color 0E

echo ========================================================
echo    LEGAL AI PLATFORM - COMPLETE INTEGRATION CHECK
echo ========================================================
echo.

echo Checking all component integrations...
echo.

:: Quick check for Node.js
where node >nul 2>&1
if %errorlevel% == 0 (
    echo [RUNNING] Full integration check with Node.js...
    node check-all-integrations.mjs
    
    echo.
    echo ========================================================
    echo Would you like to test actual data flow? (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        echo.
        echo [TESTING] Running data flow tests...
        node test-integration-flow.mjs
    )
) else (
    echo [ERROR] Node.js not found!
    echo Please install Node.js to run integration checks.
    echo.
    echo Falling back to basic status check...
    pause
    call CHECK-NATIVE-STATUS.bat
)

echo.
echo ========================================================
echo Integration check complete!
echo.
echo For detailed information, check:
echo - integration-status-report.json
echo - integration-test-results.json
echo.
echo To start all services, run:
echo START-NATIVE-WINDOWS-COMPLETE.ps1
echo ========================================================
pause
