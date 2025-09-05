@echo off
title AI Integration Complete Check
color 0A

echo ================================================
echo    COMPLETE AI INTEGRATION CHECK
echo         Installing and Verifying
echo ================================================
echo.

echo This script will:
echo   1. Install all npm packages
echo   2. Run TypeScript and Svelte checks
echo   3. Verify all services
echo   4. Check database configuration
echo   5. Generate complete report
echo.

echo Press any key to start or Ctrl+C to cancel...
pause >nul

echo.
echo Starting PowerShell script...
echo.

powershell -ExecutionPolicy Bypass -File install-and-check.ps1

echo.
echo ================================================
echo           CHECK COMPLETE
echo ================================================
echo.
echo Check the generated report for details.
echo.
pause