@echo off
title Installation and Error Check
color 0E

echo.
echo ========================================
echo    FULL INSTALLATION AND ERROR CHECK
echo ========================================
echo.
echo This will:
echo   1. Install all dependencies
echo   2. Check for TypeScript errors
echo   3. Check for Svelte errors
echo   4. Verify service status
echo   5. Run security audit
echo   6. Test production build
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul

echo.
echo Starting comprehensive check...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Run the installation and check script
node scripts\install-and-check.mjs

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo    ERRORS DETECTED
    echo ========================================
    echo.
    echo Please review the errors above and the
    echo installation-check-results.json file
    echo for detailed information.
    echo.
) else (
    echo.
    echo ========================================
    echo    ALL CHECKS PASSED
    echo ========================================
    echo.
    echo System is ready for development!
    echo.
)

echo.
pause
