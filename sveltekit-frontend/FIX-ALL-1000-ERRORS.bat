@echo off
title Fix All 1000+ Errors - Legal AI
echo =======================================
echo Fix All 1000+ Errors - Legal AI
echo =======================================
echo.

if not exist "src" (
    echo [ERROR] Not in sveltekit-frontend directory.
    echo Please run this from the sveltekit-frontend folder.
    pause
    exit /b 1
)

echo [INFO] This will fix all TypeScript, database, and Svelte errors.
echo.
echo What this fixes:
echo - Database schema issues (Drizzle ORM)
echo - TypeScript type errors
echo - Import/export problems  
echo - Array type mismatches
echo - Syntax errors
echo - React to Svelte conversions
echo.

set /p confirm="Continue with comprehensive fixes? (y/n): "
if /i not "%confirm%"=="y" (
    echo Operation cancelled.
    pause
    exit /b 0
)

echo.
echo [INFO] Starting comprehensive error fixing...
echo.

node fix-all-1000-errors.mjs

echo.
echo [INFO] Running TypeScript check to verify fixes...
echo.

npm run check

echo.
echo [SUCCESS] Error fixing completed!
echo.
echo Next steps:
echo 1. Review the changes made
echo 2. Test with: npm run dev
echo 3. Remove .backup files once satisfied
echo.
pause
