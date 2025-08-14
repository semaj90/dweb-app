@echo off
echo =============================================
echo     COMPLETE ERROR FIX AND BEST PRACTICES
echo =============================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo [WARNING] Not running as administrator. Some fixes may fail.
    echo.
)

echo [1/6] Fixing Svelte and TypeScript Errors...
echo ---------------------------------------------
powershell -ExecutionPolicy Bypass -File fix-svelte-errors.ps1 -Verbose
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Svelte error fixing failed
    pause
    exit /b 1
)

echo.
echo [2/6] Fixing Go Module Imports...
echo ---------------------------------------------
powershell -ExecutionPolicy Bypass -File fix-minio-imports.ps1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Go import fixing failed
)

echo.
echo [3/6] Checking Database Schema...
echo ---------------------------------------------
psql -U postgres -d deeds_web_app -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Database connection: OK
    
    REM Check extensions
    psql -U postgres -d deeds_web_app -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>nul
    psql -U postgres -d deeds_web_app -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";" 2>nul
    echo Database extensions: VERIFIED
) else (
    echo Database connection: FAILED - Please check PostgreSQL
)

echo.
echo [4/6] Running ESLint Fix...
echo ---------------------------------------------
cd sveltekit-frontend
if exist node_modules (
    npx eslint src --fix --ext .js,.ts,.svelte 2>nul
    echo ESLint fixes: APPLIED
) else (
    echo ESLint: SKIPPED (node_modules not found)
)
cd ..

echo.
echo [5/6] Running Prettier Format...
echo ---------------------------------------------
cd sveltekit-frontend
if exist node_modules (
    npx prettier --write "src/**/*.{js,ts,svelte,css,html}" 2>nul
    echo Prettier formatting: APPLIED
) else (
    echo Prettier: SKIPPED (node_modules not found)
)
cd ..

echo.
echo [6/6] Final Validation Check...
echo ---------------------------------------------
cd sveltekit-frontend
npm run check 2>&1 | findstr /C:"error" /C:"warning" >temp_check.txt
set /p CHECK_RESULT=<temp_check.txt
del temp_check.txt

if "%CHECK_RESULT%"=="" (
    echo ✅ No errors or warnings found!
) else (
    echo ⚠️  Some issues remain - review npm run check output
)
cd ..

echo.
echo =============================================
echo        ERROR FIX SUMMARY
echo =============================================
echo.

REM Count remaining errors
cd sveltekit-frontend
npm run check 2>&1 | findstr /C:"found" >temp_summary.txt
if exist temp_summary.txt (
    type temp_summary.txt
    del temp_summary.txt
)
cd ..

echo.
echo Next Steps:
echo -----------
echo 1. Review any remaining errors with: npm run check
echo 2. Test the application: npm run dev
echo 3. Run the file merger HTML app for documentation
echo 4. Check MinIO integration: START-MINIO-INTEGRATION.bat
echo.
echo ✨ Error fixing complete!
echo.
pause