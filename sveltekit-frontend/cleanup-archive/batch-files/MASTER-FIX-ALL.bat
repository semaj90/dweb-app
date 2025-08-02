@echo off
echo ============================================
echo    FIXING ALL SVELTEKIT ERRORS
echo ============================================
echo.

echo [1/5] Running comprehensive fixes...
node comprehensive-fix.mjs
if errorlevel 1 (
    echo Comprehensive fixes failed!
    pause
    exit /b 1
)

timeout /t 2 >nul

echo.
echo [2/5] Running TypeScript fixes...
node fix-all-typescript-errors.mjs
if errorlevel 1 (
    echo TypeScript fixes failed!
    pause
    exit /b 1
)

timeout /t 2 >nul

echo.
echo [3/5] Running CSS fixes...
node fix-css-issues.mjs
if errorlevel 1 (
    echo CSS fixes failed!
    pause
    exit /b 1
)

timeout /t 2 >nul

echo.
echo [4/5] Running import fixes...
node fix-imports.mjs
if errorlevel 1 (
    echo Import fixes failed!
    pause
    exit /b 1
)

timeout /t 2 >nul

echo.
echo [5/5] Syncing SvelteKit and checking...
call npm run prepare
if errorlevel 1 (
    echo SvelteKit sync failed!
    echo Continuing anyway...
)

echo.
echo ============================================
echo    RUNNING TYPE CHECK
echo ============================================
echo.

call npm run check > check-results.txt 2>&1
type check-results.txt | find /c "error"
echo.

echo ============================================
echo    ALL FIXES COMPLETED!
echo ============================================
echo.
echo Check the results above for any remaining issues.
echo.

echo Would you like to:
echo 1. Start the dev server
echo 2. Start the NieR showcase
echo 3. Run check again
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Starting dev server...
    npm run dev
) else if "%choice%"=="2" (
    echo Starting NieR showcase...
    npm run showcase
) else if "%choice%"=="3" (
    echo Running check again...
    npm run check
) else (
    echo Done! Your SvelteKit app should now have minimal errors.
)

pause
