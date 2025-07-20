@echo off
echo ========================================
echo Fixing all TypeScript and CSS errors...
echo ========================================
echo.

echo [1/4] Running TypeScript fixes...
node fix-all-typescript-errors.mjs
if errorlevel 1 (
    echo TypeScript fixes failed!
    pause
    exit /b 1
)

echo.
echo [2/4] Running CSS fixes...
node fix-css-issues.mjs
if errorlevel 1 (
    echo CSS fixes failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Syncing SvelteKit...
call npm run prepare
if errorlevel 1 (
    echo SvelteKit sync failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Running type check...
call npm run check > check-results.txt 2>&1
type check-results.txt

echo.
echo ========================================
echo All fixes completed!
echo Check the results above for any remaining issues.
echo ========================================
echo.

echo Would you like to start the dev server? (Y/N)
set /p response=
if /i "%response%"=="Y" (
    echo Starting dev server...
    npm run dev
) else (
    echo Done! Run "npm run dev" when ready.
)

pause
