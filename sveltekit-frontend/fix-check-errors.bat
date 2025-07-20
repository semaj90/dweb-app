@echo off
echo ====================================
echo Running Svelte Check Error Fixes
echo ====================================
echo.

echo Step 1: Applying comprehensive fixes...
node fix-svelte-check-errors.mjs

echo.
echo Step 2: Running svelte-check to verify...
echo.
call npm run check

echo.
echo ====================================
echo Fix process complete!
echo ====================================
echo.
echo Check the output above for any remaining issues.
echo Most warnings about unused CSS selectors (dark mode) can be ignored.
echo.
pause
