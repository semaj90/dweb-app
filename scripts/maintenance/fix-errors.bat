@echo off
REM Automatic error fixing for Windows

echo ========================================
echo Legal AI Assistant - Error Fix Tool
echo ========================================
echo.

REM Create logs directory
if not exist logs mkdir logs

REM Install dependencies
echo Installing missing dependencies...
cd sveltekit-frontend
call npm install fuse.js
cd ..

REM Run fix scripts
echo.
echo Running error fixes...
call node fix-specific-errors.mjs

echo.
echo Fixing TypeScript imports...
call node fix-all-typescript-imports.mjs

echo.
echo Running final check...
cd sveltekit-frontend
call npm run check
cd ..

echo.
echo ========================================
echo Fix process complete!
echo Check the output above for any remaining errors.
echo ========================================
pause
