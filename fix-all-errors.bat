@echo off
echo ==============================================
echo      Legal AI Assistant - Error Fixing Tool
echo ==============================================
echo.

cd /d "%~dp0"

echo 1. Running comprehensive fix script...
echo.
node comprehensive-fix-and-setup.mjs
echo.

echo 2. Installing/updating dependencies...
echo.
cd sveltekit-frontend
call npm install
echo.

echo 3. Running type check to verify fixes...
echo.
call npm run check
echo.

echo 4. Build test to ensure everything compiles...
echo.
call npm run build
echo.

echo ==============================================
echo           Fix Process Complete!
echo ==============================================
echo.
echo If no errors appeared above, your project is ready!
echo.
echo To start development:
echo   - cd sveltekit-frontend
echo   - npm run dev
echo.
echo To run checks manually:
echo   - npm run check
echo   - npm run type-check
echo.
pause
