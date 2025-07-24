@echo off
echo ==============================================
echo      Legal AI Assistant - Error Fixing Tool
echo ==============================================
echo.

cd /d "%~dp0"

echo 1. Checking current directory structure...
echo.
if exist "sveltekit-frontend" (
    echo ✅ Found sveltekit-frontend directory
) else (
    echo ❌ sveltekit-frontend directory not found
    echo Please run this from the deeds-web-app root directory
    pause
    exit /b 1
)

echo.
echo 2. Moving to sveltekit-frontend directory...
echo.
cd sveltekit-frontend

echo 3. Installing/updating dependencies...
echo.
call npm install

echo.
echo 4. Running type check to identify errors...
echo.
call npm run check > ../error-check-results.txt 2>&1
if %errorlevel%==0 (
    echo ✅ No TypeScript errors found!
) else (
    echo ⚠️ TypeScript errors detected, check error-check-results.txt
)

echo.
echo 5. Build test to ensure everything compiles...
echo.
call npm run build > ../build-results.txt 2>&1
if %errorlevel%==0 (
    echo ✅ Build successful!
) else (
    echo ⚠️ Build issues detected, check build-results.txt
)

cd ..

echo.
echo ==============================================
echo           Fix Process Complete!
echo ==============================================
echo.
echo Results:
echo - Error check results: error-check-results.txt
echo - Build results: build-results.txt
echo.
echo If no critical errors appeared above, your project is ready!
echo.
echo To start development:
echo   - cd sveltekit-frontend
echo   - npm run dev
echo.
echo To run checks manually:
echo   - npm run check
echo   - npm run build
echo.
pause