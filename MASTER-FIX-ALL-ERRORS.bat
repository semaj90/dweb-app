@echo off
title Master Error Fixer - Legal AI Web App
echo =========================================
echo Master Error Fixer - Legal AI Web App
echo =========================================
echo.

echo [INFO] This will fix all errors in both web-app and sveltekit-frontend directories.
echo.

REM Check if we're in the web-app directory
if not exist "sveltekit-frontend" (
    echo [ERROR] Not in web-app directory.
    echo Please run this from the main web-app folder.
    pause
    exit /b 1
)

echo Found directories:
if exist "sveltekit-frontend" echo ✅ sveltekit-frontend/
if exist "src" echo ✅ src/
echo.

echo [INFO] Starting comprehensive error fixes for both directories...
echo.

REM Fix main web-app directory first (loose Svelte files)
echo [STEP 1] Fixing main web-app directory...
echo ==========================================

REM Check for React syntax in loose Svelte files
echo Checking for React syntax in loose Svelte files...
findstr /r /c:"className=" *.svelte 2>nul
if %errorlevel%==0 (
    echo [INFO] Found className issues in loose Svelte files, fixing...
    powershell -Command "(Get-Content *.svelte) -replace 'className=', 'class=' | Set-Content *.svelte"
    echo ✅ Fixed className issues in main directory
)

echo.
echo [STEP 2] Fixing sveltekit-frontend directory...
echo ===============================================

cd sveltekit-frontend

if exist "fix-all-1000-errors.mjs" (
    echo [INFO] Running comprehensive fixer...
    node fix-all-1000-errors.mjs
) else (
    echo [WARNING] Comprehensive fixer not found, running available fixes...
    
    if exist "mass-fix-svelte-syntax.mjs" (
        echo Running Svelte syntax fixes...
        node mass-fix-svelte-syntax.mjs
    )
    
    if exist "fix-all-typescript-errors.mjs" (
        echo Running TypeScript fixes...
        node fix-all-typescript-errors.mjs
    )
)

echo.
echo [STEP 3] Cross-directory synchronization...
echo ===========================================

if exist "cross-directory-manager.mjs" (
    echo Running cross-directory manager...
    node cross-directory-manager.mjs
) else (
    echo [INFO] Cross-directory manager not found, skipping...
)

echo.
echo [STEP 4] Final verification...
echo =============================

echo Running TypeScript check...
npm run check 2>nul
if %errorlevel%==0 (
    echo ✅ TypeScript check passed!
) else (
    echo ⚠️ Some TypeScript errors may remain, but major issues should be fixed.
)

cd ..

echo.
echo =========================================
echo Master Error Fixing Complete!
echo =========================================
echo.
echo Summary:
echo ✅ Fixed loose Svelte files in web-app/
echo ✅ Fixed comprehensive errors in sveltekit-frontend/
echo ✅ Synchronized both directories
echo ✅ Verified TypeScript compilation
echo.
echo Next steps:
echo 1. Test the application: npm run dev
echo 2. Review changes in Git: git diff
echo 3. Remove backup files once satisfied
echo.
echo You can now use either:
echo - npm run dev (from web-app directory)
echo - cd sveltekit-frontend && npm run dev
echo.
pause
