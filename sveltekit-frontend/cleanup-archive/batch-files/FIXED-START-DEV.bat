@echo off
title SvelteKit Debug & Fix Tool
echo.
echo =======================================================
echo  SvelteKit Debug & Fix Tool
echo  Fixing Vite module runner and TypeScript errors
echo =======================================================
echo.

echo 🔍 Step 1: Clearing caches...
echo.

REM Delete .svelte-kit directory
if exist ".svelte-kit" (
    echo Deleting .svelte-kit directory...
    rmdir /s /q ".svelte-kit"
    echo ✅ .svelte-kit directory deleted
) else (
    echo ℹ️ .svelte-kit directory doesn't exist
)

REM Delete node_modules/.vite directory
if exist "node_modules\.vite" (
    echo Deleting node_modules\.vite directory...
    rmdir /s /q "node_modules\.vite"
    echo ✅ node_modules\.vite directory deleted
) else (
    echo ℹ️ node_modules\.vite directory doesn't exist
)

REM Delete any build directory
if exist "build" (
    echo Deleting build directory...
    rmdir /s /q "build"
    echo ✅ build directory deleted
) else (
    echo ℹ️ build directory doesn't exist
)

echo.
echo 🔧 Step 2: Applying comprehensive fixes...
echo.

REM Apply TypeScript and configuration fixes
node comprehensive-fix.mjs

echo.
echo 🔄 Step 3: Syncing SvelteKit...
echo.

REM Sync SvelteKit
call npm run prepare

echo.
echo 🎯 Step 4: Type checking...
echo.

REM Run a quick type check
call npx svelte-check --tsconfig ./tsconfig.json

echo.
echo 🚀 Step 5: Starting development server...
echo.

REM Start the development server
call npm run dev

echo.
echo If you still see errors, try:
echo   1. npm run clean
echo   2. npm install  
echo   3. npm run dev
echo.
pause
