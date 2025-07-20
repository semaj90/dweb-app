@echo off
echo Clearing SvelteKit and Vite caches...

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
echo Cache cleanup completed!
echo.
echo Now starting development server...
echo.

REM Sync SvelteKit first
call npm run prepare

REM Start the development server
call npm run dev

pause
