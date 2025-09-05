@echo off
echo ====================================
echo    SVELTEKIT CACHE CLEANER
echo ====================================
echo.
echo Stopping any running dev servers...
taskkill /F /IM node.exe 2>nul

echo.
echo Clearing SvelteKit cache...
if exist .svelte-kit rmdir /s /q .svelte-kit
echo .svelte-kit folder cleared

echo.
echo Clearing Vite cache...
if exist node_modules\.vite rmdir /s /q node_modules\.vite
echo node_modules\.vite folder cleared

echo.
echo Cache clearing complete!
echo.
echo Now run: npm run dev
echo.
pause
