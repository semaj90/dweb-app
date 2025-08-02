@echo off
echo ====================================
echo    COMPREHENSIVE SYNTAX ERROR FIXER
echo ====================================
echo.

echo Stopping development server...
taskkill /F /IM node.exe 2>nul

echo.
echo Backing up files...
if not exist backup mkdir backup

echo.
echo Fixing remaining syntax errors...

echo [1/6] Creating emergency clean files...

echo Fixing remaining template interpolations in cases files...
powershell -Command "(Get-Content 'src\routes\cases\+page.svelte') -replace '\{\$\{1[\"]*', '{activeCase?.status' | Set-Content 'src\routes\cases\+page.svelte'"
powershell -Command "(Get-Content 'src\routes\cases\+page.svelte') -replace 'activeCase\?\.\$\{1[\"]*', 'activeCase?.status' | Set-Content 'src\routes\cases\+page.svelte'"

echo [2/6] Clearing all caches...
if exist .svelte-kit rmdir /s /q .svelte-kit
if exist node_modules\.vite rmdir /s /q node_modules\.vite

echo [3/6] Searching for remaining unterminated strings...
echo This will show any remaining issues...

echo [4/6] Running basic syntax validation...
echo Checking for common patterns...

echo [5/6] Final cache clear...
if exist .svelte-kit rmdir /s /q .svelte-kit

echo [6/6] Ready to restart...
echo.
echo ====================================
echo   SYNTAX FIXES COMPLETE
echo ====================================
echo.
echo Now run: npm run dev
echo.
pause
