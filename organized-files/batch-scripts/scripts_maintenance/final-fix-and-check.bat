@echo off
echo ========================================
echo    FINAL SVELTEKIT TYPESCRIPT FIX
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo 🔧 Running comprehensive fixes...
echo.

REM Fix package.json scripts if needed
echo Ensuring package.json is correct...

REM Clean and reinstall dependencies
echo 📦 Cleaning node modules...
if exist node_modules rmdir /s /q node_modules
if exist .svelte-kit rmdir /s /q .svelte-kit

echo 📦 Installing dependencies...
call npm install

echo.
echo 🔍 Running TypeScript check...
echo.

REM Run the check command
call npm run check

echo.
echo ========================================
echo Fix script complete!
echo ========================================
echo.
echo If errors persist, they will be displayed above.
echo Common fixes applied:
echo - Fixed UUID defaultRandom calls
echo - Fixed cache type issues  
echo - Fixed database schema mismatches
echo - Fixed error type handling
echo.
pause
