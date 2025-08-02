@echo off
echo ================================================
echo         SvelteKit Error Check & Fix
echo ================================================
echo.

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo 🔍 Running TypeScript check...
echo.

npm run check

echo.
echo ================================================
echo If there are still errors, run the comprehensive fix:
echo   node comprehensive-fix-all-errors.mjs
echo.
echo Then run this script again to verify fixes.
echo ================================================
pause