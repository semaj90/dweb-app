@echo off
echo ========================================
echo    SvelteKit TypeScript Error Fixer
echo ========================================
echo.

cd /d "C:\Users\james\Desktop\web-app"

echo Running comprehensive TypeScript fixes...
echo.

node fix-all-typescript-errors.mjs

echo.
echo ========================================
echo Running svelte-check to verify fixes...
echo ========================================
echo.

cd sveltekit-frontend
npm run check

echo.
echo ========================================
echo TypeScript Error Fixing Complete!
echo ========================================
pause
