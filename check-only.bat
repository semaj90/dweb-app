@echo off
echo 🔍 SvelteKit TypeScript Check
echo ============================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo Running svelte-check...
echo.

call npm run check

echo.
echo Check completed! 
echo If you see errors above, they may be non-critical for development.
echo You can still run the development server with npm run dev
echo.

pause