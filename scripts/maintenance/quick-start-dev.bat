@echo off
echo 🚀 SvelteKit App Quick Fix and Start
echo ====================================
echo.

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo 📦 Installing dependencies...
call npm install

echo.
echo 🔧 Running TypeScript check...
call npm run check

echo.
echo 🎯 Starting development server...
echo Open your browser to http://localhost:5173
echo Press Ctrl+C to stop the server
echo.

call npm run dev

pause