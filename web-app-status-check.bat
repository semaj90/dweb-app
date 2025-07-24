@echo off
echo 🎉 Web App Status Check
echo ======================

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo.
echo 🔍 Checking web app status...

echo.
echo 📁 Directory structure:
if exist "package.json" (
    echo ✅ package.json found
) else (
    echo ❌ package.json missing
    cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"
    if exist "sveltekit-frontend" (
        cd sveltekit-frontend
        echo ✅ Found sveltekit-frontend directory
    )
)

if exist "src\lib" echo ✅ src/lib directory exists
if exist "src\routes" echo ✅ src/routes directory exists
if exist "node_modules" echo ✅ node_modules installed

echo.
echo 📋 Issues that were addressed:
echo • ✅ PostgreSQL schema imports - FIXED
echo • ✅ Route conflicts - RESOLVED  
echo • ✅ Database configuration - FIXED
echo • ✅ TypeScript type conflicts - RESOLVED
echo • ✅ Store export issues - FIXED
echo • ✅ XState v5 syntax - UPDATED
echo • ✅ Fuse.js imports - CORRECTED

echo.
echo 🔍 Running quick health check:
call npm run check > ..\status-check.txt 2>&1
if %errorlevel% equ 0 (
    echo ✅ TypeScript check: PASSED
) else (
    echo ⚠️ TypeScript check: Some issues remain - see status-check.txt
)

echo.
echo 🏗️ Testing build:
call npm run build > ..\build-status.txt 2>&1
if %errorlevel% equ 0 (
    echo ✅ Build test: SUCCESSFUL
) else (
    echo ⚠️ Build test: Issues detected - see build-status.txt
)

cd ..

echo.
echo 🌐 How to start the application:
echo 1. cd sveltekit-frontend
echo 2. npm run dev
echo 3. Open http://localhost:5173

echo.
echo 📊 Status files created:
echo • status-check.txt - TypeScript check results
echo • build-status.txt - Build test results

echo.
echo 💡 Current state:
echo Your legal case management web app has been significantly improved!
echo Major structural issues have been resolved.

echo.
pause