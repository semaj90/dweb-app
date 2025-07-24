@echo off
echo =====================================================
echo 🎯 PROSECUTOR AI - DEVELOPMENT SERVER STARTUP
echo =====================================================
echo.

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

echo 📍 Current Directory: %CD%
echo.

echo 🔍 Checking Node.js installation...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found! Please install Node.js first.
    pause
    exit /b 1
)

echo ✅ Node.js found: 
node --version
echo.

echo 🔍 Checking npm installation...
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm not found! Please install npm first.
    pause
    exit /b 1
)

echo ✅ npm found:
npm --version
echo.

echo 🚀 Starting Prosecutor AI Development Server...
echo 📱 Server will be available at: http://localhost:5173
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Try to start the development server
npm run dev

REM If that fails, try alternative approaches
if %errorlevel% neq 0 (
    echo.
    echo ⚠️  Standard dev command failed. Trying alternative...
    echo.
    
    echo 🧹 Cleaning project cache...
    npm run clean
    
    echo 🔄 Starting clean development server...
    npm run dev:clean
    
    if %errorlevel% neq 0 (
        echo.
        echo ❌ Development server failed to start.
        echo 🔧 Try running these commands manually:
        echo    npm run fix:all
        echo    npm run dev
        echo.
        pause
        exit /b 1
    )
)

echo.
echo 🎉 Development server started successfully!
pause