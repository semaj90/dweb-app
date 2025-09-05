@echo off
echo Starting Legal AI System - Enhanced File Upload Demo
echo ================================================

REM Navigate to the project directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

echo Starting development server...
echo.
echo Visit: http://localhost:5173/ai-upload-demo
echo.

REM Start the development server
call npm run dev

pause
