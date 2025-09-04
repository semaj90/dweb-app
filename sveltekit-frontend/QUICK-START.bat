@echo off
echo.
echo 🚀 STARTING REAL AI SYSTEM
echo ==========================
echo.

REM Change to project directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

REM Quick system check
echo ✅ Checking system...
if not exist "package.json" (
    echo ❌ Error: Not in correct directory
    pause
    exit /b 1
)

echo ✅ Project directory found
echo ✅ Starting development server...
echo.

REM Set environment
set NODE_ENV=development

REM Start the server
echo 🌐 Legal AI System starting...
echo 📍 Demo will be available at: http://localhost:5173/ai-upload-demo
echo.
echo ⚠️  Make sure these services are running:
echo    - PostgreSQL (port 5432)
echo    - Redis (port 6379) 
echo    - Ollama (port 11434)
echo.
echo 🎯 Press Ctrl+C to stop
echo.

npm run dev

echo.
echo 👋 AI System stopped
pause
