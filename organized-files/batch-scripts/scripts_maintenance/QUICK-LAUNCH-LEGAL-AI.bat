@echo off
cls
color 0F
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   🚀 QUICK LAUNCH SUITE                     ║
echo ║            Legal AI Case Management System                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🎯 Starting optimized Legal AI services...

echo.
echo 📋 Step 1: Starting core databases...
docker-compose -f docker-compose.optimized.yml up -d legal-postgres legal-redis legal-qdrant
if errorlevel 1 (
    echo ⚠️ Using fallback configuration...
    docker-compose up -d postgres redis qdrant
)

echo.
echo 📋 Step 2: Initializing AI services...
timeout /t 10 >nul
docker-compose -f docker-compose.optimized.yml up -d legal-ollama
if errorlevel 1 (
    echo ⚠️ Using fallback AI configuration...
    docker-compose up -d ollama
)

echo.
echo 📋 Step 3: Starting collaboration features...
docker-compose -f docker-compose.optimized.yml up -d legal-collaboration legal-document-processor
if errorlevel 1 (
    echo ⚠️ Starting basic collaboration...
    start cmd /k "cd collaboration-server && npm start"
)

echo.
echo 📋 Step 4: Launching frontend application...
cd sveltekit-frontend
start cmd /k "npm run dev"
cd ..

echo.
echo 📋 Step 5: Performing health checks...
timeout /t 15 >nul

echo.
echo ✅ Legal AI System Status:
echo ════════════════════════════════════════
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr legal- 2>nul
if errorlevel 1 (
    echo 📊 Checking standard containers...
    docker ps --format "table {{.Names}}\t{{.Status}}" | findstr -E "postgres|ollama|redis|qdrant"
)

echo.
echo 🌐 Access Points:
echo ════════════════════════════════════════
echo   • Main Application: http://localhost:5173
echo   • Collaboration Hub: http://localhost:8080/health
echo   • Document Processor: http://localhost:8081/health
echo   • Database Studio: Run 'npm run db:studio' in sveltekit-frontend
echo.

echo 🎉 Legal AI System is ready!
echo.
echo 📋 Quick Actions:
echo   [1] Open main application
echo   [2] Open database studio
echo   [3] View system logs
echo   [4] Exit
echo.

set /p choice="Select action [1-4]: "

if "%choice%"=="1" start http://localhost:5173
if "%choice%"=="2" (
    cd sveltekit-frontend
    start cmd /k "npm run db:studio"
    cd ..
)
if "%choice%"=="3" docker-compose logs -f
if "%choice%"=="4" exit /b

echo.
echo 🚀 Legal AI Case Management System is now operational!
pause