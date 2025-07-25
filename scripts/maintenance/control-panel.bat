@echo off
REM Visual Start Script for Legal AI Assistant
REM This provides a menu-driven interface for common tasks

:MENU
cls
echo ============================================
echo    Legal AI Assistant - Control Panel
echo ============================================
echo.
echo 1. Start Everything (Docker + App)
echo 2. Start Docker Services Only
echo 3. Start Development Server Only
echo 4. Run System Health Check
echo 5. Open AI Test Dashboard
echo 6. View Docker Logs
echo 7. Reset Everything (Clean Start)
echo 8. Seed Database with Demo Data
echo 9. Fix TypeScript Errors
echo 0. Exit
echo.
set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto START_ALL
if "%choice%"=="2" goto START_DOCKER
if "%choice%"=="3" goto START_DEV
if "%choice%"=="4" goto HEALTH_CHECK
if "%choice%"=="5" goto AI_TEST
if "%choice%"=="6" goto VIEW_LOGS
if "%choice%"=="7" goto RESET_ALL
if "%choice%"=="8" goto SEED_DB
if "%choice%"=="9" goto FIX_TS
if "%choice%"=="0" exit

goto MENU

:START_ALL
echo.
echo Starting all services...
docker-compose up -d
timeout /t 5
cd sveltekit-frontend
start cmd /k npm run dev
cd ..
echo.
echo Services started! Opening browser...
timeout /t 3
start http://localhost:5173
pause
goto MENU

:START_DOCKER
echo.
echo Starting Docker services...
docker-compose up -d
echo.
echo Docker services started!
pause
goto MENU

:START_DEV
echo.
echo Starting development server...
cd sveltekit-frontend
start cmd /k npm run dev
cd ..
echo.
echo Development server starting...
timeout /t 3
start http://localhost:5173
pause
goto MENU

:HEALTH_CHECK
echo.
echo Running system health check...
echo.
node test-system.mjs
echo.
pause
goto MENU

:AI_TEST
echo.
echo Opening AI Test Dashboard...
start http://localhost:5173/ai-test
echo.
echo Dashboard opened in browser!
pause
goto MENU

:VIEW_LOGS
echo.
echo Showing Docker logs (Press Ctrl+C to stop)...
echo.
docker-compose logs -f
pause
goto MENU

:RESET_ALL
echo.
echo WARNING: This will delete all data and containers!
set /p confirm="Are you sure? (yes/no): "
if /i "%confirm%"=="yes" (
    echo Resetting everything...
    docker-compose down -v
    echo Reset complete!
) else (
    echo Reset cancelled.
)
pause
goto MENU

:SEED_DB
echo.
echo Seeding database with demo data...
npm run seed
echo.
echo Database seeded!
pause
goto MENU

:FIX_TS
echo.
echo Fixing TypeScript errors...
node fix-all-typescript-imports.mjs
echo.
echo TypeScript fixes applied!
pause
goto MENU
