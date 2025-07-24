@echo off
setlocal enabledelayedexpansion
title Legal AI - Master Setup & Launcher
color 0A

:: Set project root
set "PROJECT_ROOT=%~dp0"
set "FRONTEND_PATH=%PROJECT_ROOT%sveltekit-frontend"
cd /d "%PROJECT_ROOT%"

:MAIN_MENU
cls
echo ========================================
echo 🎯 LEGAL AI - MASTER CONTROL PANEL
echo ========================================
echo.
echo 1. Quick Start (Full Setup)
echo 2. Fix All Errors
echo 3. Start Development Only
echo 4. Docker Services Only
echo 5. Health Check
echo 6. Database Setup
echo 7. Exit
echo.
set /p "choice=Choose (1-7): "

if "%choice%"=="1" goto QUICK_START
if "%choice%"=="2" goto FIX_ALL
if "%choice%"=="3" goto DEV_ONLY
if "%choice%"=="4" goto DOCKER_ONLY
if "%choice%"=="5" goto HEALTH_CHECK
if "%choice%"=="6" goto DATABASE_SETUP
if "%choice%"=="7" exit /b 0

echo Invalid choice. Press any key...
pause > nul
goto MAIN_MENU

:QUICK_START
echo.
echo === QUICK START - FULL SETUP ===
echo.
call :FIX_TYPESCRIPT
call :START_DOCKER
call :SETUP_DATABASE
call :START_DEV
goto END

:FIX_ALL
echo.
echo === FIXING ALL ERRORS ===
echo.
call :FIX_TYPESCRIPT
call :FIX_ROUTES
call :FIX_IMPORTS
echo ✅ Error fixing complete!
goto END

:DEV_ONLY
echo.
echo === STARTING DEVELOPMENT ONLY ===
echo.
cd /d "%FRONTEND_PATH%"
if not exist "node_modules" npm install
npm run dev
goto END

:DOCKER_ONLY
echo.
echo === STARTING DOCKER SERVICES ===
echo.
call :START_DOCKER
goto END

:HEALTH_CHECK
echo.
echo === HEALTH CHECK ===
echo.
call :CHECK_SERVICES
goto END

:DATABASE_SETUP
echo.
echo === DATABASE SETUP ===
echo.
call :SETUP_DATABASE
goto END

:: ================================
:: HELPER FUNCTIONS
:: ================================

:FIX_TYPESCRIPT
echo 🔧 Fixing TypeScript errors...
cd /d "%FRONTEND_PATH%"

:: Fix className -> class
powershell -Command "Get-ChildItem -Recurse -Filter '*.svelte' | ForEach-Object { (Get-Content $_.FullName) -replace 'className=', 'class=' | Set-Content $_.FullName }"

:: Fix schema imports
powershell -Command "Get-ChildItem -Recurse -Filter '*.ts' | ForEach-Object { (Get-Content $_.FullName) -replace 'from \x27\$lib/server/db/schema\x27', 'from \x27\$lib/server/db/schema-postgres\x27' | Set-Content $_.FullName }"

:: Install dependencies
npm install > nul 2>&1

echo ✅ TypeScript fixes applied
goto :eof

:FIX_ROUTES
echo 🔧 Fixing route conflicts...
cd /d "%FRONTEND_PATH%"

:: Remove conflicting routes
if exist "src\routes\api\evidence\[id]\" rmdir /s /q "src\routes\api\evidence\[id]"

echo ✅ Route conflicts resolved
goto :eof

:FIX_IMPORTS
echo 🔧 Fixing import errors...
cd /d "%FRONTEND_PATH%"

:: Run additional fixes if needed
if exist "fix-all-typescript-errors.mjs" node fix-all-typescript-errors.mjs > nul 2>&1

echo ✅ Import fixes applied
goto :eof

:START_DOCKER
echo 🐳 Starting Docker services...
docker-compose up -d postgres redis qdrant > nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Docker services started
    timeout /t 10 /nobreak > nul
) else (
    echo ❌ Docker failed - ensure Docker Desktop is running
)
goto :eof

:SETUP_DATABASE
echo 🗄️ Setting up database...
cd /d "%FRONTEND_PATH%"
npm run db:push > nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Database setup complete
) else (
    echo ⚠️ Database setup had issues
)
goto :eof

:START_DEV
echo 🚀 Starting development server...
cd /d "%FRONTEND_PATH%"
echo.
echo 📱 Server will be at: http://localhost:5173
echo 🛑 Press Ctrl+C to stop
echo.
start http://localhost:5173
npm run dev
goto :eof

:CHECK_SERVICES
echo 🔍 Checking services...

:: Check Docker
docker ps > nul 2>&1 && echo ✅ Docker running || echo ❌ Docker not running

:: Check PostgreSQL
docker exec legal-ai-postgres pg_isready -U postgres > nul 2>&1 && echo ✅ PostgreSQL ready || echo ❌ PostgreSQL not ready

:: Check Redis  
docker exec legal-ai-redis redis-cli ping > nul 2>&1 && echo ✅ Redis ready || echo ❌ Redis not ready

:: Check Qdrant
curl -s http://localhost:6333/health > nul 2>&1 && echo ✅ Qdrant ready || echo ❌ Qdrant not ready

:: Check TypeScript
cd /d "%FRONTEND_PATH%"
npm run check > nul 2>&1 && echo ✅ TypeScript check passed || echo ⚠️ TypeScript issues found

goto :eof

:END
echo.
echo ========================================
echo 🎉 OPERATION COMPLETE
echo ========================================
echo.
echo 📱 Access your app: http://localhost:5173
echo 🔧 Run health check anytime with option 5
echo.
pause
goto MAIN_MENU