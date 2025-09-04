@echo off
:: Legal AI Platform - Complete Development Environment
:: Ensures PostgreSQL connection and starts all services

echo ========================================
echo Legal AI Platform - Starting Services
echo ========================================
echo.

:: Set PostgreSQL connection string
set DATABASE_URL=postgresql://postgres:123456@localhost:5432/legal_ai_db
set PGPASSWORD=123456

:: Test PostgreSQL connection
echo [1/3] Testing PostgreSQL connection...
psql -h localhost -p 5432 -U postgres -d legal_ai_db -c "SELECT 'Connected to Legal AI Database' as status;" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Cannot connect to PostgreSQL
    echo Please ensure PostgreSQL is running on port 5432
    echo Connection string: postgresql://postgres:123456@localhost:5432/legal_ai_db
    pause
    exit /b 1
)
echo ✓ PostgreSQL connected

:: Check if we should use dev:full or dev:complete
echo.
echo [2/3] Select startup mode:
echo 1. Full Stack (dev:full) - All services with RTX enhancement
echo 2. Complete (dev:complete) - Standard development environment
echo 3. Simple Dev (dev) - Just SvelteKit frontend
echo.
choice /C 123 /N /M "Enter your choice (1-3): "

if %ERRORLEVEL% EQU 1 (
    echo.
    echo [3/3] Starting Full Stack with RTX enhancement...
    npm run dev:full
) else if %ERRORLEVEL% EQU 2 (
    echo.
    echo [3/3] Starting Complete development environment...
    npm run dev:complete
) else (
    echo.
    echo [3/3] Starting SvelteKit frontend only...
    npm run dev
)