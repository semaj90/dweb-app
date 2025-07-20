@echo off
echo ===========================================
echo  PRODUCTION DEVELOPMENT ENVIRONMENT SETUP
echo ===========================================
echo.

echo [STEP 1] Checking Docker services...
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr prosecutor
if %errorlevel% neq 0 (
    echo Starting Docker services...
    docker-compose up -d
    timeout /t 5 /nobreak >nul
) else (
    echo Docker services are running.
)
echo.

echo [STEP 2] Installing dependencies...
echo Installing root dependencies...
call npm install
echo Installing frontend dependencies...
cd sveltekit-frontend
call npm install
echo.

echo [STEP 3] Environment validation...
if not exist ".env" (
    echo WARNING: .env file not found in sveltekit-frontend
    echo Creating from .env.example...
    copy .env.example .env
)
echo Environment files checked.
echo.

echo [STEP 4] Database setup and migrations...
cd ..
echo Checking database connection...
call npm run db:check
echo Running database migrations...
call npm run db:migrate
if %errorlevel% neq 0 (
    echo ERROR: Database migration failed!
    echo Please check your database connection and try again.
    exit /b 1
)
echo Database setup complete.
echo.

echo [STEP 5] TypeScript validation...
cd sveltekit-frontend
echo Running TypeScript check...
call npm run check
if %errorlevel% neq 0 (
    echo WARNING: TypeScript errors found!
    echo You may need to fix these before production deployment.
    echo.
) else (
    echo TypeScript validation passed!
    echo.
)

echo [STEP 6] Starting development server...
echo Starting SvelteKit development server...
echo Open http://localhost:5173 in your browser
echo Press Ctrl+C to stop the server when done testing.
echo.
call npm run dev

echo ===========================================
echo  SETUP COMPLETE
echo ===========================================
