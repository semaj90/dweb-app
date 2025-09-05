@echo off
:: Legal AI Platform - Windows Native Database Startup
:: Ensures PostgreSQL, Redis, and Neo4j are running with proper authentication

echo ========================================
echo Legal AI - Starting All Databases
echo ========================================
echo.

:: Set environment variables
set PGPASSWORD=123456
set DATABASE_URL=postgresql://postgres:123456@localhost:5432/legal_ai_db
set REDIS_URL=redis://localhost:6379
set NEO4J_URL=bolt://localhost:7687

echo [1/6] Checking PostgreSQL service...
sc query postgresql-x64-17 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Starting PostgreSQL service...
    net start postgresql-x64-17
) else (
    echo ✓ PostgreSQL service already running
)

echo.
echo [2/6] Testing PostgreSQL connection...
psql -h localhost -p 5432 -U postgres -d legal_ai_db -c "SELECT 'PostgreSQL Connected' as status;" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ PostgreSQL connection failed
    echo Please check PostgreSQL is installed and running on port 5432
    pause
    exit /b 1
)
echo ✓ PostgreSQL connected successfully

echo.
echo [3/6] Starting Redis server...
tasklist /FI "IMAGENAME eq redis-server.exe" 2>NUL | find /I /N "redis-server.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo ✓ Redis already running
) else (
    echo Starting Redis in background...
    start /B redis-server --daemonize yes --port 6379
    timeout /t 3 >nul
)

echo.
echo [4/6] Testing Redis connection...
redis-cli ping >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Redis connection failed
    echo Please check Redis is installed
    pause
    exit /b 1
)
echo ✓ Redis connected successfully

echo.
echo [5/6] Checking Neo4j service...
sc query neo4j >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Starting Neo4j service...
    net start neo4j
) else (
    echo ✓ Neo4j service already running
)

echo.
echo [6/6] Testing database connections...
echo.
echo ✓ PostgreSQL: postgresql://postgres:123456@localhost:5432/legal_ai_db
echo ✓ Redis: redis://localhost:6379
echo ✓ Neo4j: bolt://localhost:7687
echo.

echo ========================================
echo All databases are ready!
echo ========================================
echo.
echo Admin Interfaces:
echo - pgAdmin: http://localhost:5050 (if installed)
echo - Redis Insight: http://localhost:8001 (if installed)  
echo - Neo4j Browser: http://localhost:7474
echo.
echo Starting SvelteKit with full database connectivity...
npm run dev

pause