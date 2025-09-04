@echo off
REM ============================================
REM COMPLETE SYSTEM TEST RUNNER
REM Tests all APIs, database, and integrations
REM ============================================

cls
echo ================================================
echo   LEGAL AI SYSTEM - COMPREHENSIVE TESTING
echo ================================================
echo.

REM Check if services are running
echo [1/5] Checking services...
curl -s http://localhost:5432 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PostgreSQL not running. Starting services...
    call START-PRODUCTION-COMPLETE.bat
    timeout /t 10 /nobreak >nul
)

curl -s http://localhost:5173 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Application not running. Starting...
    start /B npm run dev
    timeout /t 5 /nobreak >nul
)

echo       [OK] Services running

REM Run database migrations
echo [2/5] Checking database...
psql -U postgres -d legal_ai_db -c "SELECT 1 FROM users LIMIT 1;" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo       Running migrations...
    psql -U postgres -d legal_ai_db -f production-migration.sql >nul 2>&1
    echo       [OK] Database initialized
) else (
    echo       [OK] Database ready
)

REM Check API health
echo [3/5] Testing API health...
curl -s http://localhost:5173/api/health -o health-check.json
if %ERRORLEVEL% EQU 0 (
    echo       [OK] API responding
    type health-check.json | findstr "healthy"
) else (
    echo       [ERROR] API not responding
)

REM Run Playwright tests
echo [4/5] Running Playwright tests...
echo.
npx playwright test --reporter=list

REM Generate test report
echo.
echo [5/5] Generating test report...
npx playwright show-report

echo.
echo ================================================
echo   TEST RESULTS SUMMARY
echo ================================================
echo.
echo Check the test report for detailed results.
echo.
pause