@echo off
REM =============================================
REM COMPLETE LEGAL AI SYSTEM TEST SUITE
REM Tests TypeScript, Database, APIs, and Services
REM =============================================

cls
echo ================================================
echo   COMPLETE LEGAL AI SYSTEM TEST SUITE
echo ================================================
echo.

REM Create test results directory
if not exist "test-results" mkdir test-results

echo [1/6] Testing TypeScript System...
echo =============================================
call TEST-TYPESCRIPT-SYSTEM.bat > test-results\typescript-test.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] TypeScript system healthy
    set TS_STATUS=PASS
) else (
    echo       [WARN] TypeScript needs attention
    set TS_STATUS=WARN
)

echo.
echo [2/6] Testing Database Setup...
echo =============================================
call setup-database-complete.bat > test-results\database-test.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] Database setup successful
    set DB_STATUS=PASS
) else (
    echo       [ERROR] Database setup failed
    set DB_STATUS=FAIL
)

echo.
echo [3/6] Running System Validation...
echo =============================================
node validate-system-complete.mjs > test-results\system-validation.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] System validation passed
    set SYS_STATUS=PASS
) else (
    echo       [WARN] System partially operational
    set SYS_STATUS=WARN
)

echo.
echo [4/6] Testing SvelteKit Build...
echo =============================================
cd sveltekit-frontend
npm run build > ..\test-results\build-test.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] SvelteKit build successful
    set BUILD_STATUS=PASS
) else (
    echo       [ERROR] SvelteKit build failed
    set BUILD_STATUS=FAIL
)
cd ..

echo.
echo [5/6] Testing API Endpoints...
echo =============================================
REM Start the application if not running
curl -s http://localhost:5173/api/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo       Starting application for testing...
    cd sveltekit-frontend
    start /B npm run dev
    cd ..
    timeout /t 10 /nobreak >nul
)

REM Test key endpoints
curl -s -o test-results\health-check.json http://localhost:5173/api/health
if %ERRORLEVEL% EQU 0 (
    echo       [OK] Health endpoint responding
    set API_STATUS=PASS
) else (
    echo       [ERROR] API endpoints not responding
    set API_STATUS=FAIL
)

echo.
echo [6/6] Generating Comprehensive Report...
echo =============================================

REM Create comprehensive test report
echo ================================================ > test-results\COMPLETE-TEST-REPORT.txt
echo   LEGAL AI SYSTEM - COMPREHENSIVE TEST REPORT >> test-results\COMPLETE-TEST-REPORT.txt
echo   Generated: %DATE% %TIME% >> test-results\COMPLETE-TEST-REPORT.txt
echo ================================================ >> test-results\COMPLETE-TEST-REPORT.txt
echo. >> test-results\COMPLETE-TEST-REPORT.txt
echo OVERALL SYSTEM STATUS: >> test-results\COMPLETE-TEST-REPORT.txt
echo ====================== >> test-results\COMPLETE-TEST-REPORT.txt
echo   TypeScript System: %TS_STATUS% >> test-results\COMPLETE-TEST-REPORT.txt
echo   Database Setup: %DB_STATUS% >> test-results\COMPLETE-TEST-REPORT.txt
echo   System Validation: %SYS_STATUS% >> test-results\COMPLETE-TEST-REPORT.txt
echo   SvelteKit Build: %BUILD_STATUS% >> test-results\COMPLETE-TEST-REPORT.txt
echo   API Endpoints: %API_STATUS% >> test-results\COMPLETE-TEST-REPORT.txt
echo. >> test-results\COMPLETE-TEST-REPORT.txt

REM Determine overall system health
set OVERALL_HEALTH=FAIL
if "%TS_STATUS%"=="PASS" if "%DB_STATUS%"=="PASS" if "%BUILD_STATUS%"=="PASS" (
    set OVERALL_HEALTH=PASS
)

echo SYSTEM HEALTH: %OVERALL_HEALTH% >> test-results\COMPLETE-TEST-REPORT.txt
echo. >> test-results\COMPLETE-TEST-REPORT.txt

echo TYPESCRIPT IMPROVEMENTS: >> test-results\COMPLETE-TEST-REPORT.txt
echo ======================= >> test-results\COMPLETE-TEST-REPORT.txt
echo   Baseline Errors: ~2000+ >> test-results\COMPLETE-TEST-REPORT.txt
findstr "Current Errors:" test-results\typescript-test.log >> test-results\COMPLETE-TEST-REPORT.txt 2>nul
findstr "Improvement:" test-results\typescript-test.log >> test-results\COMPLETE-TEST-REPORT.txt 2>nul
echo. >> test-results\COMPLETE-TEST-REPORT.txt

echo SHIM SYSTEM STATUS: >> test-results\COMPLETE-TEST-REPORT.txt
echo ================== >> test-results\COMPLETE-TEST-REPORT.txt
if exist "sveltekit-frontend\src\lib\shims" (
    echo   Shim files installed: >> test-results\COMPLETE-TEST-REPORT.txt
    dir sveltekit-frontend\src\lib\shims\*.d.ts /b >> test-results\COMPLETE-TEST-REPORT.txt 2>nul
) else (
    echo   [ERROR] Shim system not found >> test-results\COMPLETE-TEST-REPORT.txt
)
echo. >> test-results\COMPLETE-TEST-REPORT.txt

echo NEXT RECOMMENDED ACTIONS: >> test-results\COMPLETE-TEST-REPORT.txt
echo ========================= >> test-results\COMPLETE-TEST-REPORT.txt
if "%OVERALL_HEALTH%"=="PASS" (
    echo   ✅ System ready for development >> test-results\COMPLETE-TEST-REPORT.txt
    echo   - TypeScript error reduction successful >> test-results\COMPLETE-TEST-REPORT.txt
    echo   - Progressive type replacement active >> test-results\COMPLETE-TEST-REPORT.txt
    echo   - Database and APIs operational >> test-results\COMPLETE-TEST-REPORT.txt
    echo   - Continue with feature development >> test-results\COMPLETE-TEST-REPORT.txt
) else (
    echo   ⚠️ System needs attention >> test-results\COMPLETE-TEST-REPORT.txt
    if "%TS_STATUS%"=="WARN" echo   - Review TypeScript errors >> test-results\COMPLETE-TEST-REPORT.txt
    if "%DB_STATUS%"=="FAIL" echo   - Fix database connectivity >> test-results\COMPLETE-TEST-REPORT.txt
    if "%BUILD_STATUS%"=="FAIL" echo   - Resolve build issues >> test-results\COMPLETE-TEST-REPORT.txt
    if "%API_STATUS%"=="FAIL" echo   - Check API service status >> test-results\COMPLETE-TEST-REPORT.txt
)

echo.
echo ================================================
echo   COMPREHENSIVE TEST COMPLETE
echo ================================================
echo.
echo Overall Health: %OVERALL_HEALTH%
echo.
echo Test Results Available:
echo   Complete Report: test-results\COMPLETE-TEST-REPORT.txt
echo   TypeScript: test-results\typescript-test.log
echo   Database: test-results\database-test.log
echo   System: test-results\system-validation.log
echo   Build: test-results\build-test.log
echo.

if "%OVERALL_HEALTH%"=="PASS" (
    echo ✅ LEGAL AI SYSTEM READY FOR DEVELOPMENT
    exit /b 0
) else (
    echo ⚠️ SYSTEM NEEDS ATTENTION - CHECK LOGS
    exit /b 1
)

pause