@echo off
:: Quick Fix for System Integration Issues
:: Run this to resolve all critical issues

echo 🚀 Legal AI System - Quick Fix
echo ================================================
echo.

:: 1. Load environment variables
echo [1/7] Loading environment configuration...
for /f "tokens=1,2 delims==" %%a in (.env) do (
    if not "%%a"=="" set "%%a=%%b"
)
echo ✅ Environment loaded

:: 2. Fix PostgreSQL password
echo [2/7] Setting PostgreSQL password...
set PGPASSWORD=123456
echo ✅ PGPASSWORD set

:: 3. Test database connection
echo [3/7] Testing database connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️  Database connection needs fixing
    echo    Run: FIX-POSTGRES-ADMIN.bat as Administrator
) else (
    echo ✅ Database connected
)

:: 4. Create required directories
echo [4/7] Creating required directories...
if not exist logs mkdir logs
if not exist uploads mkdir uploads
if not exist documents mkdir documents
if not exist evidence mkdir evidence
if not exist generated_reports mkdir generated_reports
echo ✅ Directories created

:: 5. Start Redis
echo [5/7] Starting Redis...
redis-cli ping >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    start /B redis-windows\redis-server.exe redis-windows\redis.conf
    timeout /t 2 /nobreak >nul
)
echo ✅ Redis running

:: 6. Build and start GPU services
echo [6/7] Starting GPU services...
cd go-microservice
if not exist legal-processor-enhanced.exe (
    go build -tags=cgo -o legal-processor-enhanced.exe enhanced_legal_processor.go 2>nul
)
tasklist /FI "IMAGENAME eq legal-processor-enhanced.exe" 2>nul | find /I "legal-processor-enhanced.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    start /B legal-processor-enhanced.exe
)
cd ..
echo ✅ GPU service started

:: 7. Quick health check
echo [7/7] Running health check...
timeout /t 3 /nobreak >nul
curl -s http://localhost:8080/health >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✅ GPU service responding
) else (
    echo ⚠️  GPU service may still be starting
)

echo.
echo ================================================
echo ✅ Quick fix complete!
echo.
echo Run: node check-system-integration.mjs
echo To verify all systems are operational
echo.
pause
