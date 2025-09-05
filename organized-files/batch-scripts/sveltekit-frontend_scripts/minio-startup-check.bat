@echo off
REM Enhanced MinIO Startup and Health Check for Legal AI Platform

echo ================================================================================
echo MINIO STARTUP VERIFICATION FOR LEGAL AI PLATFORM
echo ================================================================================

echo [1/5] Checking if MinIO is already running...
tasklist | findstr "minio" >nul
if %errorlevel% == 0 (
    echo ✅ MinIO process found
    goto :check_health
) else (
    echo ⚠️  MinIO process not found, starting MinIO...
)

echo [2/5] Creating MinIO data directory...
if not exist minio-data mkdir minio-data
if not exist minio-data\legal-documents mkdir minio-data\legal-documents
if not exist minio-data\evidence-files mkdir minio-data\evidence-files
if not exist minio-data\temp-uploads mkdir minio-data\temp-uploads

echo [3/5] Starting MinIO server...
start /min cmd /k "minio.exe server ./minio-data --address :9000 --console-address :9001"

echo [4/5] Waiting for MinIO to initialize...
timeout /t 5 /nobreak >nul

:check_health
echo [5/5] Testing MinIO health and bucket creation...

echo Testing MinIO API accessibility...
curl -s http://localhost:9000/minio/health/live >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ MinIO API is responding
) else (
    echo ❌ MinIO API not responding, checking console...
    curl -s http://localhost:9001 >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ MinIO Console accessible at http://localhost:9001
    ) else (
        echo ❌ MinIO Console not accessible
    )
)

echo Testing bucket creation via API...
curl -X PUT http://localhost:9000/legal-documents -H "Authorization: AWS4-HMAC-SHA256 Credential=minioadmin/..." >nul 2>&1

echo.
echo MinIO Status Summary:
echo ==================
echo MinIO Server: http://localhost:9000
echo MinIO Console: http://localhost:9001 (admin: minioadmin/minioadmin)
echo Data Directory: %CD%\minio-data
echo.
echo Required Buckets:
echo - legal-documents (for legal docs)
echo - evidence-files (for evidence)
echo - temp-uploads (for temp files)
echo - thumbnails (for previews)
echo.

echo Testing MinIO integration with SvelteKit...
timeout /t 2 /nobreak >nul
curl -s http://localhost:5173/api/v1/minio/health >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ SvelteKit MinIO integration working
) else (
    echo ⚠️  SvelteKit not running or MinIO integration issue
    echo Start SvelteKit with: npm run dev
)

echo.
echo ================================================================================
echo MINIO STARTUP CHECK COMPLETE
echo ================================================================================
echo.

pause