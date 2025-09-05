@echo off
echo ========================================
echo MinIO Integration Diagnostic Tool
echo ========================================
echo.

echo Checking system status...
echo.

echo [Service Status]
echo ----------------

REM Check if MinIO is running
powershell -Command "if(Get-Process minio -ErrorAction SilentlyContinue) { Write-Host 'MinIO: RUNNING' -ForegroundColor Green } else { Write-Host 'MinIO: NOT RUNNING' -ForegroundColor Red }"

REM Check if PostgreSQL is running
sc query postgresql-x64-14 | findstr RUNNING >nul
if %ERRORLEVEL% EQU 0 (
    echo PostgreSQL: RUNNING
) else (
    echo PostgreSQL: NOT RUNNING
)

REM Check if Qdrant is running
powershell -Command "if(Get-Process qdrant -ErrorAction SilentlyContinue) { Write-Host 'Qdrant: RUNNING' -ForegroundColor Green } else { Write-Host 'Qdrant: NOT RUNNING' -ForegroundColor Red }"

echo.
echo [Port Status]
echo -------------
netstat -an | findstr :9000 >nul
if %ERRORLEVEL% EQU 0 (
    echo Port 9000 (MinIO): IN USE
) else (
    echo Port 9000 (MinIO): AVAILABLE
)

netstat -an | findstr :5432 >nul
if %ERRORLEVEL% EQU 0 (
    echo Port 5432 (PostgreSQL): IN USE
) else (
    echo Port 5432 (PostgreSQL): AVAILABLE
)

netstat -an | findstr :6333 >nul
if %ERRORLEVEL% EQU 0 (
    echo Port 6333 (Qdrant): IN USE
) else (
    echo Port 6333 (Qdrant): AVAILABLE
)

netstat -an | findstr :8093 >nul
if %ERRORLEVEL% EQU 0 (
    echo Port 8093 (Upload Service): IN USE
) else (
    echo Port 8093 (Upload Service): AVAILABLE
)

echo.
echo [Go Module Status]
echo ------------------
cd go-microservice 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Go Module: FOUND
    
    REM Check for import issues
    findstr /R "github.com/deeds-web/deeds-web-app/go-microservice" cmd\upload-service\main.go >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo Import Issue: DETECTED - Run fix-minio-imports.ps1
    ) else (
        echo Import Issue: NONE
    )
    
    REM Check if binaries exist
    if exist bin\upload-service.exe (
        echo Upload Service Binary: EXISTS
    ) else (
        echo Upload Service Binary: NOT BUILT
    )
    
    if exist bin\summarizer-service.exe (
        echo Summarizer Service Binary: EXISTS
    ) else (
        echo Summarizer Service Binary: NOT BUILT
    )
    
    cd ..
) else (
    echo Go Module: NOT FOUND
)

echo.
echo [Database Status]
echo -----------------
psql -U postgres -d deeds_web_app -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Database Connection: OK
    
    REM Check for pgVector extension
    psql -U postgres -d deeds_web_app -c "SELECT * FROM pg_extension WHERE extname = 'vector';" | findstr vector >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo pgVector Extension: INSTALLED
    ) else (
        echo pgVector Extension: NOT INSTALLED
    )
    
    REM Check for document_metadata table
    psql -U postgres -d deeds_web_app -c "SELECT 1 FROM document_metadata LIMIT 1;" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo Document Metadata Table: EXISTS
    ) else (
        echo Document Metadata Table: NOT FOUND
    )
) else (
    echo Database Connection: FAILED
)

echo.
echo [Environment Variables]
echo -----------------------
if defined MINIO_ENDPOINT (
    echo MINIO_ENDPOINT: %MINIO_ENDPOINT%
) else (
    echo MINIO_ENDPOINT: NOT SET
)

if defined DATABASE_URL (
    echo DATABASE_URL: SET
) else (
    echo DATABASE_URL: NOT SET
)

if defined QDRANT_URL (
    echo QDRANT_URL: %QDRANT_URL%
) else (
    echo QDRANT_URL: NOT SET
)

echo.
echo ========================================
echo Diagnostic Complete
echo ========================================
echo.
echo Next Steps:
echo -----------
echo 1. If services are not running: Run START-MINIO-INTEGRATION.bat
echo 2. If import issues detected: Run fix-minio-imports.ps1
echo 3. If binaries not built: Run "go build" commands in go-microservice directory
echo 4. If pgVector not installed: Run install-pgvector.ps1
echo 5. If database not setup: Run setup script with schema creation
echo.
pause
