@echo off
:: Direct PostgreSQL password fix and system deployment

echo 🔧 Fixing PostgreSQL authentication...

:: Method 1: Direct psql with file execution
set PGPASSWORD=postgres
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d legal_ai_db -f fix-db-auth.sql

if %ERRORLEVEL% NEQ 0 (
    echo Trying alternate connection method...
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h 127.0.0.1 -d legal_ai_db -f fix-db-auth.sql
)

:: Verify new credentials work
set PGPASSWORD=123456
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -c "SELECT 'Auth success' as status;"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Database authentication fixed
    
    :: Apply GPU schema migration
    echo 🔄 Applying schema migration...
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -f database\gpu-schema-migration.sql
    
    :: Build services
    echo 🔨 Building GPU services...
    cd go-microservice
    go get github.com/fsnotify/fsnotify github.com/minio/simdjson-go
    go mod tidy
    call BUILD-GPU-SIMD-FIXED.bat
    cd ..
    
    :: Start system
    echo 🚀 Starting integrated system...
    call START-INTEGRATED-SYSTEM.bat
    
    :: Quick health check
    timeout /t 10 /nobreak >nul
    curl -f http://localhost:8080/health && echo ✅ System operational
    
) else (
    echo ❌ Authentication fix failed - manual intervention required
)
