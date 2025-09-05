@echo off
REM Simple database connection test with proper authentication

cls
echo Testing PostgreSQL Connection...
echo ================================

REM Set password via environment variable
set PGPASSWORD=123456

echo Testing connection to legal_ai_db...
psql -U postgres -h localhost -p 5432 -d legal_ai_db -c "SELECT 'Connection successful!' as status, version() as postgresql_version;"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Database connection successful!
    echo.
    echo Testing pgvector extension...
    psql -U postgres -h localhost -p 5432 -d legal_ai_db -c "SELECT 'pgvector installed' as status FROM pg_extension WHERE extname = 'vector';"
    
    echo.
    echo Testing tables...
    psql -U postgres -h localhost -p 5432 -d legal_ai_db -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;"
) else (
    echo.
    echo ❌ Database connection failed!
    echo.
    echo Troubleshooting:
    echo 1. Ensure PostgreSQL is running
    echo 2. Check if legal_ai_db database exists
    echo 3. Verify password is correct (currently: 123456)
    echo 4. Confirm PostgreSQL is listening on localhost:5432
)

pause