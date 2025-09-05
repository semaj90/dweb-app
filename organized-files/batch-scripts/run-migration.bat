@echo off
echo 🗄️ Running Database Schema Migration for Legal AI System
echo ==========================================================
echo.

set PGPASSWORD=123456

echo 📊 Testing database connection...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT version();" -q

if %ERRORLEVEL% EQU 0 (
    echo ✅ Database connection successful
    echo.
    echo 🔧 Running schema migration...
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -f "schema.sql" -q
    
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Schema migration completed successfully
        echo.
        echo 🧪 Testing vector extension...
        "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT * FROM pg_extension WHERE extname = 'vector';" -q
        echo.
        echo 📋 Verifying tables...
        "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "\dt" -q
    ) else (
        echo ❌ Schema migration failed
    )
) else (
    echo ❌ Database connection failed
)

echo.
pause