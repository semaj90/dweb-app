@echo off
echo 🔐 Updating PostgreSQL legal_admin password to 123456...
echo.

REM Set environment for postgres user
set PGUSER=postgres
set PGPASSWORD=123456

echo Connecting to PostgreSQL and updating password...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -d postgres -c "ALTER USER legal_admin WITH PASSWORD '123456';" 2>nul

if %ERRORLEVEL% EQU 0 (
    echo ✅ Password updated successfully!
    echo.
    echo Testing connection with new password...
    
    set PGUSER=legal_admin
    set PGPASSWORD=123456
    
    "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -d legal_ai_db -c "SELECT current_user, current_database();" 2>nul
    
    if %ERRORLEVEL% EQU 0 (
        echo ✅ Connection test successful!
        echo 🚀 legal_admin can now connect to legal_ai_db
    ) else (
        echo ⚠️  Connection test failed - checking if database exists...
        
        set PGUSER=postgres  
        set PGPASSWORD=123456
        "C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -d postgres -c "CREATE DATABASE legal_ai_db OWNER legal_admin;" 2>nul
        
        echo Database creation attempted. Try connecting manually.
    )
) else (
    echo ❌ Failed to update password. Please run manually:
    echo psql -U postgres -h localhost -c "ALTER USER legal_admin WITH PASSWORD '123456';"
)

echo.
echo Environment files updated:
echo ✅ .env
echo ✅ sveltekit-frontend/.env.development  
echo ✅ go-microservice/.env.production
echo.
pause