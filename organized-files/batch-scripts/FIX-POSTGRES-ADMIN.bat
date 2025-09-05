@echo off
:: Alternative PostgreSQL Auth Fix - Run as Administrator

echo 🔧 PostgreSQL Authentication Reset (Admin Required)
echo ====================================

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ This script requires Administrator privileges
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

:: Find PostgreSQL data directory
set PG_DATA=C:\Program Files\PostgreSQL\17\data
if not exist "%PG_DATA%\pg_hba.conf" (
    echo Searching for PostgreSQL data directory...
    for /d %%D in ("C:\Program Files\PostgreSQL\*") do (
        if exist "%%D\data\pg_hba.conf" (
            set PG_DATA=%%D\data
            echo Found: %%D\data
        )
    )
)

echo Using data directory: %PG_DATA%

:: Stop PostgreSQL (try different service names)
echo Stopping PostgreSQL service...
net stop postgresql-x64-17 2>nul || net stop postgresql 2>nul || net stop "PostgreSQL 17" 2>nul

:: Backup and modify pg_hba.conf
echo Backing up pg_hba.conf...
copy "%PG_DATA%\pg_hba.conf" "%PG_DATA%\pg_hba.conf.backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%"

:: Create trust configuration
echo Creating trust authentication...
(
echo # TYPE  DATABASE        USER            ADDRESS                 METHOD
echo local   all             all                                     trust
echo host    all             all             127.0.0.1/32            trust
echo host    all             all             ::1/128                 trust
echo host    all             all             0.0.0.0/0               trust
) > "%PG_DATA%\pg_hba.conf"

:: Start PostgreSQL
echo Starting PostgreSQL with trust auth...
net start postgresql-x64-17 2>nul || net start postgresql 2>nul || net start "PostgreSQL 17" 2>nul

:: Wait for startup
timeout /t 8 /nobreak >nul

:: Reset all passwords
echo Resetting passwords...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h 127.0.0.1 -p 5432 -c "ALTER USER postgres PASSWORD 'postgres';"
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h 127.0.0.1 -p 5432 -c "ALTER USER legal_admin PASSWORD '123456';"
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h 127.0.0.1 -p 5432 -c "CREATE USER legal_admin WITH PASSWORD '123456' CREATEDB LOGIN;" 2>nul
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h 127.0.0.1 -p 5432 -c "GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;"
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h 127.0.0.1 -p 5432 -c "ALTER DATABASE legal_ai_db OWNER TO legal_admin;"

:: Restore secure configuration
echo Restoring secure authentication...
net stop postgresql-x64-17 2>nul || net stop postgresql 2>nul || net stop "PostgreSQL 17" 2>nul

(
echo # TYPE  DATABASE        USER            ADDRESS                 METHOD
echo local   all             all                                     scram-sha-256
echo host    all             all             127.0.0.1/32            scram-sha-256
echo host    all             all             ::1/128                 scram-sha-256
) > "%PG_DATA%\pg_hba.conf"

net start postgresql-x64-17 2>nul || net start postgresql 2>nul || net start "PostgreSQL 17" 2>nul

timeout /t 5 /nobreak >nul

:: Test new credentials
echo.
echo Testing authentication...
set PGPASSWORD=123456
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -c "SELECT 'SUCCESS: legal_admin authenticated' as status;"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ PostgreSQL authentication fixed!
    echo.
    echo Credentials:
    echo   User: legal_admin
    echo   Pass: 123456
    echo   Database: legal_ai_db
    echo.
    echo Next: Run DEPLOY-GPU-SYSTEM.bat
) else (
    echo ❌ Authentication test failed
    echo Check PostgreSQL logs at: %PG_DATA%\log
)

pause
