@echo off
echo Setting up PostgreSQL for Legal AI Application...

REM Set PostgreSQL path
set PGPATH="C:\Program Files\PostgreSQL\17\bin"

REM Add PostgreSQL to PATH temporarily
set PATH=%PGPATH%;%PATH%

echo Creating database and user...
REM Run the SQL setup script
%PGPATH%\psql.exe -U postgres -h localhost -f setup-postgres.sql

if %ERRORLEVEL% EQU 0 (
    echo Database setup completed successfully!
    echo.
    echo Next steps:
    echo 1. Run: npm install
    echo 2. Run: npx drizzle-kit generate
    echo 3. Run: npx drizzle-kit migrate
    echo 4. Start the application: npm run dev
) else (
    echo Database setup failed. Please check the error messages above.
    echo.
    echo Make sure:
    echo 1. PostgreSQL service is running
    echo 2. You have administrator privileges
    echo 3. The default postgres user is accessible
)

pause