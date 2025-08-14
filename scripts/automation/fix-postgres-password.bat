@echo off
echo Fixing PostgreSQL legal_admin password...

REM Set PostgreSQL environment variables
set PGHOST=localhost
set PGPORT=5432
set PGUSER=postgres
set PGPASSWORD=postgres

echo Attempting to change legal_admin password...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -U postgres -d postgres -c "ALTER USER legal_admin WITH PASSWORD '123456';"

echo.
echo Verifying user exists...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -U postgres -d postgres -c "\du legal_admin"

echo.
echo Testing connection with new password...
set PGPASSWORD=123456
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -h localhost -U legal_admin -d legal_ai_db -c "SELECT version();"

pause