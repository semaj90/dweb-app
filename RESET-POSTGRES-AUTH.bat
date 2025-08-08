@echo off
:: PostgreSQL Password Reset - Windows Method

echo 🔧 Resetting PostgreSQL authentication...

:: Stop PostgreSQL service
net stop postgresql-x64-17

:: Backup pg_hba.conf
copy "C:\Program Files\PostgreSQL\17\data\pg_hba.conf" "C:\Program Files\PostgreSQL\17\data\pg_hba.conf.backup"

:: Create trust authentication temporarily
echo # Temporary trust authentication > "C:\Program Files\PostgreSQL\17\data\pg_hba.conf"
echo local   all             all                                     trust >> "C:\Program Files\PostgreSQL\17\data\pg_hba.conf"
echo host    all             all             127.0.0.1/32            trust >> "C:\Program Files\PostgreSQL\17\data\pg_hba.conf"
echo host    all             all             ::1/128                 trust >> "C:\Program Files\PostgreSQL\17\data\pg_hba.conf"

:: Start PostgreSQL with trust auth
net start postgresql-x64-17

:: Wait for service
timeout /t 5 /nobreak >nul

:: Reset passwords without authentication
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "ALTER USER postgres PASSWORD 'postgres';"
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "ALTER USER legal_admin PASSWORD '123456';"
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;"

:: Restore original pg_hba.conf
net stop postgresql-x64-17
copy "C:\Program Files\PostgreSQL\17\data\pg_hba.conf.backup" "C:\Program Files\PostgreSQL\17\data\pg_hba.conf"
net start postgresql-x64-17

echo ✅ Authentication reset complete
echo Testing new credentials...

:: Test connections
set PGPASSWORD=postgres
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT 'Postgres auth OK' as status;"

set PGPASSWORD=123456
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U legal_admin -h localhost -d legal_ai_db -c "SELECT 'Legal_admin auth OK' as status;"
