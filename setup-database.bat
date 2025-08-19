@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - DATABASE SETUP SCRIPT
REM ================================================================================

echo Setting up Legal AI Database...

REM Set PostgreSQL password
set PGPASSWORD=123456

echo [1/5] Creating legal_ai_db database...
"C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres -h localhost legal_ai_db 2>nul || echo Database already exists

echo [2/5] Creating legal_admin user...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "CREATE USER legal_admin WITH PASSWORD '123456';" 2>nul || echo User already exists

echo [3/5] Granting privileges...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;"

echo [4/5] Enabling pgvector extension...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo Database setup complete!
