@echo off
echo 🔧 Running Database Migrations
echo =============================

set PGPASSWORD=LegalAI2024!

echo Waiting for PostgreSQL...
timeout /t 10 /nobreak >nul

echo Running migrations...
psql -h localhost -p 5432 -U legal_admin -d legal_ai_db -f database\migrations\001_initial_schema.sql

if errorlevel 1 (
    echo ❌ Migration failed
    pause
    exit /b 1
)

echo ✅ Database migrations complete!
pause