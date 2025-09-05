@echo off
echo.
echo =====================================
echo Legal AI Database Setup
echo =====================================
echo.
echo This script will:
echo 1. Create legal_ai_db database
echo 2. Set up legal_admin user
echo 3. Create all necessary tables
echo 4. Try to install pgvector (if available)
echo 5. Insert sample data
echo.
echo You will be prompted for the PostgreSQL password.
echo Default password for postgres user is typically the one you set during installation.
echo.
pause
echo.
echo Running database setup...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -f setup-legal-ai-db.sql
echo.
echo =====================================
echo Setup completed!
echo =====================================
echo.
echo If pgvector installation failed, you can:
echo 1. Install Visual Studio Build Tools
echo 2. Run: cd pgvector && nmake /F Makefile.win install
echo 3. Or use Docker PostgreSQL with pgvector
echo.
echo Your Legal AI application will work without pgvector,
echo but some AI features will be limited.
echo.
pause