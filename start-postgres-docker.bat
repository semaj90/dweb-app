@echo off
echo Starting PostgreSQL with pgvector support using Docker...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed. Please install Docker Desktop for Windows first.
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo.
echo Stopping any existing PostgreSQL containers...
docker stop legal_ai_postgres >nul 2>&1
docker rm legal_ai_postgres >nul 2>&1

echo.
echo Starting PostgreSQL 17 with pgvector...
docker run -d ^
    --name legal_ai_postgres ^
    -e POSTGRES_USER=postgres ^
    -e POSTGRES_PASSWORD=postgres ^
    -e POSTGRES_DB=legal_ai_db ^
    -p 5432:5432 ^
    -v legal_ai_pgdata:/var/lib/postgresql/data ^
    pgvector/pgvector:pg17

echo.
echo Waiting for PostgreSQL to be ready...
:wait_loop
docker exec legal_ai_postgres pg_isready -U postgres >nul 2>&1
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

echo PostgreSQL is ready!

echo.
echo Setting up database with pgvector extension...
docker cp setup-postgres.sql legal_ai_postgres:/tmp/setup.sql
docker exec -i legal_ai_postgres psql -U postgres -f /tmp/setup.sql

echo.
echo PostgreSQL with pgvector is now running!
echo.
echo Connection details:
echo   Host: localhost
echo   Port: 5432
echo   Database: legal_ai_db
echo   User: legal_admin
echo   Password: LegalAI2024!
echo.
echo To verify pgvector installation:
echo   docker exec -it legal_ai_postgres psql -U postgres -d legal_ai_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
echo.
echo Next steps:
echo 1. Update your .env file with the connection details
echo 2. Run: npm install
echo 3. Run: npx drizzle-kit generate
echo 4. Run: npx drizzle-kit migrate
echo 5. Start the application: npm run dev
echo.
pause