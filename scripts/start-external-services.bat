@echo off
echo 🚀 Starting External Services for Legal AI Platform
echo.

REM Check if we're in the right directory
if not exist "sveltekit-frontend" (
    echo Error: Please run this script from the deeds-web-app root directory
    pause
    exit /b 1
)

echo 📦 Starting MinIO Object Storage (Port 9000)...
start "MinIO Server" cmd /k "echo Starting MinIO on port 9000 && echo Download from https://min.io/download if needed && pause"

echo 📡 Starting Redis Cache (Port 6379)...
start "Redis Server" cmd /k "echo Starting Redis on port 6379 && echo Download from https://redis.io/download if needed && pause"

echo 🗄️  PostgreSQL should be running on port 5432
echo   If not installed, download from https://www.postgresql.org/download/

echo 🔗 Neo4j should be running on port 7474
echo   If not installed, download from https://neo4j.com/download/

echo.
echo ✅ External services startup initiated
echo ⚠️  Make sure to:
echo   1. Start MinIO with: minio.exe server ./data
echo   2. Start Redis with: redis-server
echo   3. Ensure PostgreSQL is running with pgvector extension
echo   4. Ensure Neo4j is running
echo.
echo 🔄 After services are running, use: npm run dev:full
pause