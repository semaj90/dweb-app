# Start PostgreSQL with pgvector using Docker
Write-Host "Starting PostgreSQL with pgvector support using Docker..." -ForegroundColor Green

# Check if Docker is installed
$dockerVersion = docker --version 2>$null
if (-not $dockerVersion) {
    Write-Host "Docker is not installed. Please install Docker Desktop for Windows first." -ForegroundColor Red
    Write-Host "Download from: https://www.docker.com/products/docker-desktop" -ForegroundColor Cyan
    exit 1
}

Write-Host "Docker version: $dockerVersion" -ForegroundColor Gray

# Stop any existing PostgreSQL container
Write-Host "`nStopping any existing PostgreSQL containers..." -ForegroundColor Yellow
docker stop legal_ai_postgres 2>$null
docker rm legal_ai_postgres 2>$null

# Start PostgreSQL with pgvector
Write-Host "`nStarting PostgreSQL 17 with pgvector..." -ForegroundColor Green
docker run -d `
    --name legal_ai_postgres `
    -e POSTGRES_USER=postgres `
    -e POSTGRES_PASSWORD=postgres `
    -e POSTGRES_DB=legal_ai_db `
    -p 5432:5432 `
    -v legal_ai_pgdata:/var/lib/postgresql/data `
    pgvector/pgvector:pg17

# Wait for PostgreSQL to be ready
Write-Host "`nWaiting for PostgreSQL to be ready..." -ForegroundColor Yellow
$retries = 30
while ($retries -gt 0) {
    $result = docker exec legal_ai_postgres pg_isready -U postgres 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "PostgreSQL is ready!" -ForegroundColor Green
        break
    }
    Write-Host "." -NoNewline
    Start-Sleep -Seconds 1
    $retries--
}

if ($retries -eq 0) {
    Write-Host "`nPostgreSQL failed to start properly." -ForegroundColor Red
    exit 1
}

Write-Host "`n`nSetting up database with pgvector extension..." -ForegroundColor Yellow

# Copy setup script to container and execute
docker cp setup-postgres.sql legal_ai_postgres:/tmp/setup.sql
docker exec -i legal_ai_postgres psql -U postgres -f /tmp/setup.sql

Write-Host "`nPostgreSQL with pgvector is now running!" -ForegroundColor Green
Write-Host "`nConnection details:" -ForegroundColor Cyan
Write-Host "  Host: localhost" -ForegroundColor Gray
Write-Host "  Port: 5432" -ForegroundColor Gray
Write-Host "  Database: legal_ai_db" -ForegroundColor Gray
Write-Host "  User: legal_admin" -ForegroundColor Gray
Write-Host "  Password: LegalAI2024!" -ForegroundColor Gray

Write-Host "`nTo verify pgvector installation:" -ForegroundColor Yellow
Write-Host '  docker exec -it legal_ai_postgres psql -U postgres -d legal_ai_db -c "SELECT * FROM pg_extension WHERE extname = ''vector'';"' -ForegroundColor Gray

Write-Host "`nNext steps:" -ForegroundColor Green
Write-Host "1. Update your .env file with the connection details"
Write-Host "2. Run: npm install"
Write-Host "3. Run: npx drizzle-kit generate"
Write-Host "4. Run: npx drizzle-kit migrate"
Write-Host "5. Start the application: npm run dev"