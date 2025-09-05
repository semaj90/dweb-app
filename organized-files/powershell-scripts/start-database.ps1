# Database Startup Script for Windows
# This script starts the required services for the Deeds Legal AI Assistant

Write-Host "🔧 Starting Deeds Legal AI Assistant Database Services" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Change to project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "`n🐳 Checking Docker..." -ForegroundColor Yellow

# Check if Docker is installed
try {
    $dockerVersion = docker --version 2>$null
    Write-Host "✅ Docker is installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed or not in PATH" -ForegroundColor Red
    Write-Host "   Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Docker daemon is running
try {
    docker info 2>$null | Out-Null
    Write-Host "✅ Docker daemon is running" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Docker is installed but daemon is not running" -ForegroundColor Yellow
    Write-Host "   Please start Docker Desktop and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n🐘 Starting PostgreSQL database..." -ForegroundColor Yellow

# Stop any existing containers
try {
    docker compose down 2>$null
    Write-Host "   Stopped existing containers" -ForegroundColor Gray
} catch {
    # Ignore errors if no containers are running
}

# Start PostgreSQL
try {
    docker compose up -d postgres
    Write-Host "✅ PostgreSQL container started" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to start PostgreSQL container" -ForegroundColor Red
    Write-Host "   Check docker-compose.yml file and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n⏳ Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow

# Wait for database to be ready
$maxAttempts = 30
$attempt = 0

do {
    $attempt++
    try {
        docker exec prosecutor_postgres pg_isready -U postgres -d prosecutor_db 2>$null | Out-Null
        Write-Host "✅ PostgreSQL is ready and accepting connections!" -ForegroundColor Green
        $ready = $true
        break
    } catch {
        Write-Host "   Attempt $attempt/$maxAttempts - waiting..." -ForegroundColor Gray
        Start-Sleep -Seconds 2
    }
} while ($attempt -lt $maxAttempts)

if (-not $ready) {
    Write-Host "⚠️  PostgreSQL did not become ready in time" -ForegroundColor Yellow
    Write-Host "   You can continue but database features may not work" -ForegroundColor Yellow
}

Write-Host "`n📊 Running database migrations..." -ForegroundColor Yellow

# Change to frontend directory
Set-Location "web-app\sveltekit-frontend"

try {
    npx drizzle-kit migrate 2>$null
    Write-Host "✅ Database migrations completed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Migration failed - this may be expected if schema is already up to date" -ForegroundColor Yellow
}

Write-Host "`n🎉 Database setup completed!" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: npm run dev" -ForegroundColor White
Write-Host "  2. Open: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Services running:" -ForegroundColor Cyan
Write-Host "  • PostgreSQL: localhost:5432" -ForegroundColor White
Write-Host "  • Database: prosecutor_db" -ForegroundColor White
Write-Host ""
Write-Host "To stop services later: docker compose down" -ForegroundColor Gray

Read-Host "Press Enter to continue"
