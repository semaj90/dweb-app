# Production Database Migration Script
# Handles database setup, migration, and validation

param(
    [switch]$Reset,
    [switch]$SeedData,
    [string]$Environment = "development"
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " DATABASE MIGRATION MANAGER" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if Docker is running
function Test-DockerService {
    try {
        $dockerStatus = docker ps 2>&1
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

# Function to check database connection
function Test-DatabaseConnection {
    Write-Host "[CHECK] Testing database connection..." -ForegroundColor Yellow
    try {
        $result = docker exec prosecutor_postgres psql -U postgres -d prosecutor_db -c "\dt" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Database connection established!" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[ERROR] Database connection failed!" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "[ERROR] Could not connect to database!" -ForegroundColor Red
        return $false
    }
}

# Function to run migrations
function Invoke-DatabaseMigration {
    Write-Host "[MIGRATION] Running database migrations..." -ForegroundColor Yellow
    
    Set-Location "sveltekit-frontend"
    
    try {
        # Generate migration files if needed
        Write-Host "Generating migration files..." -ForegroundColor Cyan
        npx drizzle-kit generate:pg
        
        # Push schema changes
        Write-Host "Pushing schema changes..." -ForegroundColor Cyan
        npx drizzle-kit push:pg
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Database migration completed!" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[ERROR] Database migration failed!" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "[ERROR] Migration process failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
    finally {
        Set-Location ".."
    }
}

# Function to seed database
function Invoke-DatabaseSeed {
    Write-Host "[SEED] Seeding database with initial data..." -ForegroundColor Yellow
    
    try {
        npm run seed
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[SUCCESS] Database seeding completed!" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[ERROR] Database seeding failed!" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "[ERROR] Seeding process failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Main execution
Write-Host "Environment: $Environment" -ForegroundColor Cyan
Write-Host ""

# Check Docker service
if (-not (Test-DockerService)) {
    Write-Host "[ERROR] Docker is not running or not accessible!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Start Docker services if not running
Write-Host "[DOCKER] Starting Docker services..." -ForegroundColor Yellow
docker-compose up -d
Start-Sleep -Seconds 5

# Reset database if requested
if ($Reset) {
    Write-Host "[RESET] Resetting database..." -ForegroundColor Yellow
    docker exec prosecutor_postgres psql -U postgres -c "DROP DATABASE IF EXISTS prosecutor_db; CREATE DATABASE prosecutor_db;"
    Write-Host "[RESET] Database reset completed!" -ForegroundColor Green
}

# Test database connection
if (-not (Test-DatabaseConnection)) {
    Write-Host "[ERROR] Cannot proceed without database connection!" -ForegroundColor Red
    exit 1
}

# Run migrations
if (-not (Invoke-DatabaseMigration)) {
    Write-Host "[ERROR] Migration failed! Check the errors above." -ForegroundColor Red
    exit 1
}

# Seed data if requested
if ($SeedData) {
    if (-not (Invoke-DatabaseSeed)) {
        Write-Host "[WARNING] Seeding failed, but migration was successful." -ForegroundColor Yellow
    }
}

# Final validation
Write-Host ""
Write-Host "[VALIDATION] Running final database validation..." -ForegroundColor Yellow
docker exec prosecutor_postgres psql -U postgres -d prosecutor_db -c "\dt"

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host " MIGRATION COMPLETE!" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run 'npm run check' to validate TypeScript" -ForegroundColor White
Write-Host "2. Run 'npm run dev' to start development server" -ForegroundColor White
Write-Host "3. Open VS Code to test MCP server integration" -ForegroundColor White
Write-Host ""
