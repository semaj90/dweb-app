# PowerShell setup script for Legal Case Management System
# Handles environment switching and database setup

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "test", "prod")]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$false)]
    [switch]$StartDocker,
    
    [Parameter(Mandatory=$false)]
    [switch]$SetupDatabase,
    
    [Parameter(Mandatory=$false)]
    [switch]$RunMigrations
)

Write-Host "Legal Case Management System Setup" -ForegroundColor Green
Write-Host "Environment: $Environment" -ForegroundColor Yellow

# Set environment variables
switch ($Environment) {
    "dev" {
        Write-Host "Setting up DEVELOPMENT environment (SQLite)" -ForegroundColor Cyan
        Copy-Item ".env.development" ".env" -Force
        $env:NODE_ENV = "development"
    }
    "test" {
        Write-Host "Setting up TESTING environment (PostgreSQL + Docker)" -ForegroundColor Cyan
        Copy-Item ".env.testing" ".env" -Force
        $env:NODE_ENV = "testing"
    }
    "prod" {
        Write-Host "Setting up PRODUCTION environment (PostgreSQL + Qdrant)" -ForegroundColor Cyan
        Copy-Item ".env.production" ".env" -Force
        $env:NODE_ENV = "production"
    }
}

Write-Host "Environment file copied" -ForegroundColor Green

# Start Docker services if requested
if ($StartDocker -and ($Environment -eq "test" -or $Environment -eq "prod")) {
    Write-Host "🐳 Starting Docker services..." -ForegroundColor Blue
    
    # Navigate to root directory where docker-compose.yml is located
    Push-Location "../.."
    
    try {
        # Check if Docker is running
        docker ps *>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
            Write-Host "💡 You can start Docker Desktop and then run this script again with -StartDocker" -ForegroundColor Yellow
        } else {
            # Start the services
            docker compose -f docker-compose.yml up -d
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Docker services started successfully" -ForegroundColor Green
                Write-Host "📊 PostgreSQL: localhost:5432" -ForegroundColor White
                Write-Host "🔍 Qdrant: localhost:6333" -ForegroundColor White
                Write-Host "📝 Redis: localhost:6379" -ForegroundColor White
            } else {
                Write-Host "❌ Failed to start Docker services" -ForegroundColor Red
            }
        }
    } catch {
        Write-Host "❌ Docker error: $($_.Exception.Message)" -ForegroundColor Red
    } finally {
        Pop-Location
    }
}

# Setup database if requested
if ($SetupDatabase) {
    Write-Host "🗄️ Setting up database..." -ForegroundColor Blue
    
    if ($Environment -eq "dev") {
        Write-Host "📝 SQLite database will be created automatically" -ForegroundColor Yellow
    } else {
        Write-Host "⏳ Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
}

# Run migrations if requested
if ($RunMigrations) {
    Write-Host "🔄 Running database migrations..." -ForegroundColor Blue
    
    try {
        if ($Environment -eq "dev") {
            npm run db:generate:dev
            npm run db:push:dev
        } else {
            npm run db:generate:test
            npm run db:push:test
        }
        Write-Host "✅ Migrations completed successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Migration failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Green
Write-Host "  1. If using test/prod environment, start Docker Desktop" -ForegroundColor White
Write-Host "  2. Run: npm run dev:$Environment" -ForegroundColor White
Write-Host "  3. Open: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "🛠️ Available Commands:" -ForegroundColor Green
Write-Host "  npm run env:dev          - Switch to development" -ForegroundColor White
Write-Host "  npm run env:test         - Switch to testing" -ForegroundColor White
Write-Host "  npm run env:prod         - Switch to production" -ForegroundColor White
Write-Host "  npm run db:start         - Start Docker services" -ForegroundColor White
Write-Host "  npm run db:push          - Push schema changes" -ForegroundColor White
Write-Host "  npm run db:studio        - Open Drizzle Studio" -ForegroundColor White
Write-Host ""

if ($Environment -ne "dev") {
    Write-Host "🔍 Vector Search Features:" -ForegroundColor Blue
    Write-Host "  - Semantic case search using pgvector" -ForegroundColor White
    Write-Host "  - Advanced evidence search with Qdrant" -ForegroundColor White
    Write-Host "  - Redis caching for performance" -ForegroundColor White
    Write-Host "  - Hybrid text + semantic search" -ForegroundColor White
}

Write-Host ""
Write-Host "🎉 Setup complete! Happy coding!" -ForegroundColor Green
