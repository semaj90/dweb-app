# Simple PowerShell setup script for Legal Case Management System
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("dev", "test", "prod")]
    [string]$Environment = "dev"
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

Write-Host "Environment file copied successfully" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "  1. Run: npm run dev" -ForegroundColor White
Write-Host "  2. Open: http://localhost:5173" -ForegroundColor White
Write-Host ""
Write-Host "Available Commands:" -ForegroundColor Green
Write-Host "  npm run env:dev    - Switch to development" -ForegroundColor White
Write-Host "  npm run env:test   - Switch to testing" -ForegroundColor White
Write-Host "  npm run env:prod   - Switch to production" -ForegroundColor White
Write-Host ""

if ($Environment -ne "dev") {
    Write-Host "For vector search features, ensure Docker is running and run:" -ForegroundColor Blue
    Write-Host "  npm run docker:up" -ForegroundColor White
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
