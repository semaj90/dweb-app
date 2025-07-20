# PowerShell script to run tests with PostgreSQL
Write-Host "=== Legal AI Assistant Test Setup ===" -ForegroundColor Green

# Set environment for testing first
$env:NODE_ENV = "testing"
Write-Host "Setting NODE_ENV to testing..." -ForegroundColor Yellow

# Copy testing environment config
Write-Host "Copying test environment configuration..." -ForegroundColor Yellow
Copy-Item ".env.testing" ".env" -Force

# Start Docker containers
Write-Host "Starting Docker containers..." -ForegroundColor Green
docker-compose up -d

# Wait for services to be ready
Write-Host "Waiting for database to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Test database connection
Write-Host "Testing database connection..." -ForegroundColor Green
node test-db-connection.js

# Run database migrations
Write-Host "Running database migrations..." -ForegroundColor Green
$env:NODE_ENV = "testing"
npx drizzle-kit migrate

# Seed database with demo users
Write-Host "Seeding database..." -ForegroundColor Green
$env:NODE_ENV = "testing"
npm run db:seed

# Run Playwright tests
Write-Host "Running Playwright tests..." -ForegroundColor Green
$env:NODE_ENV = "testing"
npx playwright test --reporter=list

Write-Host "Tests completed!" -ForegroundColor Green
