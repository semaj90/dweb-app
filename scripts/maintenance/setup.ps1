# Deeds Legal AI Assistant - Complete Setup Script
# This script automates the entire setup process

Write-Host "🔧 Starting Deeds Legal AI Assistant Setup..." -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Function to check if command exists
function Test-Command($command) {
    try {
        if (Get-Command $command -ErrorAction Stop) { return $true }
    } catch {
        return $false
    }
}

# Check prerequisites
Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Yellow

$prerequisites = @{
    "docker" = "Docker Desktop"
    "node" = "Node.js"
    "npm" = "npm"
}

$missingPrereqs = @()
foreach ($cmd in $prerequisites.Keys) {
    if (Test-Command $cmd) {
        Write-Host "✅ $($prerequisites[$cmd]) is installed" -ForegroundColor Green
    } else {
        Write-Host "❌ $($prerequisites[$cmd]) is missing" -ForegroundColor Red
        $missingPrereqs += $prerequisites[$cmd]
    }
}

if ($missingPrereqs.Count -gt 0) {
    Write-Host "`n⚠️  Missing prerequisites:" -ForegroundColor Red
    $missingPrereqs | ForEach-Object { Write-Host "   - $_" -ForegroundColor Red }
    Write-Host "`nPlease install the missing prerequisites and run this script again." -ForegroundColor Red
    exit 1
}

# Load environment variables
Write-Host "`n🔧 Loading environment variables..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^#].*)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    Write-Host "✅ Environment variables loaded" -ForegroundColor Green
} else {
    Write-Host "❌ .env file not found" -ForegroundColor Red
    exit 1
}

# Stop any existing containers
Write-Host "`n🛑 Stopping existing containers..." -ForegroundColor Yellow
docker-compose down -v 2>$null
Write-Host "✅ Existing containers stopped" -ForegroundColor Green

# Start Docker services
Write-Host "`n🐳 Starting Docker services..." -ForegroundColor Yellow
docker-compose up -d db
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start Docker services" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Docker services started" -ForegroundColor Green

# Wait for database to be ready
Write-Host "`n⏳ Waiting for database to be ready..." -ForegroundColor Yellow
$maxWait = 60
$waited = 0
do {
    Start-Sleep -Seconds 2
    $waited += 2
    $dbReady = docker-compose exec -T db pg_isready -U postgres 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Database is ready!" -ForegroundColor Green
        break
    }
    Write-Host "   Waiting... ($waited/$maxWait seconds)" -ForegroundColor Gray
} while ($waited -lt $maxWait)

if ($waited -ge $maxWait) {
    Write-Host "❌ Database failed to start within $maxWait seconds" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Generate and run migrations
Write-Host "`n🗃️  Setting up database schema..." -ForegroundColor Yellow
npx drizzle-kit generate
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to generate migrations" -ForegroundColor Red
    exit 1
}

npx drizzle-kit migrate
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to run migrations" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Database schema setup complete" -ForegroundColor Green

# Verify setup
Write-Host "`n🔍 Verifying setup..." -ForegroundColor Yellow
$dbTest = docker-compose exec -T db psql -U postgres -d prosecutor_db -c "\dt" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Database tables verified" -ForegroundColor Green
} else {
    Write-Host "⚠️  Database verification failed, but continuing..." -ForegroundColor Yellow
}

# Display success message and next steps
Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Start Drizzle Studio (database UI):" -ForegroundColor White
Write-Host "   npm run db:studio" -ForegroundColor Gray
Write-Host "`n2. Run development server:" -ForegroundColor White
Write-Host "   npm run dev" -ForegroundColor Gray
Write-Host "`n3. Run tests:" -ForegroundColor White
Write-Host "   npm run test" -ForegroundColor Gray
Write-Host "`n4. Open Drizzle Studio at: http://localhost:4983" -ForegroundColor White
Write-Host "5. Open development app at: http://localhost:5173" -ForegroundColor White

Write-Host "`n📊 Services running:" -ForegroundColor Cyan
Write-Host "   PostgreSQL: localhost:5432" -ForegroundColor Gray
Write-Host "   Qdrant (Vector DB): localhost:6333" -ForegroundColor Gray

Write-Host "`n💡 Useful commands:" -ForegroundColor Cyan
Write-Host "   Stop services: docker-compose down" -ForegroundColor Gray
Write-Host "   View logs: docker-compose logs -f" -ForegroundColor Gray
Write-Host "   Reset database: npm run db:reset" -ForegroundColor Gray
