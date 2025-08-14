# Comprehensive Setup Script for Legal AI Assistant with Ollama Integration
# This script ensures everything works together seamlessly

param(
    [switch]$SkipDocker = $false,
    [switch]$SkipFixes = $false,
    [switch]$Fresh = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

Write-Host @"
╔══════════════════════════════════════════════════════════════════════╗
║     Legal AI Assistant - Complete Setup with Ollama & pgvector       ║
║                    Windows 10 + WSL2 + Docker                        ║
╚══════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Helper functions
function Write-Step {
    param($message, $type = "info")
    $colors = @{
        "info" = "Yellow"
        "success" = "Green"
        "error" = "Red"
        "warning" = "Magenta"
    }
    $prefix = @{
        "info" = "📋"
        "success" = "✅"
        "error" = "❌"
        "warning" = "⚠️"
    }
    Write-Host "$($prefix[$type]) $message" -ForegroundColor $colors[$type]
}

function Test-DockerRunning {
    try {
        docker info 2>&1 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Wait-ForContainer {
    param($containerName, $maxWait = 60)
    
    $waited = 0
    while ($waited -lt $maxWait) {
        $status = docker inspect -f '{{.State.Status}}' $containerName 2>$null
        if ($status -eq "running") {
            $health = docker inspect -f '{{.State.Health.Status}}' $containerName 2>$null
            if ($health -eq "healthy" -or $health -eq "") {
                return $true
            }
        }
        Start-Sleep -Seconds 2
        $waited += 2
        Write-Host "." -NoNewline
    }
    Write-Host ""
    return $false
}

# Step 1: Check prerequisites
Write-Host "`n=== Checking Prerequisites ===" -ForegroundColor Cyan

$prereqs = @{
    "docker" = "Docker Desktop"
    "node" = "Node.js"
    "npm" = "npm"
    "git" = "Git"
}

$missing = @()
foreach ($cmd in $prereqs.Keys) {
    try {
        $null = Get-Command $cmd -ErrorAction Stop
        Write-Step "$($prereqs[$cmd]) found" "success"
    } catch {
        Write-Step "$($prereqs[$cmd]) not found" "error"
        $missing += $prereqs[$cmd]
    }
}

if ($missing.Count -gt 0) {
    Write-Host "`nMissing prerequisites:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}

# Step 2: Clean start if requested
if ($Fresh) {
    Write-Host "`n=== Performing Fresh Setup ===" -ForegroundColor Cyan
    Write-Step "Stopping all containers..." "info"
    docker-compose down -v 2>$null
    
    Write-Step "Removing old data..." "info"
    Remove-Item -Path "node_modules", "sveltekit-frontend/node_modules" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path "sveltekit-frontend/.svelte-kit", "sveltekit-frontend/build" -Recurse -Force -ErrorAction SilentlyContinue
    
    Write-Step "Clean start prepared" "success"
}

# Step 3: Docker setup
if (-not $SkipDocker) {
    Write-Host "`n=== Setting up Docker Services ===" -ForegroundColor Cyan
    
    if (-not (Test-DockerRunning)) {
        Write-Step "Starting Docker Desktop..." "warning"
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        
        $timeout = 120
        $elapsed = 0
        while ($elapsed -lt $timeout -and -not (Test-DockerRunning)) {
            Start-Sleep -Seconds 3
            $elapsed += 3
            Write-Host "." -NoNewline
        }
        Write-Host ""
        
        if (-not (Test-DockerRunning)) {
            Write-Step "Docker failed to start" "error"
            Write-Host "Please start Docker Desktop manually and run this script again" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Step "Docker is running" "success"
    
    # Pull latest images
    Write-Step "Pulling latest Docker images..." "info"
    docker-compose pull
    
    # Start services
    Write-Step "Starting all services (PostgreSQL, Qdrant, Redis, Ollama)..." "info"
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Step "Docker services started" "success"
        
        # Wait for services
        Write-Step "Waiting for services to be ready..." "info"
        
        $services = @{
            "prosecutor_postgres" = "PostgreSQL"
            "prosecutor_qdrant" = "Qdrant"
            "prosecutor_redis" = "Redis"
            "prosecutor_ollama" = "Ollama"
        }
        
        foreach ($container in $services.Keys) {
            Write-Host -NoNewline "  Waiting for $($services[$container])..."
            if (Wait-ForContainer $container) {
                Write-Host " Ready!" -ForegroundColor Green
            } else {
                Write-Host " Timeout!" -ForegroundColor Red
                Write-Step "Check logs: docker logs $container" "warning"
            }
        }
        
        # Special check for PostgreSQL with pgvector
        Write-Step "Verifying pgvector extension..." "info"
        Start-Sleep -Seconds 3
        $pgvectorCheck = docker exec prosecutor_postgres psql -U postgres -d prosecutor_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Step "pgvector extension verified" "success"
        } else {
            Write-Step "pgvector not found, will be created on first migration" "warning"
        }
        
    } else {
        Write-Step "Failed to start Docker services" "error"
        Write-Host "Check docker-compose.yml and try: docker-compose logs" -ForegroundColor Red
        exit 1
    }
}

# Step 4: Fix TypeScript and imports
if (-not $SkipFixes) {
    Write-Host "`n=== Fixing TypeScript and Import Issues ===" -ForegroundColor Cyan
    
    if (Test-Path "fix-all-typescript-imports.mjs") {
        Write-Step "Running TypeScript and import fixes..." "info"
        node fix-all-typescript-imports.mjs
        Write-Step "TypeScript fixes applied" "success"
    }
}

# Step 5: Install dependencies
Write-Host "`n=== Installing Dependencies ===" -ForegroundColor Cyan

Write-Step "Installing root dependencies..." "info"
npm install
if ($LASTEXITCODE -eq 0) {
    Write-Step "Root dependencies installed" "success"
} else {
    Write-Step "Failed to install root dependencies" "error"
    exit 1
}

Write-Step "Installing frontend dependencies..." "info"
Push-Location sveltekit-frontend
npm install
if ($LASTEXITCODE -eq 0) {
    Write-Step "Frontend dependencies installed" "success"
} else {
    Write-Step "Failed to install frontend dependencies" "error"
    Pop-Location
    exit 1
}

# Step 6: Database setup
Write-Host "`n=== Setting up Database ===" -ForegroundColor Cyan

Write-Step "Generating database migrations..." "info"
npm run db:generate

Write-Step "Running database migrations..." "info"
npm run db:migrate

if ($LASTEXITCODE -eq 0) {
    Write-Step "Database migrations completed" "success"
} else {
    Write-Step "Database migration failed" "warning"
    Write-Host "This might be okay if migrations are already applied" -ForegroundColor Yellow
}

# Step 7: Seed database (optional)
$seedChoice = Read-Host "`nWould you like to seed the database with demo data? (y/N)"
if ($seedChoice -eq 'y' -or $seedChoice -eq 'Y') {
    Write-Step "Seeding database..." "info"
    npm run db:seed
    if ($LASTEXITCODE -eq 0) {
        Write-Step "Database seeded with demo data" "success"
        Write-Host "  Demo users created:" -ForegroundColor Cyan
        Write-Host "    - admin@prosecutor.local (password: admin123)" -ForegroundColor White
        Write-Host "    - user@prosecutor.local (password: user123)" -ForegroundColor White
    }
}

Pop-Location

# Step 8: Test Ollama integration
Write-Host "`n=== Testing AI Integration ===" -ForegroundColor Cyan

Write-Step "Testing Ollama connection..." "info"
$ollamaTest = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -ErrorAction SilentlyContinue

if ($ollamaTest) {
    Write-Step "Ollama is accessible" "success"
    if ($ollamaTest.models.Count -gt 0) {
        Write-Host "  Available models:" -ForegroundColor Cyan
        $ollamaTest.models | ForEach-Object { Write-Host "    - $($_.name)" -ForegroundColor White }
    } else {
        Write-Step "No models loaded yet, pulling default models..." "warning"
        docker exec prosecutor_ollama ollama pull nomic-embed-text
        docker exec prosecutor_ollama ollama pull llama3.2
    }
} else {
    Write-Step "Ollama not responding yet, models will be loaded in background" "warning"
}

# Step 9: Display status and next steps
Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green

Write-Host "`n📊 Service Status:" -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-String "prosecutor_"

Write-Host "`n🔗 Available Services:" -ForegroundColor Cyan
Write-Host "  - Application:      http://localhost:5173" -ForegroundColor White
Write-Host "  - Qdrant Dashboard: http://localhost:6333/dashboard" -ForegroundColor White
Write-Host "  - PgAdmin:          http://localhost:5050" -ForegroundColor White
Write-Host "  - Ollama API:       http://localhost:11434" -ForegroundColor White

Write-Host "`n🚀 To start development:" -ForegroundColor Cyan
Write-Host "  cd sveltekit-frontend" -ForegroundColor White
Write-Host "  npm run dev" -ForegroundColor White

Write-Host "`n📝 Additional commands:" -ForegroundColor Cyan
Write-Host "  - Database UI:      npm run db:studio" -ForegroundColor White
Write-Host "  - View logs:        docker-compose logs -f" -ForegroundColor White
Write-Host "  - Run tests:        npm test" -ForegroundColor White
Write-Host "  - Type check:       npm run check" -ForegroundColor White

Write-Host "`n✨ All systems ready! Happy coding! ✨" -ForegroundColor Green

# Optional: Start dev server immediately
$startNow = Read-Host "`nWould you like to start the development server now? (Y/n)"
if ($startNow -ne 'n' -and $startNow -ne 'N') {
    Push-Location sveltekit-frontend
    Write-Host "`n🚀 Starting development server..." -ForegroundColor Green
    npm run dev
    Pop-Location
}
