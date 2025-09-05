# Complete Development Startup Script for Deeds Legal AI Assistant
# This script handles all setup and startup tasks

param(
    [switch]$SkipDocker = $false,
    [switch]$SkipDeps = $false,
    [switch]$WithLLM = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"

Write-Host @"
╔═══════════════════════════════════════════════════════════════╗
║        Deeds Legal AI Assistant - Development Startup         ║
╚═══════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

# Helper function for logging
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

# Check if running from correct directory
if (-not (Test-Path "package.json")) {
    Write-Step "Not in project root directory!" "error"
    Write-Host "Please run this script from: C:\Users\james\Desktop\web-app" -ForegroundColor Red
    exit 1
}

# Step 1: Check prerequisites
Write-Host "`n=== Checking Prerequisites ===" -ForegroundColor Cyan

$prereqs = @{
    "docker" = "Docker Desktop"
    "node" = "Node.js"
    "npm" = "npm"
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
    Write-Host "`nPlease install missing prerequisites and try again." -ForegroundColor Red
    exit 1
}

# Step 2: Docker setup
if (-not $SkipDocker) {
    Write-Host "`n=== Setting up Docker Services ===" -ForegroundColor Cyan
    
    Write-Step "Checking Docker daemon..." "info"
    try {
        docker info | Out-Null
        Write-Step "Docker is running" "success"
    } catch {
        Write-Step "Docker is not running. Starting Docker Desktop..." "warning"
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        Write-Host "Waiting for Docker to start (this may take a minute)..." -ForegroundColor Yellow
        
        $timeout = 60
        $elapsed = 0
        while ($elapsed -lt $timeout) {
            Start-Sleep -Seconds 2
            $elapsed += 2
            try {
                docker info | Out-Null
                Write-Step "Docker started successfully" "success"
                break
            } catch {
                Write-Host "." -NoNewline
            }
        }
        
        if ($elapsed -ge $timeout) {
            Write-Step "Docker failed to start within $timeout seconds" "error"
            exit 1
        }
    }
    
    Write-Step "Starting database services..." "info"
    docker-compose up -d postgres qdrant redis
    
    if ($LASTEXITCODE -eq 0) {
        Write-Step "Docker services started" "success"
        
        # Wait for PostgreSQL to be ready
        Write-Step "Waiting for PostgreSQL to be ready..." "info"
        $maxWait = 30
        $waited = 0
        while ($waited -lt $maxWait) {
            $result = docker exec prosecutor_postgres pg_isready -U postgres 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Step "PostgreSQL is ready" "success"
                break
            }
            Start-Sleep -Seconds 2
            $waited += 2
            Write-Host "." -NoNewline
        }
        Write-Host ""
    } else {
        Write-Step "Failed to start Docker services" "error"
        Write-Host "Check docker-compose.yml and try: docker-compose logs" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Install dependencies
if (-not $SkipDeps) {
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
    Pop-Location
}

# Step 4: Database setup
Write-Host "`n=== Setting up Database ===" -ForegroundColor Cyan

Push-Location sveltekit-frontend
Write-Step "Running database migrations..." "info"
npm run db:migrate

if ($LASTEXITCODE -eq 0) {
    Write-Step "Database migrations completed" "success"
} else {
    Write-Step "Database migration failed" "warning"
    Write-Host "This might be okay if migrations are already applied" -ForegroundColor Yellow
}

# Optional: Seed database
$seedChoice = Read-Host "Would you like to seed the database with demo data? (y/N)"
if ($seedChoice -eq 'y' -or $seedChoice -eq 'Y') {
    Write-Step "Seeding database..." "info"
    npm run db:seed
    if ($LASTEXITCODE -eq 0) {
        Write-Step "Database seeded" "success"
    } else {
        Write-Step "Database seeding failed" "warning"
    }
}
Pop-Location

# Step 5: LLM Setup (if requested)
if ($WithLLM) {
    Write-Host "`n=== Setting up Local LLM ===" -ForegroundColor Cyan
    
    Write-Step "Checking Ollama installation..." "info"
    try {
        $null = Get-Command ollama -ErrorAction Stop
        Write-Step "Ollama found" "success"
        
        Write-Step "Starting Ollama service..." "info"
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
        
        Write-Step "Pulling Gemma3 Legal model (this may take a while)..." "info"
        ollama pull gemma3-legal
        
        if ($LASTEXITCODE -eq 0) {
            Write-Step "LLM setup complete" "success"
        } else {
            Write-Step "Failed to pull model" "warning"
            Write-Host "You can manually pull it later with: ollama pull gemma3-legal" -ForegroundColor Yellow
        }
    } catch {
        Write-Step "Ollama not found" "warning"
        Write-Host "Install from: https://ollama.com/download" -ForegroundColor Yellow
    }
}

# Step 6: Start the application
Write-Host "`n=== Starting Application ===" -ForegroundColor Cyan

Write-Step "Launching SvelteKit development server..." "info"
Write-Host ""
Write-Host "🚀 Application will start at: http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "Available URLs:" -ForegroundColor Cyan
Write-Host "  - Application:    http://localhost:5173" -ForegroundColor White
Write-Host "  - Database UI:    Run 'npm run db:studio' in another terminal" -ForegroundColor White
Write-Host "  - Qdrant UI:      http://localhost:6333/dashboard" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the dev server
Push-Location sveltekit-frontend
npm run dev
Pop-Location
