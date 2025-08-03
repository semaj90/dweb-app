#!/usr/bin/env pwsh

# Legal AI Assistant - Cross-Platform Docker CLI Manager
# Optimized for WSL2 and Docker Desktop CLI workflows

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "restart", "status", "logs", "clean", "rebuild", "health", "deploy", "wsl", "help")]
    [string]$Action = "help",

    [switch]$GPU,
    [switch]$Optimize,
    [switch]$Production,
    [switch]$Force,
    [string]$Service = "",
    [string]$Environment = "development"
)

# Colors for output
$Host.UI.RawUI.ForegroundColor = "Cyan"

function Write-Header {
    param($message)
    Write-Host "`n🚀 $message" -ForegroundColor Cyan
    Write-Host ("=" * ($message.Length + 3)) -ForegroundColor Cyan
}

function Write-Success {
    param($message)
    Write-Host "✅ $message" -ForegroundColor Green
}

function Write-Warning {
    param($message)
    Write-Host "⚠️  $message" -ForegroundColor Yellow
}

function Write-Error {
    param($message)
    Write-Host "❌ $message" -ForegroundColor Red
}

function Test-DockerStatus {
    try {
        $version = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            docker ps > $null 2>&1
            return $LASTEXITCODE -eq 0
        }
        return $false
    } catch {
        return $false
    }
}

function Test-WSLAvailable {
    try {
        $wslVersion = wsl --version 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Get-ComposeFile {
    if ($Production) {
        return "docker-compose.production.yml"
    } elseif ($GPU) {
        return "docker-compose.gpu.yml"
    } elseif ($Optimize) {
        return "docker-compose.optimized.yml"
    } else {
        return "docker-compose.yml"
    }
}

function Start-DockerDaemon {
    Write-Header "Starting Docker Daemon"

    if (Test-DockerStatus) {
        Write-Success "Docker daemon is already running"
        return $true
    }

    Write-Warning "Docker daemon not running. Attempting to start..."

    # Method 1: Try Docker Desktop CLI
    if (Get-Command "docker-desktop" -ErrorAction SilentlyContinue) {
        Write-Host "Using docker-desktop CLI..." -ForegroundColor Yellow
        docker-desktop start 2>$null
    }

    # Method 2: Try WSL2 Docker
    if ((Test-WSLAvailable) -and (-not (Test-DockerStatus))) {
        Write-Host "Using WSL2 Docker backend..." -ForegroundColor Yellow
        wsl -d docker-desktop -e sh -c "service docker start" 2>$null
    }

    # Method 3: Try com.docker.cli
    if (-not (Test-DockerStatus)) {
        $dockerCliPath = "C:\Program Files\Docker\Docker\resources\com.docker.cli.exe"
        if (Test-Path $dockerCliPath) {
            Write-Host "Using com.docker.cli..." -ForegroundColor Yellow
            & $dockerCliPath start 2>$null
        }
    }

    # Wait for Docker to be ready
    $timeout = 90
    $elapsed = 0

    do {
        Start-Sleep 3
        $elapsed += 3
        if ($elapsed % 15 -eq 0) {
            Write-Host "Waiting for Docker daemon... ($elapsed/$timeout seconds)" -ForegroundColor Yellow
        }
    } while ((-not (Test-DockerStatus)) -and ($elapsed -lt $timeout))

    if (Test-DockerStatus) {
        Write-Success "Docker daemon started successfully"

        # Display Docker info
        $dockerInfo = docker info --format "{{.ServerVersion}}" 2>$null
        $dockerContext = docker context show 2>$null
        Write-Host "Docker Server: $dockerInfo" -ForegroundColor Cyan
        Write-Host "Docker Context: $dockerContext" -ForegroundColor Cyan
        return $true
    } else {
        Write-Error "Failed to start Docker daemon within $timeout seconds"
        Write-Warning "Try: wsl --shutdown && wsl"
        return $false
    }
}

function Start-Services {
    Write-Header "Starting Legal AI Services"

    if (-not (Start-DockerDaemon)) {
        return
    }

    $composeFile = Get-ComposeFile
    Write-Host "Using compose file: $composeFile" -ForegroundColor Cyan

    if (-not (Test-Path $composeFile)) {
        Write-Error "Compose file not found: $composeFile"
        return
    }

    # Pull latest images
    Write-Host "Pulling latest images..." -ForegroundColor Yellow
    docker-compose -f $composeFile pull

    # Start services
    Write-Host "Starting services..." -ForegroundColor Yellow
    if ($Service) {
        docker-compose -f $composeFile up -d $Service
    } else {
        docker-compose -f $composeFile up -d
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Services started successfully"
        Start-Sleep 10
        Show-Status
    } else {
        Write-Error "Failed to start services"
    }
}

function Stop-Services {
    Write-Header "Stopping Legal AI Services"

    $composeFile = Get-ComposeFile

    if ($Service) {
        Write-Host "Stopping service: $Service" -ForegroundColor Yellow
        docker-compose -f $composeFile stop $Service
    } else {
        Write-Host "Stopping all services..." -ForegroundColor Yellow
        docker-compose -f $composeFile down --remove-orphans
    }

    if ($Force) {
        Write-Host "Force removing containers..." -ForegroundColor Yellow
        docker-compose -f $composeFile down -v --remove-orphans
    }

    Write-Success "Services stopped"
}

function Restart-Services {
    Write-Header "Restarting Legal AI Services"
    Stop-Services
    Start-Sleep 5
    Start-Services
}

function Show-Status {
    Write-Header "Legal AI Services Status"

    if (-not (Test-DockerStatus)) {
        Write-Error "Docker daemon not running"
        return
    }

    $composeFile = Get-ComposeFile

    # Show compose services
    Write-Host "`nDocker Compose Services:" -ForegroundColor Cyan
    docker-compose -f $composeFile ps

    # Test service ports
    Write-Host "`nService Health Check:" -ForegroundColor Cyan
    $services = @{
        "PostgreSQL" = 5432
        "Redis" = 6379
        "Ollama" = 11434
        "Qdrant" = 6333
        "Neo4j" = 7474
        "SvelteKit" = 5173
        "RabbitMQ" = 15672
    }

    foreach ($service in $services.GetEnumerator()) {
        $connection = Test-NetConnection -ComputerName localhost -Port $service.Value -WarningAction SilentlyContinue
        if ($connection.TcpTestSucceeded) {
            Write-Success "$($service.Key) responding on port $($service.Value)"
        } else {
            Write-Warning "$($service.Key) not responding on port $($service.Value)"
        }
    }

    # Docker system info
    Write-Host "`nDocker System Info:" -ForegroundColor Cyan
    $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
    $dockerContainers = docker ps --format "{{.Names}}" 2>$null | Measure-Object | Select-Object -ExpandProperty Count
    $dockerImages = docker images --format "{{.Repository}}" 2>$null | Measure-Object | Select-Object -ExpandProperty Count

    Write-Host "Server Version: $dockerVersion" -ForegroundColor Green
    Write-Host "Running Containers: $dockerContainers" -ForegroundColor Green
    Write-Host "Available Images: $dockerImages" -ForegroundColor Green
}

function Show-Logs {
    Write-Header "Legal AI Services Logs"

    $composeFile = Get-ComposeFile

    if ($Service) {
        Write-Host "Showing logs for: $Service" -ForegroundColor Yellow
        docker-compose -f $composeFile logs -f --tail=100 $Service
    } else {
        Write-Host "Showing logs for all services..." -ForegroundColor Yellow
        docker-compose -f $composeFile logs -f --tail=50
    }
}

function Clean-Docker {
    Write-Header "Cleaning Docker System"

    Write-Warning "This will remove unused containers, networks, images, and build cache"

    if ($Force -or (Read-Host "Continue? (y/N)") -eq 'y') {
        Write-Host "Cleaning Docker system..." -ForegroundColor Yellow
        docker system prune -af
        docker volume prune -f
        Write-Success "Docker system cleaned"
    }
}

function Rebuild-Services {
    Write-Header "Rebuilding Legal AI Services"

    $composeFile = Get-ComposeFile

    Write-Host "Stopping services..." -ForegroundColor Yellow
    docker-compose -f $composeFile down

    Write-Host "Building images..." -ForegroundColor Yellow
    docker-compose -f $composeFile build --no-cache

    Write-Host "Starting services..." -ForegroundColor Yellow
    docker-compose -f $composeFile up -d

    Write-Success "Services rebuilt and started"
    Start-Sleep 10
    Show-Status
}

function Test-Health {
    Write-Header "Legal AI Health Check"

    # Run health check script if available
    if (Test-Path "scripts/health-check.js") {
        Write-Host "Running health check script..." -ForegroundColor Yellow
        node scripts/health-check.js
    } else {
        Show-Status
    }

    # Check Ollama models
    if (Test-NetConnection -ComputerName localhost -Port 11434 -WarningAction SilentlyContinue) {
        Write-Host "`nOllama Models:" -ForegroundColor Cyan
        $models = docker exec deeds-ollama-gpu ollama list 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host $models
        } else {
            Write-Warning "Could not retrieve Ollama models"
        }
    }
}

function Deploy-Production {
    Write-Header "Deploying to Production"

    $deployArgs = @()
    if ($GPU) { $deployArgs += "-EnableGPU" }
    if ($Optimize) { $deployArgs += "-OptimizeMemory" }
    if ($Production) { $deployArgs += "-Environment production" }

    $command = "./deploy-production-clean.ps1 $($deployArgs -join ' ')"
    Write-Host "Running: $command" -ForegroundColor Yellow

    Invoke-Expression $command
}

function Start-WSLWorkflow {
    Write-Header "Starting WSL2 Workflow"

    if (-not (Test-WSLAvailable)) {
        Write-Error "WSL2 not available"
        return
    }

    if (Test-Path "./start-wsl.sh") {
        Write-Host "Running WSL startup script..." -ForegroundColor Yellow
        wsl bash ./start-wsl.sh
    } else {
        Write-Error "WSL startup script not found: ./start-wsl.sh"
    }
}

function Show-Help {
    Write-Header "Legal AI Docker CLI Manager"

    Write-Host @"

USAGE:
    .\docker-cli-manager.ps1 <action> [options]

ACTIONS:
    start       Start services
    stop        Stop services
    restart     Restart services
    status      Show service status
    logs        Show service logs
    clean       Clean Docker system
    rebuild     Rebuild and restart services
    health      Run health checks
    deploy      Deploy to production
    wsl         Start WSL2 workflow
    help        Show this help

OPTIONS:
    -GPU                Enable GPU acceleration
    -Optimize           Enable memory optimization
    -Production         Use production configuration
    -Force              Force operation (skip confirmations)
    -Service <name>     Target specific service
    -Environment <env>  Set environment (development/production)

EXAMPLES:
    .\docker-cli-manager.ps1 start -GPU
    .\docker-cli-manager.ps1 stop -Service ollama
    .\docker-cli-manager.ps1 logs -Service sveltekit
    .\docker-cli-manager.ps1 deploy -GPU -Optimize
    .\docker-cli-manager.ps1 clean -Force

ACCESS POINTS:
    • SvelteKit App:     http://localhost:5173
    • Neo4j Browser:     http://localhost:7474
    • RabbitMQ Mgmt:     http://localhost:15672
    • Qdrant Dashboard:  http://localhost:6333/dashboard

"@ -ForegroundColor White
}

# Main execution
switch ($Action) {
    "start"   { Start-Services }
    "stop"    { Stop-Services }
    "restart" { Restart-Services }
    "status"  { Show-Status }
    "logs"    { Show-Logs }
    "clean"   { Clean-Docker }
    "rebuild" { Rebuild-Services }
    "health"  { Test-Health }
    "deploy"  { Deploy-Production }
    "wsl"     { Start-WSLWorkflow }
    "help"    { Show-Help }
    default   { Show-Help }
}
