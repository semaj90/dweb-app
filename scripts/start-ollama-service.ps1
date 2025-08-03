# Ollama Service Manager for Legal AI System
# PowerShell script to manage Ollama with Docker Desktop CLI

param(
    [ValidateSet("start", "stop", "restart", "status", "models", "pull", "health", "setup")]
    [string]$Action = "start",

    [string]$Model = "gemma3-legal",
    [switch]$GPU,
    [switch]$Force
)

# Colors for output
function Write-Header {
    param($message)
    Write-Host "`n🤖 $message" -ForegroundColor Cyan
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

function Test-DockerRunning {
    try {
        docker ps > $null 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Test-OllamaHealth {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Get-OllamaContainer {
    $containers = docker ps --format "{{.Names}}" | Where-Object { $_ -match "ollama" }
    if ($containers) {
        return $containers[0]
    }
    return $null
}

function Start-OllamaService {
    Write-Header "Starting Ollama Service"

    if (-not (Test-DockerRunning)) {
        Write-Error "Docker is not running. Please start Docker Desktop first."
        return $false
    }

    $containerName = Get-OllamaContainer
    if ($containerName) {
        Write-Success "Ollama container found: $containerName"

        # Check if container is running
        $isRunning = docker ps --filter "name=$containerName" --format "{{.Names}}" | Select-String $containerName
        if ($isRunning) {
            Write-Success "Ollama is already running"
        } else {
            Write-Host "Starting Ollama container..." -ForegroundColor Yellow
            docker start $containerName
            Start-Sleep 10
        }
    } else {
        Write-Warning "No Ollama container found. Starting with Docker Compose..."

        # Determine compose file
        $composeFile = if ($GPU) { "docker-compose.gpu.yml" } else { "docker-compose.yml" }

        if (Test-Path $composeFile) {
            docker-compose -f $composeFile up -d ollama
            Start-Sleep 15
        } else {
            Write-Error "Compose file not found: $composeFile"
            return $false
        }
    }

    # Verify health
    $timeout = 60
    $elapsed = 0
    Write-Host "Waiting for Ollama to be ready..." -ForegroundColor Yellow

    do {
        Start-Sleep 5
        $elapsed += 5
        if ($elapsed % 15 -eq 0) {
            Write-Host "Still waiting... ($elapsed/$timeout seconds)" -ForegroundColor Yellow
        }
    } while ((-not (Test-OllamaHealth)) -and ($elapsed -lt $timeout))

    if (Test-OllamaHealth) {
        Write-Success "Ollama service is ready!"
        Show-Status
        return $true
    } else {
        Write-Error "Ollama failed to start within $timeout seconds"
        return $false
    }
}

function Stop-OllamaService {
    Write-Header "Stopping Ollama Service"

    $containerName = Get-OllamaContainer
    if ($containerName) {
        Write-Host "Stopping Ollama container: $containerName" -ForegroundColor Yellow
        docker stop $containerName
        Write-Success "Ollama service stopped"
    } else {
        Write-Warning "No running Ollama container found"
    }
}

function Restart-OllamaService {
    Write-Header "Restarting Ollama Service"
    Stop-OllamaService
    Start-Sleep 5
    Start-OllamaService
}

function Show-Status {
    Write-Header "Ollama Service Status"

    if (-not (Test-DockerRunning)) {
        Write-Error "Docker is not running"
        return
    }

    $containerName = Get-OllamaContainer
    if ($containerName) {
        Write-Host "Container: $containerName" -ForegroundColor Cyan

        # Container status
        $containerInfo = docker inspect $containerName --format "{{.State.Status}}" 2>$null
        Write-Host "Status: $containerInfo" -ForegroundColor (if ($containerInfo -eq "running") { "Green" } else { "Red" })

        # Health check
        if (Test-OllamaHealth) {
            Write-Success "Health: HEALTHY"

            # Show available models
            try {
                $models = docker exec $containerName ollama list 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "`nAvailable Models:" -ForegroundColor Cyan
                    Write-Host $models
                } else {
                    Write-Warning "Could not retrieve model list"
                }
            } catch {
                Write-Warning "Could not retrieve model list"
            }
        } else {
            Write-Error "Health: UNHEALTHY"
        }

        # Resource usage
        try {
            $stats = docker stats $containerName --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "`nResource Usage:" -ForegroundColor Cyan
                Write-Host $stats
            }
        } catch {
            # Ignore stats errors
        }
    } else {
        Write-Warning "No Ollama container found"
    }
}

function Show-Models {
    Write-Header "Available Ollama Models"

    $containerName = Get-OllamaContainer
    if ($containerName -and (Test-OllamaHealth)) {
        try {
            $models = docker exec $containerName ollama list
            if ($LASTEXITCODE -eq 0) {
                Write-Host $models
            } else {
                Write-Error "Failed to retrieve models"
            }
        } catch {
            Write-Error "Failed to execute ollama list command"
        }
    } else {
        Write-Error "Ollama service is not running or healthy"
    }
}

function Pull-Model {
    Write-Header "Pulling Ollama Model: $Model"

    $containerName = Get-OllamaContainer
    if ($containerName -and (Test-OllamaHealth)) {
        try {
            Write-Host "Pulling model: $Model..." -ForegroundColor Yellow
            docker exec $containerName ollama pull $Model

            if ($LASTEXITCODE -eq 0) {
                Write-Success "Model $Model pulled successfully"
            } else {
                Write-Error "Failed to pull model $Model"
            }
        } catch {
            Write-Error "Failed to execute ollama pull command"
        }
    } else {
        Write-Error "Ollama service is not running or healthy"
    }
}

function Test-Health {
    Write-Header "Ollama Health Check"

    # Docker check
    if (-not (Test-DockerRunning)) {
        Write-Error "Docker is not running"
        return $false
    }
    Write-Success "Docker is running"

    # Container check
    $containerName = Get-OllamaContainer
    if (-not $containerName) {
        Write-Error "No Ollama container found"
        return $false
    }
    Write-Success "Container found: $containerName"

    # Health check
    if (-not (Test-OllamaHealth)) {
        Write-Error "Ollama API is not responding"
        return $false
    }
    Write-Success "Ollama API is healthy"

    # Model check
    try {
        $models = docker exec $containerName ollama list 2>$null | Select-String -Pattern "gemma3|legal"
        if ($models) {
            Write-Success "Legal AI models are available"
            $models | ForEach-Object { Write-Host "  - $_" -ForegroundColor Green }
        } else {
            Write-Warning "No legal AI models found. Consider running: npm run ollama:pull gemma3-legal"
        }
    } catch {
        Write-Warning "Could not check available models"
    }

    return $true
}

function Setup-Models {
    Write-Header "Setting Up Legal AI Models"

    if (-not (Test-OllamaHealth)) {
        Write-Error "Ollama is not running. Please start it first."
        return
    }

    $modelsToSetup = @("gemma3-legal", "nomic-embed-text", "gemma3:2b")

    foreach ($model in $modelsToSetup) {
        Write-Host "Setting up model: $model..." -ForegroundColor Yellow

        $containerName = Get-OllamaContainer
        docker exec $containerName ollama pull $model

        if ($LASTEXITCODE -eq 0) {
            Write-Success "✅ $model setup complete"
        } else {
            Write-Warning "⚠️  Failed to setup $model (may not exist in registry)"
        }
    }

    Write-Success "Model setup complete!"
}

# Main execution
switch ($Action) {
    "start"   { Start-OllamaService }
    "stop"    { Stop-OllamaService }
    "restart" { Restart-OllamaService }
    "status"  { Show-Status }
    "models"  { Show-Models }
    "pull"    { Pull-Model }
    "health"  { Test-Health }
    "setup"   { Setup-Models }
    default   {
        Write-Host @"
Ollama Service Manager for Legal AI

USAGE:
    .\start-ollama-service.ps1 <action> [options]

ACTIONS:
    start       Start Ollama service
    stop        Stop Ollama service
    restart     Restart Ollama service
    status      Show service status
    models      List available models
    pull        Pull a specific model
    health      Run health checks
    setup       Setup legal AI models

OPTIONS:
    -Model <name>       Specify model name (default: gemma3-legal)
    -GPU               Use GPU acceleration
    -Force             Force operation

EXAMPLES:
    .\start-ollama-service.ps1 start -GPU
    .\start-ollama-service.ps1 pull -Model gemma3-legal
    .\start-ollama-service.ps1 setup

"@ -ForegroundColor White
    }
}
