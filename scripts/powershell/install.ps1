# Legal AI System - Complete Installation Script
# PowerShell 7+ Required
# Run as Administrator for Docker operations

param(
    [switch]$SkipDocker = $false,
    [switch]$SkipModels = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'

# Configuration
$script:Config = @{
    ProjectRoot = $PSScriptRoot
    RequiredPorts = @(5432, 6379, 6333, 11434, 7474, 7687, 8001)
    RequiredServices = @('postgres', 'redis', 'qdrant', 'ollama', 'neo4j')
    VectorDimensions = 384
    EmbeddingModel = "nomic-embed-text"
    LLMModel = "gemma3-legal"
}

# Logging functions
function Write-Step {
    param([string]$Message, [string]$Type = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Type) {
        "INFO" { "Cyan" }
        "SUCCESS" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        default { "White" }
    }
    Write-Host "[$timestamp] [$Type] $Message" -ForegroundColor $color
}

function Test-Prerequisites {
    Write-Step "Checking prerequisites..."
    
    # Check PowerShell version
    if ($PSVersionTable.PSVersion.Major -lt 7) {
        Write-Step "PowerShell 7+ required. Current version: $($PSVersionTable.PSVersion)" "ERROR"
        Write-Step "Install from: https://github.com/PowerShell/PowerShell/releases" "INFO"
        return $false
    }
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Step "Docker found: $dockerVersion" "SUCCESS"
    } catch {
        Write-Step "Docker not found. Please install Docker Desktop" "ERROR"
        return $false
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Step "Node.js found: $nodeVersion" "SUCCESS"
    } catch {
        Write-Step "Node.js not found. Please install Node.js 18+" "ERROR"
        return $false
    }
    
    # Check ports
    $blockedPorts = @()
    foreach ($port in $Config.RequiredPorts) {
        $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -InformationLevel Quiet
        if ($connection) {
            $blockedPorts += $port
        }
    }
    
    if ($blockedPorts.Count -gt 0) {
        Write-Step "Following ports are already in use: $($blockedPorts -join ', ')" "WARNING"
        Write-Step "Run 'npm run db:down' to stop existing services" "INFO"
    }
    
    return $true
}

function Install-RootDependencies {
    Write-Step "Installing root dependencies..."
    
    Push-Location $Config.ProjectRoot
    try {
        # Install root package dependencies
        if (Test-Path "package.json") {
            Write-Step "Installing npm packages..."
            npm install --force
            if ($LASTEXITCODE -ne 0) {
                throw "npm install failed"
            }
            Write-Step "Root dependencies installed" "SUCCESS"
        }
        
        # Install frontend dependencies
        $frontendPath = Join-Path $Config.ProjectRoot "sveltekit-frontend"
        if (Test-Path $frontendPath) {
            Push-Location $frontendPath
            Write-Step "Installing frontend dependencies..."
            npm install --force
            if ($LASTEXITCODE -ne 0) {
                Write-Step "Frontend npm install failed - creating TODO" "WARNING"
                Add-TodoItem "Fix frontend npm dependencies" "Check package-lock.json conflicts"
            }
            Pop-Location
        }
    } finally {
        Pop-Location
    }
}

function Start-DockerServices {
    if ($SkipDocker) {
        Write-Step "Skipping Docker services (--SkipDocker flag set)" "WARNING"
        return
    }
    
    Write-Step "Starting Docker services..."
    
    Push-Location $Config.ProjectRoot
    try {
        # Stop existing containers
        Write-Step "Stopping existing containers..."
        docker-compose down --remove-orphans
        
        # Start services
        Write-Step "Starting infrastructure services..."
        docker-compose up -d postgres redis qdrant
        
        # Wait for services to be healthy
        Write-Step "Waiting for services to be healthy..."
        $maxAttempts = 30
        $attempt = 0
        
        while ($attempt -lt $maxAttempts) {
            $healthy = $true
            
            # Check PostgreSQL
            try {
                docker exec legal_ai_postgres pg_isready -U postgres -d prosecutor_db | Out-Null
                if ($LASTEXITCODE -ne 0) { $healthy = $false }
            } catch { $healthy = $false }
            
            # Check Redis
            try {
                docker exec legal_ai_redis redis-cli ping | Out-Null
                if ($LASTEXITCODE -ne 0) { $healthy = $false }
            } catch { $healthy = $false }
            
            # Check Qdrant
            try {
                $qdrantHealth = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET
                if ($qdrantHealth.status -ne "ok") { $healthy = $false }
            } catch { $healthy = $false }
            
            if ($healthy) {
                Write-Step "All services are healthy!" "SUCCESS"
                break
            }
            
            $attempt++
            Write-Host "." -NoNewline
            Start-Sleep -Seconds 2
        }
        
        if (-not $healthy) {
            Write-Step "Services failed to start properly" "ERROR"
            docker-compose logs --tail=50
            throw "Docker services unhealthy"
        }
        
        # Initialize database
        Write-Step "Initializing database..."
        docker exec -i legal_ai_postgres psql -U postgres -d prosecutor_db < (Join-Path $Config.ProjectRoot "database\db-init.sql")
        
        # Initialize Qdrant collections
        Write-Step "Initializing Qdrant collections..."
        $qdrantConfig = Get-Content (Join-Path $Config.ProjectRoot "database\qdrant-init.json") | ConvertFrom-Json
        
        foreach ($collection in $qdrantConfig.collections) {
            try {
                $body = @{
                    vectors = @{
                        size = $collection.vectors.size
                        distance = $collection.vectors.distance
                    }
                    shard_number = $collection.shard_number
                    replication_factor = $collection.replication_factor
                    optimizers_config = $collection.optimizers_config
                    hnsw_config = $collection.hnsw_config
                } | ConvertTo-Json -Depth 10
                
                Invoke-RestMethod -Uri "http://localhost:6333/collections/$($collection.name)" -Method PUT -Body $body -ContentType "application/json"
                Write-Step "Created collection: $($collection.name)" "SUCCESS"
            } catch {
                if ($_.Exception.Response.StatusCode -eq 'Conflict') {
                    Write-Step "Collection $($collection.name) already exists" "INFO"
                } else {
                    Write-Step "Failed to create collection $($collection.name): $_" "WARNING"
                    Add-TodoItem "Fix Qdrant collection" "Collection: $($collection.name), Error: $_"
                }
            }
        }
        
    } finally {
        Pop-Location
    }
}

function Install-AIModels {
    if ($SkipModels) {
        Write-Step "Skipping AI models (--SkipModels flag set)" "WARNING"
        return
    }
    
    Write-Step "Installing AI models..."
    
    # Start Ollama service
    Write-Step "Starting Ollama service..."
    docker-compose up -d ollama
    Start-Sleep -Seconds 5
    
    # Check if Ollama is running locally
    $ollamaLocal = $false
    try {
        $localOllama = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        $ollamaLocal = $true
        Write-Step "Using local Ollama instance" "INFO"
    } catch {
        Write-Step "Using Docker Ollama instance" "INFO"
    }
    
    # Pull embedding model
    Write-Step "Pulling nomic-embed-text model..."
    if ($ollamaLocal) {
        ollama pull nomic-embed-text
    } else {
        docker exec legal_ai_ollama ollama pull nomic-embed-text
    }
    
    # Create custom legal model
    $modelfilePath = Join-Path $Config.ProjectRoot "local-models\Modelfile.gemma3-legal"
    if (Test-Path $modelfilePath) {
        Write-Step "Creating gemma3-legal model..."
        if ($ollamaLocal) {
            ollama create gemma3-legal -f $modelfilePath
        } else {
            docker cp $modelfilePath legal_ai_ollama:/tmp/Modelfile.gemma3-legal
            docker exec legal_ai_ollama ollama create gemma3-legal -f /tmp/Modelfile.gemma3-legal
        }
    } else {
        Write-Step "Modelfile not found, using base gemma model" "WARNING"
        if ($ollamaLocal) {
            ollama pull gemma:2b
        } else {
            docker exec legal_ai_ollama ollama pull gemma:2b
        }
    }
    
    Write-Step "AI models installed" "SUCCESS"
}

function Initialize-Database {
    Write-Step "Running database migrations..."
    
    Push-Location $Config.ProjectRoot
    try {
        # Run Drizzle migrations
        npm run db:push
        if ($LASTEXITCODE -ne 0) {
            Write-Step "Database migration failed" "ERROR"
            Add-TodoItem "Fix database migrations" "Check drizzle config and schema"
        } else {
            Write-Step "Database migrations completed" "SUCCESS"
        }
    } finally {
        Pop-Location
    }
}

function Add-TodoItem {
    param(
        [string]$Title,
        [string]$Description
    )
    
    $todoFile = Join-Path $Config.ProjectRoot "TODO.md"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    if (-not (Test-Path $todoFile)) {
        "# Legal AI System - TODO List`n`n" | Out-File $todoFile
    }
    
    $todoEntry = @"

## [$timestamp] $Title
- **Status**: Pending
- **Description**: $Description
- **Created by**: install.ps1
---
"@
    
    Add-Content -Path $todoFile -Value $todoEntry
    Write-Step "Added TODO: $Title" "WARNING"
}

function Test-SystemHealth {
    Write-Step "Running system health checks..."
    
    $healthResults = @{
        Overall = $true
        Services = @{}
    }
    
    # Test PostgreSQL
    try {
        docker exec legal_ai_postgres pg_isready -U postgres -d prosecutor_db | Out-Null
        $healthResults.Services["PostgreSQL"] = @{
            Status = "Healthy"
            Details = "Database accepting connections"
        }
    } catch {
        $healthResults.Services["PostgreSQL"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Test Redis
    try {
        $redisPing = docker exec legal_ai_redis redis-cli ping
        if ($redisPing -eq "PONG") {
            $healthResults.Services["Redis"] = @{
                Status = "Healthy"
                Details = "Redis responding to ping"
            }
        }
    } catch {
        $healthResults.Services["Redis"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Test Qdrant
    try {
        $qdrantHealth = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET
        $healthResults.Services["Qdrant"] = @{
            Status = "Healthy"
            Details = "Version: $($qdrantHealth.version)"
        }
    } catch {
        $healthResults.Services["Qdrant"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Test Ollama
    try {
        $ollamaModels = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        $modelCount = $ollamaModels.models.Count
        $healthResults.Services["Ollama"] = @{
            Status = "Healthy"
            Details = "Models loaded: $modelCount"
        }
    } catch {
        $healthResults.Services["Ollama"] = @{
            Status = "Unhealthy"
            Details = $_
        }
        $healthResults.Overall = $false
    }
    
    # Display results
    Write-Host "`n=== System Health Report ===" -ForegroundColor Cyan
    foreach ($service in $healthResults.Services.Keys) {
        $status = $healthResults.Services[$service]
        $color = if ($status.Status -eq "Healthy") { "Green" } else { "Red" }
        Write-Host "$service : $($status.Status) - $($status.Details)" -ForegroundColor $color
    }
    
    if ($healthResults.Overall) {
        Write-Step "All systems operational!" "SUCCESS"
    } else {
        Write-Step "Some services are unhealthy. Check logs with: docker-compose logs" "ERROR"
    }
    
    return $healthResults
}

# Main execution
function Main {
    Write-Host @"
╔═══════════════════════════════════════════════════════╗
║          Legal AI System - Installation Script         ║
║                   Version 1.0.0                        ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

    # Check prerequisites
    if (-not (Test-Prerequisites)) {
        Write-Step "Prerequisites check failed. Please install missing components." "ERROR"
        return
    }
    
    try {
        # Install dependencies
        Install-RootDependencies
        
        # Start Docker services
        Start-DockerServices
        
        # Install AI models
        Install-AIModels
        
        # Initialize database
        Initialize-Database
        
        # Run health checks
        $health = Test-SystemHealth
        
        # Final instructions
        Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
        Write-Host "To start the development server, run:" -ForegroundColor Yellow
        Write-Host "  cd sveltekit-frontend && npm run dev" -ForegroundColor White
        Write-Host "`nAccess the application at:" -ForegroundColor Yellow
        Write-Host "  http://localhost:5173" -ForegroundColor White
        Write-Host "`nView system health at any time with:" -ForegroundColor Yellow
        Write-Host "  .\test-api.ps1 -HealthCheck" -ForegroundColor White
        
        if (Test-Path "TODO.md") {
            Write-Host "`nTODO items were created. Review TODO.md for any issues." -ForegroundColor Yellow
        }
        
    } catch {
        Write-Step "Installation failed: $_" "ERROR"
        Write-Step "Check logs: docker-compose logs" "INFO"
        Add-TodoItem "Installation Error" $_
    }
}

# Run main
Main
