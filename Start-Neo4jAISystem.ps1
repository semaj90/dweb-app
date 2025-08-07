# PowerShell Script for Neo4j AI System
# Phase 14: Advanced Features & Optimization
# Requires PowerShell 7+ and Administrator privileges

#Requires -RunAsAdministrator
#Requires -Version 7.0

param(
    [switch]$SkipBuild,
    [switch]$CPUOnly,
    [switch]$Verbose,
    [switch]$Debug,
    [switch]$Production
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Configuration
$Config = @{
    ProjectRoot = $PSScriptRoot
    GoMicroservicePath = Join-Path $PSScriptRoot "go-microservice"
    DockerComposePath = Join-Path $PSScriptRoot "docker-compose.neo4j.yml"
    LogPath = Join-Path $PSScriptRoot "logs"
    Services = @{
        Neo4j = @{
            Name = "neo4j-deeds"
            HealthCheck = { docker exec neo4j-deeds cypher-shell -u neo4j -p password "RETURN 1" 2>$null }
            URL = "http://localhost:7474"
            Port = 7687
        }
        Redis = @{
            Name = "redis-deeds"
            HealthCheck = { docker exec redis-deeds redis-cli ping 2>$null }
            Port = 6379
        }
        Ollama = @{
            Name = "ollama"
            HealthCheck = { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -ErrorAction SilentlyContinue }
            URL = "http://localhost:11434"
            Port = 11434
        }
        AIMicroservice = @{
            Name = "ai-microservice"
            HealthCheck = { Invoke-RestMethod -Uri "http://localhost:8081/health" -ErrorAction SilentlyContinue }
            URL = "http://localhost:8081"
            Port = 8081
        }
    }
}

# Logging functions
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO",
        [ConsoleColor]$Color = "White"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    
    # Console output
    Write-Host $logMessage -ForegroundColor $Color
    
    # File logging
    if (-not (Test-Path $Config.LogPath)) {
        New-Item -ItemType Directory -Path $Config.LogPath -Force | Out-Null
    }
    $logFile = Join-Path $Config.LogPath "system-$(Get-Date -Format 'yyyy-MM-dd').log"
    Add-Content -Path $logFile -Value $logMessage
}

function Write-Success { Write-Log -Message $args[0] -Level "SUCCESS" -Color Green }
function Write-Warning { Write-Log -Message $args[0] -Level "WARNING" -Color Yellow }
function Write-Error { Write-Log -Message $args[0] -Level "ERROR" -Color Red }
function Write-Info { Write-Log -Message $args[0] -Level "INFO" -Color Cyan }

# System check functions
function Test-DockerInstalled {
    try {
        $dockerVersion = docker --version
        Write-Success "Docker installed: $dockerVersion"
        return $true
    } catch {
        Write-Error "Docker is not installed or not in PATH"
        return $false
    }
}

function Test-DockerRunning {
    try {
        docker info | Out-Null
        Write-Success "Docker is running"
        return $true
    } catch {
        Write-Warning "Docker is not running, attempting to start..."
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        Start-Sleep -Seconds 30
        try {
            docker info | Out-Null
            Write-Success "Docker started successfully"
            return $true
        } catch {
            Write-Error "Failed to start Docker"
            return $false
        }
    }
}

function Test-GPUAvailable {
    try {
        $nvidiaInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
        if ($nvidiaInfo) {
            Write-Success "NVIDIA GPU detected: $nvidiaInfo"
            return $true
        }
    } catch {}
    
    Write-Warning "No NVIDIA GPU detected, will use CPU mode"
    return $false
}

function Test-ServiceHealth {
    param([string]$ServiceName)
    
    $service = $Config.Services[$ServiceName]
    if (-not $service) {
        Write-Error "Unknown service: $ServiceName"
        return $false
    }
    
    try {
        $result = & $service.HealthCheck
        if ($result) {
            Write-Success "$ServiceName is healthy"
            return $true
        }
    } catch {
        Write-Warning "$ServiceName health check failed: $_"
    }
    
    return $false
}

# Service management functions
function Start-Infrastructure {
    Write-Info "Starting infrastructure services..."
    
    $services = @("neo4j", "redis", "postgres", "qdrant")
    $cmd = "docker-compose -f `"$($Config.DockerComposePath)`" up -d $($services -join ' ')"
    
    Write-Info "Executing: $cmd"
    Invoke-Expression $cmd
    
    # Wait for services to be healthy
    foreach ($serviceName in @("Neo4j", "Redis")) {
        $maxRetries = 30
        $retryCount = 0
        
        Write-Info "Waiting for $serviceName to be ready..."
        while ($retryCount -lt $maxRetries) {
            if (Test-ServiceHealth -ServiceName $serviceName) {
                break
            }
            Start-Sleep -Seconds 2
            $retryCount++
        }
        
        if ($retryCount -eq $maxRetries) {
            Write-Error "$serviceName failed to start after $maxRetries attempts"
            throw "Service startup failed"
        }
    }
}

function Build-Microservice {
    param([bool]$UseGPU = $true)
    
    if ($SkipBuild) {
        Write-Info "Skipping build (--SkipBuild flag set)"
        return
    }
    
    Push-Location $Config.GoMicroservicePath
    try {
        $binaryPath = Join-Path "build" "ai-microservice.exe"
        
        if (Test-Path $binaryPath) {
            Write-Info "Binary already exists, skipping build"
            return
        }
        
        Write-Info "Building Go microservice..."
        
        if ($UseGPU -and -not $CPUOnly) {
            Write-Info "Building with GPU support..."
            make build
        } else {
            Write-Info "Building CPU-only version..."
            make build-cpu
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed with exit code $LASTEXITCODE"
        }
        
        Write-Success "Build completed successfully"
    } finally {
        Pop-Location
    }
}

function Start-Microservice {
    Write-Info "Starting AI Microservice..."
    
    $env:PORT = "8081"
    $env:REDIS_HOST = "localhost:6379"
    $env:NEO4J_URI = "neo4j://localhost:7687"
    $env:NEO4J_USER = "neo4j"
    $env:NEO4J_PASSWORD = "password"
    $env:OLLAMA_HOST = "http://localhost:11434"
    
    if ($Production) {
        $env:GIN_MODE = "release"
    } else {
        $env:GIN_MODE = "debug"
    }
    
    $exePath = Join-Path $Config.GoMicroservicePath "build" "ai-microservice.exe"
    
    if (-not (Test-Path $exePath)) {
        Write-Error "Microservice binary not found at $exePath"
        throw "Binary not found"
    }
    
    $process = Start-Process -FilePath $exePath -PassThru -WindowStyle Hidden
    
    # Wait for service to be ready
    $maxRetries = 30
    $retryCount = 0
    
    while ($retryCount -lt $maxRetries) {
        if (Test-ServiceHealth -ServiceName "AIMicroservice") {
            Write-Success "AI Microservice started (PID: $($process.Id))"
            return $process
        }
        Start-Sleep -Seconds 1
        $retryCount++
    }
    
    Write-Error "AI Microservice failed to start"
    throw "Microservice startup failed"
}

function Start-Ollama {
    Write-Info "Checking Ollama service..."
    
    if (Test-ServiceHealth -ServiceName "Ollama") {
        Write-Success "Ollama is already running"
        return
    }
    
    Write-Info "Starting Ollama..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    
    # Pull required models
    $models = @("nomic-embed-text", "llama3")
    foreach ($model in $models) {
        Write-Info "Pulling model: $model"
        ollama pull $model 2>$null
    }
    
    Write-Success "Ollama started and models loaded"
}

function Initialize-Neo4jSchema {
    Write-Info "Initializing Neo4j schema..."
    
    $cypherScript = @'
// Create constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE;

// Create indexes
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.created_at);
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.updated_at);
CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.document_id);
CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name);

// Create full-text indexes
CREATE FULLTEXT INDEX documentSearch IF NOT EXISTS 
FOR (d:Document) ON EACH [d.title, d.content, d.summary];

CREATE FULLTEXT INDEX chunkSearch IF NOT EXISTS 
FOR (c:Chunk) ON EACH [c.text];

CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS 
FOR (e:Entity) ON EACH [e.name, e.description];

// Create vector indexes for similarity search
CREATE VECTOR INDEX documentEmbeddings IF NOT EXISTS
FOR (d:Document) ON (d.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX chunkEmbeddings IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
}};

// Create initial system node
MERGE (system:System {id: 'deeds-ai-system'})
SET system.version = '2.0.0',
    system.initialized = datetime(),
    system.last_updated = datetime();

// Return confirmation
RETURN 'Schema initialized successfully' as status;
'@
    
    $cypherScript | docker exec -i neo4j-deeds cypher-shell -u neo4j -p password
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Neo4j schema initialized successfully"
    } else {
        Write-Warning "Neo4j schema initialization had issues"
    }
}

function Show-SystemStatus {
    Write-Host "`n" -NoNewline
    Write-Host "="*60 -ForegroundColor Cyan
    Write-Host "SYSTEM STATUS DASHBOARD" -ForegroundColor Cyan
    Write-Host "="*60 -ForegroundColor Cyan
    
    # Service status
    Write-Host "`nServices:" -ForegroundColor Yellow
    foreach ($serviceName in $Config.Services.Keys) {
        $service = $Config.Services[$serviceName]
        $status = if (Test-ServiceHealth -ServiceName $serviceName) { "✓ Running" } else { "✗ Stopped" }
        $color = if ($status -eq "✓ Running") { "Green" } else { "Red" }
        Write-Host "  $serviceName : $status" -ForegroundColor $color
        if ($service.URL) {
            Write-Host "    URL: $($service.URL)" -ForegroundColor Gray
        }
    }
    
    # GPU status
    Write-Host "`nGPU Status:" -ForegroundColor Yellow
    if (Test-GPUAvailable) {
        $gpuInfo = nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader
        Write-Host "  $gpuInfo" -ForegroundColor Green
    } else {
        Write-Host "  CPU Mode - No GPU acceleration" -ForegroundColor Yellow
    }
    
    # Memory status
    Write-Host "`nSystem Resources:" -ForegroundColor Yellow
    $mem = Get-CimInstance Win32_OperatingSystem
    $memUsed = [math]::Round(($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / 1MB, 2)
    $memTotal = [math]::Round($mem.TotalVisibleMemorySize / 1MB, 2)
    Write-Host "  Memory: $memUsed GB / $memTotal GB" -ForegroundColor Gray
    
    $cpu = Get-CimInstance Win32_Processor
    Write-Host "  CPU: $($cpu.Name)" -ForegroundColor Gray
    Write-Host "  Cores: $($cpu.NumberOfCores) physical, $($cpu.NumberOfLogicalProcessors) logical" -ForegroundColor Gray
    
    Write-Host "`n" -NoNewline
    Write-Host "="*60 -ForegroundColor Cyan
}

function Start-Monitoring {
    Write-Info "Starting monitoring loop..."
    
    while ($true) {
        Start-Sleep -Seconds 30
        
        # Check critical services
        foreach ($serviceName in @("AIMicroservice", "Neo4j", "Redis")) {
            if (-not (Test-ServiceHealth -ServiceName $serviceName)) {
                Write-Warning "$serviceName is not responding, attempting recovery..."
                
                switch ($serviceName) {
                    "AIMicroservice" {
                        Start-Microservice
                    }
                    default {
                        Start-Infrastructure
                    }
                }
            }
        }
        
        # Update metrics
        if ($Verbose) {
            $metrics = Invoke-RestMethod -Uri "http://localhost:8081/metrics" -ErrorAction SilentlyContinue
            if ($metrics) {
                Write-Info "Metrics: Uptime=$($metrics.uptime), Goroutines=$($metrics.goroutines), Memory=$($metrics.memory_alloc)"
            }
        }
    }
}

# Main execution
try {
    Write-Host "`n"
    Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║     DEEDS WEB APP - NEO4J AI SYSTEM LAUNCHER            ║" -ForegroundColor Cyan
    Write-Host "║     Phase 14: Advanced Features & Optimization          ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host "`n"
    
    # System checks
    Write-Info "Performing system checks..."
    if (-not (Test-DockerInstalled)) { throw "Docker is required" }
    if (-not (Test-DockerRunning)) { throw "Docker must be running" }
    
    $useGPU = Test-GPUAvailable
    
    # Start services
    Start-Infrastructure
    Build-Microservice -UseGPU $useGPU
    Start-Ollama
    $microserviceProcess = Start-Microservice
    Initialize-Neo4jSchema
    
    # Show status
    Show-SystemStatus
    
    Write-Success "`nSystem started successfully!"
    Write-Info "Press Ctrl+C to stop all services"
    
    # Start monitoring
    if (-not $Debug) {
        Start-Monitoring
    }
    
} catch {
    Write-Error "System startup failed: $_"
    Write-Info "Check the logs at: $($Config.LogPath)"
    exit 1
} finally {
    # Cleanup on exit
    Write-Info "Shutting down services..."
    
    # Stop microservice
    if ($microserviceProcess) {
        Stop-Process -Id $microserviceProcess.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Stop Docker services
    docker-compose -f $Config.DockerComposePath down
    
    Write-Success "Shutdown complete"
}
