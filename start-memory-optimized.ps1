#!/usr/bin/env powershell
<#
.SYNOPSIS
Memory-Optimized Legal AI System Startup Script

.DESCRIPTION
Intelligent startup script that adapts to system resources and optimizes memory usage
across all components including Docker containers, VS Code extension, and caching layers.

.PARAMETER Mode
System startup mode: 'minimal', 'balanced', 'full'

.PARAMETER MonitorMode
Enable real-time memory monitoring and optimization

.PARAMETER SetupClusters
Initialize k-means clustering and SOM networks

.PARAMETER ProfileMemory
Enable detailed memory profiling
#>

param(
    [ValidateSet('minimal', 'balanced', 'full')]
    [string]$Mode = 'balanced',

    [switch]$MonitorMode,
    [switch]$SetupClusters,
    [switch]$ProfileMemory
)

# Enhanced startup configuration
$script:StartupConfig = @{
    timestamp = Get-Date
    mode = $Mode
    monitoring = $MonitorMode.IsPresent
    clustering = $SetupClusters.IsPresent
    profiling = $ProfileMemory.IsPresent
    systemMemory = 0
    availableMemory = 0
    dockerMemory = 0
    extensionMemory = 0
}

# Color coding for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [ConsoleColor]$ForegroundColor = [ConsoleColor]::White
    )

    $currentColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $Host.UI.RawUI.ForegroundColor = $currentColor
}

function Initialize-MemoryOptimizer {
    Write-ColorOutput "🧠 Initializing Memory Optimizer..." -ForegroundColor Cyan

    # Detect system memory
    $memInfo = Get-CimInstance -ClassName Win32_OperatingSystem
    $script:StartupConfig.systemMemory = [math]::Round($memInfo.TotalVisibleMemorySize / 1MB, 2)
    $script:StartupConfig.availableMemory = [math]::Round($memInfo.FreePhysicalMemory / 1MB, 2)

    Write-ColorOutput "📊 System Memory: $($script:StartupConfig.systemMemory)GB" -ForegroundColor Green
    Write-ColorOutput "💾 Available Memory: $($script:StartupConfig.availableMemory)GB" -ForegroundColor Green

    # Determine optimal configuration based on available memory
    $optimalConfig = Get-OptimalConfiguration -AvailableMemory $script:StartupConfig.availableMemory

    return $optimalConfig
}

function Get-OptimalConfiguration {
    param([double]$AvailableMemory)

    $config = @{
        dockerCompose = ""
        maxContainers = 0
        memoryLimits = @{}
        lodLevel = ""
        clusterConfig = @{}
    }

    if ($AvailableMemory -lt 4) {
        Write-ColorOutput "⚠️ Low memory detected (${AvailableMemory}GB) - Using minimal configuration" -ForegroundColor Yellow
        $config.dockerCompose = "docker-compose.lowmem.yml"
        $config.maxContainers = 4
        $config.lodLevel = "low"
        $config.memoryLimits = @{
            postgres = "256MB"
            redis = "128MB"
            qdrant = "128MB"
            ollama = "2GB"
        }
    }
    elseif ($AvailableMemory -lt 8) {
        Write-ColorOutput "📊 Medium memory detected (${AvailableMemory}GB) - Using balanced configuration" -ForegroundColor Blue
        $config.dockerCompose = "docker-compose.memory-optimized.yml"
        $config.maxContainers = 8
        $config.lodLevel = "medium"
        $config.memoryLimits = @{
            postgres = "768MB"
            redis = "512MB"
            qdrant = "384MB"
            ollama = "6GB"
        }
    }
    else {
        Write-ColorOutput "🚀 High memory detected (${AvailableMemory}GB) - Using full configuration" -ForegroundColor Green
        $config.dockerCompose = "docker-compose.memory-optimized.yml"
        $config.maxContainers = 12
        $config.lodLevel = "high"
        $config.memoryLimits = @{
            postgres = "1GB"
            redis = "1GB"
            qdrant = "768MB"
            ollama = "8GB"
        }
    }

    return $config
}

function Start-DockerServices {
    param($Config)

    Write-ColorOutput "🐳 Starting Docker services with memory optimization..." -ForegroundColor Cyan

    # Create optimized environment file
    $envContent = @"
# Memory-Optimized Legal AI Environment
COMPOSE_PROJECT_NAME=legal-ai-optimized
COMPOSE_FILE=$($Config.dockerCompose)

# Memory limits
POSTGRES_MEMORY_LIMIT=$($Config.memoryLimits.postgres)
REDIS_MEMORY_LIMIT=$($Config.memoryLimits.redis)
QDRANT_MEMORY_LIMIT=$($Config.memoryLimits.qdrant)
OLLAMA_MEMORY_LIMIT=$($Config.memoryLimits.ollama)

# LOD Configuration
LOD_LEVEL=$($Config.lodLevel)
MEMORY_OPTIMIZATION=true
CLUSTERING_ENABLED=$($script:StartupConfig.clustering)
MONITORING_ENABLED=$($script:StartupConfig.monitoring)

# Cache Configuration
CACHE_STRATEGY=aggressive
CACHE_COMPRESSION=true
CACHE_TTL_SHORT=300
CACHE_TTL_MEDIUM=3600
CACHE_TTL_LONG=86400

# Node.js optimization
NODE_OPTIONS=--max-old-space-size=1024 --optimize-for-size --gc-interval=100
"@

    $envContent | Out-File -FilePath ".env.optimized" -Encoding UTF8

    # Start core services first
    $coreServices = @("postgres", "redis")
    foreach ($service in $coreServices) {
        Write-ColorOutput "🔧 Starting $service..." -ForegroundColor Yellow
        docker-compose --env-file .env.optimized -f $Config.dockerCompose up -d $service

        # Wait for service to be healthy
        Wait-ForService -ServiceName $service -MaxWaitSeconds 60
    }

    # Start vector services
    $vectorServices = @("qdrant")
    foreach ($service in $vectorServices) {
        Write-ColorOutput "🔧 Starting $service..." -ForegroundColor Yellow
        docker-compose --env-file .env.optimized -f $Config.dockerCompose up -d $service
        Wait-ForService -ServiceName $service -MaxWaitSeconds 60
    }

    # Start LLM service if memory allows
    if ($script:StartupConfig.availableMemory -gt 4) {
        Write-ColorOutput "🔧 Starting Ollama LLM service..." -ForegroundColor Yellow
        docker-compose --env-file .env.optimized -f $Config.dockerCompose up -d ollama
        Wait-ForService -ServiceName "ollama" -MaxWaitSeconds 120
    }

    # Start application services
    $appServices = @("rag-backend", "sveltekit-frontend")
    foreach ($service in $appServices) {
        Write-ColorOutput "🔧 Starting $service..." -ForegroundColor Yellow
        docker-compose --env-file .env.optimized -f $Config.dockerCompose up -d $service
        Wait-ForService -ServiceName $service -MaxWaitSeconds 90
    }

    # Start optional monitoring services
    if ($script:StartupConfig.monitoring) {
        Write-ColorOutput "📊 Starting monitoring services..." -ForegroundColor Blue
        docker-compose --env-file .env.optimized -f $Config.dockerCompose up -d memory-optimizer prometheus grafana
    }
}

function Wait-ForService {
    param(
        [string]$ServiceName,
        [int]$MaxWaitSeconds = 60
    )

    $waited = 0
    $isHealthy = $false

    Write-ColorOutput "⏳ Waiting for $ServiceName to be healthy..." -ForegroundColor Yellow

    while (-not $isHealthy -and $waited -lt $MaxWaitSeconds) {
        try {
            $containerStatus = docker-compose ps --services --filter "status=running" | Where-Object { $_ -eq $ServiceName }
            if ($containerStatus) {
                $healthStatus = docker inspect --format='{{.State.Health.Status}}' "legal-ai-${ServiceName}-optimized" 2>$null
                if ($healthStatus -eq "healthy" -or $healthStatus -eq "") {
                    $isHealthy = $true
                    Write-ColorOutput "✅ $ServiceName is healthy" -ForegroundColor Green
                }
            }
        }
        catch {
            # Service might not have health check
        }

        if (-not $isHealthy) {
            Start-Sleep -Seconds 2
            $waited += 2
            Write-Host "." -NoNewline
        }
    }

    if (-not $isHealthy) {
        Write-ColorOutput "⚠️ $ServiceName did not become healthy within $MaxWaitSeconds seconds" -ForegroundColor Red
    }
}

function Initialize-KMeansClustering {
    Write-ColorOutput "🔄 Initializing K-means clustering system..." -ForegroundColor Cyan

    # Create clustering configuration
    $clusterConfig = @{
        enabled = $true
        k_values = @(3, 5, 8)
        max_iterations = 100
        convergence_threshold = 0.001
        distance_metric = "euclidean"
        initialization = "k-means++"
        memory_limit = "256MB"
        batch_size = 1000
    }

    $clusterConfigJson = $clusterConfig | ConvertTo-Json -Depth 3
    $clusterConfigJson | Out-File -FilePath "config/clustering-config.json" -Encoding UTF8

    Write-ColorOutput "✅ K-means clustering configuration created" -ForegroundColor Green
}

function Initialize-SOMNetwork {
    Write-ColorOutput "🧠 Initializing Self-Organizing Map network..." -ForegroundColor Cyan

    # Create SOM configuration
    $somConfig = @{
        enabled = $true
        grid_size = @{
            width = 10
            height = 10
        }
        dimensions = 384
        learning_rate = @{
            initial = 0.1
            final = 0.01
            decay = "exponential"
        }
        neighborhood = @{
            initial_radius = 3.0
            final_radius = 0.5
            decay = "linear"
        }
        training = @{
            max_epochs = 1000
            batch_size = 100
            convergence_threshold = 0.01
        }
        memory_optimization = @{
            use_sparse_representation = $true
            compress_weights = $true
            quantization_bits = 8
        }
    }

    $somConfigJson = $somConfig | ConvertTo-Json -Depth 4
    $somConfigJson | Out-File -FilePath "config/som-config.json" -Encoding UTF8

    Write-ColorOutput "✅ SOM network configuration created" -ForegroundColor Green
}

function Start-MemoryMonitoring {
    Write-ColorOutput "📊 Starting real-time memory monitoring..." -ForegroundColor Cyan

    $monitoringScript = @"
# Memory Monitoring Loop
while (`$true) {
    `$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    # System memory
    `$memInfo = Get-CimInstance -ClassName Win32_OperatingSystem
    `$totalMem = [math]::Round(`$memInfo.TotalVisibleMemorySize / 1MB, 2)
    `$freeMem = [math]::Round(`$memInfo.FreePhysicalMemory / 1MB, 2)
    `$usedMem = `$totalMem - `$freeMem
    `$memPercent = [math]::Round((`$usedMem / `$totalMem) * 100, 1)

    # Docker memory
    `$dockerStats = docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" 2>`$null

    # Log memory usage
    `$logEntry = "`$timestamp,`$usedMem,`$freeMem,`$memPercent"
    `$logEntry | Out-File -FilePath "logs/memory-usage.csv" -Append

    # Check for memory pressure
    if (`$memPercent -gt 85) {
        Write-Host "⚠️ High memory usage: `$memPercent%" -ForegroundColor Red
        # Trigger optimization
        docker exec legal-ai-memory-optimizer /app/optimize-memory.sh
    }

    Start-Sleep -Seconds 30
}
"@

    # Create monitoring directories
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null

    # Start monitoring in background
    $monitoringScript | Out-File -FilePath "scripts/memory-monitor.ps1" -Encoding UTF8

    Start-Process -FilePath "powershell.exe" -ArgumentList "-File scripts/memory-monitor.ps1" -WindowStyle Hidden

    Write-ColorOutput "✅ Memory monitoring started" -ForegroundColor Green
}

function Initialize-VSCodeExtension {
    Write-ColorOutput "🔧 Initializing VS Code MCP Extension..." -ForegroundColor Cyan

    # Check if VS Code is installed
    $vscodePath = Get-Command code -ErrorAction SilentlyContinue
    if (-not $vscodePath) {
        Write-ColorOutput "⚠️ VS Code not found in PATH" -ForegroundColor Yellow
        return
    }

    # Install or update the extension
    if (Test-Path "vscode-llm-extension") {
        Write-ColorOutput "📦 Building VS Code extension..." -ForegroundColor Blue

        Push-Location "vscode-llm-extension"
        try {
            # Install dependencies
            npm install

            # Build extension with memory optimization
            $env:NODE_OPTIONS = "--max-old-space-size=512 --optimize-for-size"
            npm run compile

            # Package extension
            if (Get-Command vsce -ErrorAction SilentlyContinue) {
                vsce package --out ../legal-ai-mcp-extension.vsix
                Write-ColorOutput "✅ Extension packaged successfully" -ForegroundColor Green
            }
        }
        finally {
            Pop-Location
        }
    }
}

function Show-StartupSummary {
    param($Config)

    Write-ColorOutput "`n🎉 Memory-Optimized Legal AI System Started Successfully!" -ForegroundColor Green
    Write-ColorOutput "=" * 60 -ForegroundColor Gray

    Write-ColorOutput "📊 System Configuration:" -ForegroundColor Cyan
    Write-ColorOutput "  Mode: $($script:StartupConfig.mode)" -ForegroundColor White
    Write-ColorOutput "  LOD Level: $($Config.lodLevel)" -ForegroundColor White
    Write-ColorOutput "  System Memory: $($script:StartupConfig.systemMemory)GB" -ForegroundColor White
    Write-ColorOutput "  Available Memory: $($script:StartupConfig.availableMemory)GB" -ForegroundColor White
    Write-ColorOutput "  Docker Compose: $($Config.dockerCompose)" -ForegroundColor White

    Write-ColorOutput "`n🔗 Service URLs:" -ForegroundColor Cyan
    Write-ColorOutput "  Frontend: http://localhost:5174" -ForegroundColor White
    Write-ColorOutput "  RAG Backend: http://localhost:3001" -ForegroundColor White
    Write-ColorOutput "  PostgreSQL: localhost:5433" -ForegroundColor White
    Write-ColorOutput "  Redis: localhost:6380" -ForegroundColor White
    Write-ColorOutput "  Qdrant: http://localhost:6334" -ForegroundColor White

    if ($script:StartupConfig.availableMemory -gt 4) {
        Write-ColorOutput "  Ollama LLM: http://localhost:11435" -ForegroundColor White
    }

    if ($script:StartupConfig.monitoring) {
        Write-ColorOutput "  Prometheus: http://localhost:9091" -ForegroundColor White
        Write-ColorOutput "  Grafana: http://localhost:3001" -ForegroundColor White
    }

    Write-ColorOutput "`n🧠 Memory Optimization Features:" -ForegroundColor Cyan
    Write-ColorOutput "  ✅ Adaptive LOD (Level of Detail)" -ForegroundColor Green
    Write-ColorOutput "  ✅ Multi-layer caching (Loki.js + Redis + Qdrant)" -ForegroundColor Green
    Write-ColorOutput "  ✅ Docker resource limits" -ForegroundColor Green

    if ($script:StartupConfig.clustering) {
        Write-ColorOutput "  ✅ K-means clustering enabled" -ForegroundColor Green
        Write-ColorOutput "  ✅ Self-Organizing Maps enabled" -ForegroundColor Green
    }

    if ($script:StartupConfig.monitoring) {
        Write-ColorOutput "  ✅ Real-time memory monitoring" -ForegroundColor Green
    }

    Write-ColorOutput "`n🚀 Quick Commands:" -ForegroundColor Cyan
    Write-ColorOutput "  View logs: docker-compose logs -f" -ForegroundColor White
    Write-ColorOutput "  Monitor memory: Get-Content logs/memory-usage.csv -Tail 10" -ForegroundColor White
    Write-ColorOutput "  Optimize memory: docker exec legal-ai-memory-optimizer /app/optimize-memory.sh" -ForegroundColor White
    Write-ColorOutput "  Stop system: docker-compose down" -ForegroundColor White

    Write-ColorOutput "=" * 60 -ForegroundColor Gray
}

function Test-SystemRequirements {
    Write-ColorOutput "🔍 Testing system requirements..." -ForegroundColor Cyan

    $requirements = @{
        docker = $false
        dockerCompose = $false
        node = $false
        npm = $false
        memory = $false
    }

    # Test Docker
    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            $requirements.docker = $true
            Write-ColorOutput "✅ Docker: $dockerVersion" -ForegroundColor Green
        }
    }
    catch {
        Write-ColorOutput "❌ Docker not found" -ForegroundColor Red
    }

    # Test Docker Compose
    try {
        $composeVersion = docker-compose --version 2>$null
        if ($composeVersion) {
            $requirements.dockerCompose = $true
            Write-ColorOutput "✅ Docker Compose: $composeVersion" -ForegroundColor Green
        }
    }
    catch {
        Write-ColorOutput "❌ Docker Compose not found" -ForegroundColor Red
    }

    # Test Node.js
    try {
        $nodeVersion = node --version 2>$null
        if ($nodeVersion) {
            $requirements.node = $true
            Write-ColorOutput "✅ Node.js: $nodeVersion" -ForegroundColor Green
        }
    }
    catch {
        Write-ColorOutput "❌ Node.js not found" -ForegroundColor Red
    }

    # Test npm
    try {
        $npmVersion = npm --version 2>$null
        if ($npmVersion) {
            $requirements.npm = $true
            Write-ColorOutput "✅ npm: v$npmVersion" -ForegroundColor Green
        }
    }
    catch {
        Write-ColorOutput "❌ npm not found" -ForegroundColor Red
    }

    # Test memory
    if ($script:StartupConfig.availableMemory -gt 2) {
        $requirements.memory = $true
        Write-ColorOutput "✅ Memory: $($script:StartupConfig.availableMemory)GB available" -ForegroundColor Green
    }
    else {
        Write-ColorOutput "⚠️ Memory: Only $($script:StartupConfig.availableMemory)GB available (minimum 2GB recommended)" -ForegroundColor Yellow
    }

    $allMet = $requirements.Values -contains $false
    if (-not $allMet) {
        Write-ColorOutput "✅ All requirements met" -ForegroundColor Green
        return $true
    }
    else {
        Write-ColorOutput "❌ Some requirements not met" -ForegroundColor Red
        return $false
    }
}

# Main execution
try {
    Write-ColorOutput "🚀 Starting Memory-Optimized Legal AI System..." -ForegroundColor Cyan
    Write-ColorOutput "Mode: $Mode | Monitoring: $($MonitorMode.IsPresent) | Clustering: $($SetupClusters.IsPresent)" -ForegroundColor Blue

    # Initialize memory optimizer
    $config = Initialize-MemoryOptimizer

    # Test system requirements
    if (-not (Test-SystemRequirements)) {
        Write-ColorOutput "❌ System requirements not met. Please install missing components." -ForegroundColor Red
        exit 1
    }

    # Create necessary directories
    $directories = @("config", "data", "logs", "cache", "scripts")
    foreach ($dir in $directories) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    # Initialize clustering if requested
    if ($SetupClusters) {
        Initialize-KMeansClustering
        Initialize-SOMNetwork
    }

    # Start Docker services
    Start-DockerServices -Config $config

    # Initialize VS Code extension
    Initialize-VSCodeExtension

    # Start memory monitoring if requested
    if ($MonitorMode) {
        Start-MemoryMonitoring
    }

    # Show startup summary
    Show-StartupSummary -Config $config

    Write-ColorOutput "`n✅ System is ready! 🎉" -ForegroundColor Green
}
catch {
    Write-ColorOutput "❌ Error during startup: $($_.Exception.Message)" -ForegroundColor Red
    Write-ColorOutput "Stack trace: $($_.ScriptStackTrace)" -ForegroundColor Red
    exit 1
}
