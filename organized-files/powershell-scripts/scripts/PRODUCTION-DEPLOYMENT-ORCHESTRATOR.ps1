# 🚀 Legal AI Production Deployment Orchestrator
# Complete Windows-native deployment with monitoring and service management
param(
    [ValidateSet("Start", "Stop", "Restart", "Status", "Health", "Install", "Uninstall")]
    [string]$Action = "Start",
    
    [ValidateSet("All", "Core", "AI", "Database", "Storage", "GPU")]
    [string]$ServiceGroup = "All",
    
    [switch]$Verbose,
    [switch]$Force,
    [switch]$DryRun,
    [switch]$MonitorMode
)

$ErrorActionPreference = "Continue"
$script:DeploymentStartTime = Get-Date

# 📋 Service Configuration
$Services = @{
    Core = @{
        SvelteKit = @{
            Name = "Legal-AI-Frontend"
            Directory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
            Command = "npm run dev"
            Port = 5173
            HealthEndpoint = "http://localhost:5173"
            Dependencies = @("PostgreSQL", "Redis", "MinIO")
            Critical = $true
        }
        EmbeddingService = @{
            Name = "Legal-AI-Embedding"
            Directory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\embedding-service"
            Command = "npm start"
            Port = 8092
            GrpcPort = 50051
            HealthEndpoint = "http://localhost:8092/health"
            Dependencies = @("Redis")
            Critical = $true
        }
        IngestWorker = @{
            Name = "Legal-AI-Ingest"
            Directory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\ingest-worker"
            Command = "npm start"
            BackgroundService = $true
            HealthEndpoint = "http://localhost:8093/health"
            Dependencies = @("PostgreSQL", "MinIO", "Neo4j", "RabbitMQ", "EmbeddingService")
            Critical = $true
        }
    }
    Database = @{
        PostgreSQL = @{
            Name = "PostgreSQL-Legal-AI"
            ServiceName = "postgresql-x64-17"
            Port = 5432
            HealthCommand = "pg_isready -h localhost -p 5432"
            DataDirectory = "C:\Program Files\PostgreSQL\17\data"
            Critical = $true
        }
        Redis = @{
            Name = "Redis-Cache"
            Command = "redis-server"
            Port = 6379
            HealthCommand = "redis-cli ping"
            ConfigFile = "C:\Program Files\Redis\redis.windows.conf"
            Critical = $true
        }
        Neo4j = @{
            Name = "Neo4j-Knowledge-Graph"
            ServiceName = "Neo4j"
            Port = 7687
            WebPort = 7474
            HealthEndpoint = "http://localhost:7474/db/data/"
            Critical = $false
        }
    }
    Storage = @{
        MinIO = @{
            Name = "MinIO-Object-Storage"
            Command = "C:\Users\james\Desktop\deeds-web\deeds-web-app\temp-services\minio.exe server C:\Users\james\Desktop\deeds-web\deeds-web-app\minio-data --console-address :9001"
            Port = 9000
            ConsolePort = 9001
            HealthEndpoint = "http://localhost:9000/minio/health/live"
            DataDirectory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\minio-data"
            Critical = $true
        }
        RabbitMQ = @{
            Name = "RabbitMQ-Message-Queue"
            ServiceName = "RabbitMQ"
            Port = 5672
            ManagementPort = 15672
            HealthEndpoint = "http://localhost:15672/api/overview"
            Critical = $false
        }
    }
    AI = @{
        Ollama = @{
            Name = "Ollama-LLM-Service"
            Command = "ollama serve"
            Port = 11434
            HealthEndpoint = "http://localhost:11434/api/version"
            Models = @("gemma3-legal", "nomic-embed-text", "llama3.2")
            Critical = $true
        }
        Qdrant = @{
            Name = "Qdrant-Vector-DB"
            Command = "qdrant"
            Port = 6333
            HealthEndpoint = "http://localhost:6333/health"
            DataDirectory = "C:\qdrant-data"
            Critical = $false
        }
    }
    GPU = @{
        GPUMonitor = @{
            Name = "GPU-Performance-Monitor"
            Command = "nvidia-ml-py3"
            BackgroundService = $true
            HealthCommand = "nvidia-smi"
            Critical = $false
        }
    }
}

# 🎨 UI Functions
function Write-Header {
    param([string]$Title, [string]$Color = "Green")
    Write-Host "`n╭─" -NoNewline -ForegroundColor $Color
    Write-Host ("─" * ($Title.Length + 2)) -NoNewline -ForegroundColor $Color
    Write-Host "─╮" -ForegroundColor $Color
    Write-Host "│ $Title │" -ForegroundColor $Color
    Write-Host "╰─" -NoNewline -ForegroundColor $Color
    Write-Host ("─" * ($Title.Length + 2)) -NoNewline -ForegroundColor $Color
    Write-Host "─╯" -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Cyan
}

function Write-Progress {
    param([string]$Activity, [string]$Status, [int]$PercentComplete)
    Write-Progress -Activity $Activity -Status $Status -PercentComplete $PercentComplete
}

# 🔍 Health Check Functions
function Test-ServiceHealth {
    param(
        [hashtable]$ServiceConfig,
        [int]$TimeoutSeconds = 30
    )
    
    $healthChecks = @()
    
    # Port connectivity check
    if ($ServiceConfig.Port) {
        $portCheck = Test-NetConnection -ComputerName "localhost" -Port $ServiceConfig.Port -WarningAction SilentlyContinue
        $healthChecks += @{
            Type = "Port"
            Port = $ServiceConfig.Port
            Result = $portCheck.TcpTestSucceeded
        }
    }
    
    # HTTP health endpoint check
    if ($ServiceConfig.HealthEndpoint) {
        try {
            $response = Invoke-WebRequest -Uri $ServiceConfig.HealthEndpoint -TimeoutSec $TimeoutSeconds -ErrorAction Stop
            $healthChecks += @{
                Type = "HTTP"
                Endpoint = $ServiceConfig.HealthEndpoint
                Result = $response.StatusCode -eq 200
                StatusCode = $response.StatusCode
            }
        } catch {
            $healthChecks += @{
                Type = "HTTP"
                Endpoint = $ServiceConfig.HealthEndpoint
                Result = $false
                Error = $_.Exception.Message
            }
        }
    }
    
    # Custom health command check
    if ($ServiceConfig.HealthCommand) {
        try {
            $result = Invoke-Expression $ServiceConfig.HealthCommand -ErrorAction Stop
            $healthChecks += @{
                Type = "Command"
                Command = $ServiceConfig.HealthCommand
                Result = $LASTEXITCODE -eq 0
                Output = $result
            }
        } catch {
            $healthChecks += @{
                Type = "Command"
                Command = $ServiceConfig.HealthCommand
                Result = $false
                Error = $_.Exception.Message
            }
        }
    }
    
    return $healthChecks
}

function Test-Prerequisites {
    Write-Header "Checking Prerequisites" "Yellow"
    
    $prerequisites = @{
        "Node.js" = { node --version }
        "npm" = { npm --version }
        "Git" = { git --version }
        "PostgreSQL" = { psql --version }
        "Redis CLI" = { redis-cli --version }
        "Ollama" = { ollama --version }
    }
    
    $allGood = $true
    
    foreach ($tool in $prerequisites.Keys) {
        try {
            $version = & $prerequisites[$tool] 2>$null
            Write-Success "$tool: $($version[0])"
        } catch {
            Write-Error "$tool: Not found or not in PATH"
            $allGood = $false
        }
    }
    
    # GPU Check
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            Write-Success "GPU: $gpuInfo"
        } else {
            Write-Warning "GPU: nvidia-smi not found or no NVIDIA GPU detected"
        }
    } catch {
        Write-Warning "GPU: Unable to check GPU status"
    }
    
    return $allGood
}

# 🚀 Service Management Functions
function Start-ServiceGroup {
    param([string]$GroupName)
    
    Write-Header "Starting Service Group: $GroupName" "Green"
    
    if ($GroupName -eq "All") {
        $servicesToStart = @("Database", "Storage", "AI", "Core")
    } else {
        $servicesToStart = @($GroupName)
    }
    
    foreach ($group in $servicesToStart) {
        if (-not $Services.ContainsKey($group)) {
            Write-Error "Unknown service group: $group"
            continue
        }
        
        Write-Info "Starting $group services..."
        
        foreach ($serviceName in $Services[$group].Keys) {
            $service = $Services[$group][$serviceName]
            
            if ($DryRun) {
                Write-Info "[DRY RUN] Would start: $($service.Name)"
                continue
            }
            
            Write-Progress -Activity "Starting Services" -Status "Starting $($service.Name)..." -PercentComplete ((($servicesToStart.IndexOf($group) + 1) / $servicesToStart.Count) * 100)
            
            try {
                Start-IndividualService -ServiceConfig $service -ServiceName $serviceName
                Start-Sleep -Seconds 2  # Allow service to initialize
            } catch {
                Write-Error "Failed to start $($service.Name): $_"
                if ($service.Critical) {
                    Write-Error "Critical service failed. Stopping deployment."
                    return $false
                }
            }
        }
    }
    
    Write-Progress -Activity "Starting Services" -Completed
    return $true
}

function Start-IndividualService {
    param(
        [hashtable]$ServiceConfig,
        [string]$ServiceName
    )
    
    # Check if already running
    $healthChecks = Test-ServiceHealth -ServiceConfig $ServiceConfig -TimeoutSeconds 5
    $isHealthy = $healthChecks | Where-Object { $_.Result -eq $true }
    
    if ($isHealthy -and -not $Force) {
        Write-Success "$($ServiceConfig.Name) is already running"
        return
    }
    
    # Windows Service
    if ($ServiceConfig.ServiceName) {
        $windowsService = Get-Service -Name $ServiceConfig.ServiceName -ErrorAction SilentlyContinue
        if ($windowsService) {
            if ($windowsService.Status -ne "Running") {
                Start-Service -Name $ServiceConfig.ServiceName
                Write-Success "Started Windows service: $($ServiceConfig.ServiceName)"
            } else {
                Write-Success "Windows service already running: $($ServiceConfig.ServiceName)"
            }
            return
        }
    }
    
    # Command-based service
    if ($ServiceConfig.Command) {
        if ($ServiceConfig.Directory) {
            $originalLocation = Get-Location
            Set-Location $ServiceConfig.Directory
        }
        
        try {
            if ($ServiceConfig.BackgroundService) {
                # Start as background job
                $job = Start-Job -ScriptBlock {
                    param($Command, $Directory)
                    if ($Directory) { Set-Location $Directory }
                    Invoke-Expression $Command
                } -ArgumentList $ServiceConfig.Command, $ServiceConfig.Directory
                
                Write-Success "Started background service: $($ServiceConfig.Name) (Job ID: $($job.Id))"
            } else {
                # Start as regular process (will need external management)
                Write-Info "Service command: $($ServiceConfig.Command)"
                Write-Warning "Manual start required for: $($ServiceConfig.Name)"
                Write-Info "Run in separate terminal: cd '$($ServiceConfig.Directory)' && $($ServiceConfig.Command)"
            }
        } finally {
            if ($ServiceConfig.Directory) {
                Set-Location $originalLocation
            }
        }
    }
}

function Stop-ServiceGroup {
    param([string]$GroupName)
    
    Write-Header "Stopping Service Group: $GroupName" "Red"
    
    if ($GroupName -eq "All") {
        $servicesToStop = @("Core", "AI", "Storage", "Database")  # Reverse order
    } else {
        $servicesToStop = @($GroupName)
    }
    
    foreach ($group in $servicesToStop) {
        if (-not $Services.ContainsKey($group)) {
            Write-Error "Unknown service group: $group"
            continue
        }
        
        Write-Info "Stopping $group services..."
        
        foreach ($serviceName in $Services[$group].Keys) {
            $service = $Services[$group][$serviceName]
            
            if ($DryRun) {
                Write-Info "[DRY RUN] Would stop: $($service.Name)"
                continue
            }
            
            try {
                Stop-IndividualService -ServiceConfig $service -ServiceName $serviceName
            } catch {
                Write-Error "Failed to stop $($service.Name): $_"
            }
        }
    }
}

function Stop-IndividualService {
    param(
        [hashtable]$ServiceConfig,
        [string]$ServiceName
    )
    
    # Windows Service
    if ($ServiceConfig.ServiceName) {
        $windowsService = Get-Service -Name $ServiceConfig.ServiceName -ErrorAction SilentlyContinue
        if ($windowsService -and $windowsService.Status -eq "Running") {
            Stop-Service -Name $ServiceConfig.ServiceName -Force
            Write-Success "Stopped Windows service: $($ServiceConfig.ServiceName)"
            return
        }
    }
    
    # Kill processes by port
    if ($ServiceConfig.Port) {
        $processes = Get-NetTCPConnection -LocalPort $ServiceConfig.Port -ErrorAction SilentlyContinue | 
                    ForEach-Object { Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue }
        
        foreach ($process in $processes) {
            if ($process) {
                Stop-Process -Id $process.Id -Force
                Write-Success "Stopped process: $($process.ProcessName) (PID: $($process.Id))"
            }
        }
    }
    
    # Stop background jobs
    $jobs = Get-Job | Where-Object { $_.Name -like "*$($ServiceConfig.Name)*" }
    foreach ($job in $jobs) {
        Stop-Job -Job $job -PassThru | Remove-Job
        Write-Success "Stopped background job: $($job.Name)"
    }
}

function Get-ServicesStatus {
    Write-Header "Services Status Report" "Cyan"
    
    $overallHealth = $true
    $statusReport = @()
    
    foreach ($groupName in $Services.Keys) {
        Write-Host "`n🔍 $groupName Services:" -ForegroundColor Cyan
        
        foreach ($serviceName in $Services[$groupName].Keys) {
            $service = $Services[$groupName][$serviceName]
            
            Write-Progress -Activity "Checking Services" -Status "Checking $($service.Name)..." -PercentComplete (($statusReport.Count + 1) / ($Services.Keys.Count * 3) * 100)
            
            $healthChecks = Test-ServiceHealth -ServiceConfig $service -TimeoutSeconds 10
            $isHealthy = ($healthChecks | Where-Object { $_.Result -eq $true }).Count -gt 0
            
            if ($isHealthy) {
                Write-Success "$($service.Name)"
                if ($service.Port) {
                    Write-Host "   └─ Port: $($service.Port) ✅" -ForegroundColor Gray
                }
                if ($service.HealthEndpoint) {
                    Write-Host "   └─ Health: $($service.HealthEndpoint) ✅" -ForegroundColor Gray
                }
            } else {
                Write-Error "$($service.Name)"
                if ($service.Critical) {
                    $overallHealth = $false
                }
                
                foreach ($check in $healthChecks) {
                    if (-not $check.Result) {
                        Write-Host "   └─ $($check.Type): ❌" -ForegroundColor Red
                        if ($check.Error) {
                            Write-Host "      Error: $($check.Error)" -ForegroundColor DarkRed
                        }
                    }
                }
            }
            
            $statusReport += @{
                Group = $groupName
                Service = $serviceName
                Name = $service.Name
                Healthy = $isHealthy
                Critical = $service.Critical
                Port = $service.Port
                HealthChecks = $healthChecks
            }
        }
    }
    
    Write-Progress -Activity "Checking Services" -Completed
    
    # Summary
    Write-Header "System Health Summary" "Magenta"
    
    $totalServices = $statusReport.Count
    $healthyServices = ($statusReport | Where-Object { $_.Healthy }).Count
    $criticalServices = ($statusReport | Where-Object { $_.Critical }).Count
    $healthyCritical = ($statusReport | Where-Object { $_.Critical -and $_.Healthy }).Count
    
    Write-Host "📊 Total Services: $totalServices" -ForegroundColor White
    Write-Host "✅ Healthy: $healthyServices" -ForegroundColor Green
    Write-Host "⚠️  Critical Services: $criticalServices" -ForegroundColor Yellow
    Write-Host "🎯 Critical Healthy: $healthyCritical" -ForegroundColor Green
    
    if ($overallHealth) {
        Write-Success "🎉 System is operational!"
    } else {
        Write-Error "💥 System has critical issues!"
    }
    
    return $statusReport
}

# 📦 Installation Functions
function Install-Prerequisites {
    Write-Header "Installing Prerequisites" "Blue"
    
    # Create directories
    $directories = @(
        "C:\Legal-AI\logs",
        "C:\Legal-AI\data",
        "C:\Legal-AI\backups",
        "C:\Legal-AI\models"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created directory: $dir"
        }
    }
    
    # Install npm dependencies
    $npmProjects = @(
        "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend",
        "C:\Users\james\Desktop\deeds-web\deeds-web-app\embedding-service",
        "C:\Users\james\Desktop\deeds-web\deeds-web-app\ingest-worker"
    )
    
    foreach ($project in $npmProjects) {
        if (Test-Path "$project\package.json") {
            Write-Info "Installing dependencies for: $project"
            Push-Location $project
            try {
                npm install
                Write-Success "Dependencies installed for: $project"
            } catch {
                Write-Error "Failed to install dependencies for: $project"
            } finally {
                Pop-Location
            }
        }
    }
}

# 🔧 Database Setup
function Initialize-Database {
    Write-Header "Initializing Database" "Blue"
    
    # PostgreSQL setup
    try {
        Write-Info "Creating pgvector extension..."
        psql -h localhost -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>$null
        
        Write-Info "Running database migrations..."
        Push-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
        npm run db:migrate
        Pop-Location
        
        Write-Success "Database initialized successfully"
    } catch {
        Write-Error "Database initialization failed: $_"
    }
}

# 📈 Monitoring Functions
function Start-MonitoringMode {
    Write-Header "🔍 Production Monitoring Mode" "Magenta"
    Write-Info "Press Ctrl+C to exit monitoring mode"
    
    try {
        while ($true) {
            Clear-Host
            Write-Header "🔄 Live System Monitor - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "Green"
            
            $status = Get-ServicesStatus
            
            # Performance metrics
            $cpu = Get-Counter "\Processor(_Total)\% Processor Time" -ErrorAction SilentlyContinue
            $memory = Get-Counter "\Memory\Available MBytes" -ErrorAction SilentlyContinue
            
            if ($cpu -and $memory) {
                Write-Host "`n💻 System Performance:" -ForegroundColor Yellow
                Write-Host "   CPU Usage: $(100 - [math]::Round($cpu.CounterSamples[0].CookedValue, 1))%" -ForegroundColor White
                Write-Host "   Available Memory: $([math]::Round($memory.CounterSamples[0].CookedValue, 0)) MB" -ForegroundColor White
            }
            
            # GPU status
            try {
                $gpuInfo = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null
                if ($gpuInfo) {
                    $gpuData = $gpuInfo.Split(',')
                    Write-Host "`n🎮 GPU Status:" -ForegroundColor Yellow
                    Write-Host "   GPU Usage: $($gpuData[0].Trim())%" -ForegroundColor White
                    Write-Host "   Memory: $($gpuData[1].Trim())MB / $($gpuData[2].Trim())MB" -ForegroundColor White
                    Write-Host "   Temperature: $($gpuData[3].Trim())°C" -ForegroundColor White
                }
            } catch {
                Write-Host "`n🎮 GPU: Not available" -ForegroundColor Gray
            }
            
            Write-Host "`n⏱️  Next update in 30 seconds..." -ForegroundColor Gray
            Start-Sleep -Seconds 30
        }
    } catch [System.Management.Automation.PipelineStoppedException] {
        Write-Info "`nMonitoring stopped by user"
    }
}

# 📝 Logging Functions
function Write-DeploymentLog {
    param([string]$Message, [string]$Level = "INFO")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Write to console
    switch ($Level) {
        "ERROR" { Write-Error $Message }
        "WARN" { Write-Warning $Message }
        "INFO" { Write-Info $Message }
        "SUCCESS" { Write-Success $Message }
    }
    
    # Write to log file
    $logFile = "C:\Legal-AI\logs\deployment-$(Get-Date -Format 'yyyyMMdd').log"
    if (-not (Test-Path (Split-Path $logFile))) {
        New-Item -ItemType Directory -Path (Split-Path $logFile) -Force | Out-Null
    }
    Add-Content -Path $logFile -Value $logEntry -ErrorAction SilentlyContinue
}

# 🎯 Main Execution
function Main {
    Write-Header "🚀 Legal AI Production Deployment Orchestrator" "Green"
    Write-Host "Action: $Action | Group: $ServiceGroup | Dry Run: $DryRun" -ForegroundColor Cyan
    Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    
    if ($Verbose) {
        $VerbosePreference = "Continue"
    }
    
    switch ($Action) {
        "Install" {
            if (-not $DryRun) {
                Install-Prerequisites
                Initialize-Database
            } else {
                Write-Info "[DRY RUN] Would install prerequisites and initialize database"
            }
        }
        
        "Start" {
            if (-not (Test-Prerequisites)) {
                Write-Error "Prerequisites not met. Run with -Action Install first."
                exit 1
            }
            
            $success = Start-ServiceGroup -GroupName $ServiceGroup
            if (-not $success) {
                Write-Error "Failed to start services"
                exit 1
            }
            
            if ($MonitorMode) {
                Start-Sleep -Seconds 5
                Start-MonitoringMode
            }
        }
        
        "Stop" {
            Stop-ServiceGroup -GroupName $ServiceGroup
        }
        
        "Restart" {
            Stop-ServiceGroup -GroupName $ServiceGroup
            Start-Sleep -Seconds 5
            Start-ServiceGroup -GroupName $ServiceGroup
        }
        
        "Status" {
            $status = Get-ServicesStatus
            return $status
        }
        
        "Health" {
            $status = Get-ServicesStatus
            $criticalIssues = $status | Where-Object { $_.Critical -and -not $_.Healthy }
            
            if ($criticalIssues.Count -eq 0) {
                Write-Success "🎉 All critical services are healthy!"
                exit 0
            } else {
                Write-Error "💥 $($criticalIssues.Count) critical service(s) unhealthy!"
                exit 1
            }
        }
        
        "Uninstall" {
            if ($Force -or (Read-Host "Are you sure you want to uninstall? (y/N)") -eq "y") {
                Stop-ServiceGroup -GroupName "All"
                Write-Warning "Uninstall completed. Manual cleanup may be required."
            }
        }
    }
    
    $duration = (Get-Date) - $script:DeploymentStartTime
    Write-Host "`n⏱️  Total execution time: $($duration.TotalSeconds.ToString('0.0'))s" -ForegroundColor Gray
}

# 🚀 Script Entry Point
try {
    Main
} catch {
    Write-Error "💥 Deployment orchestrator failed: $_"
    Write-Error $_.ScriptStackTrace
    exit 1
}