# Windows Service Installation Script
# Production-ready deployment for Legal AI Platform microservices

param(
    [Parameter(Mandatory=$false)]
    [string]$Action = "install",

    [Parameter(Mandatory=$false)]
    [string]$ServiceName = "",

    [Parameter(Mandatory=$false)]
    [string]$ConfigPath = "config\production.json"
)

# Requires Administrator privileges
#Requires -RunAsAdministrator

# Service definitions with GPU acceleration and memory optimization
$Services = @{
    "LegalAI-GPU-Orchestrator" = @{
        DisplayName = "Legal AI GPU Orchestrator"
        Description = "GPU-accelerated orchestration service for Legal AI Platform"
        BinaryPath = "gpu-orchestrator.exe"
        Port = 8094
        MemoryLimit = "4GB"
        GPUEnabled = $true
        Priority = "High"
        Dependencies = @()
    }

    "LegalAI-Enhanced-RAG" = @{
        DisplayName = "Legal AI Enhanced RAG System"
        Description = "Enhanced RAG with SOM neural networks for legal document processing"
        BinaryPath = "cmd\enhanced-rag-som\enhanced-rag-som.exe"
        Port = 8095
        MemoryLimit = "6GB"
        GPUEnabled = $true
        Priority = "High"
        Dependencies = @("LegalAI-GPU-Orchestrator")
    }

    "LegalAI-Multi-Protocol-Gateway" = @{
        DisplayName = "Legal AI Multi-Protocol Gateway"
        Description = "Multi-protocol gateway supporting REST, gRPC, QUIC, and WebSocket"
        BinaryPath = "cmd\multi-protocol-gateway\multi-protocol-gateway.exe"
        Port = 8096
        MemoryLimit = "2GB"
        GPUEnabled = $false
        Priority = "Normal"
        Dependencies = @()
    }

    "LegalAI-Protocol-Monitor" = @{
        DisplayName = "Legal AI Protocol Performance Monitor"
        Description = "Real-time protocol performance monitoring and metrics collection"
        BinaryPath = "cmd\protocol-monitor\protocol-monitor.exe"
        Port = 8097
        MemoryLimit = "1GB"
        GPUEnabled = $false
        Priority = "Normal"
        Dependencies = @()
    }

    "LegalAI-Health-Server" = @{
        DisplayName = "Legal AI Health Monitoring Server"
        Description = "Health monitoring and metrics server for all Legal AI services"
        BinaryPath = "cmd\health-server\health-server.exe"
        Port = 8099
        MemoryLimit = "512MB"
        GPUEnabled = $false
        Priority = "Normal"
        Dependencies = @()
    }
}

# Global enhancement paths & files
$Global:ServiceLogRoot = Join-Path $PWD 'logs/services'
$Global:EnvFile = Join-Path $PWD '.env.production'


# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Cyan"
}

function Write-ColorOutput($Message, $Color = "White") {
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Ensure-RedisRunning {
    param([int]$Port = 6379)
    Write-ColorOutput "Checking Redis availability on port $Port..." "Info"
    try {
        $tcp = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
        if ($tcp.TcpTestSucceeded) { Write-ColorOutput "Redis detected (port $Port)." "Success"; return $true }
        $possible = @('C:\\Program Files\\Redis\\redis-server.exe','C:\\redis\\redis-server.exe') | Where-Object { Test-Path $_ }
        if ($possible.Count -gt 0) {
            Write-ColorOutput "Attempting local Redis start: $($possible[0])" "Warning"
            Start-Process -FilePath $possible[0] -ArgumentList '--maxheap 512mb' -WindowStyle Minimized
            Start-Sleep -Seconds 3
            $tcp2 = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
            if ($tcp2.TcpTestSucceeded) { Write-ColorOutput "Redis started successfully." "Success"; return $true }
        }
        Write-ColorOutput "Redis not running; cache-dependent services may degrade." "Warning"
        return $false
    } catch { Write-ColorOutput "Redis check failed: $($_.Exception.Message)" "Warning"; return $false }
}

function Detect-GPUInfo {
    try { $gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1; if ($gpu) { return [PSCustomObject]@{ Name=$gpu.Name; MemoryGB=[math]::Round($gpu.AdapterRAM/1GB,2) } } } catch {}
    return $null
}

function Compute-AdaptiveMemoryLimit { param([double]$SystemRAMGB,[double]$GpuRAMGB) $base=[math]::Min($SystemRAMGB*0.25,8); if ($GpuRAMGB -and $GpuRAMGB -lt 6){$base=[math]::Min($base,4)}; return [int][math]::Max(2,[math]::Round($base)) }

function Load-EnvFileIfPresent { if (Test-Path $Global:EnvFile) { Write-ColorOutput "Loading env from $($Global:EnvFile)" "Info"; Get-Content $Global:EnvFile | ForEach-Object { if ($_ -match '^[#;]'){return}; if ($_ -match '^(?<k>[A-Za-z_][A-Za-z0-9_]*)=(?<v>.*)$'){ $k=$Matches.k; $v=$Matches.v -replace '"',''; [Environment]::SetEnvironmentVariable($k,$v,'Process') } } } }

function Write-ServiceLaunchWrapper {
    param([string]$ServiceName,[string]$BinaryPath,[string]$ArgumentString)
    if (!(Test-Path $Global:ServiceLogRoot)) { New-Item -ItemType Directory -Path $Global:ServiceLogRoot -Force | Out-Null }
    $wrapper = Join-Path $PWD ("start-" + ($ServiceName.ToLower()) + ".ps1")
    $logPath = Join-Path $Global:ServiceLogRoot ("$ServiceName-" + (Get-Date -Format 'yyyyMMdd') + ".log")
    @(
        "# Autogenerated launch wrapper for $ServiceName",
        "`$ts = Get-Date -Format o",
        "`$env:SERVICE_NAME='$ServiceName'",
        "Write-Host `"[$ServiceName] starting `$(`$ts)`"",
        "& `"$BinaryPath`" $ArgumentString 2>&1 | Tee-Object -FilePath `"$logPath`" -Append"
    ) | Out-File -FilePath $wrapper -Encoding UTF8
    return $wrapper
}

function Optimize-ServicesConfiguration {
    Write-ColorOutput "Optimization pass (GPU + memory tuning)" "Info"
    $sysRAM=[math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)
    $gpu=Detect-GPUInfo
    if ($gpu){ Write-ColorOutput "GPU: $($gpu.Name) ($($gpu.MemoryGB) GB)" "Success" } else { Write-ColorOutput "No NVIDIA GPU detected" "Warning" }
    $adaptive=Compute-AdaptiveMemoryLimit -SystemRAMGB $sysRAM -GpuRAMGB ($gpu?.MemoryGB)
    Write-ColorOutput "Adaptive orchestrator memory target: $adaptive GB" "Info"
    if (Test-Path $ConfigPath){ try { $json=Get-Content $ConfigPath -Raw | ConvertFrom-Json; if (-not $json.memory){ $json | Add-Member -NotePropertyName memory -NotePropertyValue (@{}) }; $json.memory.max_heap_size_mb=$adaptive*1024; if ($gpu){ $json.gpu.enabled=$true }; $json | ConvertTo-Json -Depth 12 | Out-File -FilePath $ConfigPath -Encoding UTF8; Write-ColorOutput "Updated $ConfigPath (adaptive memory)." "Success" } catch { Write-ColorOutput "Config update failed: $($_.Exception.Message)" "Warning" } }
}

function Test-AdminPrivileges {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-ServiceExists($ServiceName) {
    try {
        $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        return $service -ne $null
    }
    catch {
        return $false
    }
}

function Install-LegalAIService($ServiceName, $ServiceConfig) {
    Write-ColorOutput "Installing service: $ServiceName" "Info"

    $binaryPath = Join-Path $PWD $ServiceConfig.BinaryPath

    # Check if binary exists
    if (!(Test-Path $binaryPath)) {
        Write-ColorOutput "Error: Binary not found at $binaryPath" "Error"
        return $false
    }

    # Build service arguments with production optimizations
    $arguments = @()
    $arguments += "-service"
    $arguments += "-config=`"$ConfigPath`""
    $arguments += "-port=$($ServiceConfig.Port)"

    if ($ServiceConfig.GPUEnabled) {
        $arguments += "-gpu-enabled=true"
        $arguments += "-gpu-memory-limit=$($ServiceConfig.MemoryLimit)"
    }

    $argumentString = $arguments -join " "
    $fullBinaryPath = "`"$binaryPath`" $argumentString"
    $wrapper = Write-ServiceLaunchWrapper -ServiceName $ServiceName -BinaryPath $binaryPath -ArgumentString $argumentString
    if (Test-Path $wrapper) { $fullBinaryPath = "powershell -NoProfile -ExecutionPolicy Bypass -File `"$wrapper`"" }

    try {
        # Create the service
        New-Service -Name $ServiceName `
                   -BinaryPathName $fullBinaryPath `
                   -DisplayName $ServiceConfig.DisplayName `
                   -Description $ServiceConfig.Description `
                   -StartupType Automatic `
                   -Credential "NT AUTHORITY\LocalService"

        # Configure service for production
        $serviceKey = "HKLM:\SYSTEM\CurrentControlSet\Services\$ServiceName"

        # Set failure actions (restart on failure)
        & sc.exe failure $ServiceName reset=86400 actions=restart/5000/restart/10000/restart/30000

        # Set memory limits if specified
        if ($ServiceConfig.MemoryLimit) {
            $memoryBytes = switch -Regex ($ServiceConfig.MemoryLimit) {
                "(\d+)GB" { [int64]$Matches[1] * 1GB }
                "(\d+)MB" { [int64]$Matches[1] * 1MB }
                default { 2GB }
            }

            # Set working set limit
            & sc.exe config $ServiceName obj="NT AUTHORITY\LocalService"
        }

        # Set process priority
        if ($ServiceConfig.Priority -eq "High") {
            New-ItemProperty -Path $serviceKey -Name "PriorityClass" -Value 3 -PropertyType DWord -Force | Out-Null
        }

        # Configure dependencies
        if ($ServiceConfig.Dependencies.Count -gt 0) {
            $deps = $ServiceConfig.Dependencies -join "/"
            & sc.exe config $ServiceName depend=$deps
        }

        # Create event log source
        New-EventLog -LogName Application -Source $ServiceName -ErrorAction SilentlyContinue

        Write-ColorOutput "Service $ServiceName installed successfully" "Success"
        return $true
    }
    catch {
        Write-ColorOutput "Error installing service ${ServiceName}: $($_.Exception.Message)" "Error"
        return $false
    }
}

function Uninstall-LegalAIService($ServiceName) {
    Write-ColorOutput "Uninstalling service: $ServiceName" "Info"

    try {
        if (Test-ServiceExists $ServiceName) {
            # Stop the service first
            Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue

            # Wait for service to stop
            $timeout = 30
            while ((Get-Service -Name $ServiceName).Status -ne "Stopped" -and $timeout -gt 0) {
                Start-Sleep -Seconds 1
                $timeout--
            }

            # Remove the service
            & sc.exe delete $ServiceName

            # Remove event log source
            Remove-EventLog -Source $ServiceName -ErrorAction SilentlyContinue

            Write-ColorOutput "Service $ServiceName uninstalled successfully" "Success"
        } else {
            Write-ColorOutput "Service $ServiceName does not exist" "Warning"
        }
        return $true
    }
    catch {
        Write-ColorOutput "Error uninstalling service ${ServiceName}: $($_.Exception.Message)" "Error"
        return $false
    }
}

function Start-LegalAIService($ServiceName) {
    Write-ColorOutput "Starting service: $ServiceName" "Info"

    try {
        if (Test-ServiceExists $ServiceName) {
            Start-Service -Name $ServiceName

            # Wait for service to start
            $timeout = 30
            while ((Get-Service -Name $ServiceName).Status -ne "Running" -and $timeout -gt 0) {
                Start-Sleep -Seconds 1
                $timeout--
            }

            if ((Get-Service -Name $ServiceName).Status -eq "Running") {
                Write-ColorOutput "Service $ServiceName started successfully" "Success"
            } else {
                Write-ColorOutput "Service $ServiceName failed to start within timeout" "Error"
            }
        } else {
            Write-ColorOutput "Service $ServiceName does not exist" "Error"
        }
    }
    catch {
        Write-ColorOutput "Error starting service ${ServiceName}: $($_.Exception.Message)" "Error"
    }
}

function Stop-LegalAIService($ServiceName) {
    Write-ColorOutput "Stopping service: $ServiceName" "Info"

    try {
        if (Test-ServiceExists $ServiceName) {
            Stop-Service -Name $ServiceName -Force
            Write-ColorOutput "Service $ServiceName stopped successfully" "Success"
        } else {
            Write-ColorOutput "Service $ServiceName does not exist" "Warning"
        }
    }
    catch {
        Write-ColorOutput "Error stopping service ${ServiceName}: $($_.Exception.Message)" "Error"
    }
}

function Get-ServiceStatus($ServiceName = "") {
    if ($ServiceName) {
        if (Test-ServiceExists $ServiceName) {
            $service = Get-Service -Name $ServiceName
            $status = $service.Status
            $startType = (Get-WmiObject -Class Win32_Service -Filter "Name='$ServiceName'").StartMode

            Write-ColorOutput "Service: $ServiceName" "Info"
            Write-ColorOutput "  Status: $status" $(if ($status -eq "Running") { "Success" } else { "Warning" })
            Write-ColorOutput "  Start Type: $startType" "Info"
            Write-ColorOutput ""
        } else {
            Write-ColorOutput "Service $ServiceName does not exist" "Error"
        }
    } else {
        Write-ColorOutput "Legal AI Platform Services Status:" "Info"
        Write-ColorOutput "=================================" "Info"

        foreach ($svcName in $Services.Keys) {
            Get-ServiceStatus $svcName
        }
    }
}

function Initialize-Environment {
    Write-ColorOutput "Initializing Legal AI Platform environment..." "Info"

    # Create necessary directories
    $directories = @("config", "logs", "cache", "data")
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "Created directory: $dir" "Success"
        }
    }

    # Create production configuration if it doesn't exist
    if (!(Test-Path $ConfigPath)) {
        Write-ColorOutput "Creating production configuration..." "Info"

        $prodConfig = @{
            service_name = "LegalAI-Platform"
            service_description = "Legal AI Platform Microservices"
            gpu = @{
                enabled = $true
                device_id = 0
                memory_limit_mb = 6144
                batch_size = 32
                cuda_version = "11.8"
                optimization = "balanced"
            }
            memory = @{
                max_heap_size_mb = 4096
                gc_target_percent = 100
                enable_profiling = $false
                pool_size = [Environment]::ProcessorCount * 2
                buffer_size = 8192
            }
            cache = @{
                redis = @{
                    enabled = $true
                    host = "localhost"
                    port = 6379
                    db = 0
                    pool_size = [Environment]::ProcessorCount * 2
                }
                local = @{
                    enabled = $true
                    max_size_mb = 512
                    ttl_seconds = 3600
                }
            }
            performance = @{
                worker_count = [Environment]::ProcessorCount
                max_concurrency = [Environment]::ProcessorCount * 4
                enable_pipelining = $true
                batch_timeout_ms = 100
            }
        }

        $prodConfig | ConvertTo-Json -Depth 10 | Out-File -FilePath $ConfigPath -Encoding UTF8
        Write-ColorOutput "Production configuration created: $ConfigPath" "Success"
    }

    # Check GPU availability
    try {
        $gpuInfo = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($gpuInfo) {
            Write-ColorOutput "GPU detected: $($gpuInfo.Name)" "Success"
            Write-ColorOutput "GPU Memory: $([math]::Round($gpuInfo.AdapterRAM / 1GB, 2)) GB" "Success"
        } else {
            Write-ColorOutput "No NVIDIA GPU detected - GPU acceleration will be disabled" "Warning"
        }
    }
    catch {
        Write-ColorOutput "Could not detect GPU information" "Warning"
    }

    # Check available memory
    $totalMemory = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
    Write-ColorOutput "Total System Memory: $totalMemory GB" "Info"

    if ($totalMemory -lt 16) {
        Write-ColorOutput "Warning: Less than 16GB RAM detected. Consider reducing memory limits in configuration." "Warning"
    }
}

function Build-Services {
    Write-ColorOutput "Building all Legal AI services..." "Info"

    $buildCommands = @(
        "go build -ldflags=`"-s -w`" -o gpu-orchestrator.exe ./cmd/gpu-orchestrator",
        "go build -ldflags=`"-s -w`" -o cmd/enhanced-rag-som/enhanced-rag-som.exe ./cmd/enhanced-rag-som",
        "go build -ldflags=`"-s -w`" -o cmd/multi-protocol-gateway/multi-protocol-gateway.exe ./cmd/multi-protocol-gateway",
        "go build -ldflags=`"-s -w`" -o cmd/protocol-monitor/protocol-monitor.exe ./cmd/protocol-monitor",
        "go build -ldflags=`"-s -w`" -o cmd/health-server/health-server.exe ./cmd/health-server"
    )

    foreach ($cmd in $buildCommands) {
        Write-ColorOutput "Building: $cmd" "Info"
        Invoke-Expression $cmd
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "Build successful" "Success"
        } else {
            Write-ColorOutput "Build failed" "Error"
            return $false
        }
    }

    return $true
}

# Main execution logic
Write-ColorOutput "Legal AI Platform Windows Service Manager" "Info"
Write-ColorOutput "=========================================" "Info"

# Check for administrator privileges
if (!(Test-AdminPrivileges)) {
    Write-ColorOutput "This script requires Administrator privileges. Please run as Administrator." "Error"
    exit 1
}

switch ($Action.ToLower()) {
    "install" {
        if ($ServiceName) {
            if ($Services.ContainsKey($ServiceName)) {
                Install-LegalAIService $ServiceName $Services[$ServiceName]
            } else {
                Write-ColorOutput "Unknown service: $ServiceName" "Error"
                Write-ColorOutput "Available services: $($Services.Keys -join ', ')" "Info"
            }
        } else {
            Initialize-Environment
            Load-EnvFileIfPresent
            Ensure-RedisRunning | Out-Null
            Optimize-ServicesConfiguration

            if (Build-Services) {
                Write-ColorOutput "Installing all Legal AI services..." "Info"

                $installOrder = @(
                    "LegalAI-GPU-Orchestrator",
                    "LegalAI-Enhanced-RAG",
                    "LegalAI-Multi-Protocol-Gateway",
                    "LegalAI-Protocol-Monitor",
                    "LegalAI-Health-Server"
                )

                foreach ($svcName in $installOrder) {
                    Install-LegalAIService $svcName $Services[$svcName]
                    Start-Sleep -Seconds 2
                }

                Write-ColorOutput "All services installed. Use 'start' action to start them." "Success"
            } else {
                Write-ColorOutput "Build failed. Services not installed." "Error"
            }
        }
    }
    "optimize" { Initialize-Environment; Optimize-ServicesConfiguration }
    "redis-check" { Ensure-RedisRunning | Out-Null }

    "uninstall" {
        if ($ServiceName) {
            Uninstall-LegalAIService $ServiceName
        } else {
            Write-ColorOutput "Uninstalling all Legal AI services..." "Info"

            # Uninstall in reverse order
            $uninstallOrder = @(
                "LegalAI-Health-Server",
                "LegalAI-Protocol-Monitor",
                "LegalAI-Multi-Protocol-Gateway",
                "LegalAI-Enhanced-RAG",
                "LegalAI-GPU-Orchestrator"
            )

            foreach ($svcName in $uninstallOrder) {
                Uninstall-LegalAIService $svcName
            }
        }
    }

    "start" {
        if ($ServiceName) {
            Start-LegalAIService $ServiceName
        } else {
            Write-ColorOutput "Starting all Legal AI services..." "Info"

            $startOrder = @(
                "LegalAI-GPU-Orchestrator",
                "LegalAI-Multi-Protocol-Gateway",
                "LegalAI-Protocol-Monitor",
                "LegalAI-Enhanced-RAG",
                "LegalAI-Health-Server"
            )

            foreach ($svcName in $startOrder) {
                Start-LegalAIService $svcName
                Start-Sleep -Seconds 3
            }
        }
    }

    "stop" {
        if ($ServiceName) {
            Stop-LegalAIService $ServiceName
        } else {
            Write-ColorOutput "Stopping all Legal AI services..." "Info"

            # Stop in reverse order
            $stopOrder = @(
                "LegalAI-Health-Server",
                "LegalAI-Enhanced-RAG",
                "LegalAI-Protocol-Monitor",
                "LegalAI-Multi-Protocol-Gateway",
                "LegalAI-GPU-Orchestrator"
            )

            foreach ($svcName in $stopOrder) {
                Stop-LegalAIService $svcName
            }
        }
    }

    "status" {
        Get-ServiceStatus $ServiceName
    }

    "build" {
        Build-Services
    }

    "init" {
        Initialize-Environment
    }

    default {
        Write-ColorOutput "Usage: install-windows-services.ps1 -Action <action> [-ServiceName <name>] [-ConfigPath <path>]" "Info"
        Write-ColorOutput ""
        Write-ColorOutput "Actions:" "Info"
        Write-ColorOutput "  install   - Install services (builds if needed)" "Info"
        Write-ColorOutput "  uninstall - Uninstall services" "Info"
        Write-ColorOutput "  start     - Start services" "Info"
        Write-ColorOutput "  stop      - Stop services" "Info"
        Write-ColorOutput "  status    - Show service status" "Info"
        Write-ColorOutput "  build     - Build all executables" "Info"
        Write-ColorOutput "  init      - Initialize environment" "Info"
        Write-ColorOutput ""
        Write-ColorOutput "Available Services:" "Info"
        foreach ($svc in $Services.Keys) {
            Write-ColorOutput "  $svc" "Info"
        }
    }
}

Write-ColorOutput "Operation completed." "Success"