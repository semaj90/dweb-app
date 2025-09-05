# Legal AI Platform - Windows Services Setup Script
# Configures production deployment as Windows Services

param(
    [Parameter()]
    [ValidateSet("install", "uninstall", "start", "stop", "status", "logs")]
    [string]$Action = "status",
    
    [Parameter()]
    [string]$ServiceName = "",
    
    [Parameter()]
    [switch]$Force = $false
)

# Service configuration
$SERVICES = @{
    "LegalAI-SvelteKit" = @{
        Name = "LegalAI-SvelteKit"
        DisplayName = "Legal AI Platform - SvelteKit Frontend"
        Description = "SvelteKit frontend server for Legal AI Platform"
        ServiceType = "own"
        StartType = "Automatic"
        ExecutablePath = "node"
        Arguments = "build/index.js"
        WorkingDirectory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"
        LogPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\logs\sveltekit.log"
        Port = 5173
    }
    "LegalAI-EnhancedRAG" = @{
        Name = "LegalAI-EnhancedRAG"
        DisplayName = "Legal AI Platform - Enhanced RAG Service"
        Description = "Go microservice providing enhanced RAG functionality"
        ServiceType = "own"
        StartType = "Automatic"
        ExecutablePath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice\bin\enhanced-rag.exe"
        Arguments = ""
        WorkingDirectory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice"
        LogPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\logs\enhanced-rag.log"
        Port = 8094
    }
    "LegalAI-UploadService" = @{
        Name = "LegalAI-UploadService"
        DisplayName = "Legal AI Platform - Upload Service"
        Description = "Go microservice for file uploads and processing"
        ServiceType = "own"
        StartType = "Automatic"
        ExecutablePath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice\bin\upload-service.exe"
        Arguments = ""
        WorkingDirectory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice"
        LogPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\logs\upload-service.log"
        Port = 8093
    }
    "LegalAI-PostgreSQL" = @{
        Name = "LegalAI-PostgreSQL"
        DisplayName = "Legal AI Platform - PostgreSQL Database"
        Description = "PostgreSQL database with pgvector extension for Legal AI Platform"
        ServiceType = "own"
        StartType = "Automatic"
        ExecutablePath = "pg_ctl"
        Arguments = "start -D `"C:\Program Files\PostgreSQL\17\data`" -l `"C:\Users\james\Desktop\deeds-web\deeds-web-app\logs\postgresql.log`""
        WorkingDirectory = "C:\Program Files\PostgreSQL\17\bin"
        LogPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\logs\postgresql.log"
        Port = 5432
    }
    "LegalAI-Redis" = @{
        Name = "LegalAI-Redis"
        DisplayName = "Legal AI Platform - Redis Cache"
        Description = "Redis caching service for Legal AI Platform"
        ServiceType = "own"
        StartType = "Automatic"
        ExecutablePath = "redis-server"
        Arguments = "--service-run"
        WorkingDirectory = "C:\Users\james\Desktop\deeds-web\deeds-web-app\redis-latest"
        LogPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\logs\redis.log"
        Port = 6379
    }
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Ensure-LogDirectory {
    $logDir = "C:\Users\james\Desktop\deeds-web\deeds-web-app\logs"
    if (!(Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        Write-ColorOutput "✅ Created log directory: $logDir" "Green"
    }
}

function Install-Service {
    param(
        [string]$ServiceKey
    )
    
    $service = $SERVICES[$ServiceKey]
    $serviceName = $service.Name
    
    Write-ColorOutput "🔧 Installing service: $($service.DisplayName)" "Yellow"
    
    # Check if service already exists
    if (Get-Service -Name $serviceName -ErrorAction SilentlyContinue) {
        if ($Force) {
            Write-ColorOutput "⚠️ Service $serviceName exists. Removing..." "Yellow"
            Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
            sc.exe delete $serviceName | Out-Null
            Start-Sleep -Seconds 2
        } else {
            Write-ColorOutput "❌ Service $serviceName already exists. Use -Force to reinstall." "Red"
            return $false
        }
    }
    
    # Create service using sc.exe for better control
    $executablePath = "`"$($service.ExecutablePath)`""
    if ($service.Arguments) {
        $executablePath += " $($service.Arguments)"
    }
    
    $createResult = sc.exe create $serviceName `
        binPath= $executablePath `
        DisplayName= "`"$($service.DisplayName)`"" `
        description= "`"$($service.Description)`"" `
        start= $($service.StartType -eq "Automatic" ? "auto" : "manual") `
        type= $service.ServiceType
    
    if ($LASTEXITCODE -eq 0) {
        # Set working directory if specified
        if ($service.WorkingDirectory) {
            sc.exe config $serviceName obj= "LocalSystem" | Out-Null
        }
        
        Write-ColorOutput "✅ Service $serviceName installed successfully" "Green"
        return $true
    } else {
        Write-ColorOutput "❌ Failed to install service $serviceName" "Red"
        Write-ColorOutput "Error: $createResult" "Red"
        return $false
    }
}

function Uninstall-Service {
    param(
        [string]$ServiceKey
    )
    
    $service = $SERVICES[$ServiceKey]
    $serviceName = $service.Name
    
    Write-ColorOutput "🗑️ Uninstalling service: $($service.DisplayName)" "Yellow"
    
    # Stop service if running
    if (Get-Service -Name $serviceName -ErrorAction SilentlyContinue) {
        Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
        
        # Delete service
        $deleteResult = sc.exe delete $serviceName
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Service $serviceName uninstalled successfully" "Green"
            return $true
        } else {
            Write-ColorOutput "❌ Failed to uninstall service $serviceName" "Red"
            return $false
        }
    } else {
        Write-ColorOutput "⚠️ Service $serviceName not found" "Yellow"
        return $true
    }
}

function Start-LegalAIService {
    param(
        [string]$ServiceKey
    )
    
    $service = $SERVICES[$ServiceKey]
    $serviceName = $service.Name
    
    Write-ColorOutput "▶️ Starting service: $($service.DisplayName)" "Yellow"
    
    try {
        Start-Service -Name $serviceName -ErrorAction Stop
        Start-Sleep -Seconds 2
        
        $serviceStatus = Get-Service -Name $serviceName
        if ($serviceStatus.Status -eq "Running") {
            Write-ColorOutput "✅ Service $serviceName started successfully" "Green"
            
            # Test port if specified
            if ($service.Port) {
                Test-ServicePort -Port $service.Port -ServiceName $serviceName
            }
            
            return $true
        } else {
            Write-ColorOutput "❌ Service $serviceName failed to start (Status: $($serviceStatus.Status))" "Red"
            return $false
        }
    } catch {
        Write-ColorOutput "❌ Failed to start service $serviceName`: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Stop-LegalAIService {
    param(
        [string]$ServiceKey
    )
    
    $service = $SERVICES[$ServiceKey]
    $serviceName = $service.Name
    
    Write-ColorOutput "⏹️ Stopping service: $($service.DisplayName)" "Yellow"
    
    try {
        Stop-Service -Name $serviceName -Force -ErrorAction Stop
        Write-ColorOutput "✅ Service $serviceName stopped successfully" "Green"
        return $true
    } catch {
        Write-ColorOutput "❌ Failed to stop service $serviceName`: $($_.Exception.Message)" "Red"
        return $false
    }
}

function Test-ServicePort {
    param(
        [int]$Port,
        [string]$ServiceName
    )
    
    try {
        $connection = Test-NetConnection -ComputerName "localhost" -Port $Port -WarningAction SilentlyContinue
        if ($connection.TcpTestSucceeded) {
            Write-ColorOutput "   ✅ Port $Port is accessible" "Green"
        } else {
            Write-ColorOutput "   ⚠️ Port $Port is not accessible (service may still be starting)" "Yellow"
        }
    } catch {
        Write-ColorOutput "   ⚠️ Could not test port $Port" "Yellow"
    }
}

function Get-ServiceStatus {
    param(
        [string]$ServiceKey = ""
    )
    
    Write-ColorOutput "`n📊 Legal AI Platform Service Status" "Cyan"
    Write-ColorOutput "=====================================" "Cyan"
    
    $servicesToCheck = if ($ServiceKey) { @($ServiceKey) } else { $SERVICES.Keys }
    
    foreach ($key in $servicesToCheck) {
        $service = $SERVICES[$key]
        $serviceName = $service.Name
        
        $serviceObj = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
        
        if ($serviceObj) {
            $statusColor = switch ($serviceObj.Status) {
                "Running" { "Green" }
                "Stopped" { "Red" }
                "Paused" { "Yellow" }
                default { "Gray" }
            }
            
            Write-ColorOutput "$($service.DisplayName):" "White"
            Write-ColorOutput "   Status: $($serviceObj.Status)" $statusColor
            Write-ColorOutput "   Start Type: $($serviceObj.StartType)" "Gray"
            
            if ($service.Port -and $serviceObj.Status -eq "Running") {
                Test-ServicePort -Port $service.Port -ServiceName $serviceName
            }
            
            # Check log file
            if ($service.LogPath -and (Test-Path $service.LogPath)) {
                $logSize = (Get-Item $service.LogPath).Length
                Write-ColorOutput "   Log: $($service.LogPath) ($([math]::Round($logSize/1KB, 1)) KB)" "Gray"
            }
        } else {
            Write-ColorOutput "$($service.DisplayName):" "White"
            Write-ColorOutput "   Status: Not Installed" "Red"
        }
        
        Write-ColorOutput "" "White"
    }
}

function Show-ServiceLogs {
    param(
        [string]$ServiceKey
    )
    
    if (!$ServiceKey) {
        Write-ColorOutput "❌ Service name required for log viewing" "Red"
        return
    }
    
    $service = $SERVICES[$ServiceKey]
    if (!$service) {
        Write-ColorOutput "❌ Unknown service: $ServiceKey" "Red"
        return
    }
    
    if (Test-Path $service.LogPath) {
        Write-ColorOutput "📋 Last 50 lines of $($service.DisplayName) log:" "Cyan"
        Write-ColorOutput "================================================" "Cyan"
        Get-Content $service.LogPath -Tail 50
    } else {
        Write-ColorOutput "❌ Log file not found: $($service.LogPath)" "Red"
    }
}

function Show-Usage {
    Write-ColorOutput "`n🔧 Legal AI Platform Windows Services Manager" "Cyan"
    Write-ColorOutput "=============================================" "Cyan"
    Write-ColorOutput ""
    Write-ColorOutput "Usage: .\setup-windows-services.ps1 -Action <action> [-ServiceName <name>] [-Force]"
    Write-ColorOutput ""
    Write-ColorOutput "Actions:"
    Write-ColorOutput "  install   - Install Windows services"
    Write-ColorOutput "  uninstall - Remove Windows services"
    Write-ColorOutput "  start     - Start services"
    Write-ColorOutput "  stop      - Stop services"
    Write-ColorOutput "  status    - Show service status (default)"
    Write-ColorOutput "  logs      - Show service logs"
    Write-ColorOutput ""
    Write-ColorOutput "Service Names:"
    foreach ($key in $SERVICES.Keys) {
        Write-ColorOutput "  $key - $($SERVICES[$key].DisplayName)" "Gray"
    }
    Write-ColorOutput ""
    Write-ColorOutput "Examples:"
    Write-ColorOutput "  .\setup-windows-services.ps1 -Action install"
    Write-ColorOutput "  .\setup-windows-services.ps1 -Action start -ServiceName LegalAI-SvelteKit"
    Write-ColorOutput "  .\setup-windows-services.ps1 -Action status"
    Write-ColorOutput "  .\setup-windows-services.ps1 -Action logs -ServiceName LegalAI-EnhancedRAG"
}

# Main execution
if (!$args -and $Action -eq "status") {
    Show-Usage
    Get-ServiceStatus
    exit 0
}

# Check administrator privileges for install/uninstall operations
if ($Action -in @("install", "uninstall") -and !(Test-Administrator)) {
    Write-ColorOutput "❌ Administrator privileges required for $Action operation" "Red"
    Write-ColorOutput "Please run PowerShell as Administrator and try again." "Yellow"
    exit 1
}

# Ensure log directory exists
Ensure-LogDirectory

# Execute action
switch ($Action.ToLower()) {
    "install" {
        Write-ColorOutput "🚀 Installing Legal AI Platform Windows Services..." "Cyan"
        
        $servicesToInstall = if ($ServiceName) { @($ServiceName) } else { $SERVICES.Keys }
        $successCount = 0
        
        foreach ($key in $servicesToInstall) {
            if ($SERVICES.ContainsKey($key)) {
                if (Install-Service -ServiceKey $key) {
                    $successCount++
                }
            } else {
                Write-ColorOutput "❌ Unknown service: $key" "Red"
            }
        }
        
        Write-ColorOutput "`n✅ Installation completed: $successCount/$($servicesToInstall.Count) services installed" "Green"
    }
    
    "uninstall" {
        Write-ColorOutput "🗑️ Uninstalling Legal AI Platform Windows Services..." "Cyan"
        
        $servicesToUninstall = if ($ServiceName) { @($ServiceName) } else { $SERVICES.Keys }
        $successCount = 0
        
        foreach ($key in $servicesToUninstall) {
            if ($SERVICES.ContainsKey($key)) {
                if (Uninstall-Service -ServiceKey $key) {
                    $successCount++
                }
            } else {
                Write-ColorOutput "❌ Unknown service: $key" "Red"
            }
        }
        
        Write-ColorOutput "`n✅ Uninstallation completed: $successCount/$($servicesToUninstall.Count) services uninstalled" "Green"
    }
    
    "start" {
        Write-ColorOutput "▶️ Starting Legal AI Platform Services..." "Cyan"
        
        $servicesToStart = if ($ServiceName) { @($ServiceName) } else { $SERVICES.Keys }
        $successCount = 0
        
        foreach ($key in $servicesToStart) {
            if ($SERVICES.ContainsKey($key)) {
                if (Start-LegalAIService -ServiceKey $key) {
                    $successCount++
                }
            } else {
                Write-ColorOutput "❌ Unknown service: $key" "Red"
            }
        }
        
        Write-ColorOutput "`n✅ Start completed: $successCount/$($servicesToStart.Count) services started" "Green"
    }
    
    "stop" {
        Write-ColorOutput "⏹️ Stopping Legal AI Platform Services..." "Cyan"
        
        $servicesToStop = if ($ServiceName) { @($ServiceName) } else { $SERVICES.Keys }
        $successCount = 0
        
        foreach ($key in $servicesToStop) {
            if ($SERVICES.ContainsKey($key)) {
                if (Stop-LegalAIService -ServiceKey $key) {
                    $successCount++
                }
            } else {
                Write-ColorOutput "❌ Unknown service: $key" "Red"
            }
        }
        
        Write-ColorOutput "`n✅ Stop completed: $successCount/$($servicesToStop.Count) services stopped" "Green"
    }
    
    "status" {
        Get-ServiceStatus -ServiceKey $ServiceName
    }
    
    "logs" {
        Show-ServiceLogs -ServiceKey $ServiceName
    }
    
    default {
        Write-ColorOutput "❌ Invalid action: $Action" "Red"
        Show-Usage
        exit 1
    }
}

Write-ColorOutput "`n🎯 Legal AI Platform Windows Services Manager completed." "Cyan"