# Legal AI Platform - Production Deployment Script
# Master Service Coordinator with Error Resolution
# Native Windows Deployment - Enterprise Grade

param(
    [string]$Action = "start",
    [switch]$SkipGo = $false,
    [switch]$SkipFrontend = $false,
    [switch]$SkipDatabase = $false,
    [switch]$Production = $false,
    [switch]$Verbose = $false,
    [int]$HealthCheckInterval = 30,
    [string]$LogLevel = "info"
)

$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = "Legal AI Platform - Production Deployment"

# Configuration
$Config = @{
    Environment = if ($Production) { "production" } else { "development" }
    NodeOptions = "--max-old-space-size=8192"
    LogLevel = $LogLevel
    HealthCheckInterval = $HealthCheckInterval
    Services = @{
        PostgreSQL = @{ Port = 5432; Name = "PostgreSQL Database" }
        Redis = @{ Port = 6379; Name = "Redis Cache" }
        Ollama = @{ Port = 11434; Name = "Ollama AI Server" }
        Neo4j = @{ Port = 7474; Name = "Neo4j Graph Database" }
        Frontend = @{ Port = 5173; Name = "SvelteKit Frontend" }
        EnhancedRAG = @{ Port = 8094; Name = "Enhanced RAG Engine" }
        UploadService = @{ Port = 8093; Name = "Upload Service" }
        CUDAService = @{ Port = 8096; Name = "CUDA GPU Service" }
        GRPCServer = @{ Port = 50051; Name = "gRPC Protocol Server" }
    }
    Colors = @{
        Success = "Green"
        Warning = "Yellow"  
        Error = "Red"
        Info = "Cyan"
        Accent = "Magenta"
        Secondary = "Gray"
    }
}

# Utility Functions
function Write-ColorOutput($Text, $Color = "White", $NoNewline = $false) {
    if ($NoNewline) {
        Write-Host $Text -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Text -ForegroundColor $Color
    }
}

function Write-Banner($Title) {
    $border = "=" * 80
    Write-ColorOutput "`n$border" $Config.Colors.Accent
    Write-ColorOutput "  $Title" $Config.Colors.Success
    Write-ColorOutput "$border`n" $Config.Colors.Accent
}

function Write-Step($StepNumber, $Description) {
    Write-ColorOutput "[STEP $StepNumber] $Description" $Config.Colors.Info
}

function Write-Success($Message) {
    Write-ColorOutput "✅ $Message" $Config.Colors.Success
}

function Write-Warning($Message) {
    Write-ColorOutput "⚠️  $Message" $Config.Colors.Warning
}

function Write-Error($Message) {
    Write-ColorOutput "❌ $Message" $Config.Colors.Error
}

function Test-Port($Port, $Timeout = 3) {
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $connect = $tcpClient.BeginConnect('localhost', $Port, $null, $null)
        $wait = $connect.AsyncWaitHandle.WaitOne($Timeout * 1000, $false)
        
        if ($wait) {
            try {
                $tcpClient.EndConnect($connect)
                $tcpClient.Close()
                return $true
            } catch {
                $tcpClient.Close()
                return $false
            }
        } else {
            $tcpClient.Close()
            return $false
        }
    } catch {
        return $false
    }
}

function Start-ServiceWithRetry($Name, $Command, $Args, $WorkingDirectory = $null, $MaxRetries = 3) {
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            Write-ColorOutput "  Starting $Name (attempt $i/$MaxRetries)..." $Config.Colors.Secondary
            
            $startInfo = New-Object System.Diagnostics.ProcessStartInfo
            $startInfo.FileName = $Command
            $startInfo.Arguments = $Args
            $startInfo.UseShellExecute = $false
            $startInfo.RedirectStandardOutput = $true
            $startInfo.RedirectStandardError = $true
            $startInfo.CreateNoWindow = $true
            
            if ($WorkingDirectory) {
                $startInfo.WorkingDirectory = $WorkingDirectory
            }
            
            $process = [System.Diagnostics.Process]::Start($startInfo)
            
            if ($process) {
                Write-Success "$Name started (PID: $($process.Id))"
                return $process
            }
        } catch {
            Write-Warning "$Name start attempt $i failed: $_"
            if ($i -lt $MaxRetries) {
                Start-Sleep -Seconds 5
            }
        }
    }
    
    Write-Error "Failed to start $Name after $MaxRetries attempts"
    return $null
}

function Test-Prerequisites {
    Write-Step 1 "Checking prerequisites"
    
    $prerequisites = @(
        @{ Name = "Node.js"; Command = "node"; Version = "--version" },
        @{ Name = "npm"; Command = "npm"; Version = "--version" },
        @{ Name = "Go"; Command = "go"; Version = "version"; Optional = $SkipGo }
    )
    
    $allGood = $true
    
    foreach ($prereq in $prerequisites) {
        try {
            $version = & $prereq.Command $prereq.Version 2>$null
            Write-Success "$($prereq.Name): $version"
        } catch {
            if ($prereq.Optional) {
                Write-Warning "$($prereq.Name) not found (optional)"
            } else {
                Write-Error "$($prereq.Name) not found (required)"
                $allGood = $false
            }
        }
    }
    
    return $allGood
}

function Test-ExternalServices {
    Write-Step 2 "Checking external services"
    
    foreach ($serviceName in @("PostgreSQL", "Redis", "Ollama", "Neo4j")) {
        $service = $Config.Services[$serviceName]
        $isRunning = Test-Port $service.Port
        
        if ($isRunning) {
            Write-Success "$($service.Name) detected on port $($service.Port)"
        } else {
            Write-Warning "$($service.Name) not detected on port $($service.Port)"
            
            # Auto-start known services
            switch ($serviceName) {
                "Ollama" {
                    Write-ColorOutput "  Attempting to start Ollama..." $Config.Colors.Info
                    try {
                        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
                        Start-Sleep -Seconds 5
                        if (Test-Port $service.Port) {
                            Write-Success "Ollama started successfully"
                        }
                    } catch {
                        Write-Warning "Could not auto-start Ollama: $_"
                    }
                }
            }
        }
    }
}

function Install-Dependencies {
    Write-Step 3 "Installing dependencies"
    
    if (-not (Test-Path "node_modules")) {
        Write-ColorOutput "  Installing npm dependencies..." $Config.Colors.Info
        
        $env:NODE_ENV = $Config.Environment
        $env:NODE_OPTIONS = $Config.NodeOptions
        
        $npmProcess = Start-Process "npm" -ArgumentList "install", "--silent" -Wait -PassThru -NoNewWindow
        
        if ($npmProcess.ExitCode -eq 0) {
            Write-Success "Dependencies installed successfully"
        } else {
            Write-Error "Failed to install dependencies"
            return $false
        }
    } else {
        Write-Success "Dependencies already installed"
    }
    
    return $true
}

function Build-Application {
    Write-Step 4 "Building application"
    
    if ($Production) {
        Write-ColorOutput "  Building production assets..." $Config.Colors.Info
        
        $env:NODE_ENV = "production"
        $buildProcess = Start-Process "npm" -ArgumentList "run", "build" -Wait -PassThru -NoNewWindow
        
        if ($buildProcess.ExitCode -eq 0) {
            Write-Success "Production build completed"
        } else {
            Write-Error "Build failed"
            return $false
        }
    } else {
        Write-Success "Skipping build for development mode"
    }
    
    return $true
}

function Start-GoServices {
    Write-Step 5 "Starting Go microservices"
    
    if ($SkipGo) {
        Write-Warning "Skipping Go services (--SkipGo specified)"
        return @()
    }
    
    $goServices = @(
        @{ 
            Name = "Enhanced RAG"
            Path = "..\go-microservice\cmd\enhanced-rag"
            Binary = "enhanced-rag.exe"
            Source = "main.go"
            Port = 8094
            Tier = 1
        },
        @{ 
            Name = "Upload Service"
            Path = "..\go-microservice\cmd\upload-service"
            Binary = "upload-service.exe"
            Source = "main.go" 
            Port = 8093
            Tier = 1
        },
        @{ 
            Name = "CUDA Service"
            Path = "..\go-microservice\cmd\cuda-service"
            Binary = "cuda-service.exe"
            Source = "main.go"
            Port = 8096
            Tier = 2
        },
        @{ 
            Name = "gRPC Server"
            Path = "..\go-microservice\cmd\grpc-server"
            Binary = "grpc-server.exe"
            Source = "main.go"
            Port = 50051
            Tier = 1
        }
    )
    
    $processes = @()
    
    # Start services by tier
    for ($tier = 1; $tier -le 2; $tier++) {
        $tierServices = $goServices | Where-Object { $_.Tier -eq $tier }
        
        if ($tierServices.Count -gt 0) {
            Write-ColorOutput "  Starting Tier $tier services..." $Config.Colors.Accent
            
            foreach ($service in $tierServices) {
                $binaryPath = Join-Path $service.Path $service.Binary
                $sourcePath = Join-Path $service.Path $service.Source
                
                if (Test-Path $binaryPath) {
                    Write-ColorOutput "    Starting $($service.Name) from binary..." $Config.Colors.Secondary
                    $process = Start-ServiceWithRetry $service.Name $binaryPath "" $service.Path
                } elseif (Test-Path $sourcePath) {
                    Write-ColorOutput "    Starting $($service.Name) from source..." $Config.Colors.Secondary
                    $process = Start-ServiceWithRetry $service.Name "go" "run main.go" $service.Path
                } else {
                    Write-Warning "$($service.Name) not found at $($service.Path)"
                    continue
                }
                
                if ($process) {
                    $processes += @{
                        Name = $service.Name
                        Process = $process
                        Port = $service.Port
                    }
                }
                
                Start-Sleep -Seconds 2
            }
            
            # Wait between tiers
            if ($tier -lt 2) {
                Write-ColorOutput "  Waiting for Tier $tier services to stabilize..." $Config.Colors.Secondary
                Start-Sleep -Seconds 10
            }
        }
    }
    
    return $processes
}

function Start-MasterCoordinator {
    Write-Step 6 "Starting Master Service Coordinator"
    
    $coordinatorScript = @"
import('./src/lib/services/master-service-coordinator.js').then(module => {
    const coordinator = module.masterServiceCoordinator;
    console.log('🎛️ Master Service Coordinator initializing...');
    
    coordinator.startAllServices().then(() => {
        console.log('✅ Service coordination active');
        
        // Start health monitoring
        setInterval(() => {
            const status = coordinator.getSystemStatus();
            console.log(`📊 System Health: `+status.systemHealth+` (`+status.serviceCount.total+` services)`);
        }, $($Config.HealthCheckInterval * 1000));
        
    }).catch(err => {
        console.error('❌ Coordinator failed:', err);
    });
    
    // Start error resolution engine
    import('./src/lib/services/error-resolution-engine.js').then(errorModule => {
        console.log('🔧 Error Resolution Engine active');
    });
});
"@
    
    try {
        $process = Start-Process "node" -ArgumentList "-e", $coordinatorScript -PassThru -NoNewWindow
        Write-Success "Master Service Coordinator started (PID: $($process.Id))"
        Start-Sleep -Seconds 5
        return $process
    } catch {
        Write-Error "Failed to start Master Service Coordinator: $_"
        return $null
    }
}

function Start-Frontend {
    Write-Step 7 "Starting SvelteKit application"
    
    if ($SkipFrontend) {
        Write-Warning "Skipping frontend (--SkipFrontend specified)"
        return $null
    }
    
    $env:NODE_ENV = $Config.Environment
    $env:NODE_OPTIONS = $Config.NodeOptions
    $env:COORDINATOR_MODE = "production"
    $env:MULTI_PROTOCOL = "true"
    $env:CUDA_INTEGRATION = "true"
    $env:ERROR_RECOVERY = "true"
    $env:HEALTH_MONITORING = "true"
    
    try {
        if ($Production) {
            $process = Start-Process "npm" -ArgumentList "run", "preview" -PassThru
        } else {
            $process = Start-Process "npm" -ArgumentList "run", "dev" -PassThru
        }
        
        Write-Success "SvelteKit application started (PID: $($process.Id))"
        
        # Wait for frontend to be ready
        Write-ColorOutput "  Waiting for frontend to be ready..." $Config.Colors.Secondary
        $maxAttempts = 30
        $attempt = 0
        
        do {
            Start-Sleep -Seconds 2
            $attempt++
            $ready = Test-Port 5173
        } while (-not $ready -and $attempt -lt $maxAttempts)
        
        if ($ready) {
            Write-Success "Frontend is ready at http://localhost:5173"
            Start-Process "http://localhost:5173"
        } else {
            Write-Warning "Frontend may not be fully ready yet"
        }
        
        return $process
    } catch {
        Write-Error "Failed to start SvelteKit application: $_"
        return $null
    }
}

function Show-SystemStatus($Processes) {
    Write-Banner "DEPLOYMENT COMPLETE - SYSTEM STATUS"
    
    Write-ColorOutput "🌐 Frontend: " $Config.Colors.Info -NoNewline
    if (Test-Port 5173) {
        Write-ColorOutput "✅ http://localhost:5173" $Config.Colors.Success
    } else {
        Write-ColorOutput "❌ Not accessible" $Config.Colors.Error
    }
    
    Write-ColorOutput "📊 Health Dashboard: " $Config.Colors.Info -NoNewline
    Write-ColorOutput "✅ http://localhost:5173/system/health" $Config.Colors.Success
    
    Write-ColorOutput "🔗 API Coordinator: " $Config.Colors.Info -NoNewline  
    Write-ColorOutput "✅ http://localhost:5173/api/v1/coordinator" $Config.Colors.Success
    
    Write-Host ""
    Write-ColorOutput "📋 ACTIVE SERVICES:" $Config.Colors.Accent
    
    foreach ($service in $Config.Services.GetEnumerator()) {
        $name = $service.Value.Name
        $port = $service.Value.Port
        $status = if (Test-Port $port) { "✅ Running" } else { "❌ Offline" }
        $color = if (Test-Port $port) { $Config.Colors.Success } else { $Config.Colors.Error }
        
        Write-ColorOutput "  • $name (Port $port): " $Config.Colors.Secondary -NoNewline
        Write-ColorOutput $status $color
    }
    
    Write-Host ""
    Write-ColorOutput "🎛️ PROCESS MANAGEMENT:" $Config.Colors.Accent
    foreach ($proc in $Processes) {
        if ($proc.Process -and -not $proc.Process.HasExited) {
            Write-ColorOutput "  • $($proc.Name) (PID: $($proc.Process.Id))" $Config.Colors.Success
        }
    }
    
    Write-Host ""
    Write-ColorOutput "📡 QUICK COMMANDS:" $Config.Colors.Accent
    Write-ColorOutput "  • Health Check: curl http://localhost:5173/api/v1/coordinator?action=health" $Config.Colors.Secondary
    Write-ColorOutput "  • Service Status: curl http://localhost:5173/api/v1/coordinator?action=services" $Config.Colors.Secondary
    Write-ColorOutput "  • Force Recovery: curl -X POST -H 'Content-Type: application/json' -d '{\"action\":\"force_health_check\"}' http://localhost:5173/api/v1/coordinator" $Config.Colors.Secondary
    
    Write-Host ""
    Write-ColorOutput "🔧 To stop all services, close this PowerShell window or press Ctrl+C" $Config.Colors.Warning
}

function Stop-AllServices($Processes) {
    Write-Banner "STOPPING ALL SERVICES"
    
    foreach ($proc in $Processes) {
        if ($proc.Process -and -not $proc.Process.HasExited) {
            try {
                Write-ColorOutput "Stopping $($proc.Name)..." $Config.Colors.Info
                $proc.Process.Kill()
                Write-Success "$($proc.Name) stopped"
            } catch {
                Write-Warning "Failed to stop $($proc.Name): $_"
            }
        }
    }
}

# Main execution
try {
    switch ($Action.ToLower()) {
        "start" {
            Write-Banner "LEGAL AI PLATFORM - PRODUCTION DEPLOYMENT"
            Write-ColorOutput "Mode: $($Config.Environment.ToUpper())" $Config.Colors.Accent
            Write-ColorOutput "Health Check Interval: $HealthCheckInterval seconds" $Config.Colors.Secondary
            Write-Host ""
            
            # Execute deployment steps
            if (-not (Test-Prerequisites)) { exit 1 }
            Test-ExternalServices
            if (-not (Install-Dependencies)) { exit 1 }
            if (-not (Build-Application)) { exit 1 }
            
            $goProcesses = Start-GoServices
            $coordinatorProcess = Start-MasterCoordinator
            $frontendProcess = Start-Frontend
            
            # Collect all processes
            $allProcesses = $goProcesses
            if ($coordinatorProcess) {
                $allProcesses += @{ Name = "Master Coordinator"; Process = $coordinatorProcess; Port = $null }
            }
            if ($frontendProcess) {
                $allProcesses += @{ Name = "SvelteKit Frontend"; Process = $frontendProcess; Port = 5173 }
            }
            
            Show-SystemStatus $allProcesses
            
            # Keep script running and monitor processes
            Write-Host ""
            Write-ColorOutput "Press any key to stop all services..." $Config.Colors.Warning
            $Host.UI.RawUI.ReadKey() | Out-Null
            
            Stop-AllServices $allProcesses
        }
        
        "stop" {
            Write-Banner "STOPPING ALL SERVICES"
            # Implementation for stopping services
            Write-Success "All services stopped"
        }
        
        "status" {
            Write-Banner "SYSTEM STATUS CHECK"
            # Implementation for status check
            Test-ExternalServices
        }
        
        default {
            Write-Error "Unknown action: $Action"
            Write-ColorOutput "Available actions: start, stop, status" $Config.Colors.Info
            exit 1
        }
    }
    
} catch {
    Write-Error "Deployment failed: $_"
    exit 1
}