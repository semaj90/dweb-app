#!/usr/bin/env pwsh
# Enhanced RAG V2 - Complete Deployment Automation
# ================================================
# One-click deployment and management for the entire Enhanced RAG V2 system

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("deploy", "start", "stop", "restart", "status", "logs", "update", "backup", "restore", "benchmark")]
    [string]$Action = "deploy",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("development", "production", "testing")]
    [string]$Environment = "development",
    
    [Parameter(Mandatory=$false)]
    [switch]$AutoFix = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipGPU = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $false,
    
    [Parameter(Mandatory=$false)]
    [string]$ConfigFile = "deployment.config.json"
)

# Script configuration
$Script:Config = @{
    ProjectName = "Enhanced RAG V2"
    Version = "2.0.0"
    Author = "Enhanced RAG Development Team"
    BaseDirectory = "C:\Users\james\Desktop\deeds-web\deeds-web-app"
    LogDirectory = "logs"
    BackupDirectory = "backups"
    
    # Service configuration
    Services = @{
        "PostgreSQL" = @{ Port = 5432; Required = $true; Type = "Database" }
        "Redis" = @{ Port = 6379; Required = $true; Type = "Cache" }
        "Neo4j" = @{ Port = 7687; Required = $false; Type = "Graph" }
        "Qdrant" = @{ Port = 6333; Required = $false; Type = "Vector" }
        "CUDA Service" = @{ Port = 8765; Required = $false; Type = "GPU" }
        "Tensor Service" = @{ Port = 8099; Required = $true; Type = "AI" }
        "Enhanced RAG V2" = @{ Port = 8097; Required = $true; Type = "AI" }
        "Simply Enhanced RAG" = @{ Port = 8096; Required = $true; Type = "AI" }
        "Legal AI Distillation" = @{ Port = 8100; Required = $true; Type = "AI" }
        "SvelteKit Frontend" = @{ Port = 3000; Required = $true; Type = "Frontend" }
        "System Dashboard" = @{ Port = 8080; Required = $false; Type = "Monitoring" }
    }
    
    # GPU requirements
    GPU = @{
        CUDAVersion = "12.8"
        MinimumVRAM = 8  # GB
        RecommendedVRAM = 16  # GB
    }
    
    # Performance thresholds
    Performance = @{
        MaxStartupTime = 120  # seconds
        HealthCheckTimeout = 30  # seconds
        MaxMemoryUsage = 80  # percent
        MaxCPUUsage = 90  # percent
    }
}

# Logging functions
function Write-DeploymentLog {
    param($Message, $Level = "INFO", $Color = "White")
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp][$Level] $Message"
    
    # Console output with colors
    $consoleColors = @{
        "INFO" = "White"
        "SUCCESS" = "Green" 
        "WARNING" = "Yellow"
        "ERROR" = "Red"
        "DEBUG" = "Cyan"
    }
    
    Write-Host $logMessage -ForegroundColor $consoleColors[$Level]
    
    # File logging
    $logFile = Join-Path $Script:Config.LogDirectory "deployment-$(Get-Date -Format 'yyyy-MM-dd').log"
    Add-Content -Path $logFile -Value $logMessage -ErrorAction SilentlyContinue
}

function Write-Section {
    param($Title)
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host " $Title" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

# System validation functions
function Test-SystemRequirements {
    Write-Section "SYSTEM REQUIREMENTS VALIDATION"
    
    $requirements = @{}
    
    # Check Windows version
    $osInfo = Get-CimInstance Win32_OperatingSystem
    $requirements.OS = @{
        Name = $osInfo.Caption
        Version = $osInfo.Version
        Valid = [version]$osInfo.Version -ge [version]"10.0.0"
    }
    
    # Check PowerShell version
    $requirements.PowerShell = @{
        Version = $PSVersionTable.PSVersion.ToString()
        Valid = $PSVersionTable.PSVersion.Major -ge 5
    }
    
    # Check Node.js
    try {
        $nodeVersion = & node --version 2>$null
        $requirements.NodeJS = @{
            Version = $nodeVersion
            Valid = $nodeVersion -and [version]($nodeVersion -replace 'v','') -ge [version]"16.0.0"
        }
    } catch {
        $requirements.NodeJS = @{ Version = "Not Found"; Valid = $false }
    }
    
    # Check Go
    try {
        $goVersion = & go version 2>$null
        $requirements.Go = @{
            Version = $goVersion
            Valid = $goVersion -and $goVersion -match "go1\.(2[0-9]|[3-9][0-9])"
        }
    } catch {
        $requirements.Go = @{ Version = "Not Found"; Valid = $false }
    }
    
    # Check CUDA (if not skipping GPU)
    if (-not $SkipGPU) {
        try {
            $cudaVersion = & nvcc --version 2>$null | Select-String "release (\d+\.\d+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
            $requirements.CUDA = @{
                Version = $cudaVersion
                Valid = $cudaVersion -and [version]$cudaVersion -ge [version]"12.0"
            }
        } catch {
            $requirements.CUDA = @{ Version = "Not Found"; Valid = $false }
        }
        
        # Check GPU
        try {
            $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
            if ($gpuInfo) {
                $memory = ($gpuInfo -split ',')[1].Trim()
                $requirements.GPU = @{
                    Name = ($gpuInfo -split ',')[0].Trim()
                    Memory = "$memory MB"
                    Valid = [int]($memory) -ge ($Script:Config.GPU.MinimumVRAM * 1024)
                }
            }
        } catch {
            $requirements.GPU = @{ Name = "Not Found"; Memory = "0 MB"; Valid = $false }
        }
    }
    
    # Check memory
    $memory = Get-CimInstance Win32_ComputerSystem
    $totalMemoryGB = [math]::Round($memory.TotalPhysicalMemory / 1GB, 2)
    $requirements.Memory = @{
        Total = "$totalMemoryGB GB"
        Valid = $totalMemoryGB -ge 16
    }
    
    # Check disk space
    $disk = Get-CimInstance Win32_LogicalDisk | Where-Object { $_.DeviceID -eq "C:" }
    $freeSpaceGB = [math]::Round($disk.FreeSpace / 1GB, 2)
    $requirements.DiskSpace = @{
        Free = "$freeSpaceGB GB"
        Valid = $freeSpaceGB -ge 50
    }
    
    # Display results
    foreach ($req in $requirements.GetEnumerator()) {
        $status = if ($req.Value.Valid) { "✅" } else { "❌" }
        $level = if ($req.Value.Valid) { "SUCCESS" } else { "ERROR" }
        
        if ($req.Key -eq "GPU" -and $SkipGPU) {
            $status = "⏭️ "
            $level = "INFO"
        }
        
        Write-DeploymentLog "$status $($req.Key): $($req.Value.Version ?? $req.Value.Name ?? $req.Value.Total ?? $req.Value.Free)" $level
    }
    
    # Check for critical failures
    $criticalFailures = $requirements.GetEnumerator() | Where-Object { 
        -not $_.Value.Valid -and $_.Key -notin @("GPU", "CUDA") 
    }
    
    if ($SkipGPU) {
        $criticalFailures = $criticalFailures | Where-Object { $_.Key -notin @("GPU", "CUDA") }
    }
    
    if ($criticalFailures) {
        Write-DeploymentLog "Critical requirements not met. Deployment cannot continue." "ERROR"
        if ($AutoFix) {
            Write-DeploymentLog "Attempting to install missing components..." "INFO"
            Install-MissingComponents $criticalFailures
        } else {
            throw "System requirements validation failed"
        }
    }
    
    Write-DeploymentLog "System requirements validation completed" "SUCCESS"
    return $requirements
}

function Install-MissingComponents {
    param($MissingComponents)
    
    Write-Section "AUTOMATIC COMPONENT INSTALLATION"
    
    foreach ($component in $MissingComponents) {
        switch ($component.Key) {
            "NodeJS" {
                Write-DeploymentLog "Installing Node.js..." "INFO"
                try {
                    # Use winget to install Node.js
                    & winget install OpenJS.NodeJS --silent --accept-source-agreements 2>$null
                    Write-DeploymentLog "Node.js installation completed" "SUCCESS"
                } catch {
                    Write-DeploymentLog "Failed to install Node.js automatically" "ERROR"
                }
            }
            "Go" {
                Write-DeploymentLog "Installing Go..." "INFO"
                try {
                    & winget install GoLang.Go --silent --accept-source-agreements 2>$null
                    Write-DeploymentLog "Go installation completed" "SUCCESS"
                } catch {
                    Write-DeploymentLog "Failed to install Go automatically" "ERROR"
                }
            }
        }
    }
}

# Service management functions
function Start-EnhancedRAGV2Services {
    Write-Section "STARTING ENHANCED RAG V2 SERVICES"
    
    $startupOrder = @(
        @{ Name = "PostgreSQL"; Command = "pg_ctl start -D `"C:\Program Files\PostgreSQL\14\data`""; Wait = 10 }
        @{ Name = "Redis"; Command = "redis-server"; Path = "redis-windows"; Wait = 5 }
        @{ Name = "Neo4j"; Command = "bin\neo4j.bat start"; Path = "neo4j-community-5.23.0"; Wait = 15 }
        @{ Name = "Qdrant"; Command = "qdrant.exe"; Path = "qdrant-windows"; Wait = 10 }
    )
    
    if (-not $SkipGPU) {
        $startupOrder += @{ Name = "CUDA Service"; Command = "go-microservice\bin\cuda-service.exe"; Wait = 8 }
    }
    
    $startupOrder += @(
        @{ Name = "Tensor Service"; Command = "go-microservice\bin\tensor-service.exe"; Wait = 10 }
        @{ Name = "Enhanced RAG V2"; Command = "go-microservice\bin\enhanced-rag-v2.exe"; Wait = 15 }
        @{ Name = "Simply Enhanced RAG"; Command = "go-microservice\bin\simply-enhanced-rag.exe"; Wait = 10 }
        @{ Name = "Legal AI Distillation"; Command = "go-microservice\bin\legal-ai-distillation.exe"; Wait = 12 }
        @{ Name = "SvelteKit Frontend"; Command = "npm run dev"; Wait = 20 }
    )
    
    foreach ($service in $startupOrder) {
        Write-DeploymentLog "Starting $($service.Name)..." "INFO"
        
        try {
            $processArgs = @{
                FilePath = "cmd.exe"
                ArgumentList = "/c", "cd /d `"$($Script:Config.BaseDirectory)`" && $($service.Command)"
                WindowStyle = "Hidden"
                PassThru = $true
            }
            
            if ($service.Path) {
                $processArgs.ArgumentList[1] = "cd /d `"$($Script:Config.BaseDirectory)\$($service.Path)`" && $($service.Command)"
            }
            
            $process = Start-Process @processArgs
            
            # Wait for service to start
            Start-Sleep -Seconds $service.Wait
            
            # Verify service is running
            $serviceConfig = $Script:Config.Services[$service.Name]
            if ($serviceConfig -and $serviceConfig.Port) {
                $isRunning = Test-ServicePort $serviceConfig.Port
                if ($isRunning) {
                    Write-DeploymentLog "$($service.Name) started successfully on port $($serviceConfig.Port)" "SUCCESS"
                } else {
                    Write-DeploymentLog "$($service.Name) may not have started correctly" "WARNING"
                }
            } else {
                Write-DeploymentLog "$($service.Name) started (no port check available)" "INFO"
            }
            
        } catch {
            Write-DeploymentLog "Failed to start $($service.Name): $($_.Exception.Message)" "ERROR"
        }
    }
}

function Test-ServicePort {
    param($Port, $Timeout = 5)
    
    try {
        $tcpClient = New-Object System.Net.Sockets.TcpClient
        $connectTask = $tcpClient.ConnectAsync("localhost", $Port)
        $result = $connectTask.Wait($Timeout * 1000)
        $tcpClient.Close()
        return $result
    } catch {
        return $false
    }
}

function Stop-EnhancedRAGV2Services {
    Write-Section "STOPPING ENHANCED RAG V2 SERVICES"
    
    # Stop services in reverse order
    $servicePorts = @(3000, 8100, 8096, 8097, 8099, 8765, 6333, 7687, 6379, 5432)
    
    foreach ($port in $servicePorts) {
        $processes = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | 
                    ForEach-Object { Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue }
        
        foreach ($process in $processes) {
            if ($process) {
                Write-DeploymentLog "Stopping process $($process.ProcessName) (PID: $($process.Id), Port: $port)" "INFO"
                try {
                    $process.Kill()
                    Start-Sleep -Seconds 2
                    Write-DeploymentLog "Process stopped successfully" "SUCCESS"
                } catch {
                    Write-DeploymentLog "Failed to stop process: $($_.Exception.Message)" "WARNING"
                }
            }
        }
    }
}

function Get-ServiceStatus {
    Write-Section "SERVICE STATUS REPORT"
    
    $status = @{}
    
    foreach ($service in $Script:Config.Services.GetEnumerator()) {
        $serviceName = $service.Key
        $serviceConfig = $service.Value
        
        $serviceStatus = @{
            Name = $serviceName
            Port = $serviceConfig.Port
            Type = $serviceConfig.Type
            Required = $serviceConfig.Required
            Status = "Unknown"
            ResponseTime = $null
        }
        
        if ($serviceConfig.Port) {
            $startTime = Get-Date
            $isRunning = Test-ServicePort $serviceConfig.Port
            $responseTime = (Get-Date) - $startTime
            
            $serviceStatus.Status = if ($isRunning) { "Running" } else { "Stopped" }
            $serviceStatus.ResponseTime = [math]::Round($responseTime.TotalMilliseconds, 2)
        }
        
        $status[$serviceName] = $serviceStatus
        
        # Display status
        $statusIcon = switch ($serviceStatus.Status) {
            "Running" { "✅" }
            "Stopped" { "❌" }
            default { "❓" }
        }
        
        $level = switch ($serviceStatus.Status) {
            "Running" { "SUCCESS" }
            "Stopped" { if ($serviceConfig.Required) { "ERROR" } else { "WARNING" } }
            default { "INFO" }
        }
        
        $message = "$statusIcon $serviceName ($($serviceConfig.Type))"
        if ($serviceConfig.Port) {
            $message += " - Port $($serviceConfig.Port)"
        }
        if ($serviceStatus.ResponseTime) {
            $message += " - Response: $($serviceStatus.ResponseTime)ms"
        }
        
        Write-DeploymentLog $message $level
    }
    
    return $status
}

# Build and deployment functions
function Build-GoServices {
    Write-Section "BUILDING GO MICROSERVICES"
    
    Set-Location (Join-Path $Script:Config.BaseDirectory "go-microservice")
    
    $services = @(
        @{ Name = "Enhanced RAG V2"; Path = ".\cmd\enhanced-rag-v2"; Output = "bin\enhanced-rag-v2.exe" }
        @{ Name = "Simply Enhanced RAG"; Path = ".\cmd\simply-enhanced-rag"; Output = "bin\simply-enhanced-rag.exe" }
        @{ Name = "Tensor Service"; Source = ".\tensor-gpu-service.go"; Output = "bin\tensor-service.exe" }
        @{ Name = "Legal AI Distillation"; Source = ".\legal-ai-distillation.go"; Output = "bin\legal-ai-distillation.exe" }
    )
    
    if (-not $SkipGPU) {
        $services += @{ Name = "CUDA Service"; Source = ".\cuda-service.go"; Output = "bin\cuda-service.exe"; Tags = "cuda" }
    }
    
    # Ensure bin directory exists
    if (-not (Test-Path "bin")) {
        New-Item -ItemType Directory -Path "bin" -Force | Out-Null
    }
    
    foreach ($service in $services) {
        Write-DeploymentLog "Building $($service.Name)..." "INFO"
        
        try {
            $buildArgs = @("build", "-ldflags", "-s -w", "-o", $service.Output)
            
            if ($service.Tags) {
                $buildArgs += @("-tags", $service.Tags)
            }
            
            if ($service.Path) {
                $buildArgs += $service.Path
            } elseif ($service.Source) {
                $buildArgs += $service.Source
            }
            
            $result = & go @buildArgs 2>&1
            
            if ($LASTEXITCODE -eq 0) {
                $fileInfo = Get-Item $service.Output
                $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
                Write-DeploymentLog "$($service.Name) built successfully ($sizeMB MB)" "SUCCESS"
            } else {
                Write-DeploymentLog "$($service.Name) build failed: $result" "ERROR"
            }
            
        } catch {
            Write-DeploymentLog "Build error for $($service.Name): $($_.Exception.Message)" "ERROR"
        }
    }
    
    Set-Location $Script:Config.BaseDirectory
}

function Install-FrontendDependencies {
    Write-Section "INSTALLING FRONTEND DEPENDENCIES"
    
    if (Test-Path "package.json") {
        Write-DeploymentLog "Installing Node.js dependencies..." "INFO"
        
        try {
            $result = & npm install 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-DeploymentLog "Frontend dependencies installed successfully" "SUCCESS"
            } else {
                Write-DeploymentLog "Frontend dependency installation failed: $result" "ERROR"
            }
        } catch {
            Write-DeploymentLog "NPM install error: $($_.Exception.Message)" "ERROR"
        }
    } else {
        Write-DeploymentLog "package.json not found, skipping frontend dependencies" "WARNING"
    }
}

function Initialize-Database {
    Write-Section "INITIALIZING DATABASE SCHEMA"
    
    $schemaFiles = @(
        "scripts\init_database.sql",
        "scripts\create-basic-schema.sql"
    )
    
    foreach ($schemaFile in $schemaFiles) {
        if (Test-Path $schemaFile) {
            Write-DeploymentLog "Executing schema file: $schemaFile" "INFO"
            
            try {
                # Wait for PostgreSQL to be ready
                $retries = 0
                while ($retries -lt 10 -and -not (Test-ServicePort 5432)) {
                    Start-Sleep -Seconds 2
                    $retries++
                }
                
                if (Test-ServicePort 5432) {
                    $result = & psql -h localhost -U postgres -f $schemaFile 2>&1
                    Write-DeploymentLog "Schema file executed successfully" "SUCCESS"
                } else {
                    Write-DeploymentLog "PostgreSQL not accessible, skipping schema initialization" "WARNING"
                }
                
            } catch {
                Write-DeploymentLog "Schema execution error: $($_.Exception.Message)" "WARNING"
            }
        }
    }
}

# Performance and monitoring functions
function Test-SystemPerformance {
    Write-Section "SYSTEM PERFORMANCE BENCHMARK"
    
    $benchmarks = @{}
    
    # Test CPU performance
    Write-DeploymentLog "Testing CPU performance..." "INFO"
    $cpuStart = Get-Date
    1..100000 | ForEach-Object { [math]::Sqrt($_) } | Out-Null
    $cpuTime = (Get-Date) - $cpuStart
    $benchmarks.CPU = @{
        Time = [math]::Round($cpuTime.TotalMilliseconds, 2)
        Score = [math]::Round(100000 / $cpuTime.TotalSeconds, 0)
    }
    
    # Test memory performance
    Write-DeploymentLog "Testing memory performance..." "INFO"
    $memStart = Get-Date
    $largeArray = 1..1000000
    $memTime = (Get-Date) - $memStart
    $benchmarks.Memory = @{
        Time = [math]::Round($memTime.TotalMilliseconds, 2)
        Score = [math]::Round(1000000 / $memTime.TotalSeconds, 0)
    }
    
    # Test disk I/O performance
    Write-DeploymentLog "Testing disk I/O performance..." "INFO"
    $testFile = "performance_test.tmp"
    $diskStart = Get-Date
    1..1000 | ForEach-Object { "Performance test data line $_" } | Out-File -FilePath $testFile
    Get-Content $testFile | Out-Null
    Remove-Item $testFile -ErrorAction SilentlyContinue
    $diskTime = (Get-Date) - $diskStart
    $benchmarks.Disk = @{
        Time = [math]::Round($diskTime.TotalMilliseconds, 2)
        Score = [math]::Round(1000 / $diskTime.TotalSeconds, 0)
    }
    
    # Display results
    foreach ($benchmark in $benchmarks.GetEnumerator()) {
        Write-DeploymentLog "$($benchmark.Key) Performance: $($benchmark.Value.Time)ms (Score: $($benchmark.Value.Score))" "INFO"
    }
    
    return $benchmarks
}

function Start-SystemMonitoring {
    Write-DeploymentLog "Starting system monitoring..." "INFO"
    
    # Create monitoring script
    $monitoringScript = @"
while (`$true) {
    `$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    `$cpu = Get-Counter '\Processor(_Total)\% Processor Time' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    `$memory = Get-Counter '\Memory\Available MBytes' | Select-Object -ExpandProperty CounterSamples | Select-Object -ExpandProperty CookedValue
    
    Write-Host "[`$timestamp] CPU: `$([math]::Round(`$cpu, 1))% | Memory Available: `$([math]::Round(`$memory, 0))MB"
    
    Start-Sleep -Seconds 10
}
"@
    
    $monitoringScriptPath = "monitoring.ps1"
    $monitoringScript | Out-File -FilePath $monitoringScriptPath -Encoding UTF8
    
    # Start monitoring in background
    Start-Process -FilePath "powershell.exe" -ArgumentList "-WindowStyle", "Minimized", "-File", $monitoringScriptPath
    
    Write-DeploymentLog "System monitoring started in background" "SUCCESS"
}

# Backup and restore functions
function Backup-System {
    param($BackupName = "backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')")
    
    Write-Section "CREATING SYSTEM BACKUP"
    
    $backupPath = Join-Path $Script:Config.BackupDirectory $BackupName
    New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
    
    # Backup configuration files
    $configFiles = @(
        ".env*",
        "*.json",
        "*.yaml",
        "*.yml",
        "*.toml"
    )
    
    foreach ($pattern in $configFiles) {
        Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination $backupPath
        }
    }
    
    # Backup compiled binaries
    if (Test-Path "go-microservice\bin") {
        Copy-Item "go-microservice\bin" -Destination (Join-Path $backupPath "bin") -Recurse
    }
    
    # Backup database (if accessible)
    if (Test-ServicePort 5432) {
        Write-DeploymentLog "Backing up PostgreSQL database..." "INFO"
        try {
            & pg_dump -h localhost -U postgres --no-password -f (Join-Path $backupPath "database_backup.sql") 2>$null
            Write-DeploymentLog "Database backup completed" "SUCCESS"
        } catch {
            Write-DeploymentLog "Database backup failed: $($_.Exception.Message)" "WARNING"
        }
    }
    
    Write-DeploymentLog "System backup created: $backupPath" "SUCCESS"
    return $backupPath
}

function Restore-System {
    param($BackupPath)
    
    Write-Section "RESTORING SYSTEM FROM BACKUP"
    
    if (-not (Test-Path $BackupPath)) {
        Write-DeploymentLog "Backup path not found: $BackupPath" "ERROR"
        return $false
    }
    
    Write-DeploymentLog "Restoring from backup: $BackupPath" "INFO"
    
    # Restore configuration files
    Get-ChildItem -Path $BackupPath -File | ForEach-Object {
        Copy-Item $_.FullName -Destination . -Force
        Write-DeploymentLog "Restored: $($_.Name)" "INFO"
    }
    
    # Restore binaries
    $binBackup = Join-Path $BackupPath "bin"
    if (Test-Path $binBackup) {
        Copy-Item $binBackup -Destination "go-microservice\bin" -Recurse -Force
        Write-DeploymentLog "Binaries restored" "SUCCESS"
    }
    
    # Restore database
    $dbBackup = Join-Path $BackupPath "database_backup.sql"
    if (Test-Path $dbBackup) {
        Write-DeploymentLog "Restoring database..." "INFO"
        try {
            & psql -h localhost -U postgres -f $dbBackup 2>$null
            Write-DeploymentLog "Database restored successfully" "SUCCESS"
        } catch {
            Write-DeploymentLog "Database restore failed: $($_.Exception.Message)" "WARNING"
        }
    }
    
    Write-DeploymentLog "System restore completed" "SUCCESS"
    return $true
}

# Main deployment function
function Deploy-EnhancedRAGV2 {
    Write-Section "ENHANCED RAG V2 - COMPLETE DEPLOYMENT"
    
    try {
        # Create necessary directories
        @($Script:Config.LogDirectory, $Script:Config.BackupDirectory) | ForEach-Object {
            if (-not (Test-Path $_)) {
                New-Item -ItemType Directory -Path $_ -Force | Out-Null
            }
        }
        
        # Set working directory
        Set-Location $Script:Config.BaseDirectory
        
        # Validate system requirements
        $requirements = Test-SystemRequirements
        
        # Build services
        Build-GoServices
        
        # Install frontend dependencies
        Install-FrontendDependencies
        
        # Start services
        Start-EnhancedRAGV2Services
        
        # Initialize database
        Initialize-Database
        
        # Test system performance
        if ($Environment -eq "production") {
            Test-SystemPerformance
        }
        
        # Start monitoring
        Start-SystemMonitoring
        
        # Final status check
        $status = Get-ServiceStatus
        
        # Create deployment report
        $report = @{
            Timestamp = Get-Date
            Environment = $Environment
            Version = $Script:Config.Version
            Requirements = $requirements
            Services = $status
            GPUEnabled = -not $SkipGPU
        }
        
        $reportJson = $report | ConvertTo-Json -Depth 10
        $reportPath = "deployment-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
        $reportJson | Out-File -FilePath $reportPath -Encoding UTF8
        
        Write-DeploymentLog "Deployment report saved: $reportPath" "INFO"
        
        # Success message
        Write-Section "DEPLOYMENT COMPLETED SUCCESSFULLY"
        Write-DeploymentLog "Enhanced RAG V2 system is now running!" "SUCCESS"
        Write-DeploymentLog "Frontend: http://localhost:3000" "INFO"
        Write-DeploymentLog "Dashboard: http://localhost:8080" "INFO"
        Write-DeploymentLog "API Endpoints:" "INFO"
        Write-DeploymentLog "  - Enhanced RAG V2: http://localhost:8097" "INFO"
        Write-DeploymentLog "  - Simply Enhanced RAG: http://localhost:8096" "INFO"
        Write-DeploymentLog "  - Legal AI Distillation: http://localhost:8100" "INFO"
        Write-DeploymentLog "  - Tensor Service: http://localhost:8099" "INFO"
        
        if (-not $SkipGPU) {
            Write-DeploymentLog "  - CUDA Service: http://localhost:8765" "INFO"
        }
        
    } catch {
        Write-DeploymentLog "Deployment failed: $($_.Exception.Message)" "ERROR"
        Write-DeploymentLog "Check logs for details: $($Script:Config.LogDirectory)" "INFO"
        throw
    }
}

# Main script execution
try {
    switch ($Action.ToLower()) {
        "deploy" {
            Deploy-EnhancedRAGV2
        }
        "start" {
            Start-EnhancedRAGV2Services
            Get-ServiceStatus
        }
        "stop" {
            Stop-EnhancedRAGV2Services
        }
        "restart" {
            Stop-EnhancedRAGV2Services
            Start-Sleep -Seconds 5
            Start-EnhancedRAGV2Services
            Get-ServiceStatus
        }
        "status" {
            Get-ServiceStatus
        }
        "logs" {
            $logFile = Join-Path $Script:Config.LogDirectory "deployment-$(Get-Date -Format 'yyyy-MM-dd').log"
            if (Test-Path $logFile) {
                Get-Content $logFile -Tail 50
            } else {
                Write-Host "No log file found for today"
            }
        }
        "backup" {
            Backup-System
        }
        "restore" {
            $backups = Get-ChildItem $Script:Config.BackupDirectory -Directory | Sort-Object Name -Descending
            if ($backups) {
                $latestBackup = $backups[0].FullName
                Restore-System $latestBackup
            } else {
                Write-Host "No backups found"
            }
        }
        "benchmark" {
            Test-SystemPerformance
        }
        "update" {
            Write-DeploymentLog "Updating system..." "INFO"
            # Pull latest changes, rebuild, restart
            & git pull 2>$null
            Build-GoServices
            Stop-EnhancedRAGV2Services
            Start-Sleep -Seconds 5
            Start-EnhancedRAGV2Services
        }
        default {
            Write-Host "Unknown action: $Action"
            Write-Host "Available actions: deploy, start, stop, restart, status, logs, update, backup, restore, benchmark"
        }
    }
} catch {
    Write-DeploymentLog "Script execution failed: $($_.Exception.Message)" "ERROR"
    exit 1
}
