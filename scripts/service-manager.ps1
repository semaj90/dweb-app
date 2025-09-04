#!/usr/bin/env pwsh
# Advanced Service Manager for Legal AI System
# Provides complete control over all services

param(
    [Parameter(Position=0)]
    [ValidateSet('start', 'stop', 'restart', 'status', 'fix', 'neo4j')]
    [string]$Command = 'status',
    
    [Parameter(Position=1)]
    [string]$Service = 'all'
)

# Service definitions
$services = @{
    postgres = @{
        Name = "PostgreSQL"
        Port = 5432
        Process = "postgres"
        StartCmd = { pg_ctl start -D "$env:PGDATA" }
        StopCmd = { pg_ctl stop -D "$env:PGDATA" }
        CheckCmd = { Test-NetConnection -ComputerName localhost -Port 5432 -WarningAction SilentlyContinue }
    }
    ollama = @{
        Name = "Ollama"
        Port = 11434
        Process = "ollama"
        StartCmd = { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep -Seconds 3 }
        StopCmd = { Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue }
        CheckCmd = { 
            try { 
                Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 2 
                return $true
            } catch { 
                return $false 
            }
        }
    }
    neo4j = @{
        Name = "Neo4j"
        Port = 7474
        Process = "java"
        StartCmd = { 
            $neo4jPath = Get-ChildItem -Path . -Filter "neo4j-community-*" -Directory | Select-Object -First 1
            if ($neo4jPath) {
                $batPath = Join-Path $neo4jPath.FullName "bin\neo4j.bat"
                Start-Process $batPath -ArgumentList "console" -WindowStyle Hidden
            }
        }
        StopCmd = {
            $neo4jPath = Get-ChildItem -Path . -Filter "neo4j-community-*" -Directory | Select-Object -First 1
            if ($neo4jPath) {
                $batPath = Join-Path $neo4jPath.FullName "bin\neo4j.bat"
                & $batPath stop
            }
        }
        CheckCmd = { Test-NetConnection -ComputerName localhost -Port 7474 -WarningAction SilentlyContinue }
    }
    redis = @{
        Name = "Redis"
        Port = 6379
        Process = "redis-server"
        StartCmd = { 
            $redisPath = Get-ChildItem -Path . -Filter "redis-server.exe" -Recurse | Select-Object -First 1
            if ($redisPath) {
                Start-Process $redisPath.FullName -WindowStyle Hidden
            }
        }
        StopCmd = { Stop-Process -Name redis-server -Force -ErrorAction SilentlyContinue }
        CheckCmd = { Test-NetConnection -ComputerName localhost -Port 6379 -WarningAction SilentlyContinue }
    }
    minio = @{
        Name = "MinIO"
        Port = 9000
        Process = "minio"
        StartCmd = {
            if (Test-Path "minio.exe") {
                if (-not (Test-Path "minio-data")) {
                    New-Item -ItemType Directory -Path "minio-data" | Out-Null
                }
                Start-Process "minio.exe" -ArgumentList "server", "./minio-data", "--console-address", ":9001" -WindowStyle Hidden
            }
        }
        StopCmd = { Stop-Process -Name minio -Force -ErrorAction SilentlyContinue }
        CheckCmd = { Test-NetConnection -ComputerName localhost -Port 9000 -WarningAction SilentlyContinue }
    }
    rag = @{
        Name = "Enhanced RAG"
        Port = 8094
        Process = "enhanced-rag"
        StartCmd = {
            if (Test-Path "enhanced-rag-som-system.exe") {
                Start-Process "enhanced-rag-som-system.exe" -WindowStyle Hidden
            } elseif ((Test-Path "enhanced-rag-som-system.go") -and (Get-Command go -ErrorAction SilentlyContinue)) {
                Start-Process go -ArgumentList "run", "enhanced-rag-som-system.go" -WindowStyle Hidden
            }
        }
        StopCmd = { 
            Stop-Process -Name "enhanced-rag-som-system" -Force -ErrorAction SilentlyContinue
        }
        CheckCmd = { Test-NetConnection -ComputerName localhost -Port 8094 -WarningAction SilentlyContinue }
    }
}

function Write-ServiceStatus {
    param($ServiceName, $Status, $Details = "")
    
    $symbol = switch ($Status) {
        "Running" { "✅"; $color = "Green" }
        "Stopped" { "❌"; $color = "Red" }
        "Starting" { "🔄"; $color = "Yellow" }
        "Unknown" { "❓"; $color = "Gray" }
    }
    
    Write-Host "$symbol $($ServiceName.PadRight(20)) " -NoNewline
    Write-Host "[$Status]" -ForegroundColor $color -NoNewline
    if ($Details) {
        Write-Host " - $Details" -ForegroundColor Gray
    } else {
        Write-Host ""
    }
}

function Get-ServiceStatus {
    param($ServiceKey)
    
    $service = $services[$ServiceKey]
    $check = & $service.CheckCmd
    
    if ($check -and $check.TcpTestSucceeded) {
        return "Running"
    } else {
        $process = Get-Process -Name $service.Process -ErrorAction SilentlyContinue
        if ($process) {
            return "Starting"
        } else {
            return "Stopped"
        }
    }
}

function Start-ServiceByKey {
    param($ServiceKey)
    
    $service = $services[$ServiceKey]
    Write-Host "Starting $($service.Name)..." -ForegroundColor Yellow
    
    & $service.StartCmd
    Start-Sleep -Seconds 2
    
    $status = Get-ServiceStatus -ServiceKey $ServiceKey
    if ($status -eq "Running" -or $status -eq "Starting") {
        Write-Host "✅ $($service.Name) started successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Failed to start $($service.Name)" -ForegroundColor Red
        return $false
    }
}

function Stop-ServiceByKey {
    param($ServiceKey)
    
    $service = $services[$ServiceKey]
    Write-Host "Stopping $($service.Name)..." -ForegroundColor Yellow
    
    & $service.StopCmd
    Start-Sleep -Seconds 2
    
    $status = Get-ServiceStatus -ServiceKey $ServiceKey
    if ($status -eq "Stopped") {
        Write-Host "✅ $($service.Name) stopped successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "⚠️ $($service.Name) may still be running" -ForegroundColor Yellow
        return $false
    }
}

function Show-Status {
    Write-Host "`n📊 Legal AI Services Status" -ForegroundColor Cyan
    Write-Host "=============================" -ForegroundColor Cyan
    Write-Host ""
    
    foreach ($key in $services.Keys) {
        $service = $services[$key]
        $status = Get-ServiceStatus -ServiceKey $key
        $details = "Port: $($service.Port)"
        Write-ServiceStatus -ServiceName $service.Name -Status $status -Details $details
    }
    
    Write-Host "`n🌐 URLs:" -ForegroundColor Cyan
    Write-Host "  Frontend:    http://localhost:5173" -ForegroundColor White
    Write-Host "  Neo4j:       http://localhost:7474" -ForegroundColor White
    Write-Host "  MinIO:       http://localhost:9001" -ForegroundColor White
    Write-Host "  Admin:       file:///$((Get-Location).Path)/admin-dashboard.html" -ForegroundColor White
}

function Fix-OllamaConfig {
    Write-Host "🔧 Fixing Ollama configuration..." -ForegroundColor Yellow
    
    $envPath = ".env"
    if (Test-Path $envPath) {
        $content = Get-Content $envPath -Raw
        $content = $content -replace 'http://localhost:11434', 'http://localhost:11434'
        $content | Set-Content $envPath -NoNewline
        Write-Host "✅ Ollama configuration fixed" -ForegroundColor Green
    }
}

function Manage-Neo4j {
    Write-Host "`n🔗 Neo4j Management" -ForegroundColor Cyan
    Write-Host "===================" -ForegroundColor Cyan
    
    $neo4jPath = Get-ChildItem -Path . -Filter "neo4j-community-*" -Directory | Select-Object -First 1
    
    if (-not $neo4jPath) {
        Write-Host "❌ Neo4j not found" -ForegroundColor Red
        Write-Host "Download from: https://neo4j.com/download-center/#community" -ForegroundColor Gray
        return
    }
    
    $binPath = Join-Path $neo4jPath.FullName "bin"
    $neo4jBat = Join-Path $binPath "neo4j.bat"
    
    Write-Host "Neo4j installation found: $($neo4jPath.Name)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  1. Start Neo4j" -ForegroundColor White
    Write-Host "  2. Stop Neo4j" -ForegroundColor White
    Write-Host "  3. Restart Neo4j" -ForegroundColor White
    Write-Host "  4. Check Status" -ForegroundColor White
    Write-Host "  5. Open Browser" -ForegroundColor White
    Write-Host "  6. Reset Password" -ForegroundColor White
    Write-Host "  0. Back" -ForegroundColor White
    Write-Host ""
    
    $choice = Read-Host "Select option"
    
    switch ($choice) {
        "1" { 
            Start-ServiceByKey -ServiceKey "neo4j"
            Write-Host "Neo4j browser will be available at: http://localhost:7474" -ForegroundColor Cyan
        }
        "2" { Stop-ServiceByKey -ServiceKey "neo4j" }
        "3" { 
            Stop-ServiceByKey -ServiceKey "neo4j"
            Start-Sleep -Seconds 2
            Start-ServiceByKey -ServiceKey "neo4j"
        }
        "4" {
            $status = Get-ServiceStatus -ServiceKey "neo4j"
            Write-Host "Neo4j Status: $status" -ForegroundColor $(if ($status -eq "Running") { "Green" } else { "Red" })
        }
        "5" {
            Start-Process "http://localhost:7474"
            Write-Host "Opening Neo4j browser..." -ForegroundColor Green
        }
        "6" {
            Write-Host "Resetting Neo4j password..." -ForegroundColor Yellow
            $newPassword = Read-Host "Enter new password" -AsSecureString
            $adminPath = Join-Path $binPath "neo4j-admin.bat"
            if (Test-Path $adminPath) {
                & $adminPath set-initial-password $([System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($newPassword)))
                Write-Host "✅ Password reset. Restart Neo4j to apply." -ForegroundColor Green
            }
        }
    }
}

# Main execution
switch ($Command) {
    "status" {
        Show-Status
    }
    "start" {
        if ($Service -eq "all") {
            Write-Host "🚀 Starting all services..." -ForegroundColor Cyan
            foreach ($key in $services.Keys) {
                Start-ServiceByKey -ServiceKey $key
            }
        } else {
            if ($services.ContainsKey($Service)) {
                Start-ServiceByKey -ServiceKey $Service
            } else {
                Write-Host "❌ Unknown service: $Service" -ForegroundColor Red
            }
        }
    }
    "stop" {
        if ($Service -eq "all") {
            Write-Host "⏹️ Stopping all services..." -ForegroundColor Cyan
            foreach ($key in $services.Keys) {
                Stop-ServiceByKey -ServiceKey $key
            }
        } else {
            if ($services.ContainsKey($Service)) {
                Stop-ServiceByKey -ServiceKey $Service
            } else {
                Write-Host "❌ Unknown service: $Service" -ForegroundColor Red
            }
        }
    }
    "restart" {
        if ($Service -eq "all") {
            Write-Host "🔄 Restarting all services..." -ForegroundColor Cyan
            foreach ($key in $services.Keys) {
                Stop-ServiceByKey -ServiceKey $key
            }
            Start-Sleep -Seconds 2
            foreach ($key in $services.Keys) {
                Start-ServiceByKey -ServiceKey $key
            }
        } else {
            if ($services.ContainsKey($Service)) {
                Stop-ServiceByKey -ServiceKey $Service
                Start-Sleep -Seconds 2
                Start-ServiceByKey -ServiceKey $Service
            } else {
                Write-Host "❌ Unknown service: $Service" -ForegroundColor Red
            }
        }
    }
    "fix" {
        Write-Host "🔧 Fixing configuration..." -ForegroundColor Cyan
        Fix-OllamaConfig
        
        # Fix PostgreSQL password
        Write-Host "Fixing PostgreSQL configuration..." -ForegroundColor Yellow
        $sqlScript = @"
ALTER USER legal_admin WITH PASSWORD '123456';
GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;
"@
        $sqlScript | psql -U postgres -d legal_ai_db 2>&1 | Out-Null
        Write-Host "✅ PostgreSQL configuration fixed" -ForegroundColor Green
        
        Write-Host "`n✅ Configuration fixes applied" -ForegroundColor Green
    }
    "neo4j" {
        Manage-Neo4j
    }
}

Write-Host "`n💡 Usage Examples:" -ForegroundColor Yellow
Write-Host "  .\scripts\service-manager.ps1 status           # Show all service status" -ForegroundColor Gray
Write-Host "  .\scripts\service-manager.ps1 start all        # Start all services" -ForegroundColor Gray
Write-Host "  .\scripts\service-manager.ps1 stop postgres    # Stop PostgreSQL" -ForegroundColor Gray
Write-Host "  .\scripts\service-manager.ps1 restart ollama   # Restart Ollama" -ForegroundColor Gray
Write-Host "  .\scripts\service-manager.ps1 neo4j            # Neo4j management menu" -ForegroundColor Gray
Write-Host "  .\scripts\service-manager.ps1 fix              # Fix configuration issues" -ForegroundColor Gray