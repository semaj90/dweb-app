# 📊 Legal AI System Status Monitor
# Quick health check and status overview

param(
    [switch]$Detailed = $false,
    [switch]$Logs = $false,
    [switch]$Performance = $false
)

$Host.UI.RawUI.WindowTitle = "📊 Legal AI System Monitor"

Write-Host "📊 LEGAL AI SYSTEM STATUS MONITOR" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Color scheme
$successColor = "Green"
$warningColor = "Yellow"
$errorColor = "Red"
$infoColor = "White"

# Service Status Check Function
function Get-ServiceStatus {
    param($ServiceName, $URL, $ExpectedResponse = 200)

    try {
        $response = Invoke-WebRequest -Uri $URL -TimeoutSec 3 -UseBasicParsing
        if ($response.StatusCode -eq $ExpectedResponse) {
            return @{ Status = "✅ HEALTHY"; Color = $successColor; Details = "HTTP $($response.StatusCode)" }
        } else {
            return @{ Status = "⚠️ WARNING"; Color = $warningColor; Details = "HTTP $($response.StatusCode)" }
        }
    } catch {
        return @{ Status = "❌ DOWN"; Color = $errorColor; Details = $_.Exception.Message.Split("`n")[0] }
    }
}

# Docker Container Status Function
function Get-DockerStatus {
    param($ContainerName)

    try {
        $container = docker ps --filter "name=$ContainerName" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>$null
        if ($container -and $container -notmatch "NAMES") {
            $status = ($container -split "`n" | Select-Object -Last 1 -Skip 1).Split("`t")[1]
            if ($status -match "Up") {
                return @{ Status = "✅ RUNNING"; Color = $successColor; Details = $status }
            } else {
                return @{ Status = "⚠️ STOPPED"; Color = $warningColor; Details = $status }
            }
        } else {
            return @{ Status = "❌ NOT FOUND"; Color = $errorColor; Details = "Container not exists" }
        }
    } catch {
        return @{ Status = "❌ ERROR"; Color = $errorColor; Details = "Docker unavailable" }
    }
}

# Process Status Function
function Get-ProcessStatus {
    param($ProcessName)

    try {
        $process = Get-Process $ProcessName -ErrorAction SilentlyContinue
        if ($process) {
            $cpuUsage = [math]::Round((Get-Counter "\Process($ProcessName)\% Processor Time" -ErrorAction SilentlyContinue).CounterSamples.CookedValue, 1)
            return @{ Status = "✅ RUNNING"; Color = $successColor; Details = "PID: $($process.Id), CPU: ${cpuUsage}%" }
        } else {
            return @{ Status = "❌ NOT RUNNING"; Color = $errorColor; Details = "Process not found" }
        }
    } catch {
        return @{ Status = "❌ ERROR"; Color = $errorColor; Details = "Cannot check process" }
    }
}

# Database Connection Test
function Test-DatabaseConnection {
    try {
        # Test PostgreSQL
        $env:PGPASSWORD = "legal_ai_secure_2024"
        docker exec legal-ai-postgres psql -U postgres -d legal_ai_db -c "SELECT 1;" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{ Status = "✅ CONNECTED"; Color = $successColor; Details = "PostgreSQL + pgvector ready" }
        } else {
            return @{ Status = "❌ CONNECTION FAILED"; Color = $errorColor; Details = "Cannot connect to database" }
        }
    } catch {
        return @{ Status = "❌ ERROR"; Color = $errorColor; Details = $_.Exception.Message }
    }
}

# AI Model Status
function Get-AIModelStatus {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $models = ($response.Content | ConvertFrom-Json).models
            $modelCount = $models.Count
            return @{ Status = "✅ $modelCount MODELS"; Color = $successColor; Details = ($models.name -join ", ") }
        } else {
            return @{ Status = "⚠️ API ERROR"; Color = $warningColor; Details = "Cannot fetch models" }
        }
    } catch {
        return @{ Status = "❌ UNAVAILABLE"; Color = $errorColor; Details = "Ollama not responding" }
    }
}

# System Resources
function Get-SystemResources {
    $cpu = Get-CimInstance -ClassName Win32_Processor | Measure-Object -Property LoadPercentage -Average
    $memory = Get-CimInstance -ClassName Win32_OperatingSystem
    $memoryUsed = [math]::Round((($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / $memory.TotalVisibleMemorySize) * 100, 1)

    return @{
        CPU = "$([math]::Round($cpu.Average, 1))%"
        Memory = "${memoryUsed}%"
        FreeMemoryGB = [math]::Round($memory.FreePhysicalMemory / 1MB, 1)
    }
}

# Main Status Display
Write-Host "🔍 CORE SERVICES STATUS" -ForegroundColor Cyan

# Check each service
$services = @(
    @{ Name = "SvelteKit Frontend"; Type = "HTTP"; Target = "http://localhost:5173" }
    @{ Name = "Ollama API"; Type = "HTTP"; Target = "http://localhost:11434/api/tags" }
    @{ Name = "Qdrant Vector DB"; Type = "HTTP"; Target = "http://localhost:6333/health" }
    @{ Name = "PostgreSQL Container"; Type = "Docker"; Target = "legal-ai-postgres" }
    @{ Name = "Qdrant Container"; Type = "Docker"; Target = "legal-ai-qdrant" }
)

foreach ($service in $services) {
    $result = switch ($service.Type) {
        "HTTP" { Get-ServiceStatus -ServiceName $service.Name -URL $service.Target }
        "Docker" { Get-DockerStatus -ContainerName $service.Target }
    }

    Write-Host "   $($service.Name.PadRight(25)): $($result.Status)" -ForegroundColor $result.Color
    if ($Detailed) {
        Write-Host "      └─ $($result.Details)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "🗄️ DATABASE STATUS" -ForegroundColor Cyan
$dbResult = Test-DatabaseConnection
Write-Host "   PostgreSQL + pgvector".PadRight(26) + ": $($dbResult.Status)" -ForegroundColor $dbResult.Color
if ($Detailed) {
    Write-Host "      └─ $($dbResult.Details)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🤖 AI MODELS STATUS" -ForegroundColor Cyan
$aiResult = Get-AIModelStatus
Write-Host "   Ollama Models".PadRight(26) + ": $($aiResult.Status)" -ForegroundColor $aiResult.Color
if ($Detailed) {
    Write-Host "      └─ $($aiResult.Details)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "📊 SYSTEM RESOURCES" -ForegroundColor Cyan
$resources = Get-SystemResources
Write-Host "   CPU Usage".PadRight(26) + ": $($resources.CPU)" -ForegroundColor $(if([int]($resources.CPU -replace '%','') -gt 80){$errorColor}elseif([int]($resources.CPU -replace '%','') -gt 60){$warningColor}else{$successColor})
Write-Host "   Memory Usage".PadRight(26) + ": $($resources.Memory)" -ForegroundColor $(if([int]($resources.Memory -replace '%','') -gt 85){$errorColor}elseif([int]($resources.Memory -replace '%','') -gt 70){$warningColor}else{$successColor})
Write-Host "   Free Memory".PadRight(26) + ": $($resources.FreeMemoryGB) GB" -ForegroundColor $infoColor

# Performance Metrics (if requested)
if ($Performance) {
    Write-Host ""
    Write-Host "⚡ PERFORMANCE METRICS" -ForegroundColor Cyan

    # Docker stats
    try {
        $dockerStats = docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>$null
        if ($dockerStats) {
            Write-Host "   Docker Container Usage:" -ForegroundColor $infoColor
            $dockerStats | ForEach-Object {
                if ($_ -notmatch "CONTAINER") {
                    Write-Host "     $_" -ForegroundColor Gray
                }
            }
        }
    } catch {
        Write-Host "   Docker stats unavailable" -ForegroundColor $warningColor
    }

    # Disk usage
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 }
    Write-Host "   Disk Usage:" -ForegroundColor $infoColor
    $disk | ForEach-Object {
        $usedPercent = [math]::Round((($_.Size - $_.FreeSpace) / $_.Size) * 100, 1)
        $freeGB = [math]::Round($_.FreeSpace / 1GB, 1)
        Write-Host "     Drive $($_.DeviceID) ${usedPercent}% used, ${freeGB}GB free" -ForegroundColor Gray
    }
}

# Log Viewing (if requested)
if ($Logs) {
    Write-Host ""
    Write-Host "📋 RECENT LOGS" -ForegroundColor Cyan

    Write-Host "   PostgreSQL Logs (last 10 lines):" -ForegroundColor $infoColor
    try {
        docker logs legal-ai-postgres --tail 10 2>$null | ForEach-Object {
            Write-Host "     $_" -ForegroundColor Gray
        }
    } catch {
        Write-Host "     No logs available" -ForegroundColor $warningColor
    }

    Write-Host "   Qdrant Logs (last 10 lines):" -ForegroundColor $infoColor
    try {
        docker logs legal-ai-qdrant --tail 10 2>$null | ForEach-Object {
            Write-Host "     $_" -ForegroundColor Gray
        }
    } catch {
        Write-Host "     No logs available" -ForegroundColor $warningColor
    }
}

# Quick Actions
Write-Host ""
Write-Host "🛠️ QUICK ACTIONS" -ForegroundColor Cyan
Write-Host "   Launch System    : .\one-click-legal-ai-launcher.ps1" -ForegroundColor $infoColor
Write-Host "   Reset System     : .\one-click-legal-ai-launcher.ps1 -Reset" -ForegroundColor $infoColor
Write-Host "   View Detailed    : .\system-status-monitor.ps1 -Detailed" -ForegroundColor $infoColor
Write-Host "   View Performance : .\system-status-monitor.ps1 -Performance" -ForegroundColor $infoColor
Write-Host "   View Logs        : .\system-status-monitor.ps1 -Logs" -ForegroundColor $infoColor

Write-Host ""
Write-Host "📊 Status check completed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
