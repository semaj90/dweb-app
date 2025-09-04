#!/usr/bin/env pwsh
# Memory Optimizer Script for Legal AI System
# Monitors, logs, and optimizes system memory with crash prevention

param(
    [switch]$Monitor = $false,
    [switch]$Optimize = $false,
    [switch]$AutoOptimize = $false,
    [int]$Threshold = 80
)

$Global:LogPath = Join-Path $PSScriptRoot ".." "logs" "memory"
$Global:CrashLogPath = Join-Path $Global:LogPath "crash-prevention"

# Create log directories if they don't exist
if (-not (Test-Path $Global:LogPath)) {
    New-Item -ItemType Directory -Path $Global:LogPath -Force | Out-Null
}
if (-not (Test-Path $Global:CrashLogPath)) {
    New-Item -ItemType Directory -Path $Global:CrashLogPath -Force | Out-Null
}

function Get-MemoryStatus {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $computer = Get-CimInstance -ClassName Win32_ComputerSystem
    
    $totalMemory = [math]::Round($computer.TotalPhysicalMemory / 1GB, 2)
    $freeMemory = [math]::Round($os.FreePhysicalMemory / 1MB / 1024, 2)
    $usedMemory = [math]::Round($totalMemory - $freeMemory, 2)
    $usagePercent = [math]::Round(($usedMemory / $totalMemory) * 100, 2)
    
    # Get process count
    $processCount = (Get-Process).Count
    
    # Get top memory consumers
    $topProcesses = Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 5 | ForEach-Object {
        @{
            Name = $_.ProcessName
            Memory = [math]::Round($_.WorkingSet64 / 1MB, 2)
            CPU = [math]::Round($_.CPU, 2)
        }
    }
    
    return @{
        TotalGB = $totalMemory
        UsedGB = $usedMemory
        FreeGB = $freeMemory
        UsagePercent = $usagePercent
        ProcessCount = $processCount
        TopProcesses = $topProcesses
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
}

function Write-MemoryLog {
    param(
        [hashtable]$Status,
        [string]$Type = "INFO"
    )
    
    $logFile = Join-Path $Global:LogPath "memory-monitor-$(Get-Date -Format 'yyyy-MM-dd').log"
    $logEntry = "[$(Get-Date -Format 'HH:mm:ss')] [$Type] Memory: $($Status.UsedGB)GB/$($Status.TotalGB)GB ($($Status.UsagePercent)%) | Processes: $($Status.ProcessCount)"
    
    Add-Content -Path $logFile -Value $logEntry
    
    # Console output with color
    $color = switch ($Type) {
        "CRITICAL" { "Red" }
        "WARNING" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    
    Write-Host $logEntry -ForegroundColor $color
}

function Save-CrashPreventionLog {
    param(
        [hashtable]$Status
    )
    
    $crashLog = @{
        Timestamp = $Status.Timestamp
        MemoryStatus = @{
            TotalGB = $Status.TotalGB
            UsedGB = $Status.UsedGB
            FreeGB = $Status.FreeGB
            UsagePercent = $Status.UsagePercent
        }
        ProcessCount = $Status.ProcessCount
        TopProcesses = $Status.TopProcesses
        SystemInfo = @{
            OS = (Get-CimInstance Win32_OperatingSystem).Caption
            Uptime = (Get-Date) - (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
            CPUUsage = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
        }
        Services = @{
            PostgreSQL = (Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue).Status
            Ollama = (Get-Process "ollama" -ErrorAction SilentlyContinue) -ne $null
            Neo4j = (Get-Process "java" -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*Neo4j*" }) -ne $null
            Redis = (Get-Process "redis-server" -ErrorAction SilentlyContinue) -ne $null
        }
    }
    
    $logFileName = "crash-prevention-$(Get-Date -Format 'yyyy-MM-dd-HHmmss').json"
    $logPath = Join-Path $Global:CrashLogPath $logFileName
    
    $crashLog | ConvertTo-Json -Depth 5 | Out-File -FilePath $logPath -Encoding utf8
    
    Write-Host "⚠️ Crash prevention log saved: $logFileName" -ForegroundColor Yellow
    
    return $logPath
}

function Optimize-Memory {
    Write-Host "`n🔧 Starting Memory Optimization..." -ForegroundColor Cyan
    
    $beforeStatus = Get-MemoryStatus
    Write-Host "Before: $($beforeStatus.UsedGB)GB used ($($beforeStatus.UsagePercent)%)" -ForegroundColor Yellow
    
    # 1. Clear Windows working sets
    Write-Host "  → Clearing working sets..." -ForegroundColor Gray
    Get-Process | ForEach-Object {
        try {
            $_.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal
            [System.GC]::Collect()
            [System.GC]::WaitForPendingFinalizers()
            [System.GC]::Collect()
        } catch {}
    }
    
    # 2. Clear specific application caches
    Write-Host "  → Clearing application caches..." -ForegroundColor Gray
    
    # Node.js processes
    Get-Process node -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $_.Refresh()
        } catch {}
    }
    
    # Chrome/Chromium caches
    $chromeCachePaths = @(
        "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache",
        "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Code Cache"
    )
    
    foreach ($path in $chromeCachePaths) {
        if (Test-Path $path) {
            try {
                Remove-Item "$path\*" -Force -Recurse -ErrorAction SilentlyContinue
                Write-Host "    ✓ Cleared Chrome cache" -ForegroundColor Green
            } catch {}
        }
    }
    
    # 3. Clear temp files
    Write-Host "  → Clearing temporary files..." -ForegroundColor Gray
    $tempPaths = @(
        $env:TEMP,
        "$env:WINDIR\Temp"
    )
    
    foreach ($path in $tempPaths) {
        try {
            Get-ChildItem $path -File -Recurse -ErrorAction SilentlyContinue | 
                Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) } | 
                Remove-Item -Force -ErrorAction SilentlyContinue
        } catch {}
    }
    
    # 4. Optimize SQL Server/PostgreSQL if running
    if (Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue) {
        Write-Host "  → Optimizing PostgreSQL..." -ForegroundColor Gray
        try {
            & psql -U postgres -c "VACUUM ANALYZE;" 2>$null
            Write-Host "    ✓ PostgreSQL optimized" -ForegroundColor Green
        } catch {}
    }
    
    # 5. Clear DNS cache
    Write-Host "  → Clearing DNS cache..." -ForegroundColor Gray
    Clear-DnsClientCache
    
    # 6. Optimize page file
    Write-Host "  → Optimizing page file..." -ForegroundColor Gray
    try {
        $cs = Get-CimInstance -ClassName Win32_ComputerSystem
        $cs.AutomaticManagedPagefile = $true
        Set-CimInstance -InputObject $cs
    } catch {}
    
    # 7. Run Windows memory diagnostic
    Write-Host "  → Running memory compression..." -ForegroundColor Gray
    Start-Process -FilePath "compact" -ArgumentList "/CompactOS:always" -NoNewWindow -Wait -ErrorAction SilentlyContinue
    
    Start-Sleep -Seconds 3
    
    $afterStatus = Get-MemoryStatus
    $freedMemory = [math]::Round($beforeStatus.UsedGB - $afterStatus.UsedGB, 2)
    
    Write-Host "`n✅ Optimization Complete!" -ForegroundColor Green
    Write-Host "After: $($afterStatus.UsedGB)GB used ($($afterStatus.UsagePercent)%)" -ForegroundColor Green
    Write-Host "Freed: ${freedMemory}GB" -ForegroundColor Cyan
    
    Write-MemoryLog -Status $afterStatus -Type "SUCCESS"
    
    return @{
        Before = $beforeStatus
        After = $afterStatus
        FreedGB = $freedMemory
    }
}

function Start-MemoryMonitor {
    param(
        [int]$ThresholdPercent = 80,
        [switch]$AutoOptimize = $false
    )
    
    Write-Host "📊 Starting Memory Monitor" -ForegroundColor Cyan
    Write-Host "Threshold: $ThresholdPercent%" -ForegroundColor Yellow
    Write-Host "Auto-optimize: $AutoOptimize" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop monitoring`n" -ForegroundColor Gray
    
    $warningIssued = $false
    $criticalIssued = $false
    
    while ($true) {
        $status = Get-MemoryStatus
        
        # Display current status
        Clear-Host
        Write-Host "=====================================" -ForegroundColor Cyan
        Write-Host "     Legal AI Memory Monitor" -ForegroundColor Cyan
        Write-Host "=====================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Memory Usage: $($status.UsedGB)GB / $($status.TotalGB)GB" -ForegroundColor White
        
        # Create visual bar
        $barLength = 50
        $fillLength = [math]::Round(($status.UsagePercent / 100) * $barLength)
        $emptyLength = $barLength - $fillLength
        
        $barColor = if ($status.UsagePercent -ge 85) { "Red" }
                    elseif ($status.UsagePercent -ge 70) { "Yellow" }
                    else { "Green" }
        
        Write-Host -NoNewline "["
        Write-Host -NoNewline ("█" * $fillLength) -ForegroundColor $barColor
        Write-Host -NoNewline ("░" * $emptyLength) -ForegroundColor DarkGray
        Write-Host "] $($status.UsagePercent)%"
        
        Write-Host ""
        Write-Host "Processes: $($status.ProcessCount)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Top Memory Consumers:" -ForegroundColor Yellow
        $status.TopProcesses | ForEach-Object {
            Write-Host "  $($_.Name): $($_.Memory)MB" -ForegroundColor Gray
        }
        
        # Check thresholds
        if ($status.UsagePercent -ge 85) {
            if (-not $criticalIssued) {
                Write-Host "`n🚨 CRITICAL: Memory usage above 85%!" -ForegroundColor Red
                Save-CrashPreventionLog -Status $status
                Write-MemoryLog -Status $status -Type "CRITICAL"
                $criticalIssued = $true
                
                if ($AutoOptimize) {
                    Write-Host "Auto-optimizing in 5 seconds..." -ForegroundColor Yellow
                    Start-Sleep -Seconds 5
                    Optimize-Memory
                    $criticalIssued = $false
                    $warningIssued = $false
                }
            }
        }
        elseif ($status.UsagePercent -ge $ThresholdPercent) {
            if (-not $warningIssued) {
                Write-Host "`n⚠️ WARNING: Memory usage above threshold!" -ForegroundColor Yellow
                Write-MemoryLog -Status $status -Type "WARNING"
                $warningIssued = $true
            }
        }
        else {
            $warningIssued = $false
            $criticalIssued = $false
        }
        
        Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] Next check in 10 seconds..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 10
    }
}

function Get-MemoryReport {
    $status = Get-MemoryStatus
    $logs = Get-ChildItem $Global:LogPath -Filter "*.log" -ErrorAction SilentlyContinue | Select-Object -Last 5
    $crashLogs = Get-ChildItem $Global:CrashLogPath -Filter "*.json" -ErrorAction SilentlyContinue | Select-Object -Last 5
    
    $report = @{
        CurrentStatus = $status
        RecentLogs = $logs | ForEach-Object { $_.Name }
        CrashPreventionLogs = $crashLogs | ForEach-Object { $_.Name }
        Recommendations = @()
    }
    
    if ($status.UsagePercent -ge 85) {
        $report.Recommendations += "CRITICAL: Immediate optimization required"
        $report.Recommendations += "Consider closing unused applications"
        $report.Recommendations += "Restart memory-intensive services"
    }
    elseif ($status.UsagePercent -ge 70) {
        $report.Recommendations += "Monitor closely - approaching high usage"
        $report.Recommendations += "Schedule optimization during next maintenance"
    }
    
    Write-Host "`n📋 Memory Report" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    Write-Host "Current Usage: $($status.UsagePercent)%" -ForegroundColor $(if($status.UsagePercent -ge 85){"Red"}elseif($status.UsagePercent -ge 70){"Yellow"}else{"Green"})
    Write-Host "Processes: $($status.ProcessCount)" -ForegroundColor White
    Write-Host ""
    
    if ($report.Recommendations.Count -gt 0) {
        Write-Host "Recommendations:" -ForegroundColor Yellow
        $report.Recommendations | ForEach-Object {
            Write-Host "  • $_" -ForegroundColor Gray
        }
    }
    
    # Save report
    $reportPath = Join-Path $Global:LogPath "memory-report-$(Get-Date -Format 'yyyy-MM-dd-HHmmss').json"
    $report | ConvertTo-Json -Depth 5 | Out-File -FilePath $reportPath -Encoding utf8
    
    Write-Host "`nReport saved: $reportPath" -ForegroundColor Green
    
    return $report
}

# Main execution
if ($Monitor) {
    Start-MemoryMonitor -ThresholdPercent $Threshold -AutoOptimize:$AutoOptimize
}
elseif ($Optimize) {
    Optimize-Memory
}
else {
    # Default: Show current status and options
    $status = Get-MemoryStatus
    
    Write-Host "`n===============================================" -ForegroundColor Cyan
    Write-Host "        Legal AI Memory Optimizer" -ForegroundColor Cyan
    Write-Host "===============================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Memory bar visualization
    $barLength = 40
    $fillLength = [math]::Round(($status.UsagePercent / 100) * $barLength)
    $emptyLength = $barLength - $fillLength
    
    $barColor = if ($status.UsagePercent -ge 85) { "Red" }
                elseif ($status.UsagePercent -ge 70) { "Yellow" }
                else { "Green" }
    
    Write-Host "Memory Usage:" -ForegroundColor White
    Write-Host -NoNewline "["
    Write-Host -NoNewline ("█" * $fillLength) -ForegroundColor $barColor
    Write-Host -NoNewline ("░" * $emptyLength) -ForegroundColor DarkGray
    Write-Host "] $($status.UsagePercent)%"
    
    Write-Host ""
    Write-Host "Total:     $($status.TotalGB) GB" -ForegroundColor Gray
    Write-Host "Used:      $($status.UsedGB) GB" -ForegroundColor Gray
    Write-Host "Free:      $($status.FreeGB) GB" -ForegroundColor Gray
    Write-Host "Processes: $($status.ProcessCount)" -ForegroundColor Gray
    
    Write-Host "`nTop Memory Consumers:" -ForegroundColor Yellow
    $status.TopProcesses | ForEach-Object {
        Write-Host "  • $($_.Name): $($_.Memory) MB" -ForegroundColor Gray
    }
    
    Write-Host "`n📌 Available Commands:" -ForegroundColor Cyan
    Write-Host "  .\memory-optimizer.ps1 -Monitor              # Start monitoring" -ForegroundColor White
    Write-Host "  .\memory-optimizer.ps1 -Monitor -AutoOptimize # Monitor with auto-optimization" -ForegroundColor White
    Write-Host "  .\memory-optimizer.ps1 -Optimize             # Run optimization now" -ForegroundColor White
    Write-Host "  .\memory-optimizer.ps1                       # Show this status" -ForegroundColor White
    
    if ($status.UsagePercent -ge 70) {
        Write-Host "`n⚠️ Recommendation: Consider running optimization" -ForegroundColor Yellow
        Write-Host "   Run: .\memory-optimizer.ps1 -Optimize" -ForegroundColor Gray
    }
    
    # Generate report
    Get-MemoryReport | Out-Null
}