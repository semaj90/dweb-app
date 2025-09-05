# AI Synthesis Monitoring Dashboard
# Real-time monitoring of all AI synthesis services

param(
    [switch]$Minimal = $false,
    [int]$RefreshInterval = 5
)

$host.UI.RawUI.WindowTitle = "AI Synthesis Monitor"
$script:StartTime = Get-Date

# Colors
$colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Gray"
}

function Get-ServiceStatus {
    param(
        [string]$Name,
        [string]$Url,
        [int]$Port
    )
    
    $status = @{
        Name = $Name
        Port = $Port
        Status = "OFFLINE"
        ResponseTime = 0
        Health = $null
    }
    
    try {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 2
        $stopwatch.Stop()
        
        $status.Status = "ONLINE"
        $status.ResponseTime = $stopwatch.ElapsedMilliseconds
        $status.Health = $response
    } catch {
        $status.Status = "OFFLINE"
    }
    
    return $status
}

function Show-Dashboard {
    Clear-Host
    
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host "           AI SYNTHESIS MONITORING DASHBOARD" -ForegroundColor $colors.Header
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host ""
    
    $currentTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $uptime = (Get-Date) - $script:StartTime
    Write-Host "Time: $currentTime | Uptime: $($uptime.ToString('hh\:mm\:ss'))" -ForegroundColor $colors.Info
    Write-Host ""
    
    # Core Services
    Write-Host "🔧 CORE SERVICES" -ForegroundColor $colors.Header
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor $colors.Info
    
    $services = @(
        @{ Name = "SvelteKit Dev"; Url = "http://localhost:5173"; Port = 5173 },
        @{ Name = "AI Synthesis API"; Url = "http://localhost:5173/api/ai-synthesizer"; Port = 5173 },
        @{ Name = "Redis Cache"; Url = "http://localhost:6379"; Port = 6379 },
        @{ Name = "Ollama AI"; Url = "http://localhost:11434/api/tags"; Port = 11434 }
    )
    
    foreach ($service in $services) {
        $status = Get-ServiceStatus -Name $service.Name -Url $service.Url -Port $service.Port
        
        $statusIcon = if ($status.Status -eq "ONLINE") { "✅" } else { "❌" }
        $statusColor = if ($status.Status -eq "ONLINE") { $colors.Success } else { $colors.Error }
        
        $line = "{0} {1,-20} Port: {2,-5} Status: {3,-8}" -f $statusIcon, $status.Name, $status.Port, $status.Status
        
        if ($status.ResponseTime -gt 0) {
            $line += " Response: {0}ms" -f $status.ResponseTime
        }
        
        Write-Host $line -ForegroundColor $statusColor
    }
    
    Write-Host ""
    Write-Host "🤖 LEGAL AI SERVICES" -ForegroundColor $colors.Header
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor $colors.Info
    
    $legalServices = @(
        @{ Name = "Enhanced RAG"; Url = "http://localhost:8094/health"; Port = 8094 },
        @{ Name = "GPU Orchestrator"; Url = "http://localhost:8095/health"; Port = 8095 },
        @{ Name = "Context7 MCP"; Url = "http://localhost:4000/health"; Port = 4000 }
    )
    
    foreach ($service in $legalServices) {
        $status = Get-ServiceStatus -Name $service.Name -Url $service.Url -Port $service.Port
        
        $statusIcon = if ($status.Status -eq "ONLINE") { "✅" } else { "⚠️" }
        $statusColor = if ($status.Status -eq "ONLINE") { $colors.Success } else { $colors.Warning }
        
        $line = "{0} {1,-20} Port: {2,-5} Status: {3,-8}" -f $statusIcon, $status.Name, $status.Port, $status.Status
        
        if ($status.ResponseTime -gt 0) {
            $line += " Response: {0}ms" -f $status.ResponseTime
        }
        
        Write-Host $line -ForegroundColor $statusColor
    }
    
    # Get detailed metrics if AI Synthesis API is online
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:5173/api/ai-synthesizer" -Method Get -TimeoutSec 2
        
        if ($health.components) {
            Write-Host ""
            Write-Host "📊 PERFORMANCE METRICS" -ForegroundColor $colors.Header
            Write-Host "─────────────────────────────────────────────────────" -ForegroundColor $colors.Info
            
            if ($health.components.monitoring) {
                $mon = $health.components.monitoring
                Write-Host "Total Requests:    $($mon.counters.totalRequests)" -ForegroundColor $colors.Info
                Write-Host "Success Rate:      $($mon.rates.successRate)" -ForegroundColor $colors.Success
                Write-Host "Cache Hit Rate:    $($mon.rates.cacheHitRate)" -ForegroundColor $colors.Success
                
                if ($mon.performance.overall) {
                    Write-Host ""
                    Write-Host "Response Times:" -ForegroundColor $colors.Info
                    Write-Host "  P50: $($mon.performance.overall.p50)ms" -ForegroundColor $colors.Info
                    Write-Host "  P95: $($mon.performance.overall.p95)ms" -ForegroundColor $colors.Info
                    Write-Host "  P99: $($mon.performance.overall.p99)ms" -ForegroundColor $colors.Info
                }
            }
            
            if ($health.components.cache) {
                Write-Host ""
                Write-Host "📦 CACHE STATISTICS" -ForegroundColor $colors.Header
                Write-Host "─────────────────────────────────────────────────────" -ForegroundColor $colors.Info
                $cache = $health.components.cache
                Write-Host "Hits:         $($cache.hits)" -ForegroundColor $colors.Info
                Write-Host "Misses:       $($cache.misses)" -ForegroundColor $colors.Info
                Write-Host "Hit Rate:     $([math]::Round($cache.hitRate * 100, 2))%" -ForegroundColor $colors.Success
                Write-Host "Memory Usage: $([math]::Round($cache.memoryUsage / 1024 / 1024, 2)) MB" -ForegroundColor $colors.Info
            }
        }
    } catch {
        # Metrics not available
    }
    
    # System Resources
    Write-Host ""
    Write-Host "💻 SYSTEM RESOURCES" -ForegroundColor $colors.Header
    Write-Host "─────────────────────────────────────────────────────" -ForegroundColor $colors.Info
    
    $cpu = Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue
    $mem = Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue
    
    if ($cpu) {
        Write-Host "CPU Usage:        $([math]::Round($cpu.CounterSamples[0].CookedValue, 2))%" -ForegroundColor $colors.Info
    }
    
    if ($mem) {
        $totalMem = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum / 1MB
        $usedMem = $totalMem - $mem.CounterSamples[0].CookedValue
        $memPercent = ($usedMem / $totalMem) * 100
        Write-Host "Memory Usage:     $([math]::Round($memPercent, 2))% ($([math]::Round($usedMem, 0)) / $([math]::Round($totalMem, 0)) MB)" -ForegroundColor $colors.Info
    }
    
    # Network connections
    $connections = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | 
                   Where-Object { $_.LocalPort -in @(5173, 6379, 11434, 8094, 8095, 4000, 8200) }
    
    if ($connections) {
        Write-Host "Active Listeners: $($connections.Count) services" -ForegroundColor $colors.Success
    }
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor $colors.Header
    Write-Host "Refresh: $RefreshInterval seconds | Press Ctrl+C to exit" -ForegroundColor $colors.Info
}

# Main loop
while ($true) {
    Show-Dashboard
    Start-Sleep -Seconds $RefreshInterval
}
