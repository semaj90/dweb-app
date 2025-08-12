# Real-time GPU AI Monitor
# monitor-gpu-ai.ps1

param(
    [int]$RefreshInterval = 2,
    [switch]$LogToFile,
    [string]$LogPath = ".\gpu-ai-metrics.log"
)

# Console setup
$Host.UI.RawUI.WindowTitle = "GPU Legal AI Monitor - RTX 3060 Ti"
Clear-Host

# Initialize performance counters
$script:totalRequests = 0
$script:successfulRequests = 0
$script:failedRequests = 0
$script:avgLatency = @()
$script:startTime = Get-Date

function Get-GPUMetrics {
    try {
        $gpuQuery = & nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit --format=csv,noheader,nounits
        $metrics = $gpuQuery -split ','
        
        return @{
            Index = $metrics[0].Trim()
            Name = $metrics[1].Trim()
            Temperature = [int]$metrics[2]
            GPUUtil = [int]$metrics[3]
            MemUtil = [int]$metrics[4]
            MemTotal = [int]$metrics[5]
            MemFree = [int]$metrics[6]
            MemUsed = [int]$metrics[7]
            PowerDraw = [decimal]$metrics[8]
            PowerLimit = [decimal]$metrics[9]
        }
    } catch {
        return $null
    }
}

function Get-ServiceMetrics {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8084/api/metrics" -Method Get -ErrorAction Stop
        return $response
    } catch {
        return $null
    }
}

function Get-OllamaStatus {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction Stop
        $models = $response.models
        $gemmaModel = $models | Where-Object { $_.name -like "*gemma*" }
        return @{
            Status = "Online"
            ModelLoaded = if ($gemmaModel) { $true } else { $false }
            ModelName = if ($gemmaModel) { $gemmaModel.name } else { "Not Loaded" }
        }
    } catch {
        return @{
            Status = "Offline"
            ModelLoaded = $false
            ModelName = "N/A"
        }
    }
}

function Draw-ProgressBar {
    param(
        [int]$Value,
        [int]$Max = 100,
        [int]$Width = 30,
        [string]$ForegroundColor = "Green"
    )
    
    $percentage = [math]::Min(100, [math]::Round(($Value / $Max) * 100))
    $filled = [math]::Round(($percentage / 100) * $Width)
    $empty = $Width - $filled
    
    $color = switch ($percentage) {
        {$_ -ge 90} { "Red" }
        {$_ -ge 70} { "Yellow" }
        default { "Green" }
    }
    
    Write-Host -NoNewline "["
    Write-Host -NoNewline ("█" * $filled) -ForegroundColor $color
    Write-Host -NoNewline ("░" * $empty) -ForegroundColor DarkGray
    Write-Host -NoNewline "] "
    Write-Host "$percentage%" -ForegroundColor $color
}

function Format-Bytes {
    param([long]$Bytes)
    if ($Bytes -ge 1GB) {
        return "{0:N2} GB" -f ($Bytes / 1GB)
    } elseif ($Bytes -ge 1MB) {
        return "{0:N2} MB" -f ($Bytes / 1MB)
    } else {
        return "{0:N0} KB" -f ($Bytes / 1KB)
    }
}

# Main monitoring loop
while ($true) {
    Clear-Host
    $currentTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $uptime = (Get-Date) - $script:startTime
    
    # Header
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║         GPU-ACCELERATED LEGAL AI SERVICE MONITOR            ║" -ForegroundColor Cyan
    Write-Host "║                   RTX 3060 Ti Optimized                     ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Time: $currentTime | Uptime: $($uptime.ToString('hh\:mm\:ss'))" -ForegroundColor Gray
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    
    # Get metrics
    $gpu = Get-GPUMetrics
    $service = Get-ServiceMetrics
    $ollama = Get-OllamaStatus
    
    # GPU Section
    Write-Host ""
    Write-Host "🎮 GPU STATUS" -ForegroundColor Yellow
    if ($gpu) {
        Write-Host "  Device:      " -NoNewline; Write-Host $gpu.Name -ForegroundColor White
        Write-Host "  Temperature: " -NoNewline
        $tempColor = if ($gpu.Temperature -ge 80) { "Red" } elseif ($gpu.Temperature -ge 70) { "Yellow" } else { "Green" }
        Write-Host "$($gpu.Temperature)°C" -ForegroundColor $tempColor
        
        Write-Host "  GPU Usage:   " -NoNewline
        Draw-ProgressBar -Value $gpu.GPUUtil -Max 100
        
        Write-Host "  Memory:      " -NoNewline
        Write-Host "$($gpu.MemUsed)MB / $($gpu.MemTotal)MB " -NoNewline -ForegroundColor White
        Write-Host "($($gpu.MemUtil)%)" -ForegroundColor Gray
        
        Write-Host "  VRAM Usage:  " -NoNewline
        Draw-ProgressBar -Value $gpu.MemUsed -Max $gpu.MemTotal
        
        Write-Host "  Power:       " -NoNewline
        Write-Host "$($gpu.PowerDraw)W / $($gpu.PowerLimit)W" -ForegroundColor White
    } else {
        Write-Host "  Status: Unable to query GPU" -ForegroundColor Red
    }
    
    # Ollama Section
    Write-Host ""
    Write-Host "🤖 OLLAMA STATUS" -ForegroundColor Yellow
    Write-Host "  Status:      " -NoNewline
    $statusColor = if ($ollama.Status -eq "Online") { "Green" } else { "Red" }
    Write-Host $ollama.Status -ForegroundColor $statusColor
    Write-Host "  Model:       " -NoNewline; Write-Host $ollama.ModelName -ForegroundColor White
    Write-Host "  Port:        " -NoNewline; Write-Host "11434" -ForegroundColor White
    
    # Service Metrics Section
    Write-Host ""
    Write-Host "📊 SERVICE METRICS" -ForegroundColor Yellow
    if ($service) {
        Write-Host "  Requests:    " -NoNewline
        Write-Host "$($service.total_requests) total" -ForegroundColor White -NoNewline
        Write-Host " ($($service.successful_requests) success)" -ForegroundColor Green
        
        Write-Host "  Success Rate:" -NoNewline
        Write-Host " $($service.success_rate)" -ForegroundColor Green
        
        Write-Host "  Cache Hits:  " -NoNewline
        Write-Host "$($service.cache_hits)" -ForegroundColor White -NoNewline
        Write-Host " ($($service.cache_hit_rate))" -ForegroundColor Gray
        
        Write-Host "  Avg Latency: " -NoNewline
        Write-Host $service.average_latency -ForegroundColor White
        
        Write-Host "  Tokens/sec:  " -NoNewline
        Write-Host $service.tokens_per_second -ForegroundColor White
    } else {
        Write-Host "  Status: Service not responding" -ForegroundColor Red
    }
    
    # Active Processes
    Write-Host ""
    Write-Host "⚡ ACTIVE PROCESSES" -ForegroundColor Yellow
    $processes = @(
        @{Name="ollama"; Display="Ollama Server"},
        @{Name="redis-server"; Display="Redis Cache"},
        @{Name="postgres"; Display="PostgreSQL"},
        @{Name="go"; Display="Go Service"}
    )
    
    foreach ($proc in $processes) {
        $running = Get-Process -Name $proc.Name -ErrorAction SilentlyContinue
        Write-Host "  $($proc.Display): " -NoNewline
        if ($running) {
            $mem = [math]::Round($running.WorkingSet64 / 1MB, 0)
            Write-Host "Running " -ForegroundColor Green -NoNewline
            Write-Host "(${mem}MB)" -ForegroundColor Gray
        } else {
            Write-Host "Not Running" -ForegroundColor Red
        }
    }
    
    # Footer
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor DarkGray
    Write-Host "Commands: [Q]uit | [R]efresh | [L]og | [C]lear GPU Cache" -ForegroundColor Gray
    Write-Host "Service: http://localhost:8084 | Refresh: ${RefreshInterval}s" -ForegroundColor DarkGray
    
    # Log to file if requested
    if ($LogToFile) {
        $logEntry = @{
            Timestamp = $currentTime
            GPU = $gpu
            Service = $service
            Ollama = $ollama
        }
        $logEntry | ConvertTo-Json -Compress | Out-File -Append -FilePath $LogPath
    }
    
    # Check for user input
    $timeoutSeconds = $RefreshInterval
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    while ($stopwatch.Elapsed.TotalSeconds -lt $timeoutSeconds) {
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            switch ($key.Key) {
                'Q' { 
                    Write-Host "`nExiting monitor..." -ForegroundColor Yellow
                    exit 
                }
                'R' { 
                    break 
                }
                'C' {
                    Write-Host "`nClearing GPU cache..." -ForegroundColor Yellow
                    & nvidia-smi --gpu-reset
                    Start-Sleep -Seconds 2
                    break
                }
                'L' {
                    $script:LogToFile = -not $script:LogToFile
                    if ($script:LogToFile) {
                        Write-Host "`nLogging enabled: $LogPath" -ForegroundColor Green
                    } else {
                        Write-Host "`nLogging disabled" -ForegroundColor Yellow
                    }
                    Start-Sleep -Seconds 1
                    break
                }
            }
        }
        Start-Sleep -Milliseconds 100
    }
}