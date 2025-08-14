# XState Manager Health Monitor
# Monitors real-time LLM training and user state management

param(
    [int]$IntervalSeconds = 30,
    [switch]$Continuous = $false,
    [switch]$ShowDetails = $false
)

function Get-ServiceStatus {
    param($Name, $Url)
    
    try {
        $response = Invoke-RestMethod -Uri $Url -TimeoutSec 5 -ErrorAction Stop
        return @{
            Name = $Name
            Status = "✅ Healthy"
            Details = $response
            Available = $true
        }
    }
    catch {
        return @{
            Name = $Name
            Status = "❌ Unavailable"
            Details = $_.Exception.Message
            Available = $false
        }
    }
}

function Show-XStateMetrics {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "🧠 XState Manager Health Check - $timestamp" -ForegroundColor Cyan
    Write-Host "=" * 60
    
    # Check XState Manager
    $xstate = Get-ServiceStatus "XState Manager" "http://localhost:8095/health"
    Write-Host "$($xstate.Status) $($xstate.Name)"
    
    if ($xstate.Available -and $ShowDetails) {
        Write-Host "   Connected Users: $($xstate.Details.connectedUsers)" -ForegroundColor Green
        Write-Host "   Tracked Users: $($xstate.Details.trackedUsers)" -ForegroundColor Green
    }
    
    # Check Analytics Endpoint
    try {
        $analytics = Invoke-RestMethod -Uri "http://localhost:8095/api/learning-analytics" -TimeoutSec 5
        Write-Host "✅ Learning Analytics Available"
        
        if ($ShowDetails) {
            Write-Host "   Total Users: $($analytics.totalUsers)" -ForegroundColor Yellow
            Write-Host "   Active Users: $($analytics.activeUsers)" -ForegroundColor Yellow
            Write-Host "   Avg Session Time: $([math]::Round($analytics.avgSessionTimeHours, 2)) hours" -ForegroundColor Yellow
            Write-Host "   Total Searches: $($analytics.totalSearches)" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "❌ Learning Analytics Unavailable" -ForegroundColor Red
    }
    
    # Check supporting services
    Write-Host ""
    Write-Host "Supporting Services:" -ForegroundColor Magenta
    
    # PostgreSQL
    try {
        $null = & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT 1;" --quiet 2>$null
        Write-Host "✅ PostgreSQL: Connected"
    }
    catch {
        Write-Host "❌ PostgreSQL: Not connected"
    }
    
    # Ollama
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
        Write-Host "✅ Ollama: Running"
    }
    catch {
        Write-Host "❌ Ollama: Not running"
    }
    
    # Redis (optional)
    try {
        $result = & redis-cli ping 2>$null
        if ($result -eq "PONG") {
            Write-Host "✅ Redis: Running"
        } else {
            Write-Host "⚠️ Redis: Not running (optional)"
        }
    }
    catch {
        Write-Host "⚠️ Redis: Not running (optional)"
    }
    
    # SvelteKit Dev Server
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 3 -ErrorAction Stop
        Write-Host "✅ SvelteKit: Running"
    }
    catch {
        Write-Host "❌ SvelteKit: Not running"
    }
    
    Write-Host ""
}

function Show-QuickActions {
    Write-Host "🚀 Quick Actions:" -ForegroundColor Green
    Write-Host "  Start XState Manager: cd go-microservice && xstate-manager.exe"
    Write-Host "  View Real-time Dashboard: http://localhost:8095/api/learning-analytics"
    Write-Host "  WebSocket Test: ws://localhost:8095/ws?userId=demo"
    Write-Host "  Check Logs: type go-microservice\xstate-manager.log"
    Write-Host ""
}

# Main execution
if ($Continuous) {
    Write-Host "🔄 Continuous monitoring enabled. Press Ctrl+C to stop." -ForegroundColor Yellow
    Write-Host ""
    
    while ($true) {
        Clear-Host
        Show-XStateMetrics
        Show-QuickActions
        Start-Sleep -Seconds $IntervalSeconds
    }
} else {
    Show-XStateMetrics
    Show-QuickActions
}

# Usage examples
Write-Host "📖 Usage Examples:" -ForegroundColor Blue
Write-Host "  .\monitor-xstate-health.ps1                    # Single check"
Write-Host "  .\monitor-xstate-health.ps1 -Continuous        # Monitor every 30s"
Write-Host "  .\monitor-xstate-health.ps1 -ShowDetails       # Detailed metrics"
Write-Host "  .\monitor-xstate-health.ps1 -Continuous -ShowDetails -IntervalSeconds 10"