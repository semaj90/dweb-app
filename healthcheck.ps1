Param(
    [string[]]$Services = @(
        'http://localhost:8099/health',    # core health server
        'http://localhost:8081/health'     # redis-service default
    ),
    [int]$TimeoutSec = 3
)

Write-Host "🔍 Checking service health..." -ForegroundColor Cyan

foreach ($url in $Services) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec $TimeoutSec
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
            Write-Host "✅ $url" -ForegroundColor Green
        } else {
            Write-Host "⚠️ $url -> $($resp.StatusCode)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ $url : $($_.Exception.Message)" -ForegroundColor Red
    }
}
