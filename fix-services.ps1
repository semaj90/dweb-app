# Quick Service Fix Script
Write-Host "🔧 Fixing Legal AI Platform Services..." -ForegroundColor Cyan

# Kill conflicting gRPC process
Write-Host "1. Fixing gRPC port conflict..." -ForegroundColor Yellow
try {
    Stop-Process -Id 2532 -Force -ErrorAction SilentlyContinue
    Write-Host "✅ Killed conflicting process on port 50051" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Process may already be stopped" -ForegroundColor Yellow
}

# Wait a moment for cleanup
Start-Sleep -Seconds 2

Write-Host "2. Service status after cleanup:" -ForegroundColor Blue
$checkPorts = @(50051, 8094, 8093, 5173)
foreach ($port in $checkPorts) {
    $connection = netstat -ano | Select-String ":$port " | Select-Object -First 1
    if ($connection) {
        Write-Host "  Port $port: IN USE" -ForegroundColor Yellow
    } else {
        Write-Host "  Port $port: AVAILABLE" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "✅ Ready to restart with: npm run dev:full" -ForegroundColor Green
Write-Host "🎯 Expected improvements:" -ForegroundColor Magenta
Write-Host "  - gRPC Server should start successfully" -ForegroundColor White
Write-Host "  - Other services have 20s timeout (was 10s)" -ForegroundColor White
Write-Host "  - Should see 10+ running services instead of 6" -ForegroundColor White