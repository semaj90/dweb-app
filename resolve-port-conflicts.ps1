# Port Conflict Resolution for Legal AI Platform
# Resolves common port conflicts during service startup

Write-Host "🔍 Checking for port conflicts..." -ForegroundColor Cyan

$problematicPorts = @(50051, 8097, 8225, 8219, 8212, 8220)

foreach ($port in $problematicPorts) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    
    if ($connections) {
        Write-Host "⚠️ Port $port in use by:" -ForegroundColor Yellow
        foreach ($conn in $connections) {
            $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "  - PID: $($conn.OwningProcess) | Process: $($process.ProcessName)" -ForegroundColor Gray
                
                # Check if it's our own Go service
                if ($process.ProcessName -like "*rag*" -or $process.ProcessName -like "*grpc*" -or $process.ProcessName -like "*cuda*") {
                    Write-Host "    🎯 This is one of our services - keeping it running" -ForegroundColor Green
                } else {
                    Write-Host "    💡 Consider stopping this process if safe: Stop-Process -Id $($conn.OwningProcess)" -ForegroundColor Blue
                }
            }
        }
    } else {
        Write-Host "✅ Port $port is available" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🚀 Quick fixes:" -ForegroundColor Magenta
Write-Host "1. Kill conflicting processes: taskkill /f /pid <PID>" -ForegroundColor White
Write-Host "2. Use alternative ports in service config" -ForegroundColor White  
Write-Host "3. Restart with: npm run dev:full" -ForegroundColor White

# Check running Go services
Write-Host ""
Write-Host "🔧 Running Go services:" -ForegroundColor Blue
Get-Process | Where-Object { $_.ProcessName -like "*rag*" -or $_.ProcessName -like "*grpc*" -or $_.ProcessName -like "*cuda*" -or $_.ProcessName -like "*upload*" } | 
    Select-Object ProcessName, Id, @{Name="Memory(MB)";Expression={[math]::Round($_.WorkingSet / 1MB, 2)}} |
    Format-Table -AutoSize