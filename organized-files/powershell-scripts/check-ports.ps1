# Simple Port Check Script
Write-Host "Checking port conflicts..." -ForegroundColor Cyan

# Check specific ports
$ports = @(50051, 8097, 8225, 8219, 8212, 8220)

foreach ($port in $ports) {
    $connection = netstat -ano | Select-String ":$port "
    if ($connection) {
        Write-Host "Port $port is in use" -ForegroundColor Yellow
        Write-Host $connection -ForegroundColor Gray
    } else {
        Write-Host "Port $port is available" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Running Go services:" -ForegroundColor Blue
tasklist /fi "imagename eq enhanced-rag*" /fi "status eq running" 2>$null
tasklist /fi "imagename eq *grpc*" /fi "status eq running" 2>$null