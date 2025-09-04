# Simple Windows Native Services Check
# Checks for native PostgreSQL, Redis, Qdrant, and Ollama

param(
    [switch]$Status = $false,
    [switch]$Setup = $false,
    [switch]$GPU = $false
)

Write-Host "Legal AI - Native Windows Services Status" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

function Test-ServicePort {
    param([int]$Port, [string]$ServiceName)
    try {
        $connection = Test-NetConnection -ComputerName "localhost" -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
        if ($connection) {
            Write-Host "   [OK] $ServiceName is running on port $Port" -ForegroundColor Green
            return $true
        }
    } catch {}
    Write-Host "   [X] $ServiceName is not running on port $Port" -ForegroundColor Red
    return $false
}

function Show-ServiceStatus {
    Write-Host ""
    Write-Host "CURRENT SERVICE STATUS" -ForegroundColor Cyan
    Write-Host "=====================" -ForegroundColor Cyan

    $services = @(
        @{ Name = "PostgreSQL"; Port = 5432 },
        @{ Name = "Qdrant"; Port = 6333 },
        @{ Name = "Redis/Memurai"; Port = 6379 },
        @{ Name = "Ollama"; Port = 11434 }
    )

    foreach ($service in $services) {
        Test-ServicePort -Port $service.Port -ServiceName $service.Name
    }

    Write-Host ""
    Write-Host "Service Endpoints:" -ForegroundColor White
    Write-Host "   PostgreSQL: localhost:5432" -ForegroundColor White
    Write-Host "   Qdrant: http://localhost:6333" -ForegroundColor White
    Write-Host "   Redis: localhost:6379" -ForegroundColor White
    Write-Host "   Ollama: http://localhost:11434" -ForegroundColor White
    Write-Host ""

function Setup-NativeServices {
    Write-Host ""
    Write-Host "Setting up native Windows services..." -ForegroundColor Cyan
    Write-Host ""

    # PostgreSQL
    Write-Host "1. PostgreSQL Setup:" -ForegroundColor Yellow
    Write-Host "   - Install from: https://www.postgresql.org/download/windows/" -ForegroundColor White
    Write-Host "   - Or use: winget install PostgreSQL.PostgreSQL" -ForegroundColor White
    Write-Host "   - Or use: choco install postgresql" -ForegroundColor White
    Write-Host ""

    # Qdrant
    Write-Host "2. Qdrant Setup:" -ForegroundColor Yellow
    Write-Host "   - Download from: https://github.com/qdrant/qdrant/releases" -ForegroundColor White
    Write-Host "   - Extract qdrant.exe and run: qdrant.exe --config-path config.yaml" -ForegroundColor White
    Write-Host ""

    # Redis (Memurai)
    Write-Host "3. Redis Setup:" -ForegroundColor Yellow
    Write-Host "   - Install Memurai: https://www.memurai.com/" -ForegroundColor White
    Write-Host "   - Or use: choco install memurai" -ForegroundColor White
    Write-Host "   - Or use WSL: wsl --install && wsl -d Ubuntu apt install redis-server" -ForegroundColor White
    Write-Host ""

    # Ollama
    Write-Host "4. Ollama Setup:" -ForegroundColor Yellow
    Write-Host "   - Download from: https://ollama.com/download/windows" -ForegroundColor White
    Write-Host "   - Install and run: ollama serve" -ForegroundColor White
    if ($GPU) {
        Write-Host "   - GPU mode: CUDA will be auto-detected" -ForegroundColor White
    }
    Write-Host ""

    Write-Host "Quick setup commands:" -ForegroundColor Cyan
    Write-Host "   winget install PostgreSQL.PostgreSQL" -ForegroundColor White
    Write-Host "   choco install memurai" -ForegroundColor White
    Write-Host "   Invoke-WebRequest -Uri https://ollama.com/download/windows -OutFile ollama.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "After installation, run: npm run native:status" -ForegroundColor Green
}

# Main execution
if ($Setup) {
    Setup-NativeServices
} else {
    Show-ServiceStatus
}

Write-Host ""
Write-Host "To set up services: npm run native:setup" -ForegroundColor Yellow
Write-Host "To check status: npm run native:status" -ForegroundColor Yellow
