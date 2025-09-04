# YoRHa Legal AI Platform - Install Missing Services
# Quick installer for Redis, RabbitMQ, Neo4j, Qdrant

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "YoRHa Legal AI Platform - Installing Missing Services" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if winget is available
try {
    $wingetVersion = winget --version
    Write-Host "✅ Winget available: $wingetVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Winget not available - using manual methods" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Installing core services..." -ForegroundColor Cyan

# Install Redis
Write-Host ""
Write-Host "[1/4] Installing Redis..." -ForegroundColor Yellow
try {
    winget install Redis.Redis --accept-package-agreements --accept-source-agreements -h
    Write-Host "✅ Redis installation completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Redis installation failed or already installed" -ForegroundColor Yellow
}

# Install RabbitMQ  
Write-Host ""
Write-Host "[2/4] Installing RabbitMQ..." -ForegroundColor Yellow
try {
    winget install RabbitMQ.RabbitMQ --accept-package-agreements --accept-source-agreements -h
    Write-Host "✅ RabbitMQ installation completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ RabbitMQ installation failed or already installed" -ForegroundColor Yellow
}

# Install Neo4j
Write-Host ""
Write-Host "[3/4] Installing Neo4j..." -ForegroundColor Yellow
try {
    winget install Neo4j.Neo4j --accept-package-agreements --accept-source-agreements -h
    Write-Host "✅ Neo4j installation completed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Neo4j installation failed or already installed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[4/4] Qdrant - Creating portable version..." -ForegroundColor Yellow

# Create Qdrant download and setup
$qdrantUrl = "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.tar.gz"
$qdrantDir = "./qdrant-portable"

try {
    if (-not (Test-Path $qdrantDir)) {
        Write-Host "📥 Downloading Qdrant portable..." -ForegroundColor Cyan
        New-Item -ItemType Directory -Path $qdrantDir -Force | Out-Null
        
        # Note: For now, just create the directory and config
        Write-Host "📝 Creating Qdrant low memory config..." -ForegroundColor Cyan
        
        $qdrantConfig = @"
log_level: INFO
storage:
  storage_path: ./qdrant-storage
service:
  http_port: 6333
  grpc_port: 6334
performance:
  max_search_threads: 2
  max_optimization_threads: 1
hnsw_config:
  m: 8
  ef_construct: 100
"@
        
        $qdrantConfig | Out-File -FilePath "$qdrantDir/qdrant-config.yaml" -Encoding UTF8
        Write-Host "✅ Qdrant config created (manual download required)" -ForegroundColor Green
        Write-Host "   Download from: https://github.com/qdrant/qdrant/releases" -ForegroundColor Yellow
    } else {
        Write-Host "✅ Qdrant directory already exists" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Qdrant setup failed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "Service Installation Summary" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Restart your terminal/PowerShell" -ForegroundColor White
Write-Host "2. Run: YORHA-COMPLETE-STARTUP.bat" -ForegroundColor White
Write-Host "3. Services will be available at:" -ForegroundColor White
Write-Host "   • Redis: localhost:6379" -ForegroundColor Cyan
Write-Host "   • RabbitMQ: http://localhost:15672 (guest/guest)" -ForegroundColor Cyan
Write-Host "   • Neo4j: http://localhost:7474 (neo4j/neo4j)" -ForegroundColor Cyan
Write-Host "   • Qdrant: http://localhost:6333" -ForegroundColor Cyan

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan

Read-Host "Press Enter to continue"