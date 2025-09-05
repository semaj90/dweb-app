# Setup Optimization Services for SIMD JSON + Vector Embeddings
# PowerShell script to install and configure required services

Write-Host "🚀 Setting up SIMD JSON Optimization Services..." -ForegroundColor Cyan

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    }
    catch {
        return $false
    }
}

# Function to download and extract if not exists
function Download-And-Extract {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$ExtractPath
    )
    
    if (!(Test-Path $OutputPath)) {
        Write-Host "📥 Downloading $OutputPath..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath
    }
    
    if (!(Test-Path $ExtractPath)) {
        Write-Host "📦 Extracting to $ExtractPath..." -ForegroundColor Yellow
        Expand-Archive -Path $OutputPath -DestinationPath $ExtractPath -Force
    }
}

# Create services directory
$servicesDir = Join-Path $PSScriptRoot "services"
if (!(Test-Path $servicesDir)) {
    New-Item -ItemType Directory -Path $servicesDir -Force
}

Write-Host "`n🔧 Installing Services..." -ForegroundColor Green

# 1. Install Ollama for Local LLM Embeddings
Write-Host "`n1️⃣ Setting up Ollama (Port 11434)..." -ForegroundColor Cyan

if (!(Test-Port 11434)) {
    $ollamaPath = Join-Path $servicesDir "ollama"
    
    if (!(Test-Path "$ollamaPath\ollama.exe")) {
        Write-Host "📥 Downloading Ollama..." -ForegroundColor Yellow
        $ollamaUrl = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
        $ollamaZip = Join-Path $servicesDir "ollama.zip"
        
        try {
            Invoke-WebRequest -Uri $ollamaUrl -OutFile $ollamaZip
            Expand-Archive -Path $ollamaZip -DestinationPath $ollamaPath -Force
            Remove-Item $ollamaZip -Force
        }
        catch {
            Write-Host "⚠️ Failed to download Ollama. Please install manually from https://ollama.ai/download" -ForegroundColor Red
        }
    }
    
    # Start Ollama
    if (Test-Path "$ollamaPath\ollama.exe") {
        Write-Host "🚀 Starting Ollama..." -ForegroundColor Green
        Start-Process -FilePath "$ollamaPath\ollama.exe" -ArgumentList "serve" -WindowStyle Minimized
        Start-Sleep 5
        
        # Pull embedding model
        Write-Host "📚 Pulling nomic-embed-text model..." -ForegroundColor Yellow
        & "$ollamaPath\ollama.exe" pull nomic-embed-text
    }
    else {
        Write-Host "⚠️ Ollama not found. Downloading from ollama.ai..." -ForegroundColor Yellow
        Start-Process "https://ollama.ai/download"
    }
}
else {
    Write-Host "✅ Ollama already running on port 11434" -ForegroundColor Green
}

# 2. Install Qdrant Vector Database
Write-Host "`n2️⃣ Setting up Qdrant (Port 6333)..." -ForegroundColor Cyan

if (!(Test-Port 6333)) {
    # Check if Docker is available
    try {
        $dockerVersion = docker --version
        Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
        
        Write-Host "🐳 Starting Qdrant with Docker..." -ForegroundColor Yellow
        docker run -d --name qdrant-optimization -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
        Start-Sleep 5
        
        # Create collection for copilot embeddings
        if (Test-Port 6333) {
            Write-Host "📊 Creating copilot_embeddings collection..." -ForegroundColor Yellow
            $createCollection = @{
                vectors = @{
                    size = 384
                    distance = "Cosine"
                }
            } | ConvertTo-Json
            
            try {
                Invoke-RestMethod -Uri "http://localhost:6333/collections/copilot_embeddings" -Method PUT -Body $createCollection -ContentType "application/json"
                Write-Host "✅ Qdrant collection created successfully" -ForegroundColor Green
            }
            catch {
                Write-Host "⚠️ Collection might already exist or Qdrant is starting up" -ForegroundColor Yellow
            }
        }
    }
    catch {
        Write-Host "❌ Docker not found. Installing Qdrant standalone..." -ForegroundColor Red
        
        $qdrantPath = Join-Path $servicesDir "qdrant"
        if (!(Test-Path "$qdrantPath\qdrant.exe")) {
            Write-Host "📥 Downloading Qdrant..." -ForegroundColor Yellow
            $qdrantUrl = "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip"
            $qdrantZip = Join-Path $servicesDir "qdrant.zip"
            
            try {
                Download-And-Extract -Url $qdrantUrl -OutputPath $qdrantZip -ExtractPath $qdrantPath
                
                # Start Qdrant
                Write-Host "🚀 Starting Qdrant..." -ForegroundColor Green
                Start-Process -FilePath "$qdrantPath\qdrant.exe" -WindowStyle Minimized
                Start-Sleep 5
            }
            catch {
                Write-Host "⚠️ Manual Qdrant setup required. Please visit https://qdrant.tech/documentation/guides/installation/" -ForegroundColor Red
            }
        }
    }
}
else {
    Write-Host "✅ Qdrant already running on port 6333" -ForegroundColor Green
}

# 3. Install Redis/KeyDB for Caching
Write-Host "`n3️⃣ Setting up Redis/KeyDB (Port 6379)..." -ForegroundColor Cyan

if (!(Test-Port 6379)) {
    # Check if Docker is available first
    try {
        docker --version | Out-Null
        Write-Host "🐳 Starting Redis with Docker..." -ForegroundColor Yellow
        docker run -d --name redis-optimization -p 6379:6379 redis:latest
        Start-Sleep 3
    }
    catch {
        # Install KeyDB (Redis alternative) for Windows
        $keydbPath = Join-Path $servicesDir "keydb"
        if (!(Test-Path "$keydbPath\keydb-server.exe")) {
            Write-Host "📥 Setting up KeyDB (Redis alternative)..." -ForegroundColor Yellow
            
            # KeyDB Windows installation via Chocolatey or manual
            try {
                if (Get-Command choco -ErrorAction SilentlyContinue) {
                    choco install keydb -y
                }
                else {
                    Write-Host "⚠️ Please install Redis manually or use Docker" -ForegroundColor Yellow
                    Write-Host "   Option 1: Install Docker and run: docker run -d -p 6379:6379 redis:latest" -ForegroundColor White
                    Write-Host "   Option 2: Install from https://github.com/microsoftarchive/redis/releases" -ForegroundColor White
                }
            }
            catch {
                Write-Host "⚠️ Redis/KeyDB installation failed. Using in-memory cache only." -ForegroundColor Yellow
            }
        }
    }
}
else {
    Write-Host "✅ Redis/KeyDB already running on port 6379" -ForegroundColor Green
}

# 4. Verify Services
Write-Host "`n🔍 Verifying Services..." -ForegroundColor Cyan

$services = @(
    @{ Name = "Ollama"; Port = 11434; Url = "http://localhost:11434/api/tags" },
    @{ Name = "Qdrant"; Port = 6333; Url = "http://localhost:6333/collections" },
    @{ Name = "Redis"; Port = 6379; Url = $null }
)

foreach ($service in $services) {
    if (Test-Port $service.Port) {
        Write-Host "✅ $($service.Name) running on port $($service.Port)" -ForegroundColor Green
        
        if ($service.Url) {
            try {
                $response = Invoke-RestMethod -Uri $service.Url -TimeoutSec 5
                Write-Host "   📡 API responding correctly" -ForegroundColor Gray
            }
            catch {
                Write-Host "   ⚠️ Service running but API not ready yet" -ForegroundColor Yellow
            }
        }
    }
    else {
        Write-Host "❌ $($service.Name) not running on port $($service.Port)" -ForegroundColor Red
    }
}

# 5. Create Service Management Scripts
Write-Host "`n📝 Creating management scripts..." -ForegroundColor Cyan

$startScript = @"
# Start Optimization Services
Write-Host "🚀 Starting SIMD JSON Optimization Services..." -ForegroundColor Cyan

# Start Ollama
if (!(Test-Port 11434)) {
    if (Test-Path "$servicesDir\ollama\ollama.exe") {
        Start-Process -FilePath "$servicesDir\ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Minimized
    }
}

# Start Qdrant (Docker)
if (!(Test-Port 6333)) {
    try {
        docker start qdrant-optimization
    }
    catch {
        docker run -d --name qdrant-optimization -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
    }
}

# Start Redis (Docker)
if (!(Test-Port 6379)) {
    try {
        docker start redis-optimization
    }
    catch {
        docker run -d --name redis-optimization -p 6379:6379 redis:latest
    }
}

Write-Host "✅ Services started. Wait 10 seconds for initialization..." -ForegroundColor Green
Start-Sleep 10
"@

$stopScript = @"
# Stop Optimization Services
Write-Host "🛑 Stopping SIMD JSON Optimization Services..." -ForegroundColor Yellow

# Stop Docker containers
try {
    docker stop qdrant-optimization redis-optimization
    Write-Host "✅ Docker services stopped" -ForegroundColor Green
}
catch {
    Write-Host "⚠️ Some Docker services may not be running" -ForegroundColor Yellow
}

# Stop Ollama
Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "✅ All services stopped" -ForegroundColor Green
"@

$startScript | Out-File -FilePath (Join-Path $PSScriptRoot "start-optimization-services.ps1") -Encoding UTF8
$stopScript | Out-File -FilePath (Join-Path $PSScriptRoot "stop-optimization-services.ps1") -Encoding UTF8

# 6. Create Docker Compose for easy management
$dockerCompose = @"
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant-optimization
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

  redis:
    image: redis:latest
    container_name: redis-optimization
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  qdrant_storage:
  redis_data:
"@

$dockerCompose | Out-File -FilePath (Join-Path $PSScriptRoot "docker-compose-optimization.yml") -Encoding UTF8

Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Run: docker-compose -f docker-compose-optimization.yml up -d" -ForegroundColor White
Write-Host "   2. Or use: .\start-optimization-services.ps1" -ForegroundColor White
Write-Host "   3. Test optimization API: http://localhost:5173/dev/copilot-optimizer" -ForegroundColor White
Write-Host "   4. Pull Ollama model: ollama pull nomic-embed-text" -ForegroundColor White

Write-Host "`n🔧 Service URLs:" -ForegroundColor Cyan
Write-Host "   - Ollama: http://localhost:11434" -ForegroundColor White
Write-Host "   - Qdrant: http://localhost:6333" -ForegroundColor White
Write-Host "   - Redis: localhost:6379" -ForegroundColor White

Write-Host "`n⚠️ If services don't start automatically:" -ForegroundColor Yellow
Write-Host "   - Install Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor White
Write-Host "   - Install Ollama: https://ollama.ai/download" -ForegroundColor White
Write-Host "   - Use the management scripts created in this directory" -ForegroundColor White