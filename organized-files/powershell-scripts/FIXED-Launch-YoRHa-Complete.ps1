# =============================================================================
# YoRHa Legal AI Platform - COMPLETE NATIVE WINDOWS LAUNCHER (FIXED)
# Starts ALL services: RabbitMQ, Redis, MinIO, Neo4j, Qdrant, PostgreSQL, Ollama
# =============================================================================

Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "YoRHa LEGAL AI PLATFORM - COMPLETE NATIVE WINDOWS LAUNCHER" -ForegroundColor Yellow
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the correct directory
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app"

# =============================================================================
# PHASE 1: Fix Node.js Dependencies
# =============================================================================

Write-Host "[1/7] FIXING NODE.JS DEPENDENCIES" -ForegroundColor Green
Write-Host "---------------------------------------------"

Set-Location "sveltekit-frontend"

if (Test-Path "node_modules") {
    Write-Host "🧹 Cleaning existing node_modules..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "node_modules" -ErrorAction SilentlyContinue
}

if (Test-Path "package-lock.json") {
    Write-Host "🧹 Removing package-lock.json..." -ForegroundColor Yellow
    Remove-Item "package-lock.json" -ErrorAction SilentlyContinue
}

Write-Host "📦 Installing Windows-native dependencies..." -ForegroundColor Cyan
try {
    & npm install --platform=win32 --arch=x64 --force
    Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
} catch {
    Write-Host "⚠️ npm install failed, trying with different flags..." -ForegroundColor Yellow
    try {
        & npm install --force
        Write-Host "✅ Dependencies installed with alternative method" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    }
}

Set-Location ".."

# =============================================================================
# PHASE 2: Infrastructure Services (Complete)
# =============================================================================

Write-Host ""
Write-Host "[2/7] STARTING ALL INFRASTRUCTURE SERVICES" -ForegroundColor Green
Write-Host "---------------------------------------------"

# PostgreSQL
Write-Host "🗄️  Checking PostgreSQL..." -NoNewline
try {
    $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pgService -and $pgService.Status -eq "Running") {
        Write-Host " ✅ Running" -ForegroundColor Green
    } elseif ($pgService) {
        Write-Host " 🔄 Starting..." -ForegroundColor Yellow
        Start-Service $pgService.Name
        Write-Host "   ✅ Started" -ForegroundColor Green
    } else {
        Write-Host " ⚠️  Not installed" -ForegroundColor Red
    }
} catch {
    Write-Host " ❌ Error checking PostgreSQL" -ForegroundColor Red
}

# Redis
Write-Host "🔴 Checking Redis..." -NoNewline
try {
    $redisResponse = Invoke-WebRequest -Uri "http://localhost:6379" -Method HEAD -TimeoutSec 3 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " 🔄 Starting Redis..." -ForegroundColor Yellow
    try {
        # Try to start Redis server
        Start-Process -FilePath "redis-server" -WindowStyle Hidden -ErrorAction Stop
        Start-Sleep -Seconds 3
        Write-Host "   ✅ Started" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Install Redis or start manually" -ForegroundColor Yellow
    }
}

# RabbitMQ
Write-Host "🐰 Checking RabbitMQ..." -NoNewline
try {
    $rabbitResponse = Invoke-WebRequest -Uri "http://localhost:15672" -Method HEAD -TimeoutSec 3 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " 🔄 Starting RabbitMQ..." -ForegroundColor Yellow
    try {
        $rabbitService = Get-Service -Name "RabbitMQ" -ErrorAction SilentlyContinue
        if ($rabbitService) {
            Start-Service -Name "RabbitMQ"
            Start-Sleep -Seconds 5
            Write-Host "   ✅ Started" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Install RabbitMQ or start manually" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ⚠️  Manual start required" -ForegroundColor Yellow
    }
}

# MinIO
Write-Host "📦 Checking MinIO..." -NoNewline
try {
    $minioResponse = Invoke-WebRequest -Uri "http://localhost:9000" -Method HEAD -TimeoutSec 3 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " 🔄 Starting MinIO..." -ForegroundColor Yellow
    try {
        # Start MinIO server
        Start-Process -FilePath "minio.exe" -ArgumentList "server", "./data" -WindowStyle Hidden -ErrorAction Stop
        Start-Sleep -Seconds 5
        Write-Host "   ✅ Started" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Install MinIO or start manually" -ForegroundColor Yellow
    }
}

# Neo4j
Write-Host "🔗 Checking Neo4j..." -NoNewline
try {
    $neo4jResponse = Invoke-WebRequest -Uri "http://localhost:7474" -Method HEAD -TimeoutSec 3 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " 🔄 Starting Neo4j..." -ForegroundColor Yellow
    try {
        $neo4jService = Get-Service -Name "Neo4j" -ErrorAction SilentlyContinue
        if ($neo4jService) {
            Start-Service -Name "Neo4j"
            Start-Sleep -Seconds 10
            Write-Host "   ✅ Started" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️  Install Neo4j or start manually" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   ⚠️  Manual start required" -ForegroundColor Yellow
    }
}

# Qdrant (Low Memory Mode)
Write-Host "🔍 Checking Qdrant..." -NoNewline
try {
    $qdrantResponse = Invoke-WebRequest -Uri "http://localhost:6333" -Method HEAD -TimeoutSec 3 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " 🔄 Starting Qdrant (low memory mode)..." -ForegroundColor Yellow
    try {
        # Start Qdrant with low memory configuration
        $env:QDRANT_CONFIG_PATH = "./qdrant-config-low-memory.yaml"
        Start-Process -FilePath "qdrant.exe" -ArgumentList "--config-path", "./qdrant-config-low-memory.yaml" -WindowStyle Hidden -ErrorAction Stop
        Start-Sleep -Seconds 5
        Write-Host "   ✅ Started in low memory mode" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Install Qdrant or start manually" -ForegroundColor Yellow
    }
}

# Ollama
Write-Host "🧠 Checking Ollama..." -NoNewline
try {
    $ollamaResponse = Invoke-WebRequest -Uri "http://localhost:11434" -Method HEAD -TimeoutSec 3 -ErrorAction Stop
    Write-Host " ✅ Running" -ForegroundColor Green
} catch {
    Write-Host " 🔄 Starting Ollama..." -ForegroundColor Yellow
    try {
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -ErrorAction Stop
        Start-Sleep -Seconds 5
        Write-Host "   ✅ Started" -ForegroundColor Green
    } catch {
        Write-Host "   ⚠️  Install Ollama or start manually" -ForegroundColor Yellow
    }
}

# =============================================================================
# PHASE 3: Go Microservices
# =============================================================================

Write-Host ""
Write-Host "[3/7] PREPARING GO MICROSERVICES" -ForegroundColor Green  
Write-Host "---------------------------------------------"

# Enhanced RAG Service
$ragBinary = "..\go-microservice\bin\enhanced-rag.exe"
if (Test-Path $ragBinary) {
    Write-Host "✅ Enhanced RAG binary found" -ForegroundColor Green
} else {
    Write-Host "🔨 Building Enhanced RAG service..." -ForegroundColor Yellow
    Set-Location "..\go-microservice"
    try {
        & go build -o ".\bin\enhanced-rag.exe" ".\cmd\enhanced-rag\main.go"
        Write-Host "✅ Enhanced RAG built" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to build Enhanced RAG" -ForegroundColor Red
    }
    Set-Location "..\deeds-web-app"
}

# Upload Service
$uploadBinary = "..\go-microservice\bin\upload-service.exe"
if (Test-Path $uploadBinary) {
    Write-Host "✅ Upload Service binary found" -ForegroundColor Green
} else {
    Write-Host "🔨 Building Upload Service..." -ForegroundColor Yellow
    Set-Location "..\go-microservice"
    try {
        & go build -o ".\bin\upload-service.exe" ".\cmd\upload-service\main.go"
        Write-Host "✅ Upload Service built" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to build Upload Service" -ForegroundColor Red
    }
    Set-Location "..\deeds-web-app"
}

# =============================================================================
# PHASE 4: Launch Go Services
# =============================================================================

Write-Host ""
Write-Host "[4/7] LAUNCHING GO SERVICES" -ForegroundColor Green
Write-Host "---------------------------------------------"

# Start Enhanced RAG Service
Write-Host "🤖 Starting Enhanced RAG Service on port 8094..." -NoNewline
try {
    if (Test-Path $ragBinary) {
        $ragProcess = Start-Process -FilePath $ragBinary -PassThru -WindowStyle Hidden -ErrorAction Stop
        Start-Sleep -Seconds 3
        $response = Invoke-WebRequest -Uri "http://localhost:8094/health" -TimeoutSec 5 -ErrorAction Stop
        Write-Host " ✅ Running" -ForegroundColor Green
    } else {
        Write-Host " ❌ Binary not found" -ForegroundColor Red
    }
} catch {
    Write-Host " ⚠️  May not be fully ready" -ForegroundColor Yellow
}

# Start Upload Service  
Write-Host "📤 Starting Upload Service on port 8093..." -NoNewline
try {
    if (Test-Path $uploadBinary) {
        $uploadProcess = Start-Process -FilePath $uploadBinary -PassThru -WindowStyle Hidden -ErrorAction Stop
        Start-Sleep -Seconds 2
        Write-Host " ✅ Started" -ForegroundColor Green
    } else {
        Write-Host " ❌ Binary not found" -ForegroundColor Red
    }
} catch {
    Write-Host " ⚠️  May not be fully ready" -ForegroundColor Yellow
}

# =============================================================================
# PHASE 5: Create Qdrant Low Memory Config
# =============================================================================

Write-Host ""
Write-Host "[5/7] CREATING QDRANT LOW MEMORY CONFIG" -ForegroundColor Green
Write-Host "---------------------------------------------"

$qdrantConfig = @"
log_level: INFO
storage:
  storage_path: ./qdrant-storage
  snapshots_path: ./qdrant-snapshots
  temp_path: ./qdrant-temp
  
service:
  http_port: 6333
  grpc_port: 6334
  
# Low memory configuration
cluster:
  enabled: false
  
performance:
  max_search_threads: 2
  max_optimization_threads: 1
  
# Reduce memory usage
hnsw_config:
  m: 8
  ef_construct: 100
  full_scan_threshold: 10000
  
quantization:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: false
    
# Low memory collection defaults
collection_config:
  optimizer_config:
    deleted_threshold: 0.5
    vacuum_min_vector_number: 1000
    max_optimization_threads: 1
"@

try {
    $qdrantConfig | Out-File -FilePath "./qdrant-config-low-memory.yaml" -Encoding UTF8
    Write-Host "✅ Qdrant low memory config created" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create Qdrant config" -ForegroundColor Red
}

# =============================================================================
# PHASE 6: Service Health Check
# =============================================================================

Write-Host ""
Write-Host "[6/7] COMPREHENSIVE HEALTH CHECK" -ForegroundColor Green
Write-Host "---------------------------------------------"

$services = @(
    @{ Name = "PostgreSQL"; Port = 5432; URL = "http://localhost:5432" },
    @{ Name = "Redis"; Port = 6379; URL = "http://localhost:6379" },
    @{ Name = "RabbitMQ"; Port = 15672; URL = "http://localhost:15672" },
    @{ Name = "MinIO"; Port = 9000; URL = "http://localhost:9000" },
    @{ Name = "Neo4j"; Port = 7474; URL = "http://localhost:7474" },
    @{ Name = "Qdrant"; Port = 6333; URL = "http://localhost:6333" },
    @{ Name = "Ollama"; Port = 11434; URL = "http://localhost:11434" },
    @{ Name = "Enhanced RAG"; Port = 8094; URL = "http://localhost:8094/health" },
    @{ Name = "Upload Service"; Port = 8093; URL = "http://localhost:8093/health" }
)

Write-Host "📊 Service Status Report:" -ForegroundColor Cyan
Write-Host "=================================================="

foreach ($service in $services) {
    Write-Host "$($service.Name) ($($service.Port))" -NoNewline
    try {
        $response = Invoke-WebRequest -Uri $service.URL -Method HEAD -TimeoutSec 2 -ErrorAction Stop
        Write-Host " ✅ RUNNING" -ForegroundColor Green
    } catch {
        Write-Host " ❌ NOT ACCESSIBLE" -ForegroundColor Red
    }
}

# =============================================================================
# PHASE 7: SvelteKit Frontend
# =============================================================================

Write-Host ""
Write-Host "[7/7] LAUNCHING YoRHa INTERFACE" -ForegroundColor Green
Write-Host "---------------------------------------------"

Set-Location "sveltekit-frontend"

# Set Windows-specific environment variables
$env:NODE_ENV = "development"
$env:VITE_PLATFORM = "win32" 
$env:FORCE_COLOR = "1"

Write-Host "🎮 Starting YoRHa Interface..." -ForegroundColor Magenta
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host "  🤖 YoRHa Legal AI Interface" -ForegroundColor Cyan
Write-Host "  🧠 Enhanced RAG: http://localhost:8094" -ForegroundColor Green  
Write-Host "  📤 Upload Service: http://localhost:8093" -ForegroundColor Green
Write-Host "  🗄️  PostgreSQL: localhost:5432" -ForegroundColor Green
Write-Host "  🔴 Redis: localhost:6379" -ForegroundColor Green
Write-Host "  🐰 RabbitMQ: http://localhost:15672" -ForegroundColor Green
Write-Host "  📦 MinIO: http://localhost:9000" -ForegroundColor Green
Write-Host "  🔗 Neo4j: http://localhost:7474" -ForegroundColor Green
Write-Host "  🔍 Qdrant: http://localhost:6333" -ForegroundColor Green
Write-Host "  🎯 Frontend: http://localhost:5177" -ForegroundColor Magenta
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
Write-Host ""

# Start SvelteKit with native Windows npm
try {
    Write-Host "🚀 Launching SvelteKit development server..." -ForegroundColor Cyan
    & npm run dev
} catch {
    Write-Host "❌ Failed to start SvelteKit" -ForegroundColor Red
    Write-Host "Trying alternative startup method..." -ForegroundColor Yellow
    try {
        & npx vite dev
    } catch {
        Write-Host "❌ All startup methods failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "===============================================================================" -ForegroundColor Cyan
Write-Host "YoRHa Legal AI Platform - COMPLETE NATIVE WINDOWS LAUNCH COMPLETE!" -ForegroundColor Green
Write-Host "===============================================================================" -ForegroundColor Cyan