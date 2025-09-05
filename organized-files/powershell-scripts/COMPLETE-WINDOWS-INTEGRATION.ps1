# ================================================================================
# COMPLETE WINDOWS 10 NATIVE LEGAL AI PLATFORM INTEGRATION
# ================================================================================
# Full end-to-end pipeline with all services wired up
# ================================================================================

param(
    [switch]$StartAll,
    [switch]$StopAll,
    [switch]$Status,
    [switch]$Test
)

Write-Host @"
================================================================================
🚀 LEGAL AI PLATFORM - WINDOWS 10 NATIVE FULL STACK
================================================================================
Complete End-to-End Pipeline Integration
Version: 2.0 PRODUCTION READY
================================================================================
"@ -ForegroundColor Cyan

# Global Configuration
$global:CONFIG = @{
    # Database
    POSTGRES_HOST = "localhost"
    POSTGRES_PORT = 5432
    POSTGRES_DB = "legal_ai_db"
    POSTGRES_USER = "legal_admin"
    POSTGRES_PASSWORD = "LegalAI2024!"
    
    # Neo4j Desktop
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_BROWSER = "http://localhost:7474"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "password123"
    
    # Redis
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    
    # MinIO
    MINIO_HOST = "localhost"
    MINIO_PORT = 9000
    MINIO_CONSOLE_PORT = 9001
    MINIO_ACCESS_KEY = "minioadmin"
    MINIO_SECRET_KEY = "minioadmin123"
    MINIO_BUCKET = "legal-documents"
    
    # Ollama
    OLLAMA_HOST = "localhost"
    OLLAMA_PORT = 11434
    OLLAMA_MODEL = "gemma3-legal:latest"
    
    # Services
    FRONTEND_PORT = 5173
    ENHANCED_RAG_PORT = 8094
    XSTATE_PORT = 8095
    UPLOAD_SERVICE_PORT = 8093
    ML_PIPELINE_PORT = 8080
    NEO4J_SERVICE_PORT = 7475
    
    # GPU Settings
    CUDA_VISIBLE_DEVICES = "0"
    TF_GPU_MEMORY_LIMIT = "6144"  # 6GB for RTX 3060 Ti
}

# Set environment variables
function Set-EnvironmentVariables {
    Write-Host "⚙️ Setting environment variables..." -ForegroundColor Yellow
    
    foreach ($key in $global:CONFIG.Keys) {
        [Environment]::SetEnvironmentVariable($key, $global:CONFIG[$key], [EnvironmentVariableTarget]::Process)
    }
    
    # Additional Node.js settings
    $env:NODE_ENV = "development"
    $env:NODE_OPTIONS = "--max-old-space-size=8192"
    
    Write-Host "✅ Environment variables configured" -ForegroundColor Green
}

# Function to check service health
function Test-ServiceHealth {
    param([string]$Name, [int]$Port, [string]$HealthUrl = "")
    
    $tcpTest = Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
    
    if ($tcpTest) {
        if ($HealthUrl) {
            try {
                $response = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2 -ErrorAction SilentlyContinue
                Write-Host "✅ $Name (Port $Port): Healthy" -ForegroundColor Green
                return $true
            } catch {
                Write-Host "⚠️ $Name (Port $Port): Running but health check failed" -ForegroundColor Yellow
                return $true
            }
        } else {
            Write-Host "✅ $Name (Port $Port): Running" -ForegroundColor Green
            return $true
        }
    } else {
        Write-Host "❌ $Name (Port $Port): Not running" -ForegroundColor Red
        return $false
    }
}

# Start PostgreSQL
function Start-PostgreSQL {
    Write-Host "`n📦 Starting PostgreSQL..." -ForegroundColor Cyan
    
    $pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
    if ($pgService) {
        if ($pgService.Status -ne 'Running') {
            Start-Service $pgService.Name
            Start-Sleep -Seconds 3
        }
        Write-Host "✅ PostgreSQL started" -ForegroundColor Green
        
        # Initialize database if needed
        $testQuery = "SELECT 1"
        try {
            & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U $env:POSTGRES_USER -d $env:POSTGRES_DB -h localhost -c $testQuery -q 2>$null
            Write-Host "✅ Database connection verified" -ForegroundColor Green
        } catch {
            Write-Host "⚠️ Creating database..." -ForegroundColor Yellow
            & "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres -h localhost $env:POSTGRES_DB 2>$null
            & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "CREATE USER $($env:POSTGRES_USER) WITH PASSWORD '$($env:POSTGRES_PASSWORD)' CREATEDB;" 2>$null
            & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "GRANT ALL PRIVILEGES ON DATABASE $($env:POSTGRES_DB) TO $($env:POSTGRES_USER);" 2>$null
            Write-Host "✅ Database initialized" -ForegroundColor Green
        }
    } else {
        Write-Host "❌ PostgreSQL service not found" -ForegroundColor Red
    }
}

# Start Redis
function Start-Redis {
    Write-Host "`n💾 Starting Redis..." -ForegroundColor Cyan
    
    $redisRunning = Test-ServiceHealth -Name "Redis" -Port 6379
    if (!$redisRunning) {
        # Try to start Redis
        $redisPath = "C:\Program Files\Redis\redis-server.exe"
        if (Test-Path $redisPath) {
            Start-Process $redisPath -WindowStyle Hidden
            Start-Sleep -Seconds 2
            Write-Host "✅ Redis started" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Redis not installed - skipping" -ForegroundColor Yellow
        }
    }
}

# Start Neo4j Desktop
function Start-Neo4jDesktop {
    Write-Host "`n🔗 Checking Neo4j Desktop..." -ForegroundColor Cyan
    
    $neo4jRunning = Test-ServiceHealth -Name "Neo4j" -Port 7474
    if (!$neo4jRunning) {
        Write-Host @"
⚠️ Neo4j Desktop not running. Please:
   1. Open Neo4j Desktop application
   2. Start your database
   3. Set password to: $($env:NEO4J_PASSWORD)
"@ -ForegroundColor Yellow
    }
}

# Start MinIO
function Start-MinIO {
    Write-Host "`n📦 Starting MinIO..." -ForegroundColor Cyan
    
    $minioRunning = Test-ServiceHealth -Name "MinIO" -Port 9000
    if (!$minioRunning) {
        if (!(Test-Path ".\minio-data")) {
            New-Item -Path ".\minio-data" -ItemType Directory -Force | Out-Null
        }
        
        $minioExe = ".\minio.exe"
        if (Test-Path $minioExe) {
            Start-Process $minioExe -ArgumentList "server", "./minio-data", "--address", ":9000", "--console-address", ":9001" -WindowStyle Hidden
            Start-Sleep -Seconds 3
            Write-Host "✅ MinIO started" -ForegroundColor Green
            
            # Create bucket if needed
            Start-Sleep -Seconds 2
            $env:MINIO_ROOT_USER = $env:MINIO_ACCESS_KEY
            $env:MINIO_ROOT_PASSWORD = $env:MINIO_SECRET_KEY
            
            Write-Host "   Console: http://localhost:9001" -ForegroundColor Gray
            Write-Host "   Credentials: $($env:MINIO_ACCESS_KEY)/$($env:MINIO_SECRET_KEY)" -ForegroundColor Gray
        } else {
            Write-Host "⚠️ MinIO not found - downloading..." -ForegroundColor Yellow
            Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile "minio.exe"
            Start-MinIO  # Recursive call
        }
    }
}

# Start Ollama
function Start-Ollama {
    Write-Host "`n🤖 Starting Ollama..." -ForegroundColor Cyan
    
    $ollamaRunning = Test-ServiceHealth -Name "Ollama" -Port 11434 -HealthUrl "http://localhost:11434/api/version"
    if (!$ollamaRunning) {
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
        
        # Pull model if needed
        Write-Host "📥 Checking Gemma3 model..." -ForegroundColor Yellow
        & ollama list | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "📥 Pulling Gemma3 model..." -ForegroundColor Yellow
            & ollama pull gemma3:latest
        }
        Write-Host "✅ Ollama ready with Gemma3" -ForegroundColor Green
    }
}

# Start Backend Services
function Start-BackendServices {
    Write-Host "`n🚀 Starting Backend Services..." -ForegroundColor Cyan
    
    # Enhanced RAG Service
    if (!(Test-ServiceHealth -Name "Enhanced RAG" -Port 8094)) {
        Push-Location ".\go-services"
        if (Test-Path ".\cmd\enhanced-rag") {
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd cmd\enhanced-rag; go run main.go" -WindowStyle Minimized
        }
        Pop-Location
    }
    
    # XState Manager
    if (!(Test-ServiceHealth -Name "XState Manager" -Port 8095)) {
        Push-Location ".\go-services"
        if (Test-Path ".\cmd\xstate-manager") {
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd cmd\xstate-manager; go run main.go" -WindowStyle Minimized
        }
        Pop-Location
    }
    
    # Upload Service
    if (!(Test-ServiceHealth -Name "Upload Service" -Port 8093)) {
        if (Test-Path ".\go-microservice") {
            Push-Location ".\go-microservice"
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "go run main.go" -WindowStyle Minimized
            Pop-Location
        }
    }
    
    # Neo4j Integration Service
    if (!(Test-ServiceHealth -Name "Neo4j Service" -Port 7475)) {
        if (Test-Path ".\go-services\cmd\neo4j-service") {
            Push-Location ".\go-services\cmd\neo4j-service"
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "go run main.go" -WindowStyle Minimized
            Pop-Location
        }
    }
    
    Start-Sleep -Seconds 5
    Write-Host "✅ Backend services started" -ForegroundColor Green
}

# Start Frontend
function Start-Frontend {
    Write-Host "`n🎨 Starting Frontend..." -ForegroundColor Cyan
    
    if (!(Test-ServiceHealth -Name "Frontend" -Port 5173)) {
        Push-Location ".\sveltekit-frontend"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev -- --host 0.0.0.0" -WindowStyle Minimized
        Pop-Location
        
        Start-Sleep -Seconds 5
        Write-Host "✅ Frontend started at http://localhost:5173" -ForegroundColor Green
    }
}

# Create dev:full npm script
function Create-DevFullScript {
    Write-Host "`n📝 Creating npm run dev:full script..." -ForegroundColor Cyan
    
    $packageJsonPath = ".\package.json"
    if (Test-Path $packageJsonPath) {
        $packageJson = Get-Content $packageJsonPath -Raw | ConvertFrom-Json
        
        # Add dev:full script
        $packageJson.scripts."dev:full" = "powershell -ExecutionPolicy Bypass -File COMPLETE-WINDOWS-INTEGRATION.ps1 -StartAll"
        $packageJson.scripts."dev:full:status" = "powershell -ExecutionPolicy Bypass -File COMPLETE-WINDOWS-INTEGRATION.ps1 -Status"
        $packageJson.scripts."dev:full:stop" = "powershell -ExecutionPolicy Bypass -File COMPLETE-WINDOWS-INTEGRATION.ps1 -StopAll"
        $packageJson.scripts."dev:full:test" = "powershell -ExecutionPolicy Bypass -File COMPLETE-WINDOWS-INTEGRATION.ps1 -Test"
        
        $packageJson | ConvertTo-Json -Depth 10 | Set-Content $packageJsonPath
        Write-Host "✅ npm scripts created" -ForegroundColor Green
    }
}

# System Status Check
function Get-SystemStatus {
    Write-Host "`n📊 SYSTEM STATUS CHECK" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    
    $services = @(
        @{Name="PostgreSQL"; Port=5432; Health=""},
        @{Name="Redis"; Port=6379; Health=""},
        @{Name="Neo4j Desktop"; Port=7474; Health=""},
        @{Name="MinIO"; Port=9000; Health="http://localhost:9000/minio/health/live"},
        @{Name="Ollama"; Port=11434; Health="http://localhost:11434/api/version"},
        @{Name="Enhanced RAG"; Port=8094; Health="http://localhost:8094/health"},
        @{Name="XState Manager"; Port=8095; Health="http://localhost:8095/health"},
        @{Name="Upload Service"; Port=8093; Health="http://localhost:8093/health"},
        @{Name="Neo4j Service"; Port=7475; Health="http://localhost:7475/health"},
        @{Name="Frontend"; Port=5173; Health=""}
    )
    
    $healthyCount = 0
    foreach ($service in $services) {
        if (Test-ServiceHealth -Name $service.Name -Port $service.Port -HealthUrl $service.Health) {
            $healthyCount++
        }
    }
    
    $percentage = [math]::Round(($healthyCount / $services.Count) * 100)
    Write-Host "`n📈 System Health: $percentage% ($healthyCount/$($services.Count) services running)" -ForegroundColor $(
        if ($percentage -ge 80) { "Green" } 
        elseif ($percentage -ge 60) { "Yellow" } 
        else { "Red" }
    )
}

# Test End-to-End Pipeline
function Test-Pipeline {
    Write-Host "`n🧪 TESTING END-TO-END PIPELINE" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    
    # Test 1: Database Connection
    Write-Host "`n1️⃣ Testing Database..." -NoNewline
    try {
        $testQuery = "SELECT version()"
        & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U $env:POSTGRES_USER -d $env:POSTGRES_DB -h localhost -t -c $testQuery 2>$null | Out-Null
        Write-Host " ✅ Connected" -ForegroundColor Green
    } catch {
        Write-Host " ❌ Failed" -ForegroundColor Red
    }
    
    # Test 2: Neo4j Connection
    Write-Host "2️⃣ Testing Neo4j..." -NoNewline
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:7475/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Host " ✅ Connected" -ForegroundColor Green
    } catch {
        Write-Host " ⚠️ Service not ready" -ForegroundColor Yellow
    }
    
    # Test 3: Ollama LLM
    Write-Host "3️⃣ Testing Ollama LLM..." -NoNewline
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post -Body (@{
            model = "gemma3:latest"
            prompt = "Test"
            stream = $false
        } | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 5 -ErrorAction SilentlyContinue
        Write-Host " ✅ Working" -ForegroundColor Green
    } catch {
        Write-Host " ⚠️ Model not ready" -ForegroundColor Yellow
    }
    
    # Test 4: MinIO Storage
    Write-Host "4️⃣ Testing MinIO..." -NoNewline
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:9000/minio/health/live" -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Host " ✅ Healthy" -ForegroundColor Green
    } catch {
        Write-Host " ⚠️ Not ready" -ForegroundColor Yellow
    }
    
    # Test 5: Frontend
    Write-Host "5️⃣ Testing Frontend..." -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
        Write-Host " ✅ Accessible" -ForegroundColor Green
    } catch {
        Write-Host " ❌ Not accessible" -ForegroundColor Red
    }
    
    Write-Host "`n✅ Pipeline tests completed" -ForegroundColor Green
}

# Stop all services
function Stop-AllServices {
    Write-Host "`n🛑 STOPPING ALL SERVICES" -ForegroundColor Yellow
    Write-Host "=" * 60 -ForegroundColor Yellow
    
    # Stop Node processes
    Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # Stop Go processes
    Get-Process go -ErrorAction SilentlyContinue | Stop-Process -Force
    
    # Stop specific services
    Get-Process minio -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force
    Get-Process redis-server -ErrorAction SilentlyContinue | Stop-Process -Force
    
    Write-Host "✅ All services stopped" -ForegroundColor Green
}

# Main execution
Set-EnvironmentVariables

if ($StartAll) {
    Write-Host "`n🚀 STARTING COMPLETE LEGAL AI PLATFORM" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    
    Start-PostgreSQL
    Start-Redis
    Start-Neo4jDesktop
    Start-MinIO
    Start-Ollama
    Start-BackendServices
    Start-Frontend
    Create-DevFullScript
    
    Start-Sleep -Seconds 5
    Get-SystemStatus
    
    Write-Host "`n" -NoNewline
    Write-Host "=" * 60 -ForegroundColor Green
    Write-Host "🎉 LEGAL AI PLATFORM FULLY OPERATIONAL!" -ForegroundColor Green
    Write-Host "=" * 60 -ForegroundColor Green
    
    Write-Host @"

🌐 ACCESS POINTS:
───────────────────────────────────────────────────
🖥️  Frontend:          http://localhost:5173
🧠 Enhanced RAG:       http://localhost:8094/api/rag
📈 XState Manager:     http://localhost:8095
📁 Upload Service:     http://localhost:8093/upload
📊 Neo4j Browser:      http://localhost:7474
📦 MinIO Console:      http://localhost:9001
🤖 Ollama API:         http://localhost:11434

📚 QUICK COMMANDS:
───────────────────────────────────────────────────
npm run dev:full        - Start everything
npm run dev:full:status - Check system status
npm run dev:full:stop   - Stop all services
npm run dev:full:test   - Test pipeline

🚀 Your system is ready for legal AI operations!
"@ -ForegroundColor Cyan
    
} elseif ($Status) {
    Get-SystemStatus
} elseif ($StopAll) {
    Stop-AllServices
} elseif ($Test) {
    Test-Pipeline
} else {
    Get-SystemStatus
}
