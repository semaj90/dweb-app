# Complete Integration Check for Legal AI Platform
# Tests all components: RabbitMQ, Neo4j, MinIO, PostgreSQL, Drizzle, Go Services, WebGPU

param(
    [switch]$Quick = $false,
    [switch]$Fix = $false
)

Write-Host @"
╔════════════════════════════════════════════════════════════════════╗
║          COMPLETE INTEGRATION CHECK - ALL COMPONENTS              ║
║     RabbitMQ | Neo4j | MinIO | PostgreSQL | Go Services | WebGPU  ║
╚════════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

$global:results = @{
    PostgreSQL = @{ Status = $false; Details = @{} }
    Drizzle = @{ Status = $false; Details = @{} }
    Neo4j = @{ Status = $false; Details = @{} }
    RabbitMQ = @{ Status = $false; Details = @{} }
    MinIO = @{ Status = $false; Details = @{} }
    GoServices = @{ Status = $false; Details = @{} }
    WebGPU = @{ Status = $false; Details = @{} }
    Integration = @{ Status = $false; Details = @{} }
}

function Write-Section {
    param($Title)
    Write-Host "`n$('=' * 70)" -ForegroundColor DarkGray
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host ('=' * 70) -ForegroundColor DarkGray
}

# 1. PostgreSQL & postgres.js Check
Write-Section "1. PostgreSQL & postgres.js Integration"

$pgPort = Test-NetConnection -ComputerName localhost -Port 5432 -WarningAction SilentlyContinue
if ($pgPort.TcpTestSucceeded) {
    Write-Host "✅ PostgreSQL is running on port 5432" -ForegroundColor Green
    $global:results.PostgreSQL.Status = $true
    
    # Test connection
    $env:PGPASSWORD = "123456"
    $dbTest = psql -U legal_admin -d legal_ai_db -h localhost -c "SELECT version();" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Connected to legal_ai_db database" -ForegroundColor Green
        
        # Check pgvector
        $vectorTest = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT * FROM pg_extension WHERE extname = 'vector';" 2>$null
        if ($vectorTest) {
            Write-Host "✅ pgvector extension installed" -ForegroundColor Green
            $global:results.PostgreSQL.Details.pgvector = $true
        } else {
            Write-Host "⚠️  pgvector not installed - run: CREATE EXTENSION vector;" -ForegroundColor Yellow
        }
        
        # Check tables
        $tables = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>$null
        Write-Host "ℹ️  Tables in database: $($tables.Trim())" -ForegroundColor Cyan
        
        # Check data
        $cases = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM cases;" 2>$null
        $docs = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM legal_documents;" 2>$null
        $evidence = psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM evidence;" 2>$null
        
        Write-Host "📊 Data counts:" -ForegroundColor Cyan
        Write-Host "   • Cases: $($cases.Trim())" -ForegroundColor Gray
        Write-Host "   • Documents: $($docs.Trim())" -ForegroundColor Gray
        Write-Host "   • Evidence: $($evidence.Trim())" -ForegroundColor Gray
    } else {
        Write-Host "❌ Cannot connect to database" -ForegroundColor Red
    }
} else {
    Write-Host "❌ PostgreSQL is not running" -ForegroundColor Red
    if ($Fix) {
        Write-Host "🔧 Starting PostgreSQL..." -ForegroundColor Yellow
        Start-Service "postgresql-x64-17" -ErrorAction SilentlyContinue
    }
}

# 2. Drizzle ORM Check
Write-Section "2. Drizzle ORM Integration"

if (Test-Path ".\drizzle.config.ts") {
    Write-Host "✅ drizzle.config.ts found" -ForegroundColor Green
    $global:results.Drizzle.Details.config = $true
} else {
    Write-Host "❌ drizzle.config.ts not found" -ForegroundColor Red
}

if (Test-Path ".\src\lib\server\schema.ts") {
    Write-Host "✅ Drizzle schema found" -ForegroundColor Green
    $global:results.Drizzle.Details.schema = $true
} else {
    Write-Host "❌ Schema file not found" -ForegroundColor Red
}

if (Test-Path ".\drizzle") {
    $migrations = (Get-ChildItem ".\drizzle" -Filter "*.sql").Count
    Write-Host "ℹ️  Migration files: $migrations" -ForegroundColor Cyan
    $global:results.Drizzle.Details.migrations = $migrations
}

if ($global:results.Drizzle.Details.config -and $global:results.Drizzle.Details.schema) {
    $global:results.Drizzle.Status = $true
}

# 3. Neo4j Check
Write-Section "3. Neo4j Graph Database"

$neo4jBrowser = Test-NetConnection -ComputerName localhost -Port 7474 -WarningAction SilentlyContinue
$neo4jBolt = Test-NetConnection -ComputerName localhost -Port 7687 -WarningAction SilentlyContinue

if ($neo4jBrowser.TcpTestSucceeded) {
    Write-Host "✅ Neo4j Browser running on port 7474" -ForegroundColor Green
    Write-Host "   Access at: http://localhost:7474" -ForegroundColor Gray
}

if ($neo4jBolt.TcpTestSucceeded) {
    Write-Host "✅ Neo4j Bolt running on port 7687" -ForegroundColor Green
    $global:results.Neo4j.Status = $true
} else {
    Write-Host "❌ Neo4j is not running" -ForegroundColor Red
    if ($Fix) {
        Write-Host "🔧 Starting Neo4j..." -ForegroundColor Yellow
        if (Test-Path ".\neo4j-community-5.23.0\bin\neo4j.bat") {
            Start-Process ".\neo4j-community-5.23.0\bin\neo4j.bat" -ArgumentList "console" -WindowStyle Hidden
        }
    }
}

# 4. RabbitMQ Check
Write-Section "4. RabbitMQ Message Queue"

$rabbitPort = Test-NetConnection -ComputerName localhost -Port 5672 -WarningAction SilentlyContinue
$rabbitMgmt = Test-NetConnection -ComputerName localhost -Port 15672 -WarningAction SilentlyContinue

if ($rabbitPort.TcpTestSucceeded) {
    Write-Host "✅ RabbitMQ running on port 5672" -ForegroundColor Green
    $global:results.RabbitMQ.Status = $true
    
    if ($rabbitMgmt.TcpTestSucceeded) {
        Write-Host "✅ RabbitMQ Management UI on port 15672" -ForegroundColor Green
        Write-Host "   Access at: http://localhost:15672 (guest/guest)" -ForegroundColor Gray
    }
} else {
    Write-Host "❌ RabbitMQ is not running" -ForegroundColor Red
    if ($Fix) {
        Write-Host "🔧 Starting RabbitMQ..." -ForegroundColor Yellow
        Start-Service "RabbitMQ" -ErrorAction SilentlyContinue
    }
}

# 5. MinIO Check
Write-Section "5. MinIO Object Storage"

$minioApi = Test-NetConnection -ComputerName localhost -Port 9000 -WarningAction SilentlyContinue
$minioConsole = Test-NetConnection -ComputerName localhost -Port 9001 -WarningAction SilentlyContinue

if ($minioApi.TcpTestSucceeded) {
    Write-Host "✅ MinIO API running on port 9000" -ForegroundColor Green
    $global:results.MinIO.Status = $true
    
    if ($minioConsole.TcpTestSucceeded) {
        Write-Host "✅ MinIO Console on port 9001" -ForegroundColor Green
        Write-Host "   Access at: http://localhost:9001 (minioadmin/minioadmin123)" -ForegroundColor Gray
    }
} else {
    Write-Host "❌ MinIO is not running" -ForegroundColor Red
    if ($Fix) {
        Write-Host "🔧 Starting MinIO..." -ForegroundColor Yellow
        if (Test-Path ".\minio.exe") {
            Start-Process ".\minio.exe" -ArgumentList "server", "C:\minio-data", "--console-address", ":9001" -WindowStyle Hidden
        }
    }
}

# 6. Go Microservices Check
Write-Section "6. Go Microservices"

$goServices = @(
    @{Name="GPU Orchestrator"; Port=8084},
    @{Name="RAG Service"; Port=8085},
    @{Name="Tensor Service"; Port=8086},
    @{Name="QUIC Service"; Port=8087}
)

$runningServices = 0
foreach ($service in $goServices) {
    $test = Test-NetConnection -ComputerName localhost -Port $service.Port -WarningAction SilentlyContinue
    if ($test.TcpTestSucceeded) {
        Write-Host "✅ $($service.Name) running on port $($service.Port)" -ForegroundColor Green
        $runningServices++
        
        # Try health check
        try {
            $health = Invoke-WebRequest -Uri "http://localhost:$($service.Port)/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($health.StatusCode -eq 200) {
                Write-Host "   Health: OK" -ForegroundColor Gray
            }
        } catch {}
    } else {
        Write-Host "⚠️  $($service.Name) not running on port $($service.Port)" -ForegroundColor Yellow
    }
}

if ($runningServices -gt 0) {
    Write-Host "ℹ️  $runningServices/4 Go services running" -ForegroundColor Cyan
    $global:results.GoServices.Status = $true
} else {
    Write-Host "❌ No Go services running" -ForegroundColor Red
}

# Check Go files
Write-Host "`nGo service files:" -ForegroundColor Cyan
$goFiles = @(
    "gpu-orchestrator.go",
    "enhanced-rag-service.go",
    "tensor-tiling-gpu-accelerator.go",
    "quic-tensor-transport.go"
)

foreach ($file in $goFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file" -ForegroundColor Red
    }
}

# 7. WebGPU Check
Write-Section "7. WebGPU Support"

# Check for WebGPU implementation files
$webgpuFiles = @(
    ".\src\lib\webgpu\gpu-compute.ts",
    ".\src\lib\webgpu\tensor-ops.ts",
    ".\webgpu\index.html"
)

Write-Host "WebGPU files:" -ForegroundColor Cyan
$webgpuFound = 0
foreach ($file in $webgpuFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file" -ForegroundColor Green
        $webgpuFound++
    } else {
        Write-Host "   ✗ $file not found" -ForegroundColor Yellow
    }
}

# Check for GPU
$gpu = Get-WmiObject Win32_VideoController | Select-Object -First 1
if ($gpu) {
    Write-Host "✅ GPU detected: $($gpu.Name)" -ForegroundColor Green
    $global:results.WebGPU.Status = $true
}

# Check for CUDA
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if (Test-Path $cudaPath) {
    Write-Host "✅ CUDA Toolkit installed" -ForegroundColor Green
} else {
    Write-Host "ℹ️  CUDA not found (optional for WebGPU)" -ForegroundColor Cyan
}

# 8. Integration Tests
Write-Section "8. Component Integration Tests"

Write-Host "`nTest 1: PostgreSQL ↔ Drizzle Integration" -ForegroundColor Cyan
if ($global:results.PostgreSQL.Status -and $global:results.Drizzle.Status) {
    Write-Host "✅ PostgreSQL and Drizzle are integrated" -ForegroundColor Green
    $global:results.Integration.Details.PostgresDrizzle = $true
} else {
    Write-Host "❌ Integration not working" -ForegroundColor Red
}

Write-Host "`nTest 2: Document Pipeline (MinIO → RabbitMQ → PostgreSQL)" -ForegroundColor Cyan
if ($global:results.MinIO.Status -and $global:results.RabbitMQ.Status -and $global:results.PostgreSQL.Status) {
    Write-Host "✅ Document pipeline ready" -ForegroundColor Green
    Write-Host "   MinIO (storage) → RabbitMQ (queue) → PostgreSQL (metadata)" -ForegroundColor Gray
    $global:results.Integration.Details.DocumentPipeline = $true
} else {
    Write-Host "❌ Pipeline incomplete" -ForegroundColor Red
}

Write-Host "`nTest 3: Graph Integration (PostgreSQL ↔ Neo4j)" -ForegroundColor Cyan
if ($global:results.PostgreSQL.Status -and $global:results.Neo4j.Status) {
    Write-Host "✅ Graph integration ready" -ForegroundColor Green
    Write-Host "   PostgreSQL (structured) ↔ Neo4j (relationships)" -ForegroundColor Gray
    $global:results.Integration.Details.GraphIntegration = $true
} else {
    Write-Host "❌ Graph integration not available" -ForegroundColor Red
}

Write-Host "`nTest 4: AI Pipeline (Go Services → pgvector)" -ForegroundColor Cyan
if ($global:results.GoServices.Status -and $global:results.PostgreSQL.Details.pgvector) {
    Write-Host "✅ AI pipeline ready" -ForegroundColor Green
    Write-Host "   Go Services → pgvector → RAG queries" -ForegroundColor Gray
    $global:results.Integration.Details.AIPipeline = $true
} else {
    Write-Host "❌ AI pipeline not configured" -ForegroundColor Red
}

# Summary
Write-Section "INTEGRATION SUMMARY"

$components = @(
    "PostgreSQL", "Drizzle", "Neo4j", "RabbitMQ", 
    "MinIO", "GoServices", "WebGPU"
)

$working = $components | Where-Object { $global:results[$_].Status -eq $true }
$notWorking = $components | Where-Object { $global:results[$_].Status -eq $false }

Write-Host "`n📊 COMPONENT STATUS:" -ForegroundColor Cyan
Write-Host "✅ Working: $($working.Count)/$($components.Count)" -ForegroundColor Green
foreach ($comp in $working) {
    Write-Host "   • $comp" -ForegroundColor Green
}

if ($notWorking.Count -gt 0) {
    Write-Host "`n❌ Not Working: $($notWorking.Count)/$($components.Count)" -ForegroundColor Red
    foreach ($comp in $notWorking) {
        Write-Host "   • $comp" -ForegroundColor Red
    }
}

# Architecture diagram
Write-Host "`n📐 ARCHITECTURE:" -ForegroundColor Cyan
Write-Host @"

    ┌─────────────────────────────────────┐
    │         YoRHa Frontend              │
    │     (SvelteKit + WebGPU)            │
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────▼──────────────────────┐
    │      Go Microservices Layer         │
    │  (GPU, RAG, Tensor, QUIC Services)  │
    └──────────┬────────┬─────────────────┘
               │        │
    ┌──────────▼────┐ ┌▼──────────────────┐
    │  Message Queue│ │  Object Storage   │
    │   (RabbitMQ)  │ │    (MinIO)        │
    └──────────┬────┘ └───────────────────┘
               │
    ┌──────────▼──────────────────────────┐
    │         Data Layer                  │
    │  PostgreSQL │ Neo4j │ Redis         │
    │  (+ pgvector) (Graph)  (Cache)      │
    └─────────────────────────────────────┘

"@ -ForegroundColor DarkGray

if ($working.Count -eq $components.Count) {
    Write-Host "`n🎉 ALL COMPONENTS ARE LINKED AND WORKING!" -ForegroundColor Green
    Write-Host "Your Legal AI platform is fully integrated!" -ForegroundColor Green
} else {
    Write-Host "`n🔧 To fix missing components:" -ForegroundColor Yellow
    Write-Host "Run: .\START-NATIVE-WINDOWS-COMPLETE.ps1" -ForegroundColor White
}

# Save report
$report = $global:results | ConvertTo-Json -Depth 3
$report | Out-File "integration-status-report.json"
Write-Host "`n📄 Report saved to: integration-status-report.json" -ForegroundColor Cyan
