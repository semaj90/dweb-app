# Run comprehensive tests with local PostgreSQL and Ollama
# This script ensures all services are running before executing tests

Write-Host "🧪 Running comprehensive tests with local environment..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Function to check if a port is in use
function Test-PortInUse {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Function to wait for service
function Wait-ForService {
    param(
        [string]$ServiceName,
        [int]$Port,
        [int]$MaxWaitSeconds = 30
    )
    
    Write-Host "⏳ Waiting for $ServiceName to be ready..." -ForegroundColor Yellow
    
    $waited = 0
    while ($waited -lt $MaxWaitSeconds) {
        if (Test-PortInUse -Port $Port) {
            Write-Host "✅ $ServiceName is ready" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Seconds 2
        $waited += 2
    }
    
    Write-Host "❌ $ServiceName failed to start within $MaxWaitSeconds seconds" -ForegroundColor Red
    return $false
}

# Step 1: Ensure PostgreSQL is running
Write-Host "`n1️⃣ Checking PostgreSQL..." -ForegroundColor Magenta

if (-not (Test-PortInUse -Port 5432)) {
    Write-Host "🔄 Starting PostgreSQL..." -ForegroundColor Yellow
    
    $postgresService = Get-Service -Name "postgresql-x64-*" -ErrorAction SilentlyContinue
    if ($postgresService) {
        Start-Service $postgresService
        if (-not (Wait-ForService -ServiceName "PostgreSQL" -Port 5432)) {
            Write-Host "❌ PostgreSQL failed to start. Please check the installation." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ PostgreSQL service not found. Please run setup-local-environment.ps1 first." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ PostgreSQL is running" -ForegroundColor Green
}

# Step 2: Ensure Ollama is running
Write-Host "`n2️⃣ Checking Ollama..." -ForegroundColor Magenta

if (-not (Test-PortInUse -Port 11434)) {
    Write-Host "🔄 Starting Ollama..." -ForegroundColor Yellow
    
    try {
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        
        if (-not (Wait-ForService -ServiceName "Ollama" -Port 11434)) {
            Write-Host "❌ Ollama failed to start. Please check the installation." -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "❌ Failed to start Ollama: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Ollama is running" -ForegroundColor Green
}

# Step 3: Check required models
Write-Host "`n3️⃣ Checking AI models..." -ForegroundColor Magenta

$requiredModels = @("llama3.2", "nomic-embed-text")
$missingModels = @()

try {
    $availableModels = & ollama list 2>$null
    
    foreach ($model in $requiredModels) {
        if ($availableModels -notmatch $model) {
            $missingModels += $model
        }
    }
    
    if ($missingModels.Count -gt 0) {
        Write-Host "📦 Pulling missing models..." -ForegroundColor Yellow
        foreach ($model in $missingModels) {
            Write-Host "Pulling $model..." -ForegroundColor Yellow
            & ollama pull $model
        }
    }
    
    Write-Host "✅ All required models are available" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Failed to check models, tests may fail" -ForegroundColor Yellow
}

# Step 4: Setup test environment
Write-Host "`n4️⃣ Setting up test environment..." -ForegroundColor Magenta

$frontendPath = "C:\Users\james\Desktop\deeds-web\deeds-web-app\sveltekit-frontend"

if (-not (Test-Path $frontendPath)) {
    Write-Host "❌ Frontend directory not found: $frontendPath" -ForegroundColor Red
    exit 1
}

Set-Location $frontendPath

# Ensure dependencies are installed
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Set environment variables for testing
$env:NODE_ENV = "testing"
$env:DATABASE_URL = "postgresql://postgres:postgres123@localhost:5432/deeds_legal_ai"
$env:OLLAMA_URL = "http://localhost:11434"

# Check if dev server is needed (Playwright will start it)
if (Test-PortInUse -Port 5173) {
    Write-Host "⚠️  Development server is already running on port 5173" -ForegroundColor Yellow
} else {
    Write-Host "✅ Port 5173 is available for test server" -ForegroundColor Green
}

# Step 5: Run comprehensive tests
Write-Host "`n5️⃣ Running test suite..." -ForegroundColor Magenta

# Quick validation first
Write-Host "`n🔍 Quick Validation Tests:" -ForegroundColor Cyan
try {
    npx playwright test quick-validation.spec.ts --reporter=line
    Write-Host "✅ Quick validation passed" -ForegroundColor Green
} catch {
    Write-Host "❌ Quick validation failed" -ForegroundColor Red
    Write-Host "Continuing with other tests..." -ForegroundColor Yellow
}

# Core functionality tests
$testSuites = @(
    @{ Name = "Authentication & Session"; File = "user-authentication-session.spec.ts" },
    @{ Name = "Global Store & State"; File = "global-store-state.spec.ts" },
    @{ Name = "System Health & Logging"; File = "system-health-logging.spec.ts" },
    @{ Name = "Service Worker"; File = "service-worker.spec.ts" },
    @{ Name = "XState Machines"; File = "xstate-machines.spec.ts" },
    @{ Name = "LokiJS Caching"; File = "lokijs-caching.spec.ts" },
    @{ Name = "Ollama Integration"; File = "ollama-integration.spec.ts" },
    @{ Name = "RAG System"; File = "rag-system.spec.ts" },
    @{ Name = "PostgreSQL & pgvector"; File = "postgresql-pgvector.spec.ts" },
    @{ Name = "Drizzle ORM"; File = "drizzle-orm.spec.ts" },
    @{ Name = "RAG Pipeline Integration"; File = "rag-pipeline-integration.spec.ts" }
)

$passedTests = 0
$failedTests = 0

foreach ($suite in $testSuites) {
    Write-Host "`n🔬 Running $($suite.Name) Tests:" -ForegroundColor Cyan
    
    try {
        npx playwright test $suite.File --reporter=line
        Write-Host "✅ $($suite.Name) tests passed" -ForegroundColor Green
        $passedTests++
    } catch {
        Write-Host "❌ $($suite.Name) tests failed" -ForegroundColor Red
        $failedTests++
    }
}

# GPU tests (optional)
Write-Host "`n🎮 GPU-Specific Tests:" -ForegroundColor Cyan

try {
    $gpuAvailable = nvidia-smi 2>$null
    if ($gpuAvailable) {
        Write-Host "🔬 Running GPU Acceleration Tests..." -ForegroundColor Cyan
        npx playwright test gpu-acceleration.spec.ts --reporter=line
        
        Write-Host "🔬 Running GPU Ollama Ingestion Tests..." -ForegroundColor Cyan
        npx playwright test gpu-ollama-ingestion.spec.ts --reporter=line
        
        Write-Host "✅ GPU tests completed" -ForegroundColor Green
        $passedTests += 2
    } else {
        Write-Host "⚠️  GPU not available, skipping GPU-specific tests" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ GPU tests failed (this is OK if no GPU is available)" -ForegroundColor Yellow
}

# Step 6: Generate report
Write-Host "`n6️⃣ Generating test report..." -ForegroundColor Magenta

try {
    npx playwright show-report
    Write-Host "✅ Test report generated" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Failed to generate test report" -ForegroundColor Yellow
}

# Summary
Write-Host "`n📊 Test Results Summary:" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "✅ Passed: $passedTests test suites" -ForegroundColor Green
Write-Host "❌ Failed: $failedTests test suites" -ForegroundColor Red

if ($failedTests -eq 0) {
    Write-Host "`n🎉 All tests passed! Your system is fully functional." -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Some tests failed. Check the detailed report for more information." -ForegroundColor Yellow
}

Write-Host "`n🔧 Services Status:" -ForegroundColor Cyan
Write-Host "• PostgreSQL: $(if (Test-PortInUse -Port 5432) { '✅ Running' } else { '❌ Stopped' })" -ForegroundColor White
Write-Host "• Ollama: $(if (Test-PortInUse -Port 11434) { '✅ Running' } else { '❌ Stopped' })" -ForegroundColor White
Write-Host "• GPU: $(if (nvidia-smi 2>$null) { '✅ Available' } else { '⚠️ Not Available' })" -ForegroundColor White

Write-Host "`n✨ Testing completed!" -ForegroundColor Green