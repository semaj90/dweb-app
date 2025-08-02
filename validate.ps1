# Legal AI System - Quick Validation Script
# Validates critical system components are functioning

param(
    [switch]$Minimal = $false
)

$ErrorActionPreference = "Continue"

Write-Host @"
╔═══════════════════════════════════════════════════════╗
║        Legal AI System - Quick Validation              ║
╚═══════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

$validationResults = @{
    Docker = $false
    PostgreSQL = $false
    Redis = $false
    Qdrant = $false
    Ollama = $false
    VectorDimensions = $false
    Models = $false
    API = $false
}

# 1. Check Docker
Write-Host "`n[1/8] Checking Docker..." -ForegroundColor Yellow
try {
    $dockerPs = docker ps --format "table {{.Names}}\t{{.Status}}" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker is running" -ForegroundColor Green
        $validationResults.Docker = $true
    }
} catch {
    Write-Host "✗ Docker not available" -ForegroundColor Red
}

# 2. Check PostgreSQL
Write-Host "`n[2/8] Checking PostgreSQL..." -ForegroundColor Yellow
try {
    $pgCheck = docker exec legal_ai_postgres pg_isready -U postgres -d prosecutor_db 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ PostgreSQL is healthy" -ForegroundColor Green
        $validationResults.PostgreSQL = $true
    } else {
        Write-Host "✗ PostgreSQL not ready" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ PostgreSQL container not found" -ForegroundColor Red
}

# 3. Check Redis
Write-Host "`n[3/8] Checking Redis..." -ForegroundColor Yellow
try {
    $redisCheck = docker exec legal_ai_redis redis-cli ping 2>$null
    if ($redisCheck -eq "PONG") {
        Write-Host "✓ Redis is responding" -ForegroundColor Green
        $validationResults.Redis = $true
    } else {
        Write-Host "✗ Redis not responding" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Redis container not found" -ForegroundColor Red
}

# 4. Check Qdrant
Write-Host "`n[4/8] Checking Qdrant..." -ForegroundColor Yellow
try {
    $qdrantHealth = Invoke-RestMethod -Uri "http://localhost:6333/health" -Method GET -TimeoutSec 5
    if ($qdrantHealth.status -eq "ok") {
        Write-Host "✓ Qdrant is healthy" -ForegroundColor Green
        $validationResults.Qdrant = $true
    }
} catch {
    Write-Host "✗ Qdrant not accessible" -ForegroundColor Red
}

# 5. Check Vector Dimensions
if ($validationResults.Qdrant -and -not $Minimal) {
    Write-Host "`n[5/8] Checking vector dimensions..." -ForegroundColor Yellow
    try {
        $collections = Invoke-RestMethod -Uri "http://localhost:6333/collections" -Method GET
        $legalDocs = $collections.result.collections | Where-Object { $_.name -eq "legal_documents" }
        
        if ($legalDocs -and $legalDocs.config.params.vectors.size -eq 384) {
            Write-Host "✓ Vector dimensions correct (384)" -ForegroundColor Green
            $validationResults.VectorDimensions = $true
        } else {
            Write-Host "✗ Vector dimensions incorrect" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Cannot verify vector dimensions" -ForegroundColor Red
    }
} else {
    Write-Host "`n[5/8] Skipping vector dimension check..." -ForegroundColor Gray
}

# 6. Check Ollama
Write-Host "`n[6/8] Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5
    if ($ollamaCheck.models) {
        Write-Host "✓ Ollama is running" -ForegroundColor Green
        $validationResults.Ollama = $true
    }
} catch {
    Write-Host "✗ Ollama not accessible" -ForegroundColor Red
}

# 7. Check Models
if ($validationResults.Ollama -and -not $Minimal) {
    Write-Host "`n[7/8] Checking AI models..." -ForegroundColor Yellow
    try {
        $models = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET
        $hasEmbedding = $models.models.name -contains "nomic-embed-text"
        $hasLLM = ($models.models.name -like "*gemma*").Count -gt 0
        
        if ($hasEmbedding -and $hasLLM) {
            Write-Host "✓ Required models available" -ForegroundColor Green
            $validationResults.Models = $true
        } else {
            if (-not $hasEmbedding) { Write-Host "✗ Missing nomic-embed-text" -ForegroundColor Red }
            if (-not $hasLLM) { Write-Host "✗ Missing gemma model" -ForegroundColor Red }
        }
    } catch {
        Write-Host "✗ Cannot check models" -ForegroundColor Red
    }
} else {
    Write-Host "`n[7/8] Skipping model check..." -ForegroundColor Gray
}

# 8. Check API
Write-Host "`n[8/8] Checking API server..." -ForegroundColor Yellow
try {
    $apiHealth = Invoke-RestMethod -Uri "http://localhost:5173/api/health" -Method GET -TimeoutSec 5
    if ($apiHealth.status -eq "ok" -or $apiHealth.healthy) {
        Write-Host "✓ API server is running" -ForegroundColor Green
        $validationResults.API = $true
    }
} catch {
    Write-Host "✗ API server not running" -ForegroundColor Red
    Write-Host "  → Run: cd sveltekit-frontend && npm run dev" -ForegroundColor Gray
}

# Summary
Write-Host "`n════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan

$passed = ($validationResults.Values | Where-Object { $_ -eq $true }).Count
$total = $validationResults.Count

foreach ($component in $validationResults.Keys) {
    $status = $validationResults[$component]
    $icon = if ($status) { "✓" } else { "✗" }
    $color = if ($status) { "Green" } else { "Red" }
    Write-Host "$icon $component" -ForegroundColor $color
}

Write-Host "`nResult: $passed/$total components operational" -ForegroundColor $(if ($passed -eq $total) { "Green" } elseif ($passed -ge 5) { "Yellow" } else { "Red" })

# Quick fixes
if ($passed -lt $total) {
    Write-Host "`n💡 Quick Fixes:" -ForegroundColor Yellow
    
    if (-not $validationResults.Docker) {
        Write-Host "  1. Start Docker Desktop" -ForegroundColor White
    }
    
    if ($validationResults.Docker -and (-not $validationResults.PostgreSQL -or -not $validationResults.Redis -or -not $validationResults.Qdrant)) {
        Write-Host "  2. Run: docker-compose up -d" -ForegroundColor White
    }
    
    if (-not $validationResults.Models -and $validationResults.Ollama) {
        Write-Host "  3. Run: ollama pull nomic-embed-text && ollama pull gemma:2b" -ForegroundColor White
    }
    
    if (-not $validationResults.API) {
        Write-Host "  4. Run: cd sveltekit-frontend && npm run dev" -ForegroundColor White
    }
    
    if (-not $validationResults.VectorDimensions -and $validationResults.Qdrant) {
        Write-Host "  5. Run: .\health-check.ps1 -Fix" -ForegroundColor White
    }
}

# Test a simple operation if everything is working
if ($passed -eq $total -and -not $Minimal) {
    Write-Host "`n🧪 Running functionality test..." -ForegroundColor Cyan
    
    try {
        # Test embedding generation
        $testDoc = @{
            content = "This is a test legal document for validation"
            metadata = @{ test = $true }
        } | ConvertTo-Json
        
        $response = Invoke-RestMethod -Uri "http://localhost:5173/api/documents/embed" -Method POST -Body $testDoc -ContentType "application/json" -TimeoutSec 10
        
        if ($response.embedding -and $response.embedding.Count -eq 384) {
            Write-Host "✓ Embedding generation working (384 dimensions)" -ForegroundColor Green
            Write-Host "`n🎉 SYSTEM FULLY OPERATIONAL!" -ForegroundColor Green
        } else {
            Write-Host "✗ Embedding generation failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Functionality test failed: $_" -ForegroundColor Red
    }
}

# Exit code
exit $(if ($passed -eq $total) { 0 } else { 1 })
