# Production Integration Check for Current Legal AI System
# Tests PostgreSQL, Qdrant, Ollama, Enhanced RAG, and Upload System

Write-Host "======================================================"
Write-Host "    LEGAL AI PRODUCTION SYSTEM INTEGRATION CHECK"
Write-Host "======================================================"
Write-Host ""

$results = @{
    PostgreSQL = $false
    Qdrant = $false
    Ollama = $false
    EnhancedRAG = $false
    UploadSystem = $false
    SvelteKit = $false
}

$errors = @()

# Test 1: PostgreSQL Connection
Write-Host "1. Testing PostgreSQL Connection..." -ForegroundColor Cyan
try {
    $pgResponse = Invoke-RestMethod -Uri "http://localhost:5173/api/health" -TimeoutSec 5 -Method GET
    if ($pgResponse.database.connected) {
        Write-Host "   ✅ PostgreSQL: Connected" -ForegroundColor Green
        Write-Host "   📊 Tables: $($pgResponse.database.tables) | Extensions: pgvector" -ForegroundColor Gray
        $results.PostgreSQL = $true
    } else {
        throw "Database not connected"
    }
} catch {
    Write-Host "   ❌ PostgreSQL: Failed - $($_.Exception.Message)" -ForegroundColor Red
    $errors += "PostgreSQL connection failed"
}

# Test 2: Qdrant Vector Database
Write-Host "2. Testing Qdrant Vector Database..." -ForegroundColor Cyan
try {
    $qdrantResponse = Invoke-RestMethod -Uri "http://localhost:6333/health" -TimeoutSec 5 -Method GET
    if ($qdrantResponse.title -eq "qdrant - vector search engine") {
        Write-Host "   ✅ Qdrant: Running on port 6333" -ForegroundColor Green
        $results.Qdrant = $true
    } else {
        throw "Qdrant not responding correctly"
    }
} catch {
    Write-Host "   ❌ Qdrant: Failed - $($_.Exception.Message)" -ForegroundColor Red
    $errors += "Qdrant vector database not available"
}

# Test 3: Ollama AI Service
Write-Host "3. Testing Ollama AI Service..." -ForegroundColor Cyan
try {
    $ollamaResponse = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 10 -Method GET
    $models = $ollamaResponse.models | Where-Object { $_.name -like "*gemma*" -or $_.name -like "*legal*" }
    
    if ($models.Count -gt 0) {
        Write-Host "   ✅ Ollama: Running with $($models.Count) legal models" -ForegroundColor Green
        foreach ($model in $models) {
            Write-Host "      📋 Model: $($model.name)" -ForegroundColor Gray
        }
        $results.Ollama = $true
    } else {
        Write-Host "   ⚠️ Ollama: Running but no legal models found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ❌ Ollama: Failed - $($_.Exception.Message)" -ForegroundColor Red
    $errors += "Ollama AI service not available"
}

# Test 4: Enhanced RAG Service
Write-Host "4. Testing Enhanced RAG Service..." -ForegroundColor Cyan
try {
    $ragResponse = Invoke-RestMethod -Uri "http://localhost:8094/health" -TimeoutSec 5 -Method GET
    Write-Host "   ✅ Enhanced RAG: Running on port 8094" -ForegroundColor Green
    $results.EnhancedRAG = $true
} catch {
    Write-Host "   ⚠️ Enhanced RAG: Not available (optional service)" -ForegroundColor Yellow
}

# Test 5: Production Upload System
Write-Host "5. Testing Production Upload System..." -ForegroundColor Cyan
try {
    $uploadResponse = Invoke-RestMethod -Uri "http://localhost:5173/api/production-upload" -TimeoutSec 5 -Method GET
    Write-Host "   ✅ Upload System: Endpoint accessible" -ForegroundColor Green
    $results.UploadSystem = $true
} catch {
    Write-Host "   ❌ Upload System: Failed - $($_.Exception.Message)" -ForegroundColor Red
    $errors += "Production upload system not accessible"
}

# Test 6: SvelteKit Dev Server
Write-Host "6. Testing SvelteKit Dev Server..." -ForegroundColor Cyan
try {
    $svelteResponse = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5 -Method GET
    if ($svelteResponse.StatusCode -eq 200) {
        Write-Host "   ✅ SvelteKit: Running on port 5173" -ForegroundColor Green
        $results.SvelteKit = $true
    } else {
        throw "SvelteKit server not responding correctly"
    }
} catch {
    Write-Host "   ❌ SvelteKit: Failed - $($_.Exception.Message)" -ForegroundColor Red
    $errors += "SvelteKit development server not running"
}

# Generate Report
Write-Host ""
Write-Host "======================================================"
Write-Host "              INTEGRATION CHECK RESULTS"
Write-Host "======================================================"
Write-Host ""

$passedTests = ($results.Values | Where-Object { $_ -eq $true }).Count
$totalTests = $results.Keys.Count

Write-Host "📊 Overall Status: $passedTests/$totalTests tests passed" -ForegroundColor $(if ($passedTests -eq $totalTests) { "Green" } else { "Yellow" })
Write-Host ""

Write-Host "🔍 Detailed Results:" -ForegroundColor Cyan
foreach ($test in $results.Keys) {
    $status = if ($results[$test]) { "✅ PASS" } else { "❌ FAIL" }
    $color = if ($results[$test]) { "Green" } else { "Red" }
    Write-Host "   $test`: $status" -ForegroundColor $color
}

if ($errors.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️ Issues Found:" -ForegroundColor Yellow
    foreach ($error in $errors) {
        Write-Host "   - $error" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🚀 Production System URLs:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:5173" -ForegroundColor Gray
Write-Host "   Upload Test: http://localhost:5173/upload-test" -ForegroundColor Gray
Write-Host "   API Health: http://localhost:5173/api/health" -ForegroundColor Gray
Write-Host "   Qdrant: http://localhost:6333/dashboard" -ForegroundColor Gray
Write-Host "   Ollama: http://localhost:11434" -ForegroundColor Gray

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Green
if ($passedTests -eq $totalTests) {
    Write-Host "   ✅ All systems operational! Ready for document processing." -ForegroundColor Green
    Write-Host "   📤 Test file upload: http://localhost:5173/upload-test" -ForegroundColor Green
    Write-Host "   💬 Test AI chat: Visit frontend and use legal AI assistant" -ForegroundColor Green
} else {
    Write-Host "   🔧 Fix failing services before proceeding" -ForegroundColor Yellow
    Write-Host "   📖 Check logs for detailed error information" -ForegroundColor Yellow
    Write-Host "   🔄 Run START-LEGAL-AI.bat to start missing services" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Integration check completed at $(Get-Date)" -ForegroundColor Gray