# Test-Complete-Integration.ps1
# Comprehensive test script for Phase 14 Legal AI System

$ErrorActionPreference = "Continue"
Write-Host "========================================" -ForegroundColor Blue
Write-Host " LEGAL AI SYSTEM - INTEGRATION TEST" -ForegroundColor White
Write-Host " Phase 14 - Complete Stack Validation" -ForegroundColor Gray
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

$testResults = @()
$baseUrl = "http://localhost:8080"

# Test 1: Health Check
Write-Host "[TEST 1] Health Check..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    if ($health.status -eq "healthy") {
        Write-Host "  ✅ Server is healthy" -ForegroundColor Green
        Write-Host "  - Version: $($health.version)" -ForegroundColor Gray
        Write-Host "  - Workers: $($health.workers)" -ForegroundColor Gray
        Write-Host "  - Redis: $($health.services.redis)" -ForegroundColor Gray
        Write-Host "  - PostgreSQL: $($health.services.postgresql)" -ForegroundColor Gray
        Write-Host "  - Neo4j: $($health.services.neo4j)" -ForegroundColor Gray
        Write-Host "  - Qdrant: $($health.services.qdrant)" -ForegroundColor Gray
        Write-Host "  - Ollama: $($health.services.ollama)" -ForegroundColor Gray
        $testResults += @{Test="Health Check"; Status="PASS"}
    } else {
        Write-Host "  ❌ Server unhealthy" -ForegroundColor Red
        $testResults += @{Test="Health Check"; Status="FAIL"}
    }
} catch {
    Write-Host "  ❌ Health check failed: $_" -ForegroundColor Red
    $testResults += @{Test="Health Check"; Status="FAIL"}
}

Write-Host ""

# Test 2: GPU Status
Write-Host "[TEST 2] GPU Status..." -ForegroundColor Cyan
try {
    $gpu = Invoke-RestMethod -Uri "$baseUrl/gpu-status" -Method Get
    if ($gpu.gpu_available) {
        Write-Host "  ✅ GPU Available" -ForegroundColor Green
        Write-Host "  - CUDA Enabled: $($gpu.cuda_enabled)" -ForegroundColor Gray
        $testResults += @{Test="GPU Status"; Status="PASS"}
    } else {
        Write-Host "  ⚠️ GPU not available - CPU mode" -ForegroundColor Yellow
        $testResults += @{Test="GPU Status"; Status="WARN"}
    }
} catch {
    Write-Host "  ❌ GPU status check failed: $_" -ForegroundColor Red
    $testResults += @{Test="GPU Status"; Status="FAIL"}
}

Write-Host ""

# Test 3: Document Processing
Write-Host "[TEST 3] Document Processing..." -ForegroundColor Cyan
$testDoc = @{
    document_id = "test-$(Get-Date -Format 'yyyyMMddHHmmss')"
    content = "This is a test legal document. The agreement between Party A and Party B states that payment of `$10,000 is due by December 31, 2024. This contract is governed by the laws of California."
    document_type = "contract"
    case_id = "TEST-001"
    user_id = "test-user"
    options = @{
        extract_entities = $true
        generate_summary = $true
        assess_risk = $true
        generate_embedding = $true
        use_qdrant = $true
        update_neo4j = $true
        cache_results = $true
        gpu_accelerated = $true
    }
} | ConvertTo-Json

try {
    $result = Invoke-RestMethod -Uri "$baseUrl/process-document" `
        -Method Post `
        -ContentType "application/json" `
        -Body $testDoc
    
    if ($result.success) {
        Write-Host "  ✅ Document processed successfully" -ForegroundColor Green
        Write-Host "  - Job ID: $($result.job_id)" -ForegroundColor Gray
        Write-Host "  - Processing Time: $($result.processing_time)" -ForegroundColor Gray
        if ($result.results.summary) {
            Write-Host "  - Summary Generated: Yes" -ForegroundColor Gray
        }
        if ($result.results.entities) {
            Write-Host "  - Entities Extracted: $($result.results.entities.Count)" -ForegroundColor Gray
        }
        if ($result.results.embedding) {
            Write-Host "  - Embedding Generated: Yes" -ForegroundColor Gray
        }
        $testResults += @{Test="Document Processing"; Status="PASS"}
    } else {
        Write-Host "  ❌ Document processing failed" -ForegroundColor Red
        $testResults += @{Test="Document Processing"; Status="FAIL"}
    }
} catch {
    Write-Host "  ❌ Document processing error: $_" -ForegroundColor Red
    $testResults += @{Test="Document Processing"; Status="FAIL"}
}

Write-Host ""

# Test 4: AI Assistant
Write-Host "[TEST 4] AI Assistant..." -ForegroundColor Cyan
$aiRequest = @{
    user_id = "test-user"
    session_id = "test-session-$(Get-Date -Format 'yyyyMMddHHmmss')"
    query = "What are the key terms in the contract?"
    context = @()
    state = "idle"
    ui_context = @{
        component = "bits-ui"
        action = "query"
        data = @{
            caseId = "TEST-001"
        }
    }
} | ConvertTo-Json

try {
    $aiResponse = Invoke-RestMethod -Uri "$baseUrl/ai-assistant" `
        -Method Post `
        -ContentType "application/json" `
        -Body $aiRequest
    
    if ($aiResponse.success) {
        Write-Host "  ✅ AI Assistant responded" -ForegroundColor Green
        $responseText = $aiResponse.response.text
        if ($responseText.Length -gt 100) {
            $responseText = $responseText.Substring(0, 100) + "..."
        }
        Write-Host "  - Response: $responseText" -ForegroundColor Gray
        $testResults += @{Test="AI Assistant"; Status="PASS"}
    } else {
        Write-Host "  ❌ AI Assistant failed" -ForegroundColor Red
        $testResults += @{Test="AI Assistant"; Status="FAIL"}
    }
} catch {
    Write-Host "  ❌ AI Assistant error: $_" -ForegroundColor Red
    $testResults += @{Test="AI Assistant"; Status="FAIL"}
}

Write-Host ""

# Test 5: Vector Search (Qdrant)
Write-Host "[TEST 5] Vector Search (Qdrant)..." -ForegroundColor Cyan
$searchRequest = @{
    query = "legal contract payment terms"
    limit = 5
} | ConvertTo-Json

try {
    $searchResults = Invoke-RestMethod -Uri "$baseUrl/qdrant-search" `
        -Method Post `
        -ContentType "application/json" `
        -Body $searchRequest
    
    if ($searchResults.success) {
        Write-Host "  ✅ Vector search completed" -ForegroundColor Green
        Write-Host "  - Results found: $($searchResults.results.Count)" -ForegroundColor Gray
        $testResults += @{Test="Vector Search"; Status="PASS"}
    } else {
        Write-Host "  ⚠️ Vector search returned no results" -ForegroundColor Yellow
        $testResults += @{Test="Vector Search"; Status="WARN"}
    }
} catch {
    Write-Host "  ❌ Vector search error: $_" -ForegroundColor Red
    $testResults += @{Test="Vector Search"; Status="FAIL"}
}

Write-Host ""

# Test 6: Recommendations
Write-Host "[TEST 6] Recommendations..." -ForegroundColor Cyan
$recRequest = @{
    user_id = "test-user"
    case_id = "TEST-001"
    action_type = "upload"
    entity_ids = @()
} | ConvertTo-Json

try {
    $recommendations = Invoke-RestMethod -Uri "$baseUrl/recommendations" `
        -Method Post `
        -ContentType "application/json" `
        -Body $recRequest
    
    if ($recommendations.success) {
        Write-Host "  ✅ Recommendations generated" -ForegroundColor Green
        Write-Host "  - Count: $($recommendations.recommendations.Count)" -ForegroundColor Gray
        $testResults += @{Test="Recommendations"; Status="PASS"}
    } else {
        Write-Host "  ⚠️ No recommendations generated" -ForegroundColor Yellow
        $testResults += @{Test="Recommendations"; Status="WARN"}
    }
} catch {
    Write-Host "  ❌ Recommendations error: $_" -ForegroundColor Red
    $testResults += @{Test="Recommendations"; Status="FAIL"}
}

Write-Host ""

# Test 7: Cache Operations
Write-Host "[TEST 7] Cache Operations..." -ForegroundColor Cyan
$cacheData = @{
    key = "test:cache:$(Get-Date -Format 'yyyyMMddHHmmss')"
    value = @{
        test = "data"
        timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
    ttl = 3600
} | ConvertTo-Json

try {
    # Set cache
    $setCacheResult = Invoke-RestMethod -Uri "$baseUrl/cache" `
        -Method Post `
        -ContentType "application/json" `
        -Body $cacheData
    
    if ($setCacheResult.success) {
        # Get cache
        $cacheKey = ($cacheData | ConvertFrom-Json).key
        $getCacheResult = Invoke-RestMethod -Uri "$baseUrl/cache/$cacheKey" -Method Get
        
        if ($getCacheResult.value) {
            Write-Host "  ✅ Cache operations working" -ForegroundColor Green
            $testResults += @{Test="Cache Operations"; Status="PASS"}
        } else {
            Write-Host "  ❌ Cache retrieval failed" -ForegroundColor Red
            $testResults += @{Test="Cache Operations"; Status="FAIL"}
        }
    } else {
        Write-Host "  ❌ Cache set failed" -ForegroundColor Red
        $testResults += @{Test="Cache Operations"; Status="FAIL"}
    }
} catch {
    Write-Host "  ❌ Cache operations error: $_" -ForegroundColor Red
    $testResults += @{Test="Cache Operations"; Status="FAIL"}
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Blue
Write-Host " TEST SUMMARY" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

$passCount = ($testResults | Where-Object { $_.Status -eq "PASS" }).Count
$failCount = ($testResults | Where-Object { $_.Status -eq "FAIL" }).Count
$warnCount = ($testResults | Where-Object { $_.Status -eq "WARN" }).Count

foreach ($result in $testResults) {
    $color = switch ($result.Status) {
        "PASS" { "Green" }
        "FAIL" { "Red" }
        "WARN" { "Yellow" }
    }
    $icon = switch ($result.Status) {
        "PASS" { "✅" }
        "FAIL" { "❌" }
        "WARN" { "⚠️" }
    }
    Write-Host "$icon $($result.Test): $($result.Status)" -ForegroundColor $color
}

Write-Host ""
Write-Host "Results: $passCount Passed, $failCount Failed, $warnCount Warnings" -ForegroundColor $(if ($failCount -eq 0) { "Green" } else { "Red" })

if ($failCount -eq 0) {
    Write-Host ""
    Write-Host "🎉 All critical tests passed! System is operational." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Open http://localhost:5173 in your browser" -ForegroundColor White
    Write-Host "  2. Try uploading a document" -ForegroundColor White
    Write-Host "  3. Use the AI Assistant (bottom right)" -ForegroundColor White
    Write-Host "  4. Check recommendations in the dashboard" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "⚠️ Some tests failed. Please check:" -ForegroundColor Yellow
    Write-Host "  1. All services are running (use COMPLETE-AI-SYSTEM-STARTUP.bat)" -ForegroundColor White
    Write-Host "  2. Database migrations are applied" -ForegroundColor White
    Write-Host "  3. Ollama models are installed (gemma3-legal, nomic-embed-text)" -ForegroundColor White
    Write-Host "  4. Check logs for detailed error messages" -ForegroundColor White
}

Write-Host ""
Write-Host "Test completed at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
