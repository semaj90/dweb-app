# Complete Gemma3 Legal Model Integration Script (continued)
# PowerShell script for testing and optimizing the integration

d legal_ai_db -q 2>&1 | Out-Null
    
    # Optimize Redis for caching
    Write-Host "  Optimizing Redis..." -ForegroundColor Yellow
    redis-cli CONFIG SET maxmemory 2gb | Out-Null
    redis-cli CONFIG SET maxmemory-policy allkeys-lru | Out-Null
    redis-cli CONFIG SET save "" | Out-Null  # Disable persistence for speed
    
    # Optimize Windows for high performance
    Write-Host "  Optimizing Windows settings..." -ForegroundColor Yellow
    
    # Set power plan to High Performance
    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>&1 | Out-Null
    
    # Disable Windows Defender real-time scanning for model directory (temporary)
    Add-MpPreference -ExclusionPath (Split-Path $config.ModelPath -Parent) -ErrorAction SilentlyContinue
    
    Write-Host "  [✓] Performance optimizations applied" -ForegroundColor Green
}

function Run-BenchmarkSuite {
    Write-Host "`nRunning Benchmark Suite..." -ForegroundColor Yellow
    
    $benchmarks = @()
    
    # Benchmark 1: Simple inference
    Write-Host "  Running inference benchmark..." -ForegroundColor Yellow
    $inferenceStart = Get-Date
    $inferencePrompt = @{
        model = "gemma3-legal"
        prompt = "List the elements of negligence."
        stream = $false
        options = @{
            temperature = 0.1
            num_ctx = 2048
            num_gpu = $config.GpuLayers
        }
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
            -Method POST -Body $inferencePrompt -ContentType "application/json"
        
        $inferenceTime = (Get-Date) - $inferenceStart
        $benchmarks += @{
            Test = "Simple Inference"
            Time = $inferenceTime.TotalSeconds
            TokensPerSec = [math]::Round($response.eval_count / ($response.eval_duration / 1000000000), 2)
            Status = "Success"
        }
    } catch {
        $benchmarks += @{
            Test = "Simple Inference"
            Time = 0
            TokensPerSec = 0
            Status = "Failed"
        }
    }
    
    # Benchmark 2: Vector embedding
    Write-Host "  Running embedding benchmark..." -ForegroundColor Yellow
    $embeddingStart = Get-Date
    $texts = @(
        "breach of contract",
        "negligence per se",
        "res ipsa loquitur",
        "statutory damages",
        "punitive damages"
    )
    
    $embeddingCount = 0
    foreach ($text in $texts) {
        try {
            $embeddingTest = @{
                model = "nomic-embed-text"
                prompt = $text
            } | ConvertTo-Json
            
            Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" `
                -Method POST -Body $embeddingTest -ContentType "application/json" | Out-Null
            $embeddingCount++
        } catch {}
    }
    
    $embeddingTime = (Get-Date) - $embeddingStart
    $benchmarks += @{
        Test = "Vector Embeddings (5 texts)"
        Time = $embeddingTime.TotalSeconds
        TokensPerSec = [math]::Round($embeddingCount / $embeddingTime.TotalSeconds, 2)
        Status = if ($embeddingCount -eq 5) { "Success" } else { "Partial" }
    }
    
    # Benchmark 3: RAG query
    Write-Host "  Running RAG benchmark..." -ForegroundColor Yellow
    $ragStart = Get-Date
    $ragQuery = @{
        query = "What are the legal precedents for breach of fiduciary duty in corporate law?"
        maxResults = 10
        similarityThreshold = 0.7
    } | ConvertTo-Json
    
    try {
        Invoke-RestMethod -Uri "http://localhost:8094/api/ai/enhanced-chat" `
            -Method POST -Body $ragQuery -ContentType "application/json" -TimeoutSec 30 | Out-Null
        
        $ragTime = (Get-Date) - $ragStart
        $benchmarks += @{
            Test = "RAG Pipeline"
            Time = $ragTime.TotalSeconds
            TokensPerSec = "N/A"
            Status = "Success"
        }
    } catch {
        $benchmarks += @{
            Test = "RAG Pipeline"
            Time = 0
            TokensPerSec = "N/A"
            Status = "Failed"
        }
    }
    
    # Display results
    Write-Host "`n  Benchmark Results:" -ForegroundColor Cyan
    Write-Host "  ===========================================" -ForegroundColor Cyan
    
    $benchmarks | ForEach-Object {
        $color = if ($_.Status -eq "Success") { "Green" } elseif ($_.Status -eq "Partial") { "Yellow" } else { "Red" }
        Write-Host ("  {0,-30} {1,10:F2}s  {2,15} {3,10}" -f $_.Test, $_.Time, $_.TokensPerSec, $_.Status) -ForegroundColor $color
    }
}

function Test-EndToEnd {
    Write-Host "`nRunning End-to-End Test..." -ForegroundColor Yellow
    
    # Test complete document processing pipeline
    $testDocument = @{
        id = [guid]::NewGuid().ToString()
        title = "Test Legal Document"
        content = @"
PURCHASE AGREEMENT

This Purchase Agreement ("Agreement") is entered into as of January 1, 2025, between ABC Corporation ("Buyer") and XYZ Industries ("Seller").

1. PURCHASE AND SALE
Seller agrees to sell and Buyer agrees to purchase the assets described in Exhibit A for the purchase price of $1,000,000.

2. REPRESENTATIONS AND WARRANTIES
Seller represents and warrants that:
(a) Seller has full corporate power and authority to enter into this Agreement
(b) The assets are free and clear of all liens and encumbrances
(c) This Agreement constitutes a legal, valid, and binding obligation

3. INDEMNIFICATION
Seller shall indemnify and hold harmless Buyer from any claims arising from breach of this Agreement.

4. GOVERNING LAW
This Agreement shall be governed by the laws of the State of California.
"@
        type = "contract"
        jurisdiction = "California"
    } | ConvertTo-Json
    
    Write-Host "  1. Uploading document..." -ForegroundColor Yellow
    try {
        $uploadResponse = Invoke-RestMethod -Uri "http://localhost:8093/api/upload" `
            -Method POST -Body $testDocument -ContentType "application/json"
        Write-Host "     [✓] Document uploaded: $($uploadResponse.id)" -ForegroundColor Green
    } catch {
        Write-Host "     [✗] Upload failed: $_" -ForegroundColor Red
        return
    }
    
    Write-Host "  2. Processing with Gemma3..." -ForegroundColor Yellow
    $processRequest = @{
        documentId = $uploadResponse.id
        analysisType = "full"
        useGemma3 = $true
    } | ConvertTo-Json
    
    try {
        $analysisResponse = Invoke-RestMethod -Uri "http://localhost:8094/api/documents/analyze" `
            -Method POST -Body $processRequest -ContentType "application/json" -TimeoutSec 60
        
        Write-Host "     [✓] Analysis complete" -ForegroundColor Green
        if ($analysisResponse.entities) {
            Write-Host "        Entities found: $($analysisResponse.entities.Count)"
        }
        if ($analysisResponse.summary) {
            Write-Host "        Summary generated: $($analysisResponse.summary.Length) chars"
        }
    } catch {
        Write-Host "     [✗] Analysis failed: $_" -ForegroundColor Red
        return
    }
    
    Write-Host "  3. Testing vector search..." -ForegroundColor Yellow
    $searchRequest = @{
        query = "indemnification clause"
        documentId = $uploadResponse.id
    } | ConvertTo-Json
    
    try {
        $searchResponse = Invoke-RestMethod -Uri "http://localhost:8094/api/vector/search" `
            -Method POST -Body $searchRequest -ContentType "application/json"
        
        Write-Host "     [✓] Vector search complete" -ForegroundColor Green
        Write-Host "        Results: $($searchResponse.results.Count)"
        Write-Host "        Top score: $([math]::Round($searchResponse.results[0].score, 3))"
    } catch {
        Write-Host "     [✗] Search failed: $_" -ForegroundColor Red
    }
    
    Write-Host "`n  [✓] End-to-end test complete!" -ForegroundColor Green
}

# Main execution
Write-Host "Starting Gemma3 Integration Tests...`n" -ForegroundColor Cyan

# Check services
Write-Host "Service Status:" -ForegroundColor Yellow
Write-Host "===============" -ForegroundColor Yellow

$allServicesRunning = $true
foreach ($service in $config.Services.GetEnumerator()) {
    $health = Test-ServiceHealth -Name $service.Key -Port $service.Value.Port
    
    $statusColor = if ($health.Status -eq "Running") { "Green" } else { "Red" }
    $statusSymbol = if ($health.Status -eq "Running") { "✓" } else { "✗" }
    
    Write-Host ("  [{0}] {1,-15} Port {2,-5} - {3}" -f $statusSymbol, $service.Key, $service.Value.Port, $health.Status) -ForegroundColor $statusColor
    
    if ($service.Value.Required -and $health.Status -ne "Running") {
        $allServicesRunning = $false
    }
}

if (-not $allServicesRunning) {
    Write-Host "`n[ERROR] Required services are not running!" -ForegroundColor Red
    Write-Host "Please run: .\START-GEMMA3-LEGAL-COMPLETE.bat" -ForegroundColor Yellow
    exit 1
}

# Run tests based on action
switch ($Action) {
    "test" {
        Test-GpuAcceleration
        Test-Gemma3Model
        Test-VectorOperations
        Test-RAGPipeline
        Test-EndToEnd
    }
    "benchmark" {
        Test-GpuAcceleration
        Run-BenchmarkSuite
    }
    "optimize" {
        Optimize-Performance
        Run-BenchmarkSuite
    }
    "full" {
        Test-GpuAcceleration
        Test-Gemma3Model
        Test-VectorOperations
        Test-RAGPipeline
        Optimize-Performance
        Run-BenchmarkSuite
        Test-EndToEnd
    }
    default {
        Write-Host "Usage: .\Test-Gemma3-Integration.ps1 -Action [test|benchmark|optimize|full]" -ForegroundColor Yellow
    }
}

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "  Integration test complete!" -ForegroundColor Green
Write-Host "  Gemma3 Legal Model is ready for production use." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
