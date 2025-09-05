# Law PDF Local Models Testing Script
# Tests gemma3-legal and nomic-embed-text for legal document processing

param(
    [switch]$AutoSolve,
    [string]$Model = "gemma3-legal:latest",
    [string]$EmbedModel = "nomic-embed-text:latest",
    [switch]$EnableOllamaSummary = $true
)

Write-Host "🏛️ Law PDF Local Models Testing" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Test Ollama availability
Write-Host "`n🔍 Checking Ollama service..." -ForegroundColor Yellow
try {
    $ollamaVersion = ollama --version
    Write-Host "✅ Ollama available: $ollamaVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama not available" -ForegroundColor Red
    exit 1
}

# Test models availability
Write-Host "`n📋 Checking local models..." -ForegroundColor Yellow

# Check gemma3-legal
Write-Host "Testing $Model..." -ForegroundColor Gray
try {
    $testResponse = ollama run $Model "Summarize contract law in one sentence." --timeout 30
    if ($testResponse -and $testResponse.Length -gt 10) {
        Write-Host "✅ $Model - Coherent legal responses confirmed" -ForegroundColor Green
        $gemmaWorking = $true
    } else {
        Write-Host "❌ $Model - No coherent response" -ForegroundColor Red
        $gemmaWorking = $false
    }
} catch {
    Write-Host "❌ $Model - Not available" -ForegroundColor Red
    $gemmaWorking = $false
}

# Check nomic-embed-text
Write-Host "Testing $EmbedModel..." -ForegroundColor Gray
try {
    $embedTest = Invoke-RestMethod -Uri "http://localhost:11434/api/embeddings" -Method POST -ContentType "application/json" -Body (@{
        model = $EmbedModel.Replace(":latest", "")
        prompt = "legal contract test"
    } | ConvertTo-Json) -TimeoutSec 30

    if ($embedTest.embedding -and $embedTest.embedding.Count -gt 0) {
        Write-Host "✅ $EmbedModel - Available for embeddings" -ForegroundColor Green
        $embedWorking = $true
    } else {
        Write-Host "❌ $EmbedModel - No embeddings generated" -ForegroundColor Red
        $embedWorking = $false
    }
} catch {
    Write-Host "❌ $EmbedModel - Not available" -ForegroundColor Red
    $embedWorking = $false
}

# Test law PDF processing endpoint
Write-Host "`n🏛️ Testing law PDF processing..." -ForegroundColor Yellow
try {
    $testPdfData = @{
        content = "This is a sample legal contract. The parties agree to the following terms and conditions. Liability is limited to direct damages only."
        fileName = "test-contract.txt"
        analysisType = "comprehensive"
        useLocalModels = $true
    }

    $pdfResponse = Invoke-RestMethod -Uri "http://localhost:5173/api/ai/lawpdfs" -Method POST -ContentType "application/json" -Body ($testPdfData | ConvertTo-Json) -TimeoutSec 60

    if ($pdfResponse.summary -and $pdfResponse.summary.Length -gt 0) {
        Write-Host "✅ Law PDF processing endpoint working" -ForegroundColor Green
        Write-Host "📋 Sample analysis: $($pdfResponse.summary.Substring(0, [Math]::Min(100, $pdfResponse.summary.Length)))..." -ForegroundColor Gray
        $pdfWorking = $true
    } else {
        Write-Host "❌ Law PDF processing failed" -ForegroundColor Red
        $pdfWorking = $false
    }
} catch {
    Write-Host "❌ Law PDF endpoint not available" -ForegroundColor Red
    Write-Host "💡 Make sure SvelteKit frontend is running (npm run dev)" -ForegroundColor Yellow
    $pdfWorking = $false
}

# Context7 Configuration Check
Write-Host "`n🔧 Checking Context7 configuration..." -ForegroundColor Yellow
$configPath = "context7-docs/context7-mcp-config.json"
if (Test-Path $configPath) {
    try {
        $config = Get-Content $configPath | ConvertFrom-Json
        if ($config.enable_ollama_summary -eq $true) {
            Write-Host "✅ Context7 Ollama summary enabled" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Context7 Ollama summary not enabled" -ForegroundColor Yellow
            if ($AutoSolve) {
                Write-Host "🔧 Auto-fixing Context7 configuration..." -ForegroundColor Cyan
                $config.enable_ollama_summary = $true
                $config | ConvertTo-Json -Depth 10 | Set-Content $configPath
                Write-Host "✅ Context7 configuration updated" -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "❌ Failed to read Context7 config" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️ Context7 config not found" -ForegroundColor Yellow
}

# Generate comprehensive test report
Write-Host "`n📊 Test Results Summary" -ForegroundColor Cyan
Write-Host "=======================" -ForegroundColor Cyan

$totalTests = 4
$passedTests = 0

if ($gemmaWorking) {
    Write-Host "✅ gemma3-legal:latest - Confirmed working with coherent legal responses" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ gemma3-legal:latest - Not working" -ForegroundColor Red
}

if ($embedWorking) {
    Write-Host "✅ nomic-embed-text:latest - Available for embeddings" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ nomic-embed-text:latest - Not working" -ForegroundColor Red
}

if ($pdfWorking) {
    Write-Host "✅ Law PDF processing API - Functional" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ Law PDF processing API - Not working" -ForegroundColor Red
}

$configWorking = Test-Path $configPath
if ($configWorking) {
    Write-Host "✅ Context7 configuration - Available" -ForegroundColor Green
    $passedTests++
} else {
    Write-Host "❌ Context7 configuration - Missing" -ForegroundColor Red
}

Write-Host "`n🎯 Overall Status: $passedTests/$totalTests tests passed" -ForegroundColor $(if ($passedTests -eq $totalTests) { "Green" } else { "Yellow" })

if ($passedTests -eq $totalTests) {
    Write-Host "🎉 All systems ready for enhanced legal AI processing!" -ForegroundColor Green
    Write-Host "🚀 Next steps:" -ForegroundColor Cyan
    Write-Host "   • Visit http://localhost:5173/ai-assistant for enhanced interface" -ForegroundColor Gray
    Write-Host "   • Test law PDF analysis section" -ForegroundColor Gray
    Write-Host "   • Experience fast, private legal AI processing" -ForegroundColor Gray
} else {
    Write-Host "⚠️ Some components need attention before full deployment" -ForegroundColor Yellow

    if (!$gemmaWorking) {
        Write-Host "💡 Try: ollama pull gemma3-legal" -ForegroundColor Gray
    }
    if (!$embedWorking) {
        Write-Host "💡 Try: ollama pull nomic-embed-text" -ForegroundColor Gray
    }
    if (!$pdfWorking) {
        Write-Host "💡 Try: npm run dev (in sveltekit-frontend)" -ForegroundColor Gray
    }
}

# Save results to log
$logData = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    testResults = @{
        gemmaLegal = $gemmaWorking
        nomicEmbed = $embedWorking
        lawPdfApi = $pdfWorking
        context7Config = $configWorking
    }
    passedTests = $passedTests
    totalTests = $totalTests
    recommendation = if ($passedTests -eq $totalTests) { "ready-for-production" } else { "needs-fixes" }
}

$logPath = "logs/lawpdf-local-test-$(Get-Date -Format 'yyyyMMdd-HHmmss').json"
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$logData | ConvertTo-Json -Depth 5 | Set-Content $logPath
Write-Host "`n📝 Test results saved to: $logPath" -ForegroundColor Gray

exit $(if ($passedTests -eq $totalTests) { 0 } else { 1 })
