# GEMMA3 SYSTEM VALIDATOR
# Comprehensive validation script for all Gemma3 integration components

Write-Host "🔍 Gemma3 Integration System Validator" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

$validationResults = @{}
$overallStatus = $true

# 1. CHECK GGUF MODEL FILE
Write-Host "`n📁 Validating GGUF Model File..." -ForegroundColor Yellow
$modelPath = ".\gemma3Q4_K_M\mohf16-Q4_K_M.gguf"
if (Test-Path $modelPath) {
    $fileSize = (Get-Item $modelPath).Length
    $fileSizeMB = [math]::Round($fileSize / 1MB, 2)

    if ($fileSize -gt 1MB) {
        Write-Host "✅ Model file found: $modelPath ($fileSizeMB MB)" -ForegroundColor Green
        $validationResults["model_file"] = @{ status = "OK"; message = "File found ($fileSizeMB MB)" }
    } else {
        Write-Host "⚠️  Model file too small: $fileSizeMB MB (might be corrupted)" -ForegroundColor Yellow
        $validationResults["model_file"] = @{ status = "WARNING"; message = "File too small ($fileSizeMB MB)" }
    }
} else {
    Write-Host "❌ Model file not found: $modelPath" -ForegroundColor Red
    $validationResults["model_file"] = @{ status = "ERROR"; message = "File not found" }
    $overallStatus = $false
}

# 2. CHECK MODELFILE SYNTAX
Write-Host "`n📝 Validating Modelfile..." -ForegroundColor Yellow
$modelfilePath = ".\Modelfile-Gemma3-Legal"
if (Test-Path $modelfilePath) {
    $modelfileContent = Get-Content $modelfilePath -Raw

    if ($modelfileContent -match "FROM.*mohf16-Q4_K_M\.gguf") {
        Write-Host "✅ Modelfile points to correct GGUF file" -ForegroundColor Green
        $validationResults["modelfile"] = @{ status = "OK"; message = "Syntax valid, correct GGUF reference" }
    } else {
        Write-Host "❌ Modelfile has incorrect GGUF reference" -ForegroundColor Red
        $validationResults["modelfile"] = @{ status = "ERROR"; message = "Incorrect GGUF reference" }
        $overallStatus = $false
    }
} else {
    Write-Host "❌ Modelfile not found: $modelfilePath" -ForegroundColor Red
    $validationResults["modelfile"] = @{ status = "ERROR"; message = "File not found" }
    $overallStatus = $false
}

# 3. CHECK OLLAMA SERVICE
Write-Host "`n🔗 Testing Ollama Service..." -ForegroundColor Yellow
try {
    $ollamaTest = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -Method GET -TimeoutSec 5
    Write-Host "✅ Ollama service running: v$($ollamaTest.version)" -ForegroundColor Green
    $validationResults["ollama_service"] = @{ status = "OK"; message = "Running v$($ollamaTest.version)" }
} catch {
    Write-Host "❌ Ollama service not accessible on localhost:11434" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
    $validationResults["ollama_service"] = @{ status = "ERROR"; message = "Service not accessible" }
    $overallStatus = $false
}

# 4. CHECK MODEL AVAILABILITY IN OLLAMA
Write-Host "`n🤖 Checking Model in Ollama..." -ForegroundColor Yellow
try {
    $modelsList = & ollama list 2>&1
    if ($modelsList -match "gemma3-legal") {
        Write-Host "✅ Gemma3-legal model loaded in Ollama" -ForegroundColor Green
        $validationResults["ollama_model"] = @{ status = "OK"; message = "Model loaded and available" }
    } else {
        Write-Host "❌ Gemma3-legal model not found in Ollama" -ForegroundColor Red
        Write-Host "   Available models:" -ForegroundColor Gray
        Write-Host "   $modelsList" -ForegroundColor Gray
        $validationResults["ollama_model"] = @{ status = "ERROR"; message = "Model not loaded" }
        $overallStatus = $false
    }
} catch {
    Write-Host "❌ Error checking Ollama models: $($_.Exception.Message)" -ForegroundColor Red
    $validationResults["ollama_model"] = @{ status = "ERROR"; message = "Error checking models" }
    $overallStatus = $false
}

# 5. TEST MODEL RESPONSE
Write-Host "`n💬 Testing Model Response..." -ForegroundColor Yellow
if ($validationResults["ollama_model"].status -eq "OK") {
    try {
        $testPrompt = "What is contract law? Please provide a brief explanation."
        Write-Host "   Testing with prompt: $testPrompt" -ForegroundColor Gray

        $testResponse = & ollama run gemma3-legal $testPrompt 2>&1
        if ($LASTEXITCODE -eq 0 -and $testResponse -and $testResponse.Length -gt 10) {
            Write-Host "✅ Model responds correctly" -ForegroundColor Green
            Write-Host "   Response preview: $($testResponse.ToString().Substring(0, [Math]::Min(100, $testResponse.ToString().Length)))..." -ForegroundColor Gray
            $validationResults["model_response"] = @{ status = "OK"; message = "Model generates responses" }
        } else {
            Write-Host "❌ Model response test failed" -ForegroundColor Red
            Write-Host "   Output: $testResponse" -ForegroundColor Gray
            $validationResults["model_response"] = @{ status = "ERROR"; message = "No valid response" }
            $overallStatus = $false
        }
    } catch {
        Write-Host "❌ Error testing model response: $($_.Exception.Message)" -ForegroundColor Red
        $validationResults["model_response"] = @{ status = "ERROR"; message = "Response test failed" }
        $overallStatus = $false
    }
} else {
    Write-Host "⏭️  Skipping response test (model not available)" -ForegroundColor Yellow
    $validationResults["model_response"] = @{ status = "SKIPPED"; message = "Model not available" }
}

# 6. CHECK API ENDPOINTS
Write-Host "`n🌐 Validating API Endpoints..." -ForegroundColor Yellow
$apiEndpoints = @(
    ".\sveltekit-frontend\src\routes\api\ai\chat\+server.ts",
    ".\sveltekit-frontend\src\routes\api\ai\stream\+server.ts"
)

foreach ($endpoint in $apiEndpoints) {
    $endpointName = ($endpoint -split '\\')[-2]
    if (Test-Path $endpoint) {
        $content = Get-Content $endpoint -Raw
        if ($content -match "gemma3-legal" -and $content -match "localhost:11434") {
            Write-Host "✅ API endpoint $endpointName: configured correctly" -ForegroundColor Green
            $validationResults["api_$endpointName"] = @{ status = "OK"; message = "Endpoint configured" }
        } else {
            Write-Host "⚠️  API endpoint $endpointName: missing Gemma3 configuration" -ForegroundColor Yellow
            $validationResults["api_$endpointName"] = @{ status = "WARNING"; message = "Configuration incomplete" }
        }
    } else {
        Write-Host "❌ API endpoint $endpointName: file not found" -ForegroundColor Red
        $validationResults["api_$endpointName"] = @{ status = "ERROR"; message = "File not found" }
        $overallStatus = $false
    }
}

# 7. CHECK CHAT STORE
Write-Host "`n💾 Validating Chat Store..." -ForegroundColor Yellow
$chatStorePath = ".\sveltekit-frontend\src\lib\stores\chatStore.ts"
if (Test-Path $chatStorePath) {
    $storeContent = Get-Content $chatStorePath -Raw

    $hasXStateInterface = $storeContent -match "useChatActor"
    $hasCorrectEndpoint = $storeContent -match "/api/ai/chat"

    if ($hasXStateInterface -and $hasCorrectEndpoint) {
        Write-Host "✅ Chat store: XState interface and API endpoint configured" -ForegroundColor Green
        $validationResults["chat_store"] = @{ status = "OK"; message = "Store configured correctly" }
    } else {
        $issues = @()
        if (-not $hasXStateInterface) { $issues += "missing XState interface" }
        if (-not $hasCorrectEndpoint) { $issues += "incorrect API endpoint" }

        Write-Host "⚠️  Chat store: $($issues -join ', ')" -ForegroundColor Yellow
        $validationResults["chat_store"] = @{ status = "WARNING"; message = $issues -join ', ' }
    }
} else {
    Write-Host "❌ Chat store file not found" -ForegroundColor Red
    $validationResults["chat_store"] = @{ status = "ERROR"; message = "File not found" }
    $overallStatus = $false
}

# 8. CHECK UI COMPONENTS
Write-Host "`n🎨 Checking UI Components..." -ForegroundColor Yellow
$chatComponents = @(
    ".\sveltekit-frontend\src\lib\components\ai\AIChat.svelte",
    ".\sveltekit-frontend\src\routes\test-gemma3\+page.svelte"
)

$componentStatus = $true
foreach ($component in $chatComponents) {
    $componentName = ($component -split '\\')[-1]
    if (Test-Path $component) {
        $content = Get-Content $component -Raw
        if ($content -match "useChatActor") {
            Write-Host "✅ Component $componentName: uses XState interface" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Component $componentName: may not use XState interface" -ForegroundColor Yellow
            $componentStatus = $false
        }
    } else {
        Write-Host "❌ Component $componentName: not found" -ForegroundColor Red
        $componentStatus = $false
    }
}

$validationResults["ui_components"] = @{
    status = if ($componentStatus) { "OK" } else { "WARNING" };
    message = if ($componentStatus) { "Components configured" } else { "Some components missing/misconfigured" }
}

# SUMMARY REPORT
Write-Host "`n" -NoNewline
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "🎯 VALIDATION SUMMARY" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

foreach ($key in $validationResults.Keys) {
    $result = $validationResults[$key]
    $statusColor = switch ($result.status) {
        "OK" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        "SKIPPED" { "Gray" }
        default { "White" }
    }

    $statusIcon = switch ($result.status) {
        "OK" { "✅" }
        "WARNING" { "⚠️ " }
        "ERROR" { "❌" }
        "SKIPPED" { "⏭️ " }
        default { "❓" }
    }

    Write-Host "$statusIcon $($key.Replace('_', ' ').ToUpper()): $($result.message)" -ForegroundColor $statusColor
}

Write-Host "`n" -NoNewline
if ($overallStatus) {
    Write-Host "🎉 OVERALL STATUS: SYSTEM READY" -ForegroundColor Green
    Write-Host "`n📋 All critical components validated. Your Gemma3 integration should work!" -ForegroundColor Green
    Write-Host "`n🚀 Next Steps:" -ForegroundColor Cyan
    Write-Host "   1. Run: npm run dev" -ForegroundColor White
    Write-Host "   2. Navigate to: http://localhost:5173/test-gemma3" -ForegroundColor White
    Write-Host "   3. Test the chat interface" -ForegroundColor White
} else {
    Write-Host "⚠️  OVERALL STATUS: ISSUES DETECTED" -ForegroundColor Red
    Write-Host "`n🔧 Critical issues found. Run COMPLETE_GEMMA3_FIX.ps1 to resolve them." -ForegroundColor Yellow
    Write-Host "`n🛠️  Fix Command:" -ForegroundColor Cyan
    Write-Host "   .\FIX-GEMMA3-COMPLETE.bat" -ForegroundColor White
}

Write-Host "`n💡 For detailed troubleshooting, check:" -ForegroundColor Cyan
Write-Host "   - TROUBLESHOOTING_GUIDE.md" -ForegroundColor White
Write-Host "   - GEMMA3_INTEGRATION_COMPLETE_GUIDE.md" -ForegroundColor White
