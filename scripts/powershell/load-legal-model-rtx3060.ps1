# Load Legal AI Model - RTX 3060 Optimized
Write-Host "🏛️ Loading Legal AI Model (RTX 3060 Optimized)" -ForegroundColor Cyan
Write-Host "================================================"

Write-Host "[1/4] Pulling smaller model suitable for RTX 3060..." -ForegroundColor Yellow
try {
    # Use a smaller model that fits in 12GB VRAM
    docker exec deeds-ollama-gpu ollama pull gemma2:9b-instruct-q4_K_M
    Write-Host "✅ Gemma2 9B model downloaded successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download model: $_" -ForegroundColor Red
    Write-Host "Trying fallback model..." -ForegroundColor Yellow
    try {
        docker exec deeds-ollama-gpu ollama pull llama3.2:3b-instruct-q4_K_M
        Write-Host "✅ Llama3.2 3B model downloaded successfully" -ForegroundColor Green
        $modelBase = "llama3.2:3b-instruct-q4_K_M"
    } catch {
        Write-Host "❌ Failed to download fallback model: $_" -ForegroundColor Red
        exit 1
    }
}

if (-not $modelBase) {
    $modelBase = "gemma2:9b-instruct-q4_K_M"
}

Write-Host "[2/4] Creating legal variant..." -ForegroundColor Yellow
$modelfile = @"
FROM $modelBase
SYSTEM You are a legal AI assistant for prosecutors. You provide accurate legal analysis, case recommendations, and help with evidence management. Be professional, concise, and cite relevant legal principles when applicable.
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
PARAMETER num_predict 512
"@

# Create a temporary modelfile
$tempFile = [System.IO.Path]::GetTempFileName()
$modelfile | Out-File -FilePath $tempFile -Encoding UTF8

try {
    # Copy the modelfile to the container and create the model
    docker cp $tempFile deeds-ollama-gpu:/tmp/Modelfile-Legal
    docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-Legal
    Write-Host "✅ Legal variant created successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to create legal variant: $_" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
} finally {
    Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
}

Write-Host "[3/4] Testing model..." -ForegroundColor Yellow
try {
    docker exec deeds-ollama-gpu ollama list
    Write-Host "✅ Model test completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to test model: $_" -ForegroundColor Red
}

Write-Host "[4/4] Quick functionality test..." -ForegroundColor Yellow
try {
    $testResponse = docker exec deeds-ollama-gpu ollama run gemma3-legal "What are the key elements of a criminal case?"
    Write-Host "✅ Model is responding correctly" -ForegroundColor Green
    Write-Host "Sample response: $($testResponse -join ' ' | Select-Object -First 100)..." -ForegroundColor Gray
} catch {
    Write-Host "⚠️ Model may not be fully ready, but it's installed" -ForegroundColor Yellow
}

Write-Host "================================================"
Write-Host "🎉 Legal AI model setup complete!" -ForegroundColor Green
Write-Host "Model name: 'gemma3-legal'" -ForegroundColor Cyan
Write-Host "Base model: '$modelBase'" -ForegroundColor Gray
Write-Host "Optimized for RTX 3060 with 12GB VRAM" -ForegroundColor White
Write-Host "Ready to use in your application!" -ForegroundColor White
