# Load Gemma3 Legal Model
Write-Host "🏛️ Loading Gemma3 Legal Model" -ForegroundColor Cyan
Write-Host "================================"

Write-Host "[1/3] Pulling Gemma3 model..." -ForegroundColor Yellow
try {
    docker exec deeds-ollama-gpu ollama pull gemma2:27b
    Write-Host "✅ Model downloaded successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to download model: $_" -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Creating legal variant..." -ForegroundColor Yellow
$modelfile = @"
FROM ./gemma3Q4_K_M/mo16.gguf
SYSTEM You are a legal AI assistant for prosecutors. Provide accurate legal analysis and case recommendations.
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
} finally {
    Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
}

Write-Host "[3/3] Testing model..." -ForegroundColor Yellow
try {
    docker exec deeds-ollama-gpu ollama list
    Write-Host "✅ Model test completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to test model: $_" -ForegroundColor Red
}

Write-Host "================================"
Write-Host "🎉 Gemma3 legal model setup complete!" -ForegroundColor Green
Write-Host "Model name: 'gemma3-legal'" -ForegroundColor Cyan
Write-Host "Ready to use in your application!" -ForegroundColor White
