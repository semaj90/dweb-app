# Setup Custom Gemma3 Legal AI Model for Ollama
# This script creates an optimized model for the interactive canvas system

Write-Host "🏛️ Setting up Gemma3 Legal AI Model for Interactive Canvas" -ForegroundColor Cyan
Write-Host "=" * 60

# Check if Ollama is running
Write-Host "📡 Checking Ollama status..." -ForegroundColor Yellow
try {
    $ollamaStatus = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -ErrorAction Stop
    Write-Host "✅ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama is not running. Please start Ollama first:" -ForegroundColor Red
    Write-Host "   ollama serve" -ForegroundColor Gray
    exit 1
}

# Verify model file exists
$modelPath = ".\gemma3Q4_K_M\mo16.gguf"
if (-not (Test-Path $modelPath)) {
    Write-Host "❌ Model file not found at: $modelPath" -ForegroundColor Red
    Write-Host "Please verify the path to your Gemma3 GGUF file" -ForegroundColor Gray
    exit 1
}

Write-Host "✅ Found model file: $modelPath" -ForegroundColor Green

# Create the custom legal AI model
Write-Host "🔧 Creating custom legal AI model..." -ForegroundColor Yellow
Write-Host "This may take a few minutes as Ollama processes the model..." -ForegroundColor Gray

try {
    $result = & ollama create gemma3-legal -f "Modelfile-Ultra-Light" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully created 'gemma3-legal' model!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create model. Error:" -ForegroundColor Red
        Write-Host $result -ForegroundColor Gray
        exit 1
    }
} catch {
    Write-Host "❌ Error creating model: $_" -ForegroundColor Red
    exit 1
}

# Verify the model was created
Write-Host "🔍 Verifying model creation..." -ForegroundColor Yellow
try {
    $models = & ollama list
    if ($models -match "gemma3-legal") {
        Write-Host "✅ Model 'gemma3-legal' is available!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Model may not have been created properly" -ForegroundColor Yellow
        Write-Host "Available models:" -ForegroundColor Gray
        & ollama list
    }
} catch {
    Write-Host "⚠️ Could not verify model list" -ForegroundColor Yellow
}

# Test the model with a legal query
Write-Host "🧪 Testing model with a legal query..." -ForegroundColor Yellow
$testPrompt = "Analyze the key elements needed for a strong evidence timeline in a criminal case."

try {
    Write-Host "Sending test prompt..." -ForegroundColor Gray
    $testResponse = & ollama run gemma3-legal $testPrompt --format json

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Model responds correctly!" -ForegroundColor Green
        Write-Host "Sample response preview:" -ForegroundColor Gray
        Write-Host $testResponse.Substring(0, [Math]::Min(200, $testResponse.Length)) + "..." -ForegroundColor Gray
    } else {
        Write-Host "⚠️ Model test had issues but may still work" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not test model, but it may still work in the application" -ForegroundColor Yellow
}

Write-Host "`n🎉 Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60
Write-Host "Your custom legal AI model is ready!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. The interactive canvas will now use the optimized 'gemma3-legal' model" -ForegroundColor Gray
Write-Host "2. Start your SvelteKit dev server: npm run dev" -ForegroundColor Gray
Write-Host "3. Open the interactive canvas: http://localhost:5173/interactive-canvas" -ForegroundColor Gray
Write-Host "4. Test the AI assistant with legal queries" -ForegroundColor Gray
Write-Host ""
Write-Host "Model Features:" -ForegroundColor White
Write-Host "- Optimized for legal case management" -ForegroundColor Gray
Write-Host "- Professional, consistent responses" -ForegroundColor Gray
Write-Host "- Proper Gemma3 prompt formatting" -ForegroundColor Gray
Write-Host "- Large context window for documents" -ForegroundColor Gray
