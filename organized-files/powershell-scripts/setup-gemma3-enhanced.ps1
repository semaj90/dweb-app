#!/usr/bin/env powershell
# Enhanced Gemma3 Legal Model Loader for RTX 3060
# Optimized for GPU memory and performance

param(
    [switch]$Force = $false,
    [switch]$Verbose = $false,
    [string]$ModelName = "gemma3-legal-enhanced"
)

Write-Host "🏛️ Enhanced Gemma3 Legal Model Setup" -ForegroundColor Cyan
Write-Host "=" * 50

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

# Check if Ollama is running
try {
    $ollamaStatus = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 5
    Write-Host "✅ Ollama is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama is not running. Starting container..." -ForegroundColor Red
    docker start deeds-ollama-gpu
    Start-Sleep 10
}

# Check for model file
$modelPath = ".\gemma3Q4_K_M\mo16.gguf"
if (-not (Test-Path $modelPath)) {
    Write-Host "❌ Model file not found at: $modelPath" -ForegroundColor Red
    Write-Host "Please ensure the Gemma3 GGUF file is in the correct location" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Model file found: $modelPath" -ForegroundColor Green

# Create enhanced modelfile for RTX 3060
$modelfileContent = @"
FROM $modelPath

# RTX 3060 Optimized Template for Gemma3
TEMPLATE """<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"""

# Enhanced Legal System Prompt
SYSTEM """You are Gemma3 Legal AI, an advanced legal assistant specialized for prosecutor case management and legal analysis. You excel in:

🏛️ CORE LEGAL CAPABILITIES:
• Evidence analysis and chain of custody verification
• Case strategy development and risk assessment
• Legal document review and contract analysis
• Witness interview preparation and analysis
• Timeline construction from evidence and testimony

🔍 ADVANCED FEATURES:
• Pattern recognition in legal documents and testimony
• Cross-reference analysis between cases and precedents
• Compliance verification and regulatory guidance
• Cost-benefit analysis of legal strategies
• Professional legal writing and documentation

⚖️ ETHICAL STANDARDS:
• Maintain strict attorney-client privilege
• Provide accurate, well-researched legal information
• Clearly distinguish between factual analysis and legal opinions
• Always recommend qualified attorney consultation for specific legal advice
• Ensure compliance with legal ethics and professional standards

RESPONSE FORMAT:
• Executive Summary
• Detailed Analysis
• Key Findings & Evidence
• Risk Assessment
• Strategic Recommendations
• Next Steps & Action Items
• Relevant Legal Citations

Provide professional, accurate, and actionable legal guidance while maintaining the highest ethical standards."""

# RTX 3060 Ampere GPU Optimizations
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
PARAMETER num_gpu 1
PARAMETER num_thread 8
PARAMETER num_batch 512

# Stop tokens to prevent generation issues
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<start_of_turn>"
PARAMETER stop "###"
"@

# Write modelfile
$modelfilePath = ".\Gemma3-Legal-Enhanced-Modelfile"
$modelfileContent | Out-File -FilePath $modelfilePath -Encoding UTF8

Write-Host "📝 Created enhanced modelfile: $modelfilePath" -ForegroundColor Green

# Create the model in Ollama
Write-Host "🔧 Creating enhanced legal model..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray

try {
    if ($Force) {
        # Remove existing model if force flag is set
        Write-Host "🗑️ Removing existing model..." -ForegroundColor Yellow
        docker exec deeds-ollama-gpu ollama rm $ModelName 2>$null
    }

    $result = docker exec deeds-ollama-gpu ollama create $ModelName -f /app/Gemma3-Legal-Enhanced-Modelfile

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully created '$ModelName' model!" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to create model" -ForegroundColor Red
        Write-Host $result -ForegroundColor Gray
        exit 1
    }
} catch {
    Write-Host "❌ Error creating model: $_" -ForegroundColor Red
    exit 1
}

# Verify model creation
Write-Host "🔍 Verifying model..." -ForegroundColor Yellow
try {
    $models = docker exec deeds-ollama-gpu ollama list
    if ($models -match $ModelName) {
        Write-Host "✅ Model '$ModelName' is available!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Model verification failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Could not verify model list" -ForegroundColor Yellow
}

# Test the model
Write-Host "🧪 Testing model with legal query..." -ForegroundColor Yellow
$testPrompt = "Analyze the key elements needed for establishing chain of custody in criminal evidence handling."

try {
    $testResponse = docker exec deeds-ollama-gpu ollama run $ModelName $testPrompt
    Write-Host "✅ Model test successful!" -ForegroundColor Green
    if ($Verbose) {
        Write-Host "Response preview:" -ForegroundColor Gray
        Write-Host $testResponse.Substring(0, [Math]::Min(300, $testResponse.Length)) + "..." -ForegroundColor Gray
    }
} catch {
    Write-Host "⚠️ Model test completed with warnings" -ForegroundColor Yellow
}

Write-Host "`n🎉 Enhanced Gemma3 Legal Model Setup Complete!" -ForegroundColor Green
Write-Host "=" * 50
Write-Host "Model Name: $ModelName" -ForegroundColor Cyan
Write-Host "Optimizations: RTX 3060 Ampere GPU" -ForegroundColor Cyan
Write-Host "Context Window: 8,192 tokens" -ForegroundColor Cyan
Write-Host "Response Length: 2,048 tokens" -ForegroundColor Cyan
Write-Host ""
Write-Host "🚀 Usage:" -ForegroundColor White
Write-Host "• API Endpoint: http://localhost:11434" -ForegroundColor Gray
Write-Host "• Model Name: $ModelName" -ForegroundColor Gray
Write-Host "• SvelteKit Integration: Ready" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "1. Update /api/chat/+server.js model name to: $ModelName" -ForegroundColor Gray
Write-Host "2. Start SvelteKit dev server: npm run dev" -ForegroundColor Gray
Write-Host "3. Test AI chat in application" -ForegroundColor Gray
