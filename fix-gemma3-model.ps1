# Gemma3 GGUF Model Fix Script
# Fixes loading issues and creates proper Ollama integration

param(
    [string]$ModelPath = "gemma3Q4_K_M",
    [string]$ModelName = "gemma3-legal-ai",
    [switch]$UseFallback = $false
)

Write-Host "🔧 Fixing Gemma3 GGUF Model Loading..." -ForegroundColor Cyan

# Step 1: Validate model files
Write-Host "`n📋 Step 1: Validating model files..." -ForegroundColor Yellow
$modelFiles = Get-ChildItem -Path $ModelPath -Filter "*.gguf" | Sort-Object Length -Descending

if ($modelFiles.Count -eq 0) {
    Write-Host "❌ No GGUF files found in $ModelPath" -ForegroundColor Red
    exit 1
}

$bestModel = $null
foreach ($file in $modelFiles) {
    $sizeGB = [math]::Round($file.Length / 1GB, 2)
    Write-Host "  📄 Checking $($file.Name) ($sizeGB GB)..." -ForegroundColor White

    # Check GGUF header
    try {
        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)[0..7]
        $header = [System.Text.Encoding]::ASCII.GetString($bytes)
        if ($header.StartsWith("GGUF") -and $file.Length -gt 1GB -and $file.Length -lt 20GB) {
            Write-Host "    ✅ Valid GGUF file" -ForegroundColor Green
            $bestModel = $file
            break
        } else {
            Write-Host "    ❌ Invalid or corrupted" -ForegroundColor Red
        }
    } catch {
        Write-Host "    ❌ Cannot read file" -ForegroundColor Red
    }
}

if (-not $bestModel) {
    Write-Host "❌ No valid GGUF model found. Using fallback..." -ForegroundColor Red
    $UseFallback = $true
}

# Step 2: Create proper Modelfile
Write-Host "`n📝 Step 2: Creating Ollama Modelfile..." -ForegroundColor Yellow

if ($UseFallback) {
    $modelfilePath = "Modelfile-Gemma3-Legal-Fallback"
    $modelfileContent = @"
FROM gemma2:9b

TEMPLATE """
<bos><start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192

SYSTEM """You are Gemma3 Legal AI, an advanced legal assistant specialized in case analysis, document review, and legal research. You have expertise in:

• Legal document analysis and summarization
• Case law research and precedent identification
• Contract review and risk assessment
• Regulatory compliance guidance
• Legal writing and brief preparation

Provide accurate, well-researched legal information while noting that responses are for informational purposes only and not legal advice. Always cite relevant laws, cases, or regulations when applicable."""
"@
} else {
    $modelfilePath = "Modelfile-Gemma3-Legal-Custom"
    $absolutePath = (Resolve-Path $bestModel.FullName).Path
    $modelfileContent = @"
FROM $absolutePath

TEMPLATE """
<bos><start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER num_predict 2048

SYSTEM """You are Gemma3 Legal AI, a fine-tuned legal assistant specialized in case analysis, document review, and legal research. Your training includes legal documents, case law, and regulatory frameworks.

Key capabilities:
• Legal document analysis and summarization
• Case law research and precedent identification
• Contract review and risk assessment
• Regulatory compliance guidance
• Legal writing and brief preparation
• Evidence evaluation and case strategy

Provide accurate, well-researched legal information while clearly stating that responses are for informational purposes only and do not constitute legal advice. Always cite relevant authorities when applicable."""
"@
}

$modelfileContent | Out-File -FilePath $modelfilePath -Encoding UTF8
Write-Host "  ✅ Created: $modelfilePath" -ForegroundColor Green

# Step 3: Create the model in Ollama
Write-Host "`n🚀 Step 3: Creating Ollama model..." -ForegroundColor Yellow

try {
    # Start Ollama if not running
    $ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaProcess) {
        Write-Host "  🔄 Starting Ollama service..." -ForegroundColor Yellow
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
    }

    # Create the model
    Write-Host "  🔄 Creating model '$ModelName'..." -ForegroundColor Yellow
    $createResult = ollama create $ModelName -f $modelfilePath 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Model '$ModelName' created successfully!" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Failed to create model: $createResult" -ForegroundColor Red
        throw "Model creation failed"
    }

    # Test the model
    Write-Host "  🧪 Testing model..." -ForegroundColor Yellow
    $testResult = ollama run $ModelName "Hello, can you introduce yourself as a legal AI?" 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Model test successful!" -ForegroundColor Green
        Write-Host "  Response preview: $($testResult | Select-Object -First 100)..." -ForegroundColor White
    } else {
        Write-Host "  ❌ Model test failed: $testResult" -ForegroundColor Red
    }

} catch {
    Write-Host "  ❌ Error: $($_.Exception.Message)" -ForegroundColor Red

    # Fallback to gemma2:9b
    Write-Host "`n🔄 Falling back to gemma2:9b..." -ForegroundColor Yellow
    try {
        ollama pull gemma2:9b
        Write-Host "  ✅ Fallback model ready" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Fallback failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Step 4: Update configuration
Write-Host "`n⚙️  Step 4: Updating configuration..." -ForegroundColor Yellow

$configContent = @"
// Gemma3 Legal AI Configuration
export const GEMMA3_CONFIG = {
    model: {
        name: '$ModelName',
        fallback: 'gemma2:9b',
        endpoint: 'http://localhost:11434',
        useFallback: $($UseFallback.ToString().ToLower())
    },
    parameters: {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        num_ctx: 8192,
        stream: true
    },
    features: {
        streaming: true,
        contextInjection: true,
        ragIntegration: true,
        modelSelection: true
    }
};

export const LEGAL_AI_PROMPTS = {
    caseAnalysis: 'Analyze the following legal case and provide key insights:',
    documentReview: 'Review this legal document and identify important clauses:',
    research: 'Research the following legal topic and provide relevant information:',
    contractAnalysis: 'Analyze this contract for potential risks and key terms:'
};
"@

$configContent | Out-File -FilePath "src/lib/config/gemma3-legal-config.ts" -Encoding UTF8 -Force
Write-Host "  ✅ Updated: src/lib/config/gemma3-legal-config.ts" -ForegroundColor Green

Write-Host "`n🎉 Gemma3 Legal AI setup complete!" -ForegroundColor Green
Write-Host "📋 Summary:" -ForegroundColor Cyan
Write-Host "  • Model: $ModelName" -ForegroundColor White
if ($UseFallback) {
    Write-Host "  • Status: Using fallback (gemma2:9b)" -ForegroundColor Yellow
} else {
    Write-Host "  • Status: Using custom GGUF model" -ForegroundColor Green
}
Write-Host "  • Endpoint: http://localhost:11434" -ForegroundColor White
Write-Host "  • Config: src/lib/config/gemma3-legal-config.ts" -ForegroundColor White

Write-Host "`n🚀 Next steps:" -ForegroundColor Green
Write-Host "1. Run: npm run dev" -ForegroundColor White
Write-Host "2. Test the AI chat interface" -ForegroundColor White
Write-Host "3. Check logs: ollama logs $ModelName" -ForegroundColor White
