# GGUF Model Diagnostic Script
# Fixes GGUF loading errors and validates model files

Write-Host "🔍 Diagnosing GGUF Model Loading Issues..." -ForegroundColor Cyan

$modelDir = "gemma3Q4_K_M"
if (-not (Test-Path $modelDir)) {
    Write-Host "❌ Model directory not found: $modelDir" -ForegroundColor Red
    exit 1
}

Write-Host "`n📁 Checking model files..." -ForegroundColor Yellow
Get-ChildItem -Path $modelDir -Filter "*.gguf" | ForEach-Object {
    $file = $_
    $sizeGB = [math]::Round($file.Length / 1GB, 2)
    Write-Host "  📄 $($file.Name): $sizeGB GB" -ForegroundColor White

    # Check if file is too small (likely corrupted)
    if ($file.Length -lt 100MB) {
        Write-Host "    ⚠️  File seems too small, likely corrupted" -ForegroundColor Red
    } elseif ($file.Length -gt 50GB) {
        Write-Host "    ⚠️  File seems too large for Q4_K_M quantization" -ForegroundColor Red
    } else {
        Write-Host "    ✅ File size looks reasonable" -ForegroundColor Green
    }

    # Try to read first few bytes to check GGUF header
    try {
        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)[0..7]
        $header = [System.Text.Encoding]::ASCII.GetString($bytes)
        if ($header.StartsWith("GGUF")) {
            Write-Host "    ✅ GGUF header found" -ForegroundColor Green
        } else {
            Write-Host "    ❌ Invalid GGUF header: $header" -ForegroundColor Red
        }
    } catch {
        Write-Host "    ❌ Cannot read file: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n🔧 Checking Ollama installation..." -ForegroundColor Yellow
try {
    $ollamaVersion = ollama --version
    Write-Host "  ✅ Ollama found: $ollamaVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Ollama not found or not in PATH" -ForegroundColor Red
    Write-Host "  Install Ollama from: https://ollama.ai" -ForegroundColor Yellow
}

Write-Host "`n📋 Current Ollama models:" -ForegroundColor Yellow
try {
    ollama list
} catch {
    Write-Host "  ❌ Cannot list models - Ollama may not be running" -ForegroundColor Red
}

Write-Host "`n🔄 Recommended fixes:" -ForegroundColor Cyan
Write-Host "1. If model file is corrupted, re-download or reconvert" -ForegroundColor White
Write-Host "2. Try the larger mohf16-Q4_K_M.gguf file instead" -ForegroundColor White
Write-Host "3. Use fallback to gemma2:9b while fixing the custom model" -ForegroundColor White
Write-Host "4. Ensure Ollama service is running: ollama serve" -ForegroundColor White

Write-Host "`n🚀 Next steps:" -ForegroundColor Green
Write-Host "Run: ./fix-gemma3-model.ps1" -ForegroundColor White
