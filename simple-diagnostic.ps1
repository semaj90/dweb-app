# Simple GGUF Model Diagnostic
# Basic check for GGUF model loading issues

Write-Host "Diagnosing GGUF Model Loading Issues..." -ForegroundColor Cyan

$modelDir = "gemma3Q4_K_M"
if (-not (Test-Path $modelDir)) {
    Write-Host "ERROR: Model directory not found: $modelDir" -ForegroundColor Red
    exit 1
}

Write-Host "`nChecking model files..." -ForegroundColor Yellow
$modelFiles = Get-ChildItem -Path $modelDir -Filter "*.gguf"

if ($modelFiles.Count -eq 0) {
    Write-Host "ERROR: No GGUF files found" -ForegroundColor Red
    exit 1
}

foreach ($file in $modelFiles) {
    $sizeGB = [math]::Round($file.Length / 1GB, 2)
    Write-Host "File: $($file.Name) - Size: $sizeGB GB" -ForegroundColor White

    # Check if file is reasonable size
    if ($file.Length -lt 100MB) {
        Write-Host "  WARNING: File seems too small, likely corrupted" -ForegroundColor Red
    } elseif ($file.Length -gt 50GB) {
        Write-Host "  WARNING: File seems too large for Q4_K_M quantization" -ForegroundColor Red
    } else {
        Write-Host "  OK: File size looks reasonable" -ForegroundColor Green
    }

    # Try to read GGUF header
    try {
        $bytes = [System.IO.File]::ReadAllBytes($file.FullName)[0..7]
        $header = [System.Text.Encoding]::ASCII.GetString($bytes)
        if ($header.StartsWith("GGUF")) {
            Write-Host "  OK: Valid GGUF header found" -ForegroundColor Green
        } else {
            Write-Host "  ERROR: Invalid GGUF header: $header" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ERROR: Cannot read file: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`nChecking Ollama..." -ForegroundColor Yellow
try {
    $ollamaVersion = ollama --version
    Write-Host "Ollama found: $ollamaVersion" -ForegroundColor Green

    Write-Host "`nCurrent models:" -ForegroundColor Yellow
    ollama list
} catch {
    Write-Host "Ollama not found or not running" -ForegroundColor Red
}

Write-Host "`nRecommendations:" -ForegroundColor Cyan
Write-Host "1. If model file is corrupted, re-download or reconvert" -ForegroundColor White
Write-Host "2. Try the larger mohf16-Q4_K_M.gguf file if available" -ForegroundColor White
Write-Host "3. Use fallback to gemma2:9b while fixing the custom model" -ForegroundColor White
Write-Host "4. Run: ./complete-gemma3-setup.ps1 to proceed with setup" -ForegroundColor White
