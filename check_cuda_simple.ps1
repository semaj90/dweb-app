# Simple CUDA environment check
Write-Host "=== CUDA Environment Check ===" -ForegroundColor Cyan

# Check nvcc
Write-Host "`nChecking nvcc..." -ForegroundColor Yellow
$nvccOutput = & nvcc --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ nvcc found and working" -ForegroundColor Green
    Write-Host $nvccOutput
} else {
    Write-Host "✗ nvcc not found" -ForegroundColor Red
    Write-Host $nvccOutput
}

# Check cl.exe
Write-Host "`nChecking Visual Studio Build Tools..." -ForegroundColor Yellow
$clOutput = & cl 2>&1
if ($clOutput -match "Microsoft") {
    Write-Host "✓ Visual Studio Build Tools available" -ForegroundColor Green
} else {
    Write-Host "⚠ cl.exe not found - needed for NVCC" -ForegroundColor Yellow
}

# Check clang (optional)
Write-Host "`nChecking clang..." -ForegroundColor Yellow
$clangOutput = & clang --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ clang found" -ForegroundColor Green
} else {
    Write-Host "⚠ clang not found (optional)" -ForegroundColor Yellow
}

Write-Host "`n=== Check Complete ===" -ForegroundColor Cyan