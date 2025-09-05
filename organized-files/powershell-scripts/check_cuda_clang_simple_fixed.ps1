# Simple CUDA & Clang Environment Check
Write-Host "=== CUDA & Clang Environment Check ===" -ForegroundColor Cyan

Write-Host ""
Write-Host "Checking nvcc (CUDA compiler)..." -ForegroundColor Yellow
try {
    $nvcc = nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "? nvcc found and working" -ForegroundColor Green
    } else {
        Write-Host "? nvcc failed to run" -ForegroundColor Red
    }
} catch {
    Write-Host "? nvcc not found in PATH. Install CUDA Toolkit and add to PATH." -ForegroundColor Red
}

Write-Host ""
Write-Host "Checking clang..." -ForegroundColor Yellow
try {
    $clang = clang --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "? clang found" -ForegroundColor Green
    } else {
        Write-Host "? clang failed to run" -ForegroundColor Yellow
    }
} catch {
    Write-Host "? clang not found. Install LLVM for Windows if needed." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Checking Visual Studio Build Tools..." -ForegroundColor Yellow
try {
    $cl = cl 2>&1
    if ($LASTEXITCODE -eq 0 -or $cl -match "Microsoft") {
        Write-Host "? Visual Studio Build Tools (cl.exe) available" -ForegroundColor Green
    } else {
        Write-Host "? cl.exe not found. NVCC needs Visual Studio Build Tools" -ForegroundColor Yellow
    }
} catch {
    Write-Host "? cl.exe not found. Install Visual Studio Build Tools for NVCC" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Environment Check Complete ===" -ForegroundColor Cyan
