# check_cuda_clang.ps1
# Verifies CUDA Toolkit (nvcc) and Clang availability on Windows

Write-Host "=== CUDA & Clang Environment Check ===" -ForegroundColor Cyan

Write-Host ""
Write-Host "Checking nvcc (CUDA compiler)..." -ForegroundColor Yellow
try {
    $nvcc = & nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ nvcc found and working:" -ForegroundColor Green
        Write-Host $nvcc
    } else {
        Write-Host "✗ nvcc failed to run" -ForegroundColor Red
        Write-Host $nvcc
    }
} catch {
    Write-Host "✗ nvcc not found in PATH. Install CUDA Toolkit and add to PATH." -ForegroundColor Red
}

Write-Host ""
Write-Host "Checking clang..." -ForegroundColor Yellow
try {
    $clang = & clang --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ clang found:" -ForegroundColor Green
        Write-Host $clang
    } else {
        Write-Host "⚠ clang failed to run" -ForegroundColor Yellow
        Write-Host $clang
    }
} catch {
    Write-Host "⚠ clang not found. Install LLVM for Windows if needed." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Checking Visual Studio Build Tools..." -ForegroundColor Yellow
try {
    $cl = & cl 2>&1
    if ($LASTEXITCODE -eq 0 -or $cl -match "Microsoft") {
        Write-Host "✓ Visual Studio Build Tools (cl.exe) available" -ForegroundColor Green
    } else {
        Write-Host "⚠ cl.exe not found. NVCC needs Visual Studio Build Tools" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ cl.exe not found. Install Visual Studio Build Tools for NVCC" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Testing CUDA compilation and execution..." -ForegroundColor Yellow

$testCUDASrc = @'
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel(char *out) {
    int idx = threadIdx.x;
    if (idx == 0) {
        out[0] = 'O'; 
        out[1] = 'K'; 
        out[2] = '\0';
    }
}

int main() {
    char *d_output;
    cudaError_t err = cudaMalloc((void**)&d_output, 4);
    if (err != cudaSuccess) { 
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); 
        return 1; 
    }
    
    hello_kernel<<<1, 1>>>(d_output);
    cudaDeviceSynchronize();
    
    char h_output[4] = {0};
    cudaMemcpy(h_output, d_output, 3, cudaMemcpyDeviceToHost);
    printf("%s\n", h_output);
    
    cudaFree(d_output);
    return 0;
}
'@

$srcFile = "cuda_test.cu"
Set-Content -Path $srcFile -Value $testCUDASrc -Encoding UTF8

try {
    Write-Host "Compiling CUDA test..." -ForegroundColor Cyan
    & nvcc $srcFile -o cuda_test.exe 2>&1
    
    if ($LASTEXITCODE -eq 0 -and (Test-Path "cuda_test.exe")) {
        Write-Host "✓ CUDA compilation successful" -ForegroundColor Green
        
        Write-Host "Running CUDA test..." -ForegroundColor Cyan
        $result = & .\cuda_test.exe 2>&1
        
        if ($LASTEXITCODE -eq 0 -and $result -eq "OK") {
            Write-Host "✓ CUDA execution successful: $result" -ForegroundColor Green
            Write-Host ""
            Write-Host "🎉 CUDA environment is ready for GPU acceleration!" -ForegroundColor Green
        } else {
            Write-Host "✗ CUDA test execution failed: $result" -ForegroundColor Red
        }
    } else {
        Write-Host "✗ CUDA compilation failed" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ CUDA test failed: $_" -ForegroundColor Red
} finally {
    # Cleanup
    if (Test-Path $srcFile) { Remove-Item $srcFile -Force }
    if (Test-Path "cuda_test.exe") { Remove-Item "cuda_test.exe" -Force }
}

Write-Host ""
Write-Host "=== Environment Check Complete ===" -ForegroundColor Cyan