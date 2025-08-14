@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

where clang >nul 2>&1 && echo [OK] LLVM/Clang installed || echo [FAIL] LLVM not found

set CC=clang
set CGO_ENABLED=1
set "CGO_CFLAGS=-O3"
set "CGO_LDFLAGS=-L""C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64"" -lcudart -lcublas"

echo Building CUDA test...
go build -x -o cuda_test.exe cuda_check.go 2>&1 | findstr "error"

if exist cuda_test.exe (
    echo [OK] Build successful
    cuda_test.exe
) else (
    echo [FAIL] CUDA build failed
    
    echo.
    echo Checking CUDA installation...
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" (
        echo [OK] CUDA toolkit found
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" --version
    ) else (
        echo [FAIL] CUDA toolkit not found
    )
    
    echo.
    echo Checking environment...
    echo CUDA_PATH=%CUDA_PATH%
    
    echo.
    echo Try manual test:
    nvcc --version
    nvidia-smi
)
