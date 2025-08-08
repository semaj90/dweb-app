@echo off
setlocal EnableDelayedExpansion

echo 🚀 Building Production GPU+SIMD Legal Processor
echo ================================================

:: Set environment variables
set CGO_ENABLED=1
set CC=clang
set CXX=clang++

:: CUDA paths (adjust version as needed)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
set CGO_CFLAGS=-I"%CUDA_PATH%\include" -mavx2 -mfma
set CGO_LDFLAGS=-L"%CUDA_PATH%\lib\x64" -lcudart -lcublas

echo 📋 Environment Check:
echo CGO_ENABLED: %CGO_ENABLED%
echo CC: %CC%
echo CUDA_PATH: %CUDA_PATH%

:: Check CUDA availability
echo 🔍 Checking CUDA...
nvcc --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ CUDA not found! Installing CPU-only build...
    set CGO_LDFLAGS=-mavx2 -mfma
    set BUILD_TYPE=CPU_SIMD
) else (
    echo ✅ CUDA found
    set BUILD_TYPE=GPU_SIMD
)

:: Check dependencies
echo 🔍 Checking Go dependencies...
go mod tidy
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Go mod tidy failed
    exit /b 1
)

:: Clean previous builds
echo 🧹 Cleaning previous builds...
go clean -cache
del /Q *.exe 2>nul

:: Build with optimizations
echo 🔨 Building legal processor (%BUILD_TYPE%)...
set BUILD_FLAGS=-tags=cgo -ldflags="-s -w -X main.buildType=%BUILD_TYPE%" -gcflags="-N -l"

go build %BUILD_FLAGS% -o legal-processor-gpu.exe legal_processor_gpu_simd.go
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Build failed!
    echo.
    echo 🔧 Troubleshooting tips:
    echo 1. Verify CUDA installation: nvcc --version
    echo 2. Check clang: clang --version  
    echo 3. Ensure CGO is enabled: go env CGO_ENABLED
    echo 4. Try CPU-only build: BUILD-CPU-ONLY.bat
    exit /b 1
)

echo ✅ Build successful! Binary: legal-processor-gpu.exe
echo.

:: Optional: Test the binary
echo 🧪 Testing binary...
echo package main > test.go
echo import "C" >> test.go  
echo func main() {} >> test.go
go build -tags=cgo test.go
if %ERRORLEVEL% EQU 0 (
    echo ✅ CGO linking works
    del test.go test.exe 2>nul
) else (
    echo ⚠️  CGO test failed - binary may not work correctly
)

echo.
echo 🎯 Next Steps:
echo 1. Start the processor: legal-processor-gpu.exe
echo 2. Test with: curl http://localhost:8080/health
echo 3. Check logs for GPU detection status
echo 4. Monitor performance with: nvidia-smi

endlocal
