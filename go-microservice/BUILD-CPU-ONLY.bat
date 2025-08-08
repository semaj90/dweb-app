@echo off
:: CPU-only build for systems without CUDA
set CGO_ENABLED=1
set CC=clang
set CXX=clang++
set CGO_CFLAGS=-mavx2 -mfma -O3
set CGO_LDFLAGS=-mavx2 -mfma

echo Building CPU SIMD Legal Processor...
go clean -cache
go build -tags=cgo -ldflags="-s -w -X main.buildType=CPU_SIMD" -o legal-processor-cpu.exe legal_processor_gpu_simd.go

if %ERRORLEVEL% EQU 0 (
    echo ✅ CPU SIMD build successful
    legal-processor-cpu.exe --version 2>nul
) else (
    echo ❌ Build failed
    exit /b 1
)
