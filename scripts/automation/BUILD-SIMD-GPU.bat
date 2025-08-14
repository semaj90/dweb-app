@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Build with AVX2 SIMD + CUDA
set CC=clang
set CGO_ENABLED=1
set CGO_CFLAGS=-O3 -march=native -mavx2 -mfma -IC:\Progra~1\NVIDIA~2\CUDA\v12.9\include
set CGO_LDFLAGS=-LC:\Progra~1\NVIDIA~2\CUDA\v12.9\lib\x64 -lcudart_static -lcublas

go build -tags "cuda avx2" -o simd-gpu.exe simd_gpu_parser.go

if exist simd-gpu.exe (
    taskkill /F /IM simd-gpu.exe 2>nul
    start /B simd-gpu.exe
    timeout /t 1 >nul
    
    REM Benchmark
    echo Testing SIMD performance...
    curl -X POST http://localhost:8080/parse/simd -H "Content-Type: application/json" -d @test.json
) else (
    echo Build failed. Using simdjson-go fallback.
    go build -o simd-fallback.exe main.go
    start /B simd-fallback.exe
)
