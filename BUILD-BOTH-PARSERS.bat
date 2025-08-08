@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo [1/2] Building homemade SIMD with LLVM/Clang...
set CC=clang
set CGO_ENABLED=1
set CGO_CFLAGS=-O3 -march=native -mavx2 -mfma
go build -o homemade_simd.exe homemade_simd.go 2>build_homemade.log

if exist homemade_simd.exe (
    echo [OK] Homemade SIMD built
) else (
    echo [FAIL] CGO build failed, check build_homemade.log
)

echo.
echo [2/2] Building simple SIMD (pure Go)...
set CGO_ENABLED=0
go build -ldflags "-s -w" -o simple_simd.exe simple_simd.go

if exist simple_simd.exe (
    echo [OK] Simple SIMD built
    simple_simd.exe
) else (
    echo [FAIL] Build failed
)
