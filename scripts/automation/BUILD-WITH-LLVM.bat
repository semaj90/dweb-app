@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Add LLVM to PATH for this session
set PATH=C:\Program Files\LLVM\bin;%PATH%
set CC=clang
set CXX=clang++

REM Test basic compilation
echo Testing LLVM setup...
clang --version

REM Build simple SIMD (guaranteed to work)
set CGO_ENABLED=0
go build -ldflags "-s -w" -o simple_simd.exe simple_simd.go
if exist simple_simd.exe (
    echo [OK] Simple SIMD built
    start /B simple_simd.exe
    timeout /t 1 >nul
    curl -s http://localhost:8080/health
    echo.
) else (
    echo [FAIL] Build failed
    exit /b 1
)

REM Attempt homemade build with proper LLVM config
echo.
echo Building homemade SIMD with LLVM...
set CGO_ENABLED=1
set "CGO_CFLAGS=-O3 -march=native -mavx2"
set "CGO_LDFLAGS=-L""C:\Program Files\LLVM\lib"" -lclang"

go build -x -o homemade_simd.exe homemade_simd.go 2>llvm_build.log
if exist homemade_simd.exe (
    echo [OK] Homemade SIMD compiled
) else (
    echo [INFO] CGO build failed (expected on Windows)
    echo Check llvm_build.log for details
)

echo.
echo Production recommendation: Use simple_simd.exe
echo Service running on :8080
