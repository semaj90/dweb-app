@echo off
echo 🔥 Testing Go CGO + CUDA Setup
echo ================================

REM Set environment variables for CGO and CUDA
set "PATH=%PATH%;C:\Program Files\LLVM\bin"
set "CC=clang"
set "CXX=clang++"
set "CGO_ENABLED=1"
set "CGO_CFLAGS=-I\"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include\""
set "CGO_LDFLAGS=-L\"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64\" -lcudart"

echo Environment Check:
echo PATH includes LLVM: %PATH%
echo CC: %CC%
echo CGO_ENABLED: %CGO_ENABLED%

cd go-microservice

echo.
echo 🏗️ Building simple CUDA test...
go build -tags=cgo -o test-cuda-simple.exe test_cuda_simple.go
if %ERRORLEVEL% EQU 0 (
    echo ✅ Build successful!
    echo.
    echo 🧪 Running CUDA test...
    test-cuda-simple.exe
) else (
    echo ❌ Build failed
)

echo.
echo 🚀 Testing legal AI server build...
go build -tags=cgo -o legal-ai-server.exe legal-ai-server.go
if %ERRORLEVEL% EQU 0 (
    echo ✅ Legal AI server build successful!
) else (
    echo ❌ Legal AI server build failed
)

pause