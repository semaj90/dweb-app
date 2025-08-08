@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Use MinGW instead of Clang - it works with Go's linker
set CC=gcc
set CGO_ENABLED=1
set CGO_CFLAGS=-IC:/Progra~1/NVIDIA~2/CUDA/v12.8/include
set CGO_LDFLAGS=-LC:/Progra~1/NVIDIA~2/CUDA/v12.8/lib/x64 -lcudart_static -lcublas

echo Testing CUDA with MinGW...
go build -x -o test_cuda.exe test_cuda.go 2>&1 | findstr "error"

if exist test_cuda.exe (
    test_cuda.exe
) else (
    echo MinGW build failed. Installing TDM-GCC...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/jmeubank/tdm-gcc/releases/download/v10.3.0-tdm64-2/tdm64-gcc-10.3.0-2.exe' -OutFile tdm.exe"
    tdm.exe /S /D=C:\TDM-GCC-64
    set PATH=C:\TDM-GCC-64\bin;%PATH%
    set CC=gcc
    go build -o test_cuda.exe test_cuda.go
    test_cuda.exe
)
