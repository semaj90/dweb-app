@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo Cleaning previous builds...
del ai-microservice.exe 2>nul
go clean -cache

echo Installing dependencies...
go mod tidy

echo Building with CUDA support...
set CC=clang
set CGO_ENABLED=1
set "CGO_CFLAGS=-O3 -march=native -IC:/Progra~1/NVIDIA~2/CUDA/v12.9/include"
set "CGO_LDFLAGS=-LC:/Progra~1/NVIDIA~2/CUDA/v12.9/lib/x64 -lcudart -lcublas -lcublasLt"

go build -ldflags "-s -w" -o ai-microservice.exe main_cuda.go

if not exist ai-microservice.exe (
    echo CUDA build failed, building CPU-only version...
    set CGO_ENABLED=0
    go build -ldflags "-s -w" -o ai-microservice.exe main_simple.go
)

if exist ai-microservice.exe (
    echo Build successful
    taskkill /F /IM ai-microservice.exe 2>nul
    start ai-microservice.exe
    timeout /t 2 >nul
    echo.
    echo Testing endpoints...
    curl http://localhost:8080/health
    echo.
    curl http://localhost:8080/gpu/metrics
) else (
    echo Build failed completely
    exit /b 1
)
