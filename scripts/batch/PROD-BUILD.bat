@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Production-ready build with proper toolchain
where gcc >nul 2>&1 || (
    echo Installing TDM-GCC for proper CGO support...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/jmeubank/tdm-gcc/releases/download/v10.3.0-tdm64-2/tdm64-gcc-10.3.0-2.exe' -OutFile tdm-gcc.exe"
    tdm-gcc.exe /S
    set PATH=C:\TDM-GCC-64\bin;%PATH%
)

REM Clean module cache and broken deps
go clean -modcache
go mod edit -droprequire github.com/bytedance/sonic
go mod edit -droprequire github.com/NVIDIA/go-nvml
go mod download

REM Production build with CUDA
set CC=gcc
set CGO_ENABLED=1
set CGO_CFLAGS=-O3 -march=native -I%CUDA_PATH%\include
set CGO_LDFLAGS=-L%CUDA_PATH%\lib\x64 -lcudart -lcublas -lcublasLt

REM Build with fallback
go build -tags cuda,windows -ldflags "-s -w" -o ai-service.exe main_cuda.go 2>nul || (
    set CGO_ENABLED=0
    go build -ldflags "-s -w" -o ai-service.exe main.go
)

REM Single port cleanup
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
    tasklist /FI "PID eq %%a" 2>nul | findstr legal-ai-server.exe >nul && taskkill /F /PID %%a
)

REM Production launch
set GIN_MODE=release
set GOMAXPROCS=%NUMBER_OF_PROCESSORS%
start /B ai-service.exe

timeout /t 1 >nul
curl -s http://localhost:8080/health
