@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Proper Clang+CUDA setup for Windows
set CC=clang
set CXX=clang++
set CGO_ENABLED=1
set CGO_CFLAGS=-O3 -march=native -I%CUDA_PATH%\include
set CGO_LDFLAGS=-L%CUDA_PATH%\lib\x64 -lcudart -lcublas -lcublasLt -Wl,--allow-multiple-definition

REM Fix module deps
go mod edit -droprequire github.com/bytedance/sonic
go mod edit -droprequire github.com/NVIDIA/go-nvml
go mod tidy

REM Build with explicit Windows target
go build -tags "windows cuda" -buildmode=exe -ldflags "-s -w -extldflags '-static'" -o ai-service.exe main_cuda.go || (
    set CGO_ENABLED=0
    go build -o ai-service.exe main.go
)

taskkill /F /IM ai-service.exe 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

start /B ai-service.exe
timeout /t 1 >nul
curl http://localhost:8080/health
