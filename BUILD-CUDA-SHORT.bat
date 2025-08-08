@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

for %%I in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8") do set CUDA_SHORT=%%~sI

set CC=clang
set CGO_ENABLED=1
set CGO_CFLAGS=-I%CUDA_SHORT%\include
set CGO_LDFLAGS=-L%CUDA_SHORT%\lib\x64 -lcudart_static -lcublas

go build -x -o cuda_service.exe cuda_service.go 2>&1

if exist cuda_service.exe (
    cuda_service.exe
) else (
    echo CGO failed. Alternative: Use Ollama for GPU.
    curl http://localhost:11434/api/tags
)
