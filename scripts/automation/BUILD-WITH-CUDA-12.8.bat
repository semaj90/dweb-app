@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%

set CC=clang
set CGO_ENABLED=1
set CGO_CFLAGS=-I"%CUDA_PATH%\include"
set CGO_LDFLAGS=-L"%CUDA_PATH%\lib\x64" -lcudart_static -lcublas

go build -o cuda_service.exe cuda_service.go

if exist cuda_service.exe (
    taskkill /F /IM cuda_service.exe 2>nul
    cuda_service.exe
) else (
    echo Build failed. Running without CUDA.
    set CGO_ENABLED=0
    go run main.go
)
