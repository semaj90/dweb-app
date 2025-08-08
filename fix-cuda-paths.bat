@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Fix path spaces with short names
for %%I in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8") do set CUDA_SHORT=%%~sI

REM Remove go-nvml entirely - Windows incompatible
go get -u github.com/NVIDIA/go-nvml/pkg/nvml@none

REM Working CGO config for Windows
set CC=clang
set CGO_ENABLED=1
set CGO_CFLAGS=-I%CUDA_SHORT%\include
set CGO_LDFLAGS=-L%CUDA_SHORT%\lib\x64 -lcudart_static -lcublas

REM Build
go build -o service.exe main.go

REM Already running on 8080 - just restart
taskkill /F /IM service.exe 2>nul
start /B service.exe
