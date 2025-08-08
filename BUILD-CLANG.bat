@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo Installing Clang...
winget install LLVM.LLVM --silent 2>nul

echo Configuring Clang for CGO...
set CC=clang
set CXX=clang++
set CGO_ENABLED=1
set CGO_CFLAGS=-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"
set CGO_LDFLAGS=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" -lcudart -lcublas

echo Building with Clang...
go build -tags cuda -o ai-microservice.exe .

if %errorlevel% neq 0 (
    echo Fallback: Building without CUDA
    set CGO_ENABLED=0
    go build -tags nocuda -o ai-microservice.exe .
)

taskkill /F /IM ai-microservice.exe 2>nul
start /B ai-microservice.exe

timeout /t 2 /nobreak >nul
curl http://localhost:8080/health
