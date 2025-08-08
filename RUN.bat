@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Use MinGW as fallback if clang fails
where clang >nul 2>&1
if %errorlevel% neq 0 (
    echo Using MinGW instead...
    set CC=gcc
    set CXX=g++
) else (
    set CC=clang
    set CXX=clang++
)

set CGO_ENABLED=1
set CGO_CFLAGS=-IC:\Progra~1\NVIDIA~2\CUDA\v12.9\include
set CGO_LDFLAGS=-LC:\Progra~1\NVIDIA~2\CUDA\v12.9\lib\x64 -lcudart -lcublas

echo Compiler: %CC%
go build -x -o ai-microservice.exe . 2>&1

if exist ai-microservice.exe (
    echo Build successful
    taskkill /F /IM ai-microservice.exe 2>nul
    start ai-microservice.exe
    timeout /t 2 >nul
    curl http://localhost:8080/health
) else (
    echo Build failed - running existing service
    curl http://localhost:8080/health
)
