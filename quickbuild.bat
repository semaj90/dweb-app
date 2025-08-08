@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Direct build - no installation needed
set CC=clang
set CGO_ENABLED=1
set "CGO_CFLAGS=-IC:/Progra~1/NVIDIA~2/CUDA/v12.9/include"
set "CGO_LDFLAGS=-LC:/Progra~1/NVIDIA~2/CUDA/v12.9/lib/x64 -lcudart -lcublas"

REM Kill any existing process
taskkill /F /IM ai-microservice.exe 2>nul

REM Build
go build -ldflags "-s -w" -o ai-microservice.exe .

REM Start
start /B ai-microservice.exe

REM Verify
timeout /t 2 /nobreak >nul
curl http://localhost:8080/health
