@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Fix path spaces issue with 8.3 naming
for %%I in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9") do set "CUDA_PATH=%%~sI"

set CC=clang
set CXX=clang++
set CGO_ENABLED=1
set "CGO_CFLAGS=-I%CUDA_PATH%\include"
set "CGO_LDFLAGS=-L%CUDA_PATH%\lib\x64 -lcudart -lcublas"

echo Building with: %CC%
echo CUDA: %CUDA_PATH%

go build -o ai-microservice.exe .

if %errorlevel% equ 0 (
    taskkill /F /IM ai-microservice.exe 2>nul
    start /B ai-microservice.exe
    timeout /t 2 /nobreak >nul
    curl http://localhost:8080/health
) else (
    echo Build failed. Trying without CUDA...
    set CGO_ENABLED=0
    go build -o ai-microservice.exe .
    start /B ai-microservice.exe
)
