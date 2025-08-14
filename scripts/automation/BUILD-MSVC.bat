@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

REM Kill port 8080
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul

REM Use MSVC instead of clang for Windows CUDA
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if %errorlevel% neq 0 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
)

set CGO_ENABLED=1
set CC=cl
set "CGO_CFLAGS=/IC:\Progra~1\NVIDIA~2\CUDA\v12.9\include"
set "CGO_LDFLAGS=-LC:\Progra~1\NVIDIA~2\CUDA\v12.9\lib\x64 cudart.lib cublas.lib"

go build -o ai-microservice.exe main_cuda.go 2>build_error.log

if not exist ai-microservice.exe (
    echo CUDA build failed. Error log:
    type build_error.log
    echo.
    echo Building without CUDA...
    set CGO_ENABLED=0
    go build -o ai-microservice.exe main_simple.go
)

if exist ai-microservice.exe (
    start ai-microservice.exe
    timeout /t 2 >nul
    curl http://localhost:8080/health
) else (
    echo No build succeeded. Check build_error.log
    exit /b 1
)
