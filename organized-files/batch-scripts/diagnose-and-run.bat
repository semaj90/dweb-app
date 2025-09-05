@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

echo Diagnosing build environment...
where clang >nul 2>&1 && echo [OK] Clang found || echo [MISSING] Clang
where gcc >nul 2>&1 && echo [OK] GCC found || echo [MISSING] GCC  
where cl >nul 2>&1 && echo [OK] MSVC found || echo [MISSING] MSVC
if exist "C:\Progra~1\NVIDIA~2\CUDA\v12.9\lib\x64\cudart.lib" (echo [OK] CUDA libs) else (echo [MISSING] CUDA libs)

echo.
echo Testing CGO with simplest possible build...
echo package main > test.go
echo // #include ^<stdio.h^> >> test.go
echo import "C" >> test.go
echo func main() { println("CGO works") } >> test.go

set CGO_ENABLED=1
set CC=gcc
go build -x test.go 2>&1 | findstr "error"
if %errorlevel% equ 0 (
    echo CGO compilation failed. Using pure Go.
    set CGO_ENABLED=0
) else (
    echo CGO works.
)
del test.go test.exe 2>nul

echo.
echo Building service...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

if "%CGO_ENABLED%"=="0" (
    go build -o ai-microservice.exe main_simple.go
) else (
    go build -o ai-microservice.exe doc_processor.go
)

if exist ai-microservice.exe (
    start ai-microservice.exe
    timeout /t 2 >nul
    curl http://localhost:8080/health
) else (
    echo Direct run fallback...
    go run main_simple.go
)
