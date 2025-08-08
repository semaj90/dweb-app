@echo off
cd /d C:\Users\james\Desktop\deeds-web\deeds-web-app\go-microservice

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do taskkill /F /PID %%a 2>nul

REM Install MinGW if needed
where gcc >nul 2>&1 || (
    echo Installing MinGW...
    winget install mingw -e --silent
)

set CC=gcc
set CGO_ENABLED=1
set "CGO_CFLAGS=-I%CUDA_PATH%\include"
set "CGO_LDFLAGS=-L%CUDA_PATH%\lib\x64 -lcudart -lcublas"

go build -x -o ai-microservice.exe 2>build.log

if not exist ai-microservice.exe (
    echo Build failed. Check build.log
    echo Running without CGO...
    set CGO_ENABLED=0
    go build -o ai-microservice.exe
)

start ai-microservice.exe
timeout /t 2 >nul
curl http://localhost:8080/health
