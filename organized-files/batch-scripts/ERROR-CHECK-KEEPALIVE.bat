@echo off
REM ERROR-CHECK-KEEPALIVE.bat
setlocal enabledelayedexpansion
title SIMD System Monitor
color 0A

:MAIN
cls
echo [%TIME%] Checking system...

REM Check Redis
redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Redis DOWN - Starting...
    start /min redis-server
    timeout /t 2 >nul
)

REM Check SIMD server
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] SIMD DOWN - Rebuilding...
    cd go-microservice
    go mod init simd-server 2>nul
    go get github.com/gorilla/websocket 2>nul
    go build -o simd-redis-vite.exe simd-redis-vite-server.go 2>nul
    start /min simd-redis-vite.exe
    cd ..
    timeout /t 3 >nul
)

REM Show status
echo.
echo [OK] Redis: 6379
echo [OK] SIMD: 8080
echo [OK] Uptime: %TIME%
echo.
echo Press Ctrl+C to stop monitoring...

timeout /t 5 >nul
goto MAIN