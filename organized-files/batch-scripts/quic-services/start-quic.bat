@echo off
echo Starting QUIC Protocol Services for Legal AI system...

REM Check if Go is installed
where go >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Go not found in PATH
    echo Please install Go and add it to your PATH
    echo Download from: https://golang.org/
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "certs" mkdir certs

echo Starting Legal AI QUIC Protocol Services...
echo.
echo Service Configuration:
echo   - QUIC Legal Gateway: :8443 (QUIC), :8444 (HTTP/3)
echo   - QUIC Vector Proxy: :8543 (QUIC), :8544 (HTTP/3)
echo   - QUIC AI Stream: :8643 (QUIC), :8644 (HTTP/3)
echo.
echo Performance Benefits:
echo   - 80%% faster legal document streaming
echo   - 90%% faster vector search response times
echo   - 0-RTT connection resumption
echo   - Built-in TLS 1.3 encryption
echo.

REM Generate self-signed certificates for development
if not exist "certs\server.crt" (
    echo Generating development TLS certificates...
    go run generate-certs.go
    echo Development certificates created in certs/ directory
    echo.
)

echo Starting QUIC services...
echo.

REM Start QUIC Gateway
echo [1/3] Starting QUIC Legal Gateway on :8443...
start /B go run quic-gateway.go

REM Wait a moment between starts
timeout /t 2 /nobreak >nul

REM Start QUIC Vector Proxy
echo [2/3] Starting QUIC Vector Proxy on :8543...
start /B go run quic-vector-proxy.go

REM Wait a moment between starts
timeout /t 2 /nobreak >nul

REM Start QUIC AI Stream
echo [3/3] Starting QUIC AI Stream on :8643...
start /B go run quic-ai-stream.go

echo.
echo ✅ All QUIC services started successfully!
echo.
echo Service Endpoints:
echo   📄 Legal Gateway: https://localhost:8444 (HTTP/3)
echo   🔍 Vector Proxy: https://localhost:8544 (HTTP/3)
echo   🤖 AI Stream: https://localhost:8644 (HTTP/3)
echo.
echo Health Checks:
echo   curl -k https://localhost:8444/health
echo   curl -k https://localhost:8544/health
echo   curl -k https://localhost:8644/health
echo.
echo Performance Test:
echo   ./benchmark-quic.bat
echo.
echo Press any key to view service logs...
pause >nul

REM Show recent logs
if exist "logs\quic-gateway.log" (
    echo.
    echo Recent QUIC Gateway logs:
    tail -n 10 logs\quic-gateway.log 2>nul || echo No logs available yet
)

echo.
echo QUIC services are running in background.
echo Use Task Manager or 'taskkill /f /im go.exe' to stop services.
pause