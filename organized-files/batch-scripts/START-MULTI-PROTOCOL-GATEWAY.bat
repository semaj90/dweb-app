@echo off
REM Enhanced Multi-Protocol Gateway Startup Script
REM Starts all components of the multi-protocol communication architecture
REM For Legal AI Platform - Production Ready

echo.
echo ========================================
echo   MULTI-PROTOCOL GATEWAY STARTUP
echo   Legal AI Platform - Production Ready
echo ========================================
echo.

REM Set environment variables
set GATEWAY_HTTP_PORT=8230
set GATEWAY_GRPC_PORT=50050
set GATEWAY_QUIC_PORT=4433
set MONITOR_HTTP_PORT=8240
set REDIS_ADDR=localhost:6379
set GATEWAY_ENABLE_FALLBACK=true
set LOAD_BALANCING_STRATEGY=adaptive
set METRICS_ENABLED=true
set ENABLE_ALERTS=true

echo [1/6] Starting Redis (if not running)...
redis-server --port 6379 --daemonize yes 2>nul || echo Redis already running or not installed

echo [2/6] Building Multi-Protocol Gateway...
go build -o multi-protocol-gateway.exe multi-protocol-enhanced-gateway.go
if errorlevel 1 (
    echo ERROR: Failed to build multi-protocol gateway
    pause
    exit /b 1
)

echo [3/6] Building Performance Monitor...
go build -o protocol-monitor.exe protocol-performance-monitor.go
if errorlevel 1 (
    echo ERROR: Failed to build performance monitor
    pause
    exit /b 1
)

echo [4/6] Starting Performance Monitor...
start "Protocol Monitor" cmd /k "echo Starting Protocol Performance Monitor... && protocol-monitor.exe"
timeout /t 3 /nobreak >nul

echo [5/6] Starting Multi-Protocol Gateway...
start "Multi-Protocol Gateway" cmd /k "echo Starting Enhanced Multi-Protocol Gateway... && multi-protocol-gateway.exe"
timeout /t 5 /nobreak >nul

echo [6/6] Starting SvelteKit Frontend with Multi-Protocol Support...
cd sveltekit-frontend
start "SvelteKit Frontend" cmd /k "echo Starting SvelteKit Frontend... && npm run dev"
cd ..

echo.
echo ========================================
echo   MULTI-PROTOCOL GATEWAY STARTED
echo ========================================
echo.
echo Services Status:
echo   Multi-Protocol Gateway: http://localhost:8230
echo   Performance Monitor:    http://localhost:8240
echo   SvelteKit Frontend:     http://localhost:5173
echo.
echo Protocol Endpoints:
echo   QUIC (HTTP/3):         https://localhost:4433
echo   gRPC:                  localhost:50050
echo   HTTP:                  http://localhost:8230
echo.
echo Management APIs:
echo   Gateway Health:        http://localhost:8230/api/gateway/health
echo   Protocol Metrics:      http://localhost:8240/api/metrics/protocols
echo   Service Discovery:     http://localhost:8230/api/gateway/services
echo   Circuit Breakers:      http://localhost:8230/api/circuit-breaker/status
echo.
echo API Examples:
echo   curl http://localhost:8230/api/gateway/health
echo   curl http://localhost:8240/api/metrics/protocols
echo   curl -X POST http://localhost:8230/api/gateway/fallback \
echo        -H "Content-Type: application/json" \
echo        -d "{\"service\":\"enhanced-rag\",\"preferred_protocol\":\"quic\",\"method\":\"GET\",\"path\":\"/health\",\"enable_fallback\":true}"
echo.
echo Press any key to view real-time logs...
pause >nul

REM Show logs
echo.
echo Starting real-time log viewer...
powershell -Command "Get-Content -Path '*.log' -Tail 10 -Wait" 2>nul || echo No log files found

pause