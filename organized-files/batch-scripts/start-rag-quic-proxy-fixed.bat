@echo off
echo Starting RAG QUIC Proxy with fixed configuration...

rem Set environment variables for correct ports
set RAG_QUIC_FRONT_PORT=8451
set RAG_QUIC_FALLBACK_PORT=8452
set RAG_BACKEND_URL=http://localhost:8093
set RAG_QUIC_ENABLE_FALLBACK=true

echo Configuration:
echo   QUIC Port (UDP): %RAG_QUIC_FRONT_PORT%
echo   HTTP/2 Fallback (TCP): %RAG_QUIC_FALLBACK_PORT%
echo   Backend Target: %RAG_BACKEND_URL%

echo Testing backend connectivity...
curl -s --connect-timeout 2 http://localhost:8093/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Backend not responding. Please start upload service first.
    exit /b 1
)
echo Backend healthy.

echo Starting RAG QUIC Proxy...
go-microservice\bin\rag-quic-proxy.exe