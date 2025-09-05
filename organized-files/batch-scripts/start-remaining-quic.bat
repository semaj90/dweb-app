@echo off
echo Starting remaining QUIC services...

echo Starting QUIC Gateway on port 8450...
set QUIC_GATEWAY_PORT=8450
set BACKEND_URL=http://localhost:5173
start "QUIC Gateway" go-microservice\bin\quic-gateway.exe

echo Starting RAG QUIC Proxy on port 8451...  
set RAG_QUIC_FRONT_PORT=8451
set RAG_QUIC_FALLBACK_PORT=8452
start "RAG QUIC Proxy" go-microservice\bin\rag-quic-proxy.exe

echo Waiting for services to start...
timeout /t 3 /nobreak >nul

echo Services started. Check logs for status.