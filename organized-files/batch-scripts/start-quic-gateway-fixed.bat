@echo off
echo Starting QUIC Gateway with correct SvelteKit backend...

set QUIC_GATEWAY_PORT=8450
set BACKEND_URL=http://localhost:5173
echo QUIC_GATEWAY_PORT=%QUIC_GATEWAY_PORT%
echo BACKEND_URL=%BACKEND_URL%

echo Starting QUIC Gateway...
start "QUIC Gateway" go-microservice\bin\quic-gateway.exe

echo QUIC Gateway started on port 8450 -> SvelteKit on port 5173