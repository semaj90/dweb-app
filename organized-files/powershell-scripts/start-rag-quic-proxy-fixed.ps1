# RAG QUIC Proxy Startup Script with Correct Port Configuration
Write-Host "Starting RAG QUIC Proxy with fixed configuration..." -ForegroundColor Green

# Set environment variables
$env:RAG_QUIC_FRONT_PORT = "8451"
$env:RAG_QUIC_FALLBACK_PORT = "8452" 
$env:RAG_BACKEND_URL = "http://localhost:8093"
$env:RAG_QUIC_ENABLE_FALLBACK = "true"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  QUIC Port: 8451" -ForegroundColor White
Write-Host "  HTTP/2 Fallback: 8452" -ForegroundColor White
Write-Host "  Backend: http://localhost:8093" -ForegroundColor White

# Change directory
Set-Location "C:\Users\james\Desktop\deeds-web\deeds-web-app"

# Test backend
Write-Host "Testing backend connectivity..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8093/health" -TimeoutSec 3 -UseBasicParsing
    Write-Host "Backend healthy: $($response.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "Backend not responding" -ForegroundColor Red
    exit 1
}

# Start service
Write-Host "Starting RAG QUIC Proxy..." -ForegroundColor Yellow
& ".\go-microservice\bin\rag-quic-proxy.exe"