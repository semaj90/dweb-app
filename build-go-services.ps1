# Fixed Build Script for Go Microservices
Write-Host "🔨 Building Go microservices..." -ForegroundColor Cyan

# Change to go-microservice directory
cd go-microservice

# Create bin directory if it doesn't exist
if (!(Test-Path bin)) { 
    New-Item -ItemType Directory -Path bin | Out-Null 
    Write-Host "  Created bin directory" -ForegroundColor Gray
}

# Tidy modules
go mod tidy

# Build services that actually exist
Write-Host "  Building enhanced-rag service..." -ForegroundColor Yellow
go build -o ./bin/enhanced-rag.exe ./cmd/enhanced-rag

Write-Host "  Building upload service..." -ForegroundColor Yellow
go build -o ./bin/upload-service.exe ./cmd/upload-service

Write-Host "  Building grpc server..." -ForegroundColor Yellow
go build -o ./bin/grpc-server.exe ./cmd/grpc-server

Write-Host "  Building cluster service..." -ForegroundColor Yellow
go build -o ./bin/cluster-http.exe ./cmd/cluster-service

Write-Host "  Building summarizer service..." -ForegroundColor Yellow
go build -o ./bin/summarizer-http.exe ./cmd/summarizer-service

Write-Host "  Building vector service..." -ForegroundColor Yellow
go build -o ./bin/vector-service.exe ./cmd/vector-service

Write-Host "  Building production-rag service..." -ForegroundColor Yellow
go build -o ./bin/production-rag.exe ./cmd/production-rag

Write-Host "  Building rag-kratos service..." -ForegroundColor Yellow
go build -o ./bin/rag-kratos.exe ./cmd/rag-kratos

Write-Host "  Building recommendations service..." -ForegroundColor Yellow
go build -o ./bin/recommendations-service.exe ./cmd/recommendations-service

Write-Host "  Building SIMD health service..." -ForegroundColor Yellow
go build -o ./bin/simd-health.exe ./cmd/simd-health

# Build CUDA services if they exist
if (Test-Path "./cmd/cuda-ai-service") {
    Write-Host "  Building CUDA AI service..." -ForegroundColor Yellow
    go build -o ./bin/cuda-ai-service.exe ./cmd/cuda-ai-service
}

if (Test-Path "./cmd/gpu-tensor-service") {
    Write-Host "  Building GPU tensor service..." -ForegroundColor Yellow
    go build -o ./bin/gpu-tensor-service.exe ./cmd/gpu-tensor-service
}

Write-Host "✅ Go microservices built successfully" -ForegroundColor Green

# Now build QUIC services
Write-Host ""
Write-Host "🌐 Building QUIC services..." -ForegroundColor Cyan

# Change to quic-services directory
cd ../quic-services

# Tidy modules
go mod tidy

Write-Host "  Building QUIC gateway..." -ForegroundColor Yellow
go build -o ../go-microservice/bin/quic-gateway.exe ./quic-gateway.go

Write-Host "  Building QUIC vector proxy..." -ForegroundColor Yellow
go build -o ../go-microservice/bin/quic-vector-proxy.exe ./quic-vector-proxy.go

Write-Host "  Building QUIC AI stream..." -ForegroundColor Yellow
go build -o ../go-microservice/bin/quic-ai-stream.exe ./quic-ai-stream.go

# Build additional QUIC services if they exist
if (Test-Path "./quic-tensor-server.go") {
    Write-Host "  Building QUIC tensor server..." -ForegroundColor Yellow
    go build -o ../go-microservice/bin/quic-tensor-server.exe ./quic-tensor-server.go
}

Write-Host "✅ QUIC services built successfully" -ForegroundColor Green

# Return to original directory
cd ..

Write-Host ""
Write-Host "🎯 Build Summary:" -ForegroundColor Magenta
Write-Host "  All Go microservices compiled to: go-microservice/bin/" -ForegroundColor White
Write-Host "  Use 'ls go-microservice/bin/' to see all built executables" -ForegroundColor Gray

# List built executables
Write-Host ""
Write-Host "📁 Built executables:" -ForegroundColor Blue
Get-ChildItem go-microservice/bin/*.exe | ForEach-Object {
    $size = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  $($_.Name) ($size MB)" -ForegroundColor Green
}