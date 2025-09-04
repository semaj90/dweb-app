@echo off
REM START-SIMPLE-CUDA-SERVER.bat - Start Simple CUDA gRPC Server with Cache Integration

echo 🚀 Starting Simple Legal CUDA Server with Enhanced Cache System
echo.

REM Set environment variables
set HTTP_PORT=8080
set GRPC_PORT=50052
set ENVIRONMENT=development
set POSTGRES_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set REDIS_URL=localhost:6379
set CUDA_DEVICE_ID=0
set CUDA_STREAMS=8
set TENSOR_CORES=true

echo 📊 Configuration:
echo    HTTP Port: %HTTP_PORT%
echo    gRPC Port: %GRPC_PORT%
echo    Environment: %ENVIRONMENT%
echo    CUDA Device: %CUDA_DEVICE_ID%
echo    CUDA Streams: %CUDA_STREAMS%
echo    Tensor Cores: %TENSOR_CORES%
echo.

REM Check if server executable exists
if exist "go-microservice\cuda-server\simple-legal-cuda-server.exe" (
    echo ✅ Simple CUDA server executable found
) else (
    echo ⚙️  Building Simple CUDA server...
    cd go-microservice\cuda-server
    go build -o simple-legal-cuda-server.exe simple_legal_cuda_server.go
    if %errorlevel% neq 0 (
        echo ❌ Build failed
        pause
        exit /b 1
    )
    cd ..\..
    echo ✅ Build successful
)

echo.
echo 🔧 Starting Simple Legal CUDA Server...
echo.
echo 📍 Endpoints:
echo    🌐 HTTP Server: http://localhost:%HTTP_PORT%
echo    📊 Health Check: http://localhost:%HTTP_PORT%/health
echo    📈 Metrics: http://localhost:%HTTP_PORT%/metrics
echo    💾 Cache Stats: http://localhost:%HTTP_PORT%/cache/stats  
echo    🖥️  GPU Status: http://localhost:%HTTP_PORT%/gpu/status
echo    🧠 CUDA Embed API: POST http://localhost:%HTTP_PORT%/api/cuda/embed
echo.
echo ⚠️  Prerequisites:
echo    - PostgreSQL running on port 5432
echo    - Redis running on port 6379 (optional)
echo    - CUDA Toolkit installed
echo.

REM Start the server
cd go-microservice\cuda-server
start "Simple Legal CUDA Server" simple-legal-cuda-server.exe

echo ✅ Simple Legal CUDA Server started!
echo.
echo 🧪 Test the server:
echo    curl -X POST http://localhost:%HTTP_PORT%/api/cuda/embed -H "Content-Type: application/json" -d "{\"text\":\"legal contract analysis\"}"
echo.
echo Press any key to open health check...
pause >nul

start http://localhost:%HTTP_PORT%/health