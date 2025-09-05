@echo off
REM Start SIMD JSON Optimization Services
echo 🚀 Starting SIMD JSON Optimization Services...

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker not found. Please install Docker Desktop first.
    echo    https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo ✅ Docker found. Starting services...

REM Start services with Docker Compose

if %errorlevel% equ 0 (
    echo ✅ Services started successfully!
    echo.
    echo 🔧 Service URLs:
    echo    - Qdrant Vector DB: http://localhost:6333
    echo    - Redis Cache: localhost:6379
    echo.
    echo ⏳ Waiting 15 seconds for initialization...
    timeout /t 15 /nobreak >nul
    
    echo.
    echo 🧪 Testing services...
    
    REM Test Qdrant
    curl -s http://localhost:6333/health >nul
    if %errorlevel% equ 0 (
        echo ✅ Qdrant is healthy
    ) else (
        echo ⚠️ Qdrant may still be starting up
    )
    
    REM Test Redis
    docker exec redis-optimization redis-cli ping >nul 2>&1
    if %errorlevel% equ 0 (
        echo ✅ Redis is healthy
    ) else (
        echo ⚠️ Redis may still be starting up
    )
    
    echo.
    echo 📋 Next steps:
    echo    1. Install Ollama: https://ollama.ai/download
    echo    2. Pull embedding model: ollama pull nomic-embed-text
    echo    3. Test optimization: http://localhost:5173/dev/copilot-optimizer
    echo.
    echo 🛑 To stop services: docker-compose -f docker-compose-optimization.yml down
    
) else (
    echo ❌ Failed to start services. Check Docker and try again.
)

pause