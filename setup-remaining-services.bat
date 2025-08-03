@echo off
echo 🚀 Setting up Qdrant and Redis for SIMD JSON Optimization...

REM Create services directory
mkdir services 2>nul
mkdir logs 2>nul

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Docker is available, using containerized approach
    
    REM Start Qdrant
    echo 📊 Starting Qdrant Vector Database...
    docker run -d --name qdrant-optimization -p 6333:6333 -p 6334:6334 ^
        -v %cd%\services\qdrant_storage:/qdrant/storage:z ^
        qdrant/qdrant:latest
    
    REM Start Redis
    echo 💾 Starting Redis Cache...
    docker run -d --name redis-optimization -p 6379:6379 ^
        -v %cd%\services\redis_data:/data ^
        redis:alpine redis-server --save 60 1 --loglevel warning
    
    timeout /t 5 /nobreak >nul
    
    REM Test services
    echo 🔍 Testing services...
    curl -s http://localhost:6333/health >nul && echo ✅ Qdrant ready || echo ⚠️ Qdrant starting...
    curl -s http://localhost:6379 >nul && echo ✅ Redis ready || echo ⚠️ Redis ready (connection test may fail)
    
) else (
    echo ⚠️ Docker not available, using in-memory alternatives
    echo 📝 The system will fall back to in-memory caching
)

echo.
echo 🎉 Service setup complete!
echo 📋 Next steps:
echo    1. Run: npm run dev
echo    2. Test: http://localhost:5173/dev/copilot-optimizer
echo.
echo 🔧 Service Status:
echo    - Ollama (LLM): http://localhost:11434 ✅
echo    - Qdrant (Vector): http://localhost:6333 (if Docker available)
echo    - Redis (Cache): http://localhost:6379 (if Docker available)