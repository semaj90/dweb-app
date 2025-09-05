@echo off
cls
color 0C
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                    🐳 DOCKER HEALTH CHECK FIX SCRIPT                        ║
echo ║                    Fixing Unhealthy Container Issues                        ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

echo 📋 Diagnosing Docker container health issues...
echo ════════════════════════════════════════════════════════════════════════════════

echo.
echo [STEP 1] 🔍 Current Container Status:
echo ────────────────────────────────────────────────────────────────────────────────
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo [STEP 2] 🩺 Checking Health Details for Unhealthy Containers:
echo ────────────────────────────────────────────────────────────────────────────────

echo.
echo 🤖 Ollama GPU Container Health:
docker inspect legal-ollama-gpu --format="{{range .State.Health.Log}}{{.Output}}{{end}}" 2>nul
if errorlevel 1 (
    echo ❌ Could not inspect legal-ollama-gpu
) else (
    echo ✅ Health check details retrieved
)

echo.
echo 🔍 Qdrant Container Health:
docker inspect legal-qdrant-optimized --format="{{range .State.Health.Log}}{{.Output}}{{end}}" 2>nul
if errorlevel 1 (
    echo ❌ Could not inspect legal-qdrant-optimized
) else (
    echo ✅ Health check details retrieved
)

echo.
echo [STEP 3] 🔧 Applying Fixes:
echo ────────────────────────────────────────────────────────────────────────────────

echo.
echo 🤖 Fixing Ollama GPU Container...
echo Checking if Ollama service is responding...
docker exec legal-ollama-gpu ollama list 2>nul
if errorlevel 1 (
    echo ⚠️ Ollama not responding, restarting service...
    docker exec legal-ollama-gpu pkill ollama 2>nul
    timeout /t 3 >nul
    docker exec legal-ollama-gpu nohup ollama serve ^&
    timeout /t 5 >nul
) else (
    echo ✅ Ollama service is responding
)

echo.
echo 🔍 Fixing Qdrant Container...
echo Checking if Qdrant service is responding...
docker exec legal-qdrant-optimized curl -s http://localhost:6333/health 2>nul
if errorlevel 1 (
    echo ⚠️ Qdrant not responding, checking logs...
    docker logs legal-qdrant-optimized --tail 10
    echo.
    echo Attempting to restart Qdrant service...
    docker restart legal-qdrant-optimized
    timeout /t 10 >nul
) else (
    echo ✅ Qdrant service is responding
)

echo.
echo [STEP 4] 🔄 Refreshing Health Checks:
echo ────────────────────────────────────────────────────────────────────────────────

echo Waiting for health checks to update...
timeout /t 15 >nul

echo.
echo 📊 Updated Container Status:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo [STEP 5] 🧪 Testing Service Connectivity:
echo ────────────────────────────────────────────────────────────────────────────────

echo.
echo 🤖 Testing Ollama API:
docker exec legal-ollama-gpu curl -s http://localhost:11434/api/tags 2>nul
if errorlevel 1 (
    echo ❌ Ollama API not responding
    echo.
    echo 🔧 Advanced Ollama Fix:
    echo Checking GPU access...
    docker exec legal-ollama-gpu nvidia-smi 2>nul
    if errorlevel 1 (
        echo ⚠️ GPU not accessible, may need CPU-only mode
        echo Switching to CPU mode...
        docker stop legal-ollama-gpu
        docker run -d --name legal-ollama-cpu ^
            --restart unless-stopped ^
            -p 11434:11434 ^
            -v ollama:/root/.ollama ^
            ollama/ollama
        echo ✅ Started Ollama in CPU mode
    ) else (
        echo ✅ GPU accessible
        echo Restarting Ollama service with proper GPU support...
        docker exec legal-ollama-gpu ollama serve
    )
) else (
    echo ✅ Ollama API responding correctly
)

echo.
echo 🔍 Testing Qdrant API:
curl -s http://localhost:6333/health 2>nul
if errorlevel 1 (
    echo ❌ Qdrant API not accessible from host
    echo Checking port mapping...
    docker port legal-qdrant-optimized
    echo.
    echo 🔧 Advanced Qdrant Fix:
    echo Checking internal connectivity...
    docker exec legal-qdrant-optimized curl -s http://localhost:6333/health
    if errorlevel 1 (
        echo ❌ Qdrant service internal error
        echo Recreating Qdrant container...
        docker stop legal-qdrant-optimized
        docker rm legal-qdrant-optimized
        docker run -d --name legal-qdrant-optimized ^
            --restart unless-stopped ^
            -p 6333:6333 ^
            -v qdrant_storage:/qdrant/storage ^
            qdrant/qdrant:latest
        timeout /t 10 >nul
        echo ✅ Qdrant container recreated
    ) else (
        echo ⚠️ Internal service OK, checking port mapping
    )
) else (
    echo ✅ Qdrant API responding correctly
)

echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                           📈 HEALTH FIX SUMMARY                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

echo 🎯 Final Container Status:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo 📋 Service Endpoints:
echo ┌─────────────────────┬─────────────────────┬──────────────────────┐
echo │ Service             │ Internal            │ External             │
echo ├─────────────────────┼─────────────────────┼──────────────────────┤
echo │ PostgreSQL          │ 5432                │ localhost:5432       │
echo │ Redis Cluster       │ 6379                │ localhost:6379       │
echo │ Qdrant Vector       │ 6333                │ localhost:6333       │
echo │ Ollama LLM          │ 11434               │ localhost:11434      │
echo └─────────────────────┴─────────────────────┴──────────────────────┘

echo.
echo 🔍 Next Steps:
echo ════════════════════════════════════════════════════════════════════════════════
echo.
echo 1. ✅ Test your application connection to fixed services
echo 2. 🔄 If issues persist, check application logs for connection errors  
echo 3. 🐳 Monitor container logs: docker logs [container-name]
echo 4. 🚀 Start your SvelteKit app: npm run dev
echo.

echo 💡 Troubleshooting Commands:
echo ════════════════════════════════════════════════════════════════════════════════
echo • Check all logs:        docker-compose logs -f
echo • Restart all services: docker-compose restart  
echo • Rebuild containers:   docker-compose up --build -d
echo • Remove and recreate:  docker-compose down ^&^& docker-compose up -d
echo.

echo Would you like to:
echo 1. 🚀 Start the SvelteKit development server
echo 2. 📊 Show detailed container logs
echo 3. 🔄 Restart all containers
echo 4. ❌ Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting SvelteKit development server...
    echo ════════════════════════════════════════════════════════════════════════════════
    cd /d "%~dp0"
    npm run dev
) else if "%choice%"=="2" (
    echo.
    echo 📊 Showing container logs...
    echo ════════════════════════════════════════════════════════════════════════════════
    docker-compose logs -f --tail=50
) else if "%choice%"=="3" (
    echo.
    echo 🔄 Restarting all containers...
    echo ════════════════════════════════════════════════════════════════════════════════
    docker-compose restart
    echo ✅ All containers restarted
    timeout /t 5 >nul
    docker ps
) else (
    echo.
    echo 👋 Docker health check complete!
    echo ════════════════════════════════════════════════════════════════════════════════
    echo Your containers should now be healthier. 
    echo Run this script again if health issues persist.
)

pause