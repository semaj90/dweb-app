@echo off
title Enhanced Context7 Multicore Legal AI System
color 0A
echo.
echo ===============================================================================
echo 🚀 ENHANCED CONTEXT7 MULTICORE LEGAL AI SYSTEM STARTUP
echo ===============================================================================
echo.
echo Starting complete multicore system with:
echo   ⚡ Go-SIMD JSON parsing
echo   🧠 Go-LLAMA integration  
echo   🔧 Tensor processing
echo   📊 MCP Context7 integration
echo   🌐 8-worker multicore processing
echo   🎯 Load balancing with GPU awareness
echo.

:: Set comprehensive environment variables
set NODE_ENV=production
set MCP_DEBUG=true
set MCP_MULTICORE=true
set CUDA_ENABLED=true
set GPU_ACCELERATION=true
set GPU_MEMORY_LIMIT=6GB
set LOAD_BALANCER_STRATEGY=gpu_aware
set LEGAL_BERT_ENABLED=true
set GOLLAMA_ENABLED=true
set ENABLE_SIMD=true
set ENABLE_TENSOR_OPS=true
set ENABLE_LLAMA=true
set MAX_CONCURRENCY=100
set OLLAMA_ENDPOINT=http://localhost:11434
set HEALTH_CHECK_PERIOD=30
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db

echo 🔧 Environment Configuration:
echo    NODE_ENV: %NODE_ENV%
echo    CUDA: %CUDA_ENABLED%
echo    GPU Memory: %GPU_MEMORY_LIMIT%
echo    Load Balancer Strategy: %LOAD_BALANCER_STRATEGY%
echo    Legal-BERT: %LEGAL_BERT_ENABLED%
echo    Go-LLAMA: %GOLLAMA_ENABLED%
echo    SIMD JSON: %ENABLE_SIMD%
echo    Tensor Ops: %ENABLE_TENSOR_OPS%
echo    Max Concurrency: %MAX_CONCURRENCY%
echo.

:: Start Go Microservices First
echo 📊 Starting Go Load Balancer (Port 8099)...
cd go-microservice
if exist bin\load-balancer.exe (
    start /B /MIN bin\load-balancer.exe
    echo    ✅ Load balancer started
) else (
    echo    ❌ Load balancer not found - building...
    go build -o bin\load-balancer.exe load-balancer.go
    if exist bin\load-balancer.exe (
        start /B /MIN bin\load-balancer.exe
        echo    ✅ Load balancer built and started
    ) else (
        echo    ❌ Failed to build load balancer
    )
)

echo 🚀 Starting Enhanced Context7 Multicore Service (Port 8095)...
if exist cmd\enhanced-context7-multicore\main.go (
    set PORT=8095
    set WORKER_ID=context7-main
    go run cmd\enhanced-context7-multicore\main.go &
    echo    ✅ Enhanced multicore service started
) else (
    echo    ❌ Enhanced multicore service not found
)

echo 💡 Starting Enhanced RAG Service (Port 8094)...
if exist bin\enhanced-rag.exe (
    start /B /MIN bin\enhanced-rag.exe
    echo    ✅ Enhanced RAG started
) else (
    echo    ⚠️ Enhanced RAG service not available
)

echo 🔍 Starting Recommendation Service (Port 8096)...
if exist bin\recommendation-service.exe (
    start /B /MIN bin\recommendation-service.exe
    echo    ✅ Recommendation service started
) else (
    echo    ⚠️ Recommendation service not available
)

echo 📊 Starting GPU Indexer Service (Port 8097)...
if exist bin\gpu-indexer-service.exe (
    set PORT=8097
    start /B /MIN bin\gpu-indexer-service.exe
    echo    ✅ GPU indexer started
) else (
    echo    ⚠️ GPU indexer service not available
)

cd ..

:: Start Context7 Workers (8 workers on ports 4100-4107)
echo ⚡ Starting Context7 Workers (8 workers: ports 4100-4107)...
for /l %%i in (1,1,8) do (
    set /a PORT=4099+%%i
    set WORKER_ID=worker_%%i
    set WORKER_PORT=!PORT!
    echo    - Starting Context7 worker %%i on port !PORT!
    start /B /MIN node scripts\context7-worker.js
)

:: Start MCP Servers
echo 🖥️ Starting MCP Context7 Multicore Server...
cd mcp-servers
start /B /MIN node mcp-context7-wrapper.js
echo    ✅ MCP wrapper started
cd ..

echo 🌐 Starting MCP Context7 Multicore Integration...
if exist mcp\context7-multicore.js (
    cd mcp
    start /B /MIN node context7-multicore.js
    echo    ✅ MCP multicore integration started
    cd ..
) else (
    echo    ⚠️ MCP multicore integration not found
)

:: Start Error Processing Daemon
echo 🤖 Starting Error Processor Daemon...
if exist scripts\error-processor-daemon.js (
    start /B /MIN node scripts\error-processor-daemon.js
    echo    ✅ Error processor started
) else (
    echo    ⚠️ Error processor daemon not found
)

:: Start Queue Workers
echo 📋 Starting Queue Workers...
if exist scripts\queue-worker.ts (
    start /B /MIN npx tsx scripts\queue-worker.ts
    echo    ✅ Queue workers started
) else (
    echo    ⚠️ Queue workers not found
)

:: Start SvelteKit Frontend
echo 🌐 Starting SvelteKit Frontend with multicore integration...
cd sveltekit-frontend
start /B npm run dev
echo    ✅ SvelteKit frontend starting...
cd ..

:: Wait for services to initialize
echo.
echo ⏳ Waiting for services to initialize (10 seconds)...
timeout /t 10 /nobreak >nul

:: Health check
echo.
echo 🔍 Performing health checks...

:: Check Go services
echo.
echo 📊 Checking Go Services:
curl -s http://localhost:8099/status >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✅ Load Balancer: Healthy
) else (
    echo    ❌ Load Balancer: Not responding
)

curl -s http://localhost:8095/health >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✅ Enhanced Multicore Service: Healthy
) else (
    echo    ❌ Enhanced Multicore Service: Not responding
)

curl -s http://localhost:8094/health >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✅ Enhanced RAG: Healthy
) else (
    echo    ❌ Enhanced RAG: Not responding
)

:: Check Context7 Workers
echo.
echo ⚡ Checking Context7 Workers:
set HEALTHY_WORKERS=0
for /l %%i in (4100,1,4107) do (
    curl -s http://localhost:%%i/health >nul 2>&1
    if !errorlevel! equ 0 (
        set /a HEALTHY_WORKERS+=1
        echo    ✅ Worker on port %%i: Healthy
    ) else (
        echo    ❌ Worker on port %%i: Not responding
    )
)

echo.
echo 📊 Worker Health Summary: %HEALTHY_WORKERS%/8 workers healthy

:: Check Frontend
echo.
echo 🌐 Checking Frontend:
timeout /t 5 /nobreak >nul
curl -s http://localhost:5173 >nul 2>&1
if %errorlevel% equ 0 (
    echo    ✅ SvelteKit Frontend: Available
) else (
    echo    ⚠️ SvelteKit Frontend: Still starting...
)

echo.
echo ===============================================================================
echo ✅ ENHANCED CONTEXT7 MULTICORE SYSTEM STARTUP COMPLETE
echo ===============================================================================
echo.
echo 🌐 System URLs:
echo    Frontend:                 http://localhost:5173
echo    Load Balancer:           http://localhost:8099
echo    Enhanced Multicore:      http://localhost:8095
echo    Enhanced RAG:            http://localhost:8094
echo    Recommendation Service:  http://localhost:8096
echo    GPU Indexer:             http://localhost:8097
echo    Context7 Workers:        http://localhost:4100-4107
echo.
echo 🔧 Management URLs:
echo    Load Balancer Status:    http://localhost:8099/status
echo    Load Balancer Metrics:   http://localhost:8099/metrics
echo    Multicore Health:        http://localhost:8095/health
echo    Multicore Metrics:       http://localhost:8095/metrics
echo    Worker Health Check:     http://localhost:4100/health (example)
echo.
echo 🧪 Testing Commands:
echo    Test JSON SIMD parsing:
echo      curl -X POST http://localhost:8095/parse/json ^
echo           -H "Content-Type: application/json" ^
echo           -d "{\"data\":\"{\\\"test\\\":true}\",\"parser\":\"simd\"}"
echo.
echo    Test tensor processing:
echo      curl -X POST http://localhost:8095/parse/tensor ^
echo           -H "Content-Type: application/json" ^
echo           -d "{\"shape\":[2,2],\"data\":[1,2,3,4],\"op\":\"create\"}"
echo.
echo    Test LLAMA generation:
echo      curl -X POST http://localhost:8095/llama ^
echo           -H "Content-Type: application/json" ^
echo           -d "{\"prompt\":\"Explain legal liability\",\"model\":\"gemma2:2b\"}"
echo.
echo    Test recommendations:
echo      curl -X POST http://localhost:8095/recommendation ^
echo           -H "Content-Type: application/json" ^
echo           -d "{\"context\":\"typescript error\",\"priority\":\"high\"}"
echo.
echo 📊 System Monitoring:
echo    Check all service health:     curl http://localhost:8099/status
echo    Get performance metrics:      curl http://localhost:8095/metrics
echo    View worker pool stats:       curl http://localhost:8095/health
echo.
echo 🔍 Error Debugging:
echo    If services fail to start, check:
echo    1. Prerequisites: PostgreSQL, Redis, Ollama running
echo    2. GPU drivers and CUDA installation
echo    3. Go modules: cd go-microservice && go mod tidy
echo    4. Node modules: npm install
echo    5. Port conflicts: netstat -an | findstr ":8095"
echo.
echo ⚠️ Remember:
echo    - This system uses 8 CPU cores for multicore processing
echo    - GPU acceleration requires CUDA and compatible GPU
echo    - Context7 MCP integration provides Claude Code tools
echo    - All services have built-in health checks and metrics
echo.
echo Press any key to open system monitoring dashboard...
pause >nul

:: Open monitoring dashboard
start http://localhost:8099/status
start http://localhost:8095/health
start http://localhost:5173

echo.
echo 🎉 Enhanced Context7 Multicore Legal AI System is now running!
echo    Monitoring dashboards opened in your browser.
echo.
echo Press any key to exit...
pause >nul