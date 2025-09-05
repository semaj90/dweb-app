@echo off
echo 🚀 Starting Enhanced Multi-Core Legal AI System with Context7 Integration
echo ================================================================================

:: Set comprehensive environment variables
set CUDA_ENABLED=true
set GPU_MEMORY_LIMIT=6GB
set LOAD_BALANCER_STRATEGY=gpu_aware
set LEGAL_BERT_ENABLED=true
set GOLLAMA_ENABLED=true
set MCP_MULTICORE=true
set SIMD_OPTIMIZED=true
set TENSOR_CACHE_SIZE=100
set JSON_BUFFER_SIZE=4096
set WORKER_COUNT=8
set CONTEXT7_BASE_PORT=4100
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set REDIS_URL=redis://localhost:6379
set OLLAMA_ENDPOINT=http://localhost:11434

echo 🔧 Enhanced Environment Configuration:
echo    - CUDA: %CUDA_ENABLED%
echo    - GPU Memory: %GPU_MEMORY_LIMIT%
echo    - Load Balancer: %LOAD_BALANCER_STRATEGY%
echo    - Legal-BERT: %LEGAL_BERT_ENABLED%
echo    - GoLlama: %GOLLAMA_ENABLED%
echo    - MCP Multicore: %MCP_MULTICORE%
echo    - SIMD Optimized: %SIMD_OPTIMIZED%
echo    - Worker Count: %WORKER_COUNT%
echo    - Context7 Base Port: %CONTEXT7_BASE_PORT%
echo.

:: Verify prerequisites
echo 🔍 Checking prerequisites...
if not exist "go-microservice\bin" (
    echo ❌ Go microservice binaries not found. Please build them first.
    pause
    exit /b 1
)

if not exist "mcp-servers\context7-multicore.js" (
    echo ❌ MCP Context7 server not found
    pause
    exit /b 1
)

if not exist "scripts\context7-worker.js" (
    echo ❌ Context7 worker not found
    pause
    exit /b 1
)
echo ✅ Prerequisites verified

:: Start core infrastructure services
echo 🏗️ Starting Core Infrastructure Services...

echo 📊 Starting Enhanced Go Load Balancer on port 8099...
cd go-microservice
if exist "bin\load-balancer.exe" (
    start /B bin\load-balancer.exe
    echo    ✅ Load balancer started
) else (
    echo    ⚠️ Load balancer not found, skipping
)
cd ..

echo 🔧 Starting Enhanced Multicore Service on port 8098...
cd go-microservice
if exist "enhanced-multicore-service.go" (
    start /B go run enhanced-multicore-service.go
    echo    ✅ Enhanced multicore service started
) else (
    echo    ⚠️ Enhanced multicore service not found, skipping
)
cd ..

:: Start Context7 MCP servers
echo 🧩 Starting Context7 MCP Servers...

echo 🖥️ Starting Primary MCP Context7 Server (port 4000)...
cd mcp
if exist "context7-multicore.js" (
    start /B node context7-multicore.js
    echo    ✅ Primary MCP server started
) else (
    echo    ⚠️ Primary MCP server not found, skipping
)
cd ..

echo 🖥️ Starting Secondary MCP Context7 Server (mcp-servers)...
cd mcp-servers
if exist "context7-multicore.js" (
    start /B node context7-multicore.js
    echo    ✅ Secondary MCP server started
) else (
    echo    ⚠️ Secondary MCP server not found, skipping
)
cd ..

:: Start Context7 Workers with coordination
echo ⚡ Starting Context7 Enhanced Workers (%WORKER_COUNT% workers: ports %CONTEXT7_BASE_PORT%-4107)...
for /l %%i in (1,1,%WORKER_COUNT%) do (
    set /a PORT=%CONTEXT7_BASE_PORT%+%%i-1
    set WORKER_ID=worker_%%i
    set WORKER_PORT=!PORT!
    echo    - Starting enhanced worker %%i on port !PORT!
    start /B node scripts\context7-worker.js
)

:: Start specialized Go services
echo 🤖 Starting Specialized Go Services...

echo 🧠 Starting Enhanced RAG Service (port 8094)...
cd go-microservice
if exist "bin\enhanced-rag.exe" (
    start /B bin\enhanced-rag.exe
    echo    ✅ Enhanced RAG started
) else (
    echo    ⚠️ Enhanced RAG not found, skipping
)
cd ..

echo 💡 Starting Recommendation Service (port 8096)...
cd go-microservice
if exist "bin\recommendation-service.exe" (
    start /B bin\recommendation-service.exe
    echo    ✅ Recommendation service started
) else (
    echo    ⚠️ Recommendation service not found, skipping
)
cd ..

echo ⚡ Starting SIMD Parser Service (port 8097)...
cd go-microservice
if exist "bin\simd-parser.exe" (
    start /B bin\simd-parser.exe
    echo    ✅ SIMD parser started
) else (
    echo    ⚠️ SIMD parser not found, skipping
)
cd ..

echo 🦙 Starting Go-Llama Chat Service (port 8099)...
if exist "go-llama-chat-service.go" (
    start /B go run go-llama-chat-service.go
    echo    ✅ Go-Llama chat service started
) else (
    echo    ⚠️ Go-Llama chat service not found, skipping
)

:: Start Node.js coordination services
echo 🔗 Starting Node.js Coordination Services...

echo 🤖 Starting Error Processor Daemon...
if exist "scripts\error-processor-daemon.js" (
    start /B node scripts\error-processor-daemon.js
    echo    ✅ Error processor started
) else (
    echo    ⚠️ Error processor not found, skipping
)

echo 📊 Starting Service Worker Manager...
if exist "scripts\service-worker-manager.js" (
    start /B node scripts\service-worker-manager.js
    echo    ✅ Service worker manager started
) else (
    echo    ⚠️ Service worker manager not found, skipping
)

:: Start frontend
echo 🌐 Starting Enhanced SvelteKit Frontend...
if exist "sveltekit-frontend\package.json" (
    cd sveltekit-frontend
    start /B npm run dev
    cd ..
    echo    ✅ SvelteKit frontend started
) else (
    echo    ⚠️ SvelteKit frontend not found, skipping
)

:: Health checks and status
echo 🏥 Performing Health Checks...
timeout /t 5 /nobreak >nul

echo 🔍 Testing service endpoints...
for %%p in (8099 8094 8096 8097 8098 4100 4101 4102 4103) do (
    curl -s http://localhost:%%p/status >nul 2>&1
    if !errorlevel! equ 0 (
        echo    ✅ Service on port %%p is responding
    ) else (
        echo    ⚠️ Service on port %%p is not responding
    )
)

echo.
echo ✅ Enhanced Multi-Core Legal AI System Startup Complete!
echo ================================================================================
echo.
echo 🌐 System URLs:
echo    - Load Balancer:           http://localhost:8099
echo    - Load Balancer Status:    http://localhost:8099/status
echo    - Load Balancer Metrics:   http://localhost:8099/metrics
echo    - Enhanced Multicore:      http://localhost:8098
echo    - Context7 Workers:        http://localhost:4100-4107
echo    - Enhanced RAG:            http://localhost:8094
echo    - Recommendation Service:  http://localhost:8096
echo    - SIMD Parser:             http://localhost:8097
echo    - Go-Llama Chat:           http://localhost:8099
echo    - SvelteKit Frontend:      http://localhost:5173
echo.
echo 🔧 Advanced Testing Commands:
echo    - Test Context7 Integration:   npm run context7:test
echo    - Test Multicore Processing:   npm run multicore:test
echo    - System Health Check:         curl http://localhost:8099/health
echo    - Context7 Worker Status:      curl http://localhost:4100/status
echo    - MCP Server Health:           curl http://localhost:4000/health
echo.
echo 📊 Monitoring Commands:
echo    - Real-time Metrics:           curl http://localhost:8099/metrics
echo    - Worker Performance:          curl http://localhost:4100/metrics
echo    - Service Discovery:           curl http://localhost:8099/services
echo.
echo 🧪 Demo and Testing:
echo    - Context7 Documentation:     npm run context7:docs
echo    - Multicore Benchmark:        npm run multicore:benchmark
echo    - GPU Acceleration Test:      npm run gpu:test
echo    - SIMD Performance Test:      npm run simd:test

echo.
echo ✅ All services started! System URLs:
echo    - Load Balancer: http://localhost:8099
echo    - Load Balancer Status: http://localhost:8099/status
echo    - Load Balancer Metrics: http://localhost:8099/metrics
echo    - Context7 Workers: http://localhost:4100-4107
echo    - Enhanced RAG: http://localhost:8094
echo    - Recommendation Service: http://localhost:8096
echo    - SvelteKit Frontend: http://localhost:5173
echo.
echo 🔧 To test the error-to-recommendation system:
echo    npm run check:auto
echo.
echo 📊 To check system status:
echo    curl http://localhost:8099/status
echo.
echo Press any key to continue...
pause >nul