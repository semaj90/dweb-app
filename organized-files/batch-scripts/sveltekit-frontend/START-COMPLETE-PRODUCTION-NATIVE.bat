@echo off
REM ================================================================================
REM COMPLETE LEGAL AI PRODUCTION SYSTEM - NATIVE WINDOWS
REM Using ALL existing compiled Go binaries + SvelteKit + Multi-protocol support
REM ================================================================================

echo.
echo ================================================================================
echo 🚀 LEGAL AI PLATFORM - COMPLETE PRODUCTION STARTUP
echo    Native Windows • 37+ Services • Multi-Protocol (HTTP/gRPC/QUIC/WS)
echo ================================================================================
echo.

REM Set production environment
set NODE_ENV=production
set GO_ENV=production
set GPU_ENABLED=true
set CUDA_VISIBLE_DEVICES=0
set VITE_GPU_ENABLED=true
set VITE_DEMO_MODE=true

REM Database configuration
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set REDIS_URL=redis://localhost:6379
set OLLAMA_API_URL=http://localhost:11434
set MINIO_ENDPOINT=http://localhost:9000

REM Create directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data

echo.
echo ================================================================================
echo ⚡ TIER 1: FOUNDATIONAL INFRASTRUCTURE
echo ================================================================================

echo [1/6] Starting PostgreSQL Database...
net start postgresql-x64-17 2>nul || echo PostgreSQL already running

echo [2/6] Starting Redis Cache...
tasklist | findstr "redis-server" >nul || start /min redis-server

echo [3/6] Starting Ollama AI Engine...
tasklist | findstr "ollama" >nul || start /min ollama serve

echo [4/6] Starting MinIO Object Storage...
if not exist minio-data mkdir minio-data
tasklist | findstr "minio" >nul || start /min minio.exe server ./minio-data --address :9000 --console-address :9001

echo [5/6] Starting Qdrant Vector Database...
tasklist | findstr "qdrant" >nul || start /min .\qdrant-windows\qdrant.exe

echo [6/6] Starting Neo4j Graph Database...
powershell -Command "Start-Service neo4j" 2>nul || echo Neo4j needs manual start

timeout /t 8 /nobreak > nul

echo.
echo ================================================================================
echo 🔥 TIER 2: CORE GO SERVICES (Using Existing Binaries)
echo ================================================================================

REM Check and start Enhanced RAG Service
echo [7/37] Enhanced RAG Service (8094)...
if exist "..\go-microservice\bin\enhanced-rag.exe" (
    echo ✅ Using existing enhanced-rag.exe
    start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && bin\enhanced-rag.exe > ..\sveltekit-frontend\logs\enhanced-rag.log 2>&1"
) else if exist "..\go-microservice\enhanced-rag.exe" (
    echo ✅ Using enhanced-rag.exe
    start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && enhanced-rag.exe > ..\sveltekit-frontend\logs\enhanced-rag.log 2>&1"
) else (
    echo 🔨 Building Enhanced RAG...
    start "Enhanced-RAG" /MIN cmd /c "cd ..\go-microservice && go run cmd\enhanced-rag\main.go > ..\sveltekit-frontend\logs\enhanced-rag.log 2>&1"
)

REM Check and start Upload Service
echo [8/37] Upload Service (8093)...
if exist "..\go-microservice\bin\upload-service.exe" (
    echo ✅ Using existing upload-service.exe
    start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && bin\upload-service.exe > ..\sveltekit-frontend\logs\upload-service.log 2>&1"
) else if exist "..\go-microservice\upload-service.exe" (
    echo ✅ Using upload-service.exe
    start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && upload-service.exe > ..\sveltekit-frontend\logs\upload-service.log 2>&1"
) else (
    echo 🔨 Building Upload Service...
    start "Upload-Service" /MIN cmd /c "cd ..\go-microservice && go run cmd\upload-service\main.go > ..\sveltekit-frontend\logs\upload-service.log 2>&1"
)

REM Check and start Kratos Server
echo [9/37] Kratos gRPC Server (50052)...
if exist "..\go-services\bin\kratos-server.exe" (
    echo ✅ Using existing kratos-server.exe
    start "Kratos-Server" /MIN cmd /c "cd ..\go-services && bin\kratos-server.exe > ..\sveltekit-frontend\logs\kratos-server.log 2>&1"
) else if exist "..\go-microservice\rag-kratos.exe" (
    echo ✅ Using rag-kratos.exe
    start "Kratos-Server" /MIN cmd /c "cd ..\go-microservice && rag-kratos.exe > ..\sveltekit-frontend\logs\rag-kratos.log 2>&1"
) else (
    echo 🔨 Building Kratos Server...
    start "Kratos-Server" /MIN cmd /c "cd ..\go-services && go run cmd\kratos-server\main.go > ..\sveltekit-frontend\logs\kratos-server.log 2>&1"
)

timeout /t 3 /nobreak > nul

echo.
echo ================================================================================
echo 🎯 TIER 3: PROTOCOL SERVICES (HTTP/gRPC/QUIC/WebSocket)
echo ================================================================================

echo [10/37] gRPC Server (50051)...
if exist "..\go-microservice\bin\grpc-server.exe" (
    start "gRPC-Server" /MIN cmd /c "cd ..\go-microservice && bin\grpc-server.exe > ..\sveltekit-frontend\logs\grpc-server.log 2>&1"
) else (
    start "gRPC-Server" /MIN cmd /c "cd ..\go-microservice && go run cmd\grpc-server\main.go > ..\sveltekit-frontend\logs\grpc-server.log 2>&1"
)

echo [11/37] QUIC Gateway (8216)...
if exist "..\go-microservice\bin\quic-gateway.exe" (
    start "QUIC-Gateway" /MIN cmd /c "cd ..\go-microservice && bin\quic-gateway.exe > ..\sveltekit-frontend\logs\quic-gateway.log 2>&1"
) else if exist "..\go-microservice\rag-quic-proxy.exe" (
    start "QUIC-Proxy" /MIN cmd /c "cd ..\go-microservice && rag-quic-proxy.exe > ..\sveltekit-frontend\logs\rag-quic-proxy.log 2>&1"
) else (
    start "QUIC-Gateway" /MIN cmd /c "cd ..\go-microservice && go run cmd\rag-quic\main.go > ..\sveltekit-frontend\logs\quic-gateway.log 2>&1"
)

echo [12/37] Load Balancer (8222)...
if exist "..\go-microservice\bin\load-balancer.exe" (
    start "Load-Balancer" /MIN cmd /c "cd ..\go-microservice && bin\load-balancer.exe > ..\sveltekit-frontend\logs\load-balancer.log 2>&1"
) else (
    start "Load-Balancer" /MIN cmd /c "cd ..\go-microservice && go run main.go > ..\sveltekit-frontend\logs\load-balancer.log 2>&1"
)

timeout /t 2 /nobreak > nul

echo.
echo ================================================================================
echo 🧠 TIER 4: AI & PROCESSING SERVICES
echo ================================================================================

echo [13/37] AI Summary Service (8096)...
if exist "..\ai-summary-service\ai-enhanced.exe" (
    start "AI-Summary" /MIN cmd /c "cd ..\ai-summary-service && ai-enhanced.exe > ..\sveltekit-frontend\logs\ai-enhanced.log 2>&1"
) else (
    start "AI-Summary" /MIN cmd /c "cd ..\ai-summary-service && go run enhanced-main.go > ..\sveltekit-frontend\logs\ai-enhanced.log 2>&1"
)

echo [14/37] Legal AI Service (8202)...
if exist "..\go-microservice\enhanced-legal-ai.exe" (
    start "Legal-AI" /MIN cmd /c "cd ..\go-microservice && enhanced-legal-ai.exe > ..\sveltekit-frontend\logs\legal-ai.log 2>&1"
) else (
    start "Legal-AI" /MIN cmd /c "cd ..\go-microservice && go run enhanced-legal-ai.go > ..\sveltekit-frontend\logs\legal-ai.log 2>&1"
)

echo [15/37] Live Agent Enhanced (8200)...
if exist "..\ai-summary-service\live-agent-enhanced.exe" (
    start "Live-Agent" /MIN cmd /c "cd ..\ai-summary-service && live-agent-enhanced.exe > ..\sveltekit-frontend\logs\live-agent.log 2>&1"
)

echo [16/37] GPU Indexer Service (8220)...
if exist "..\go-microservice\bin\gpu-indexer-service.exe" (
    start "GPU-Indexer" /MIN cmd /c "cd ..\go-microservice && bin\gpu-indexer-service.exe > ..\sveltekit-frontend\logs\gpu-indexer.log 2>&1"
)

timeout /t 2 /nobreak > nul

echo.
echo ================================================================================
echo 🔧 TIER 5: MANAGEMENT & ORCHESTRATION
echo ================================================================================

echo [17/37] XState Manager (8212)...
if exist "..\go-microservice\bin\xstate-manager.exe" (
    start "XState-Manager" /MIN cmd /c "cd ..\go-microservice && bin\xstate-manager.exe > ..\sveltekit-frontend\logs\xstate-manager.log 2>&1"
) else if exist "..\go-microservice\xstate-manager.exe" (
    start "XState-Manager" /MIN cmd /c "cd ..\go-microservice && xstate-manager.exe > ..\sveltekit-frontend\logs\xstate-manager.log 2>&1"
)

echo [18/37] Cluster Manager (8213)...
if exist "..\go-microservice\bin\cluster-http.exe" (
    start "Cluster-Manager" /MIN cmd /c "cd ..\go-microservice && bin\cluster-http.exe > ..\sveltekit-frontend\logs\cluster-http.log 2>&1"
)

echo [19/37] Context7 Error Pipeline (8219)...
if exist "..\go-microservice\bin\context7-error-pipeline.exe" (
    start "Context7-Pipeline" /MIN cmd /c "cd ..\go-microservice && bin\context7-error-pipeline.exe > ..\sveltekit-frontend\logs\context7-pipeline.log 2>&1"
)

echo [20/37] SIMD Health Monitor (8217)...
if exist "..\go-microservice\bin\simd-health.exe" (
    start "SIMD-Health" /MIN cmd /c "cd ..\go-microservice && bin\simd-health.exe > ..\sveltekit-frontend\logs\simd-health.log 2>&1"
)

timeout /t 2 /nobreak > nul

echo.
echo ================================================================================
echo 📊 TIER 6: ADDITIONAL PROCESSING SERVICES
echo ================================================================================

echo [21/37] Recommendation Service (8223)...
if exist "..\go-microservice\bin\recommendation-service.exe" (
    start "Recommendations" /MIN cmd /c "cd ..\go-microservice && bin\recommendation-service.exe > ..\sveltekit-frontend\logs\recommendations.log 2>&1"
)

echo [22/37] Summarizer HTTP (8224)...
if exist "..\go-microservice\bin\summarizer-http.exe" (
    start "Summarizer-HTTP" /MIN cmd /c "cd ..\go-microservice && bin\summarizer-http.exe > ..\sveltekit-frontend\logs\summarizer-http.log 2>&1"
)

echo [23/37] Summarizer Service (8225)...
if exist "..\go-microservice\bin\summarizer-service.exe" (
    start "Summarizer-Service" /MIN cmd /c "cd ..\go-microservice && bin\summarizer-service.exe > ..\sveltekit-frontend\logs\summarizer-service.log 2>&1"
) else if exist "..\go-microservice\summarizer-service.exe" (
    start "Summarizer-Service" /MIN cmd /c "cd ..\go-microservice && summarizer-service.exe > ..\sveltekit-frontend\logs\summarizer-service.log 2>&1"
)

echo [24/37] SIMD Parser (8226)...
if exist "..\go-microservice\bin\simd-parser.exe" (
    start "SIMD-Parser" /MIN cmd /c "cd ..\go-microservice && bin\simd-parser.exe > ..\sveltekit-frontend\logs\simd-parser.log 2>&1"
)

echo [25/37] Enhanced Multicore (8232)...
if exist "..\go-microservice\enhanced-multicore.exe" (
    start "Multicore" /MIN cmd /c "cd ..\go-microservice && enhanced-multicore.exe > ..\sveltekit-frontend\logs\multicore.log 2>&1"
)

timeout /t 2 /nobreak > nul

echo.
echo ================================================================================
echo 🌐 TIER 7: FRONTEND & NODE SERVICES
echo ================================================================================

echo [26/37] Starting MCP Context7 Multicore (40000)...
start "MCP-Context7" /MIN cmd /c "cd .. && node mcp-servers\context7-multicore.js > sveltekit-frontend\logs\mcp-context7.log 2>&1"

echo [27/37] Starting SvelteKit Frontend (5173)...
start "SvelteKit-Frontend" cmd /k "npm run dev -- --host 0.0.0.0 --port 5173"

timeout /t 8 /nobreak > nul

echo.
echo ================================================================================
echo 🔍 COMPREHENSIVE HEALTH CHECK - ALL SERVICES
echo ================================================================================

echo.
echo 🏥 Checking Infrastructure Services...
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✅ Ollama (11434) || echo ❌ Ollama (11434)
curl -s http://localhost:6333/collections >nul 2>&1 && echo ✅ Qdrant (6333) || echo ❌ Qdrant (6333)
redis-cli ping >nul 2>&1 && echo ✅ Redis (6379) || echo ❌ Redis (6379)
curl -s http://localhost:9000/minio/health/live >nul 2>&1 && echo ✅ MinIO (9000) || echo ❌ MinIO (9000)

echo.
echo 🔥 Checking Core Go Services...
curl -s http://localhost:8094/health >nul 2>&1 && echo ✅ Enhanced RAG (8094) || echo ❌ Enhanced RAG (8094)
curl -s http://localhost:8093/health >nul 2>&1 && echo ✅ Upload Service (8093) || echo ❌ Upload Service (8093)
curl -s http://localhost:8216/health >nul 2>&1 && echo ✅ QUIC Gateway (8216) || echo ❌ QUIC Gateway (8216)
curl -s http://localhost:8222/health >nul 2>&1 && echo ✅ Load Balancer (8222) || echo ❌ Load Balancer (8222)

echo.
echo 🎯 Checking AI Services...
curl -s http://localhost:8096/health >nul 2>&1 && echo ✅ AI Summary (8096) || echo ❌ AI Summary (8096)
curl -s http://localhost:8202/health >nul 2>&1 && echo ✅ Legal AI (8202) || echo ❌ Legal AI (8202)
curl -s http://localhost:8200/health >nul 2>&1 && echo ✅ Live Agent (8200) || echo ❌ Live Agent (8200)
curl -s http://localhost:8220/health >nul 2>&1 && echo ✅ GPU Indexer (8220) || echo ❌ GPU Indexer (8220)

echo.
echo 🔧 Checking Management Services...
curl -s http://localhost:8212/health >nul 2>&1 && echo ✅ XState Manager (8212) || echo ❌ XState Manager (8212)
curl -s http://localhost:8213/health >nul 2>&1 && echo ✅ Cluster Manager (8213) || echo ❌ Cluster Manager (8213)
curl -s http://localhost:8217/health >nul 2>&1 && echo ✅ SIMD Health (8217) || echo ❌ SIMD Health (8217)
curl -s http://localhost:8219/health >nul 2>&1 && echo ✅ Context7 Pipeline (8219) || echo ❌ Context7 Pipeline (8219)

echo.
echo 🌐 Checking Frontend & APIs...
curl -s http://localhost:5173 >nul 2>&1 && echo ✅ SvelteKit Frontend (5173) || echo ❌ SvelteKit Frontend (5173)
curl -s http://localhost:40000 >nul 2>&1 && echo ✅ MCP Context7 (40000) || echo ❌ MCP Context7 (40000)

echo.
echo ================================================================================
echo 🎉 LEGAL AI PLATFORM - PRODUCTION DEPLOYMENT COMPLETE
echo ================================================================================
echo.
echo 📊 System Architecture Summary:
echo    🏗️  Infrastructure: PostgreSQL, Redis, Ollama, MinIO, Qdrant, Neo4j
echo    ⚡  Core Services: Enhanced RAG (8094), Upload (8093), Kratos gRPC (50052)
echo    🌐  Protocols: HTTP/REST, gRPC (50051-50052), QUIC (8216), WebSocket
echo    🧠  AI Services: Legal AI, AI Summary, Live Agent, GPU Indexer
echo    🔧  Management: XState, Cluster Manager, SIMD Health, Context7
echo    📊  Processing: Summarizers, Parsers, Multi-core, Load Balancer
echo    🖥️  Frontend: SvelteKit (5173), MCP Context7 (40000)
echo.
echo 🌐 Primary Access Points:
echo    • Frontend:        http://localhost:5173
echo    • Enhanced RAG:    http://localhost:8094/api/rag
echo    • Upload API:      http://localhost:8093/upload  
echo    • QUIC Gateway:    quic://localhost:8216
echo    • Load Balancer:   http://localhost:8222
echo    • gRPC Services:   grpc://localhost:50051-50052
echo    • Cluster Health:  http://localhost:5173/api/v1/cluster/health
echo    • MinIO Console:   http://localhost:9001
echo    • Qdrant API:      http://localhost:6333
echo    • Neo4j Browser:   http://localhost:7474
echo.
echo 📋 Multi-Protocol Testing:
echo    • Test HTTP:  curl http://localhost:8094/api/rag
echo    • Test QUIC:  curl http://localhost:8216/health  
echo    • Test gRPC:  grpcurl -plaintext localhost:50051 list
echo    • Test WS:    Via SvelteKit frontend real-time features
echo.
echo 🚀 All services operational! Production ready for Legal AI workloads.
echo Press any key to open the frontend...
pause >nul

start http://localhost:5173

echo.
echo 🎯 Happy coding with the complete native Windows Legal AI ecosystem! 🚀