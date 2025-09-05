@echo off
REM ================================================================================
REM PRODUCTION SERVICE STARTUP - 37 Go Binaries Orchestration
REM ================================================================================
echo 🚀 Starting Legal AI Production Service Matrix...

REM Create log directory
if not exist "logs" mkdir logs

REM ================================================================================
REM TIER 1: CORE SERVICES (Must Start First)
REM ================================================================================
echo.
echo ⚡ TIER 1: Starting Core Services...

echo Starting Enhanced RAG (8094)...
start "Enhanced-RAG" /MIN cmd /c "..\go-microservice\bin\enhanced-rag.exe > logs\enhanced-rag.log 2>&1"
timeout /t 3 /nobreak > nul

echo Starting Upload Service (8093)...
start "Upload-Service" /MIN cmd /c "..\go-microservice\bin\upload-service.exe > logs\upload-service.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting gRPC Server (50051)...
start "gRPC-Server" /MIN cmd /c "..\go-microservice\bin\grpc-server.exe > logs\grpc-server.log 2>&1"
timeout /t 2 /nobreak > nul

REM ================================================================================
REM TIER 2: ENHANCED SERVICES (Performance Layer)
REM ================================================================================
echo.
echo 🔥 TIER 2: Starting Enhanced Services...

echo Starting QUIC Proxy (8216)...
start "QUIC-Proxy" /MIN cmd /c "..\go-microservice\rag-quic-proxy.exe > logs\rag-quic-proxy.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting AI Summary (8096)...
start "AI-Summary" /MIN cmd /c "..\ai-summary-service\ai-enhanced.exe > logs\ai-enhanced.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting Cluster Manager (8213)...
start "Cluster-Manager" /MIN cmd /c "..\go-microservice\bin\cluster-http.exe > logs\cluster-http.log 2>&1"
timeout /t 2 /nobreak > nul

REM ================================================================================
REM TIER 3: SPECIALIZED SERVICES (Feature Layer)
REM ================================================================================
echo.
echo 🎯 TIER 3: Starting Specialized Services...

echo Starting Live Agent (8200)...
start "Live-Agent" /MIN cmd /c "..\ai-summary-service\live-agent-enhanced.exe > logs\live-agent.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting Legal AI (8202)...
start "Legal-AI" /MIN cmd /c "..\go-microservice\enhanced-legal-ai.exe > logs\legal-ai.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting XState Manager (8212)...
start "XState-Manager" /MIN cmd /c "..\go-microservice\bin\xstate-manager.exe > logs\xstate-manager.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting Kratos Server (50052)...
start "Kratos-Server" /MIN cmd /c "..\go-microservice\rag-kratos.exe > logs\rag-kratos.log 2>&1"
timeout /t 2 /nobreak > nul

REM ================================================================================
REM TIER 4: INFRASTRUCTURE SERVICES (Support Layer)
REM ================================================================================
echo.
echo 🔧 TIER 4: Starting Infrastructure Services...

echo Starting Load Balancer (8222)...
start "Load-Balancer" /MIN cmd /c "..\go-microservice\bin\load-balancer.exe > logs\load-balancer.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting GPU Indexer (8220)...
start "GPU-Indexer" /MIN cmd /c "..\go-microservice\bin\gpu-indexer-service.exe > logs\gpu-indexer.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting Production Cluster (8215)...
start "Production-Cluster" /MIN cmd /c "..\indexing-system\modular-cluster-service-production.exe > logs\cluster-production.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting SIMD Health Monitor (8217)...
start "SIMD-Health" /MIN cmd /c "..\go-microservice\bin\simd-health.exe > logs\simd-health.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting Context7 Error Pipeline (8219)...
start "Context7-Pipeline" /MIN cmd /c "..\go-microservice\bin\context7-error-pipeline.exe > logs\context7-pipeline.log 2>&1"
timeout /t 2 /nobreak > nul

REM ================================================================================
REM ADDITIONAL SPECIALIZED SERVICES
REM ================================================================================
echo.
echo 🚀 Starting Additional Services...

echo Starting Recommendation Service (8223)...
start "Recommendations" /MIN cmd /c "..\go-microservice\bin\recommendation-service.exe > logs\recommendations.log 2>&1"
timeout /t 1 /nobreak > nul

echo Starting Summarizer Services...
start "Summarizer-HTTP" /MIN cmd /c "..\go-microservice\bin\summarizer-http.exe > logs\summarizer-http.log 2>&1"
start "Summarizer-Service" /MIN cmd /c "..\go-microservice\bin\summarizer-service.exe > logs\summarizer-service.log 2>&1"
timeout /t 2 /nobreak > nul

echo Starting Alternative Upload Services...
start "Gin-Upload" /MIN cmd /c "..\go-microservice\bin\gin-upload.exe > logs\gin-upload.log 2>&1"
start "Simple-Upload" /MIN cmd /c "..\go-microservice\bin\simple-upload.exe > logs\simple-upload.log 2>&1"
timeout /t 2 /nobreak > nul

REM ================================================================================
REM WAIT FOR SERVICES TO INITIALIZE
REM ================================================================================
echo.
echo ⏳ Waiting for services to initialize...
timeout /t 10 /nobreak > nul

REM ================================================================================
REM SERVICE HEALTH CHECK
REM ================================================================================
echo.
echo 🔍 Performing Health Checks...

echo Checking Core Services...
curl -s http://localhost:8094/health > nul && echo ✅ Enhanced RAG (8094) || echo ❌ Enhanced RAG (8094)
curl -s http://localhost:8093/health > nul && echo ✅ Upload Service (8093) || echo ❌ Upload Service (8093)

echo Checking Enhanced Services...
curl -s http://localhost:8096/health > nul && echo ✅ AI Summary (8096) || echo ❌ AI Summary (8096)
curl -s http://localhost:8213/health > nul && echo ✅ Cluster Manager (8213) || echo ❌ Cluster Manager (8213)

echo Checking Specialized Services...
curl -s http://localhost:8202/health > nul && echo ✅ Legal AI (8202) || echo ❌ Legal AI (8202)
curl -s http://localhost:8212/health > nul && echo ✅ XState Manager (8212) || echo ❌ XState Manager (8212)

REM ================================================================================
REM COMPLETION SUMMARY
REM ================================================================================
echo.
echo 🎉 ================================================================================
echo 🎉 PRODUCTION SERVICE MATRIX STARTUP COMPLETE
echo 🎉 ================================================================================
echo.
echo 📊 Service Tiers Started:
echo    ⚡ Tier 1 - Core Services: Enhanced RAG, Upload, gRPC
echo    🔥 Tier 2 - Enhanced Services: QUIC, AI Summary, Cluster Manager  
echo    🎯 Tier 3 - Specialized Services: Live Agent, Legal AI, XState, Kratos
echo    🔧 Tier 4 - Infrastructure: Load Balancer, GPU Indexer, Production Cluster
echo.
echo 🌐 Protocol Support:
echo    • HTTP/JSON APIs on ports 8093-8226
echo    • gRPC Services on ports 50051-50052
echo    • QUIC Protocol on port 8216
echo    • WebSocket Real-time on various ports
echo.
echo 📋 Next Steps:
echo    1. Start SvelteKit Frontend: npm run dev
echo    2. Test API endpoints: /api/v1/cluster/health
echo    3. Monitor logs in /logs directory
echo    4. Access UI at http://localhost:5173
echo.
echo 🚀 Ready for Production Traffic!

pause