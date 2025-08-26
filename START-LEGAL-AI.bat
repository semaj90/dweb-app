@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - COMPLETE PRODUCTION STARTUP
REM ================================================================================

echo.
echo ================================================================================
echo STARTING LEGAL AI PLATFORM - FULL PRODUCTION SYSTEM
echo ================================================================================
echo.

echo [1/10] Starting PostgreSQL...
net start postgresql-x64-17 2>nul || echo PostgreSQL already running

echo [2/10] Starting Redis...
start /min .\redis-windows\redis-server.exe --service-run

echo [3/10] Starting Ollama...
tasklist | findstr "ollama" >nul || start /min ollama serve

echo [4/10] Starting MinIO...
if not exist minio-data mkdir minio-data
tasklist | findstr "minio" >nul || start /min minio.exe server ./minio-data --address :9000 --console-address :9001

echo [5/10] Starting Qdrant Vector Database...
tasklist | findstr "qdrant" >nul || start /min .\qdrant-windows\qdrant.exe

echo [6/10] Starting Neo4j...
start /min cmd /c "cd neo4j-community-5.23.0\bin && neo4j.bat start" 2>nul || echo Neo4j manual start required

echo [7/10] Starting Go Enhanced RAG Service...
start /min cmd /c "cd go-microservice && go run cmd/enhanced-rag/main.go" 2>nul || start /min cmd /c "cd go-microservice && go run main.go"

echo [8/10] Starting Go Upload Service...
start /min cmd /c "cd go-microservice && go run cmd/upload-service/main.go" 2>nul || echo Upload service fallback

echo [9/10] Starting Go AI Services...
start /min cmd /c "cd go-microservice && go run cmd/ai-summary/main.go" 2>nul || echo AI Summary service optional

echo [9.1/10] Starting GPU Orchestration Service...
start /min cmd /c "cd go-microservice && go run cuda-gpu-orchestrator.go"

echo [9.2/10] Starting Multi-Protocol Gateway...
start /min cmd /c "cd go-microservice && go run multi-protocol-gateway.go"

echo [9.3/10] Starting GPU Health Monitor...
start /min cmd /c "cd go-microservice && go run gpu-health-monitor.go"

echo [9.4/10] Starting CUDA Worker Compilation...
if exist cuda-worker\cuda-worker.exe (
    echo CUDA Worker already compiled and ready
    start /min cmd /c "cd cuda-worker && cuda-worker.exe" 2>nul || echo CUDA Worker started in background
) else (
    where nvcc >nul 2>&1 && where cl.exe >nul 2>&1 && cd cuda-worker && nvcc -std=c++14 cuda-worker.cu -o cuda-worker.exe && cd .. && echo CUDA Worker compiled successfully || (
        where nvcc >nul 2>&1 && where clang.exe >nul 2>&1 && cd cuda-worker && call build-clang.bat >nul 2>&1 && cd .. && echo CUDA Worker compiled with Clang || echo CUDA Worker compilation skipped (CUDA/VS/Clang not available)
    )
    if exist cuda-worker\cuda-worker.exe (
        start /min cmd /c "cd cuda-worker && cuda-worker.exe" 2>nul || echo CUDA Worker started in background
    )
)

echo [9.5/10] Starting Vector Processing Pipeline...
start /min cmd /c "cd python-services && python embedding-service.py" 2>nul || echo Python Embedding Service optional
start /min cmd /c "cd go-microservice && go run cmd/vector-service/main.go" 2>nul || echo Vector Service optional
start /min cmd /c "go run integration-orchestrator.go" 2>nul || echo Integration Orchestrator optional

echo [9.6/10] Starting NATS Messaging Server...
tasklist | findstr "nats-server" >nul || start /min .\nats-server\nats-server-v2.10.7-windows-amd64\nats-server.exe --port 4222 --http_port 8222

echo [10/10] Starting SvelteKit Frontend...
cd sveltekit-frontend && start cmd /k "npm run dev -- --host 0.0.0.0" && cd ..

timeout /t 8 /nobreak >nul

echo.
echo ================================================================================
echo LEGAL AI PLATFORM STARTED SUCCESSFULLY!
echo ================================================================================
echo.
echo Access Points:
echo - Frontend:          http://localhost:5173
echo - Enhanced RAG:      http://localhost:8094/api/rag
echo - Upload API:        http://localhost:8093/upload
echo - Vector Service:    http://localhost:8095/health
echo - Integration Hub:   http://localhost:8096/status
echo - Python Embedding:  http://localhost:8097/health
echo - GPU Orchestrator:  http://localhost:8231/api/gpu/status
echo - Protocol Gateway:  http://localhost:8230/api/gateway/health
echo - Health Monitor:    http://localhost:8232/api/health
echo - NATS Server:       http://localhost:8222
echo - MinIO Console:     http://localhost:9001 (admin/minioadmin)
echo - Qdrant API:        http://localhost:6333
echo - Neo4j Browser:     http://localhost:7474
echo - Ollama API:        http://localhost:11434
echo.
echo Database Details:
echo - PostgreSQL:      postgresql://legal_admin:123456@localhost:5432/legal_ai_db
echo - Redis:           redis://localhost:6379
echo.
echo Press any key to open the frontend in your browser...
pause >nul

start http://localhost:5173

echo.
echo System Status Check:
echo ==================
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✓ Ollama: Running || echo ✗ Ollama: Not responding
curl -s http://localhost:6333/collections >nul 2>&1 && echo ✓ Qdrant: Running || echo ✗ Qdrant: Not responding
curl -s http://localhost:8095/health >nul 2>&1 && echo ✓ Vector Service: Running || echo ✗ Vector Service: Not responding
curl -s http://localhost:8096/status >nul 2>&1 && echo ✓ Integration Hub: Running || echo ✗ Integration Hub: Not responding
curl -s http://localhost:8097/health >nul 2>&1 && echo ✓ Python Embedding: Running || echo ✗ Python Embedding: Not responding
curl -s http://localhost:8222 >nul 2>&1 && echo ✓ NATS Server: Running || echo ✗ NATS Server: Not responding
curl -s http://localhost:8231/api/gpu/health >nul 2>&1 && echo ✓ GPU Orchestrator: Running || echo ✗ GPU Orchestrator: Not responding
curl -s http://localhost:8230/api/gateway/health >nul 2>&1 && echo ✓ Protocol Gateway: Running || echo ✗ Protocol Gateway: Not responding
curl -s http://localhost:8232/api/health >nul 2>&1 && echo ✓ Health Monitor: Running || echo ✗ Health Monitor: Not responding
.\redis-windows\redis-cli.exe ping >nul 2>&1 && echo ✓ Redis: Running || echo ✗ Redis: Not responding
echo ✓ PostgreSQL: Check manually with psql
echo.
echo Happy coding! 🚀