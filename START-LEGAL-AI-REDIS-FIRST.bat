@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - REDIS-FIRST STARTUP (Neo4j Optional)
REM ================================================================================

echo.
echo ================================================================================
echo STARTING LEGAL AI PLATFORM - REDIS-FIRST CONFIGURATION
echo ================================================================================
echo.

echo [1/9] Starting PostgreSQL...
net start postgresql-x64-17 2>nul || echo PostgreSQL already running or not available

echo [2/9] Starting Redis (Primary Cache)...
tasklist | findstr "redis-server" >nul || start /min .\redis-windows\redis-server.exe

echo [3/9] Waiting for Redis to be ready...
timeout /t 3 /nobreak >nul
.\redis-windows\redis-cli.exe ping >nul 2>&1 && echo ✓ Redis: Connected || echo ✗ Redis: Connection failed

echo [4/9] Starting Ollama (AI Engine)...
tasklist | findstr "ollama" >nul || start /min ollama serve

echo [5/9] Starting MinIO (File Storage)...
if not exist minio-data mkdir minio-data
tasklist | findstr "minio" >nul || start /min minio.exe server ./minio-data --address :9000 --console-address :9001

echo [6/9] Starting Qdrant (Vector Database)...
tasklist | findstr "qdrant" >nul || start /min .\qdrant-windows\qdrant.exe

echo [7/9] Starting Core Go Services...
start /min cmd /c "cd go-microservice && go run cmd/enhanced-rag/main.go" 2>nul || start /min cmd /c "cd go-microservice && go run main.go"
start /min cmd /c "cd go-microservice && go run cmd/upload-service/main.go" 2>nul || echo Upload service will start with RAG

echo [8/9] Starting NATS Messaging...
tasklist | findstr "nats-server" >nul || start /min .\nats-server\nats-server-v2.10.7-windows-amd64\nats-server.exe --port 4222 --http_port 8222

echo [9/9] Starting SvelteKit Frontend...
cd sveltekit-frontend && start cmd /k "npm run dev -- --host 0.0.0.0" && cd ..

echo.
echo Waiting for services to initialize...
timeout /t 10 /nobreak >nul

echo.
echo ================================================================================
echo LEGAL AI PLATFORM STARTED - REDIS-FIRST MODE
echo ================================================================================
echo.
echo ✅ Primary Services:
echo - Frontend:          http://localhost:5173
echo - Redis Cache:       redis://localhost:6379 
echo - Enhanced RAG:      http://localhost:8094/api/rag
echo - Upload Service:    http://localhost:8093/upload
echo - Ollama API:        http://localhost:11434
echo - MinIO Console:     http://localhost:9001
echo - NATS Server:       http://localhost:8222
echo.
echo ⚠️  Neo4j Skipped: Java not available (using Redis for graph caching)
echo.
echo System Health Check:
echo ==================
.\redis-windows\redis-cli.exe ping >nul 2>&1 && echo ✓ Redis: Running || echo ✗ Redis: Not responding
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✓ Ollama: Running || echo ✗ Ollama: Not responding
curl -s http://localhost:6333/collections >nul 2>&1 && echo ✓ Qdrant: Running || echo ✗ Qdrant: Not responding
curl -s http://localhost:8222 >nul 2>&1 && echo ✓ NATS Server: Running || echo ✗ NATS Server: Not responding
echo.
echo 🚀 System ready for Redis-based operations!
echo.
pause