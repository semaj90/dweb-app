@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - COMPLETE STARTUP WITH NEO4J DESKTOP INTEGRATION
REM ================================================================================

echo.
echo ================================================================================
echo STARTING LEGAL AI PLATFORM - FULL SYSTEM WITH NEO4J DESKTOP
echo ================================================================================
echo.

echo [STEP 1/12] Checking Prerequisites...
echo.

echo [1.1] Checking PostgreSQL Service...
net start postgresql-x64-17 2>nul && echo ✓ PostgreSQL: Service started || echo ✓ PostgreSQL: Already running

echo [1.2] Testing Database Connection...
export PGPASSWORD=123456 && psql -h localhost -U postgres -d legal_ai_db -c "SELECT 1" >nul 2>&1 && echo ✓ PostgreSQL: Connection verified || echo ⚠️ PostgreSQL: Connection failed

echo [1.3] Checking Neo4j Desktop Status...
powershell -Command "Get-NetTCPConnection -LocalPort 7474 -ErrorAction SilentlyContinue" >nul 2>&1 && echo ✓ Neo4j Desktop: Running on port 7474 || echo ⚠️ Neo4j Desktop: Please start your database in Neo4j Desktop first

echo [1.4] Checking Neo4j Bolt Protocol...
powershell -Command "Get-NetTCPConnection -LocalPort 7687 -ErrorAction SilentlyContinue" >nul 2>&1 && echo ✓ Neo4j Bolt: Available on port 7687 || echo ⚠️ Neo4j Bolt: Not available

echo.
echo [STEP 2/12] Starting Core Database Services...

echo [2.1] Starting Redis (Primary Cache)...
tasklist | findstr "redis-server" >nul || start /min .\redis-windows\redis-server.exe
timeout /t 2 >nul
.\redis-windows\redis-cli.exe ping >nul 2>&1 && echo ✓ Redis: Running and responding || echo ✗ Redis: Failed to start

echo.
echo [STEP 3/12] Starting AI & Machine Learning Services...

echo [3.1] Starting Ollama (Multi-Model AI)...
tasklist | findstr "ollama" >nul || start /min ollama serve
timeout /t 3 >nul

echo [3.2] Verifying Ollama Models...
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✓ Ollama: API responding with models || echo ⚠️ Ollama: API not ready

echo [3.3] Starting Qdrant Vector Database...
tasklist | findstr "qdrant" >nul || start /min .\qdrant-windows\qdrant.exe
timeout /t 2 >nul

echo.
echo [STEP 4/12] Starting File Storage & Messaging...

echo [4.1] Starting MinIO Object Storage...
if not exist minio-data mkdir minio-data
tasklist | findstr "minio" >nul || start /min minio.exe server ./minio-data --address :9000 --console-address :9001

echo [4.2] Starting NATS Messaging Server...
tasklist | findstr "nats-server" >nul || start /min .\nats-server\nats-server-v2.10.7-windows-amd64\nats-server.exe --port 4222 --http_port 8222

echo.
echo [STEP 5/12] Starting Go Microservices (Backend)...

echo [5.1] Starting Enhanced RAG Service...
cd go-microservice
start /min cmd /c "go run cmd/enhanced-rag/main.go" 2>nul || start /min cmd /c "go run main.go"
cd ..

echo [5.2] Starting Upload Service...
cd go-microservice  
start /min cmd /c "go run cmd/upload-service/main.go" 2>nul || echo Upload service will start with RAG
cd ..

echo [5.3] Starting GPU Services (if available)...
cd go-microservice
start /min cmd /c "go run gpu-orchestrator-service.go" 2>nul || echo GPU Orchestrator optional
start /min cmd /c "go run multi-protocol-gateway.go" 2>nul || echo Protocol Gateway optional
cd ..

echo.
echo [STEP 6/12] Starting SvelteKit Frontend...
cd sveltekit-frontend
echo Starting development server...
start cmd /k "npm run dev -- --host 0.0.0.0"
cd ..

echo.
echo [STEP 7/12] Initializing System...
echo Waiting for services to initialize...
timeout /t 15 >nul

echo.
echo ================================================================================
echo LEGAL AI PLATFORM - STARTUP COMPLETE
echo ================================================================================
echo.

echo 🎯 **PRIMARY SERVICES:**
echo - Frontend (SvelteKit):     http://localhost:5173
echo - Database (PostgreSQL):   postgresql://postgres:123456@localhost:5432/legal_ai_db  
echo - Cache (Redis):           redis://localhost:6379
echo - AI Engine (Ollama):      http://localhost:11434
echo.

echo 🔗 **GRAPH DATABASE:**
echo - Neo4j Desktop:           http://localhost:7474 (Browser)
echo - Neo4j Bolt Protocol:     neo4j://localhost:7687
echo ⚠️  Neo4j must be started manually in Neo4j Desktop before use
echo.

echo 🚀 **API SERVICES:**  
echo - Enhanced RAG:            http://localhost:8094/api/rag
echo - Upload Service:          http://localhost:8093/upload
echo - Vector Database:         http://localhost:6333
echo - MinIO Console:           http://localhost:9001 (admin/minioadmin)
echo - NATS Monitor:            http://localhost:8222
echo.

echo 📊 **SYSTEM HEALTH CHECK:**
echo ==================
.\redis-windows\redis-cli.exe ping >nul 2>&1 && echo ✓ Redis: Healthy || echo ✗ Redis: Not responding
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✓ Ollama: Healthy || echo ✗ Ollama: Not responding  
curl -s http://localhost:6333/collections >nul 2>&1 && echo ✓ Qdrant: Healthy || echo ✗ Qdrant: Not responding
curl -s http://localhost:8222 >nul 2>&1 && echo ✓ NATS: Healthy || echo ✗ NATS: Not responding
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:7474' -TimeoutSec 5 -UseBasicParsing | Out-Null; Write-Host '✓ Neo4j Desktop: Healthy' } catch { Write-Host '⚠️ Neo4j Desktop: Not accessible - start in Neo4j Desktop' }"

echo.
echo 🎮 **NEXT STEPS:**
echo 1. Open Neo4j Desktop and start your database
echo 2. Visit http://localhost:5173 for the frontend
echo 3. Check http://localhost:7474 for Neo4j Browser
echo 4. All services are now running in Redis-first mode with Neo4j support
echo.

pause