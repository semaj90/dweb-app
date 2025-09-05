@echo off
REM ================================================================================
REM LEGAL AI PLATFORM - COMPLETE WIRED PRODUCTION STARTUP
REM ================================================================================

echo.
echo ================================================================================
echo STARTING LEGAL AI PLATFORM - COMPLETE WIRED SYSTEM
echo ================================================================================
echo.

REM Check for required directories
if not exist go-microservice\bin mkdir go-microservice\bin
if not exist minio-data mkdir minio-data
if not exist redis-windows mkdir redis-windows
if not exist qdrant-windows mkdir qdrant-windows

echo [1/15] Building Go Services...
cd go-microservice

REM Build core services if they don't exist or are outdated
if not exist bin\upload-service.exe (
    echo Building Upload Service...
    go build -o bin/upload-service.exe ./cmd/upload-service/main.go
)

if not exist bin\grpc-server.exe (
    echo Building gRPC Server...
    go build -o bin/grpc-server.exe ./cmd/grpc-server/main.go
)

if not exist bin\main-service.exe (
    echo Building Main Service...
    go build -o bin/main-service.exe ./main.go
)

if not exist bin\summarizer-service.exe (
    echo Building Summarizer Service...
    go build -o bin/summarizer-service.exe ./cmd/summarizer-service/main.go
)

cd ..

echo [2/15] Starting PostgreSQL Database...
net start postgresql-x64-17 2>nul || echo PostgreSQL already running or needs manual start

echo [3/15] Starting Redis Cache...
start /min "Redis Server" cmd /c "redis-server 2>nul || .\redis-windows\redis-server.exe"

echo [4/15] Starting Ollama AI Engine...
tasklist | findstr "ollama" >nul || start /min "Ollama" ollama serve

echo [5/15] Starting MinIO Object Storage...
tasklist | findstr "minio" >nul || start /min "MinIO" minio.exe server ./minio-data --address :9000 --console-address :9001

echo [6/15] Starting Qdrant Vector Database...
tasklist | findstr "qdrant" >nul || start /min "Qdrant" .\qdrant-windows\qdrant.exe

echo [7/15] Starting Neo4j Graph Database...
powershell -Command "Start-Service neo4j" 2>nul || echo Neo4j requires manual start

REM Wait for databases to initialize
echo [8/15] Waiting for services to initialize...
timeout /t 5 /nobreak >nul

echo [9/15] Starting Go Upload Service (Port 8093)...
start /min "Upload Service" cmd /c "cd go-microservice && bin\upload-service.exe"

echo [10/15] Starting Go gRPC Server (Port 8084)...
start /min "gRPC Server" cmd /c "cd go-microservice && bin\grpc-server.exe"

echo [11/15] Starting Go Main Service (Port 8080)...
start /min "Main Service" cmd /c "cd go-microservice && bin\main-service.exe"

echo [12/15] Starting Go Summarizer Service (Port 8092)...
start /min "Summarizer Service" cmd /c "cd go-microservice && bin\summarizer-service.exe"

REM Start additional pre-built services if they exist
if exist go-microservice\bin\enhanced-rag.exe (
    echo [13/15] Starting Enhanced RAG Service (Port 8094)...
    start /min "Enhanced RAG" cmd /c "cd go-microservice && bin\enhanced-rag.exe"
) else (
    echo [13/15] Enhanced RAG Service not available - using main service
)

if exist go-microservice\bin\load-balancer.exe (
    echo [14/15] Starting Load Balancer (Port 8099)...
    start /min "Load Balancer" cmd /c "cd go-microservice && bin\load-balancer.exe"
) else (
    echo [14/15] Load Balancer not available - direct connections used
)

echo [15/15] Starting SvelteKit Frontend...
cd sveltekit-frontend && start "SvelteKit Frontend" cmd /k "npm run dev -- --host 0.0.0.0"
cd ..

REM Wait for frontend to start
timeout /t 8 /nobreak >nul

echo.
echo ================================================================================
echo LEGAL AI PLATFORM WIRED AND RUNNING!
echo ================================================================================
echo.

echo Database Services:
echo - PostgreSQL:       postgresql://legal_admin:123456@localhost:5432/legal_ai_db
echo - Redis:            redis://localhost:6379
echo - Neo4j Browser:    http://localhost:7474
echo - Qdrant API:       http://localhost:6333
echo.

echo AI Services:
echo - Ollama API:       http://localhost:11434
echo - MinIO Console:    http://localhost:9001 (admin/minioadmin)
echo.

echo Go Microservices:
echo - Main Service:     http://localhost:8080
echo - gRPC Server:      http://localhost:8084
echo - Summarizer:       http://localhost:8092
echo - Upload Service:   http://localhost:8093
echo - Enhanced RAG:     http://localhost:8094
echo - Load Balancer:    http://localhost:8099
echo.

echo Frontend:
echo - SvelteKit App:    http://localhost:5173
echo.

echo.
echo System Health Check:
echo ==================

REM Health checks
timeout /t 3 /nobreak >nul
curl -s http://localhost:11434/api/tags >nul 2>&1 && echo ✓ Ollama: Running || echo ✗ Ollama: Not responding
curl -s http://localhost:6333/collections >nul 2>&1 && echo ✓ Qdrant: Running || echo ✗ Qdrant: Not responding
curl -s http://localhost:6379 >nul 2>&1 && echo ✓ Redis: Running || echo ✗ Redis: Not responding
curl -s http://localhost:8080/health >nul 2>&1 && echo ✓ Main Service: Running || echo ✗ Main Service: Not responding
curl -s http://localhost:8093/health >nul 2>&1 && echo ✓ Upload Service: Running || echo ✗ Upload Service: Not responding
curl -s http://localhost:5173 >nul 2>&1 && echo ✓ SvelteKit: Running || echo ✗ SvelteKit: Not responding

echo.
echo Press any key to open the frontend...
pause >nul

start http://localhost:5173

echo.
echo Complete Legal AI Platform is now running! 🚀
echo Use Ctrl+C in any service window to stop that service.
echo.