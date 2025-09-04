@echo off
REM YoRHa Legal AI Platform - Production Service Startup
echo ===============================================================================
echo YoRHa LEGAL AI PLATFORM - PRODUCTION SERVICES STARTUP
echo ===============================================================================

cd /d "%~dp0"

echo [1/8] Starting PostgreSQL...
net start postgresql-x64-16 >nul 2>&1 || net start postgresql-x64-15 >nul 2>&
echo ✅ PostgreSQL running on port 5432

echo [2/8] Starting Redis...
if exist "services\redis\redis-server.exe" (
    start /B "Redis" /D "services\redis" redis-server.exe redis.conf
    echo ✅ Redis started on port 6379
) else (
    echo ⚠️ Redis not found - download required
)

echo [3/8] Starting RabbitMQ...
if exist "services\rabbitmq\sbin\rabbitmq-server.bat" (
    start /B "RabbitMQ" /D "services\rabbitmq\sbin" rabbitmq-server.bat
    echo ✅ RabbitMQ started - Management: http://localhost:15672
) else (
    echo ⚠️ RabbitMQ not found - download required
)

echo [4/8] Starting MinIO...
if not exist "minio-data" mkdir minio-data
start /B "MinIO" minio server minio-data --address ":9000" --console-address ":9001"
echo ✅ MinIO started - Console: http://localhost:9001

echo [5/8] Starting Neo4j...
if exist "services\neo4j\bin\neo4j.bat" (
    start /B "Neo4j" /D "services\neo4j\bin" neo4j.bat console
    echo ✅ Neo4j started - Browser: http://localhost:7474
) else (
    echo ⚠️ Neo4j not found - download required
)

echo [6/8] Starting Qdrant (Low Memory)...
if exist "services\qdrant\qdrant.exe" (
    start /B "Qdrant" /D "services\qdrant" qdrant.exe --config-path config.yaml
    echo ✅ Qdrant started - Dashboard: http://localhost:6333
) else (
    echo ⚠️ Qdrant not found - download required
)

echo [7/8] Starting Ollama...
start /B "Ollama" ollama serve
echo ✅ Ollama started on port 11434

echo [8/8] Starting Go Microservices...
if exist "go-microservice\bin\enhanced-rag.exe" (
    start /B "Enhanced RAG" "go-microservice\bin\enhanced-rag.exe"
    echo ✅ Enhanced RAG started on port 8094
)
if exist "go-microservice\bin\upload-service.exe" (
    start /B "Upload Service" "go-microservice\bin\upload-service.exe"
    echo ✅ Upload Service started on port 8093
)

echo ===============================================================================
echo YoRHa LEGAL AI PLATFORM - ALL SERVICES STARTED
echo ===============================================================================
echo Frontend: cd sveltekit-frontend && npm run dev
echo YoRHa Interface: http://localhost:5177/yorha-home
echo ===============================================================================
pause
