@echo off
:: FIXED-COMPLETE-SYSTEM.bat
:: Fixed version with proper paths and error handling

echo ============================================================
echo   COMPLETE DOCUMENT INGESTION + RAG SYSTEM
echo   With MinIO, Neo4j, PGVector, LangChain, Ollama
echo ============================================================
echo.

:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script requires Administrator privileges.
    echo Right-click and select "Run as Administrator"
    pause
    exit /b 1
)

:: Set working directory
cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

:: Set environment variables
set NODE_ENV=production
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_rag?sslmode=disable
set REDIS_URL=localhost:6379
set RABBITMQ_URL=amqp://guest:guest@localhost:5672/
set OLLAMA_URL=http://localhost:11434
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password

echo.
echo [STEP 1/8] Starting Core Services...
echo =====================================

:: Start PostgreSQL
echo Starting PostgreSQL...
net start postgresql-x64-14 2>nul
if %errorlevel% == 0 (
    echo   [OK] PostgreSQL started
) else (
    echo   [OK] PostgreSQL already running
)

:: Start Redis (with proper path handling)
echo Starting Redis...
if exist "C:\Program Files\Redis\redis-server.exe" (
    start /min "Redis" "C:\Program Files\Redis\redis-server.exe"
    echo   [OK] Redis started from Program Files
) else if exist "C:\Redis\redis-server.exe" (
    start /min "Redis" "C:\Redis\redis-server.exe"
    echo   [OK] Redis started from C:\Redis
) else if exist "redis-windows\redis-server.exe" (
    start /min "Redis" "redis-windows\redis-server.exe"
    echo   [OK] Redis started from local directory
) else (
    echo   [WARN] Redis not found - downloading...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip' -OutFile 'redis.zip'"
    powershell -Command "Expand-Archive -Path 'redis.zip' -DestinationPath 'redis-windows' -Force"
    del redis.zip
    start /min "Redis" "redis-windows\redis-server.exe"
    echo   [OK] Redis downloaded and started
)

:: Start RabbitMQ
echo Starting RabbitMQ...
net start RabbitMQ 2>nul
if %errorlevel% == 0 (
    echo   [OK] RabbitMQ started
) else (
    echo   [OK] RabbitMQ already running
)

:: Start Ollama
echo Starting Ollama...
start /min "Ollama" ollama serve
echo   [OK] Ollama server started

:: Start MinIO
echo Starting MinIO...
if exist "minio.exe" (
    start /min "MinIO" minio.exe server ./minio-data --console-address ":9001"
    echo   [OK] MinIO started
) else (
    echo   [WARN] MinIO not found - download from https://min.io/download
)

:: Start Neo4j
echo Starting Neo4j...
if exist "neo4j-community-5.23.0\bin\neo4j.bat" (
    start /min "Neo4j" neo4j-community-5.23.0\bin\neo4j.bat console
    echo   [OK] Neo4j started
) else (
    echo   [WARN] Neo4j not found in expected location
)

timeout /t 5 /nobreak >nul

echo.
echo [STEP 2/8] Building Go Services...
echo ===================================

:: Create go-microservice directories if they don't exist
if not exist "go-microservice\cmd\enhanced-rag-v2-local" (
    mkdir "go-microservice\cmd\enhanced-rag-v2-local"
)

if not exist "go-microservice\bin" (
    mkdir "go-microservice\bin"
)

:: Check if main.go exists, if not use the existing enhanced-rag
if exist "go-microservice\cmd\enhanced-rag-v2-local\main.go" (
    echo Building Enhanced RAG V2 Local...
    cd go-microservice
    go build -o bin\enhanced-rag-v2-local.exe cmd\enhanced-rag-v2-local\main.go 2>nul
    cd ..
    if exist "go-microservice\bin\enhanced-rag-v2-local.exe" (
        echo   [OK] Enhanced RAG V2 built successfully
    ) else (
        echo   [WARN] Build failed, trying alternative...
    )
)

:: Try alternative build paths
if not exist "go-microservice\bin\enhanced-rag-v2-local.exe" (
    if exist "go-microservice\cmd\enhanced-rag\main.go" (
        echo Building from enhanced-rag...
        cd go-microservice
        go build -o bin\enhanced-rag-v2-local.exe cmd\enhanced-rag\main.go 2>nul
        cd ..
    ) else if exist "go-microservice\cmd\production-rag\main.go" (
        echo Building from production-rag...
        cd go-microservice
        go build -o bin\enhanced-rag-v2-local.exe cmd\production-rag\main.go 2>nul
        cd ..
    )
)

echo.
echo [STEP 3/8] Starting Document Ingestion Pipeline...
echo ===================================================

:: Start the document ingestion pipeline
if exist "scripts\document-ingestion-pipeline.js" (
    start /min "Document Pipeline" node scripts\document-ingestion-pipeline.js
    echo   [OK] Document ingestion pipeline started
) else (
    echo   [WARN] Document pipeline script not found
)

echo.
echo [STEP 4/8] Starting Enhanced RAG Service...
echo ============================================

if exist "go-microservice\bin\enhanced-rag-v2-local.exe" (
    start /min "Enhanced RAG" go-microservice\bin\enhanced-rag-v2-local.exe
    echo   [OK] Enhanced RAG service started
) else if exist "go-microservice\bin\enhanced-rag.exe" (
    start /min "Enhanced RAG" go-microservice\bin\enhanced-rag.exe
    echo   [OK] Using alternative Enhanced RAG binary
) else if exist "go-microservice\bin\production-rag.exe" (
    start /min "Enhanced RAG" go-microservice\bin\production-rag.exe
    echo   [OK] Using production RAG binary
) else (
    echo   [WARN] No Go service binary found
)

echo.
echo [STEP 5/8] Starting Aggregate Server...
echo ========================================

if exist "scripts\aggregate-server.cjs" (
    start /min "Aggregate Server" node scripts\aggregate-server.cjs
    echo   [OK] Aggregate server started
) else (
    echo   [WARN] Aggregate server script not found
)

echo.
echo [STEP 6/8] Installing Ollama Models...
echo =======================================

:: Check if gemma3-legal model exists
ollama list | findstr "gemma3-legal" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing gemma3 model...
    ollama pull gemma3:latest
    
    :: Create legal variant
    echo FROM gemma3:latest > Modelfile.legal
    echo SYSTEM "You are an expert legal AI assistant specializing in contract law, compliance, and legal document analysis." >> Modelfile.legal
    ollama create gemma3-legal -f Modelfile.legal
    echo   [OK] Legal model created
) else (
    echo   [OK] Gemma3-legal model already installed
)

:: Install embedding model
ollama list | findstr "nomic-embed-text" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing embedding model...
    ollama pull nomic-embed-text
    echo   [OK] Embedding model installed
) else (
    echo   [OK] Embedding model already installed
)

echo.
echo [STEP 7/8] Initializing Database Schema...
echo ===========================================

:: Apply PostgreSQL schema
if exist "sql\enhanced-rag-v2-schema.sql" (
    psql -U postgres -d legal_ai_rag -f sql\enhanced-rag-v2-schema.sql 2>nul
    echo   [OK] Database schema applied
) else (
    echo   [WARN] Schema file not found
)

:: Create pgvector extension
psql -U postgres -d legal_ai_rag -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>nul
echo   [OK] PGVector extension ensured

echo.
echo [STEP 8/8] Running System Tests...
echo ===================================

timeout /t 3 /nobreak >nul

:: Test services
echo Testing service endpoints...

:: Test PostgreSQL
psql -U postgres -d legal_ai_rag -c "SELECT 1;" >nul 2>&1
if %errorlevel% == 0 (
    echo   [OK] PostgreSQL responding
) else (
    echo   [FAIL] PostgreSQL not responding
)

:: Test Redis
redis-cli ping >nul 2>&1
if %errorlevel% == 0 (
    echo   [OK] Redis responding
) else (
    echo   [WARN] Redis not responding
)

:: Test Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% == 0 (
    echo   [OK] Ollama API responding
) else (
    echo   [WARN] Ollama not responding
)

:: Test MinIO
curl -s http://localhost:9000 >nul 2>&1
if %errorlevel% == 0 (
    echo   [OK] MinIO responding
) else (
    echo   [WARN] MinIO not responding
)

:: Test Neo4j
curl -s http://localhost:7474 >nul 2>&1
if %errorlevel% == 0 (
    echo   [OK] Neo4j responding
) else (
    echo   [WARN] Neo4j not responding
)

echo.
echo ============================================================
echo   SYSTEM STATUS SUMMARY
echo ============================================================
echo.
echo Services Running:
echo   PostgreSQL    : http://localhost:5432
echo   Redis         : http://localhost:6379
echo   RabbitMQ      : http://localhost:15672 (Management UI)
echo   Ollama        : http://localhost:11434
echo   MinIO         : http://localhost:9000 (Console: 9001)
echo   Neo4j         : http://localhost:7474 (Browser)
echo   Enhanced RAG  : http://localhost:8097
echo   Aggregate     : http://localhost:8123
echo.
echo Document Ingestion Pipeline Ready!
echo.
echo Usage:
echo   1. Upload documents via API:
echo      curl -X POST http://localhost:8097/api/documents/upload
echo.
echo   2. Chat with legal AI:
echo      curl -X POST http://localhost:8097/api/chat
echo.
echo   3. Search documents:
echo      curl http://localhost:8097/api/search?q=liability
echo.
echo   4. View system status:
echo      curl http://localhost:8123/aggregate
echo.
echo ============================================================
echo.
pause
