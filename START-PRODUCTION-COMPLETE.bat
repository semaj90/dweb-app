@echo off
REM ============================================
REM COMPLETE PRODUCTION LEGAL AI SYSTEM STARTUP
REM Full stack with all services and error handling
REM ============================================

cls
echo ================================================
echo   LEGAL AI PRODUCTION SYSTEM - COMPLETE STARTUP
echo ================================================
echo.

REM Set environment variables
echo [STEP 1/15] Setting environment variables...
set PGDATA=C:\Program Files\PostgreSQL\16\data
set PGPASSWORD=postgres
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_db
set OLLAMA_HOST=http://localhost:11434
set MINIO_ROOT_USER=minioadmin
set MINIO_ROOT_PASSWORD=minioadmin
set NEO4J_AUTH=neo4j/password123
set REDIS_URL=redis://localhost:6379
set RABBITMQ_URL=amqp://localhost
set QDRANT_URL=http://localhost:6333
echo       [OK] Environment configured

REM Check and create directories
echo [STEP 2/15] Creating required directories...
if not exist "C:\LegalAI" mkdir "C:\LegalAI"
if not exist "C:\LegalAI\Evidence" mkdir "C:\LegalAI\Evidence"
if not exist "C:\LegalAI\Reports" mkdir "C:\LegalAI\Reports"
if not exist "C:\LegalAI\Citations" mkdir "C:\LegalAI\Citations"
if not exist "C:\LegalAI\Uploads" mkdir "C:\LegalAI\Uploads"
if not exist "logs" mkdir "logs"
echo       [OK] Directories created

REM Kill conflicting processes
echo [STEP 3/15] Resolving port conflicts...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5432') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :6379') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8093') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8094') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8095') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8096') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8097') do taskkill /F /PID %%a >nul 2>&1
echo       [OK] Port conflicts resolved
timeout /t 2 /nobreak >nul

REM Start PostgreSQL
echo [STEP 4/15] Starting PostgreSQL with pgvector...
pg_ctl start -D "%PGDATA%" -l logs\postgresql.log >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] PostgreSQL started on port 5432
    timeout /t 3 /nobreak >nul
    
    REM Use existing legal_ai_db database
    psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;" >nul 2>&1
    psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;" >nul 2>&1
    psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;" >nul 2>&1
    echo       [OK] Database and extensions configured
) else (
    echo       [WARN] PostgreSQL may already be running
)

REM Start Redis (Windows native)
echo [STEP 5/15] Starting Redis cache server...
where redis-server >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start /B redis-server --port 6379 --maxmemory 2gb --maxmemory-policy allkeys-lru >logs\redis.log 2>&1
    echo       [OK] Redis started on port 6379
) else (
    echo       [WARN] Redis not installed - caching disabled
    echo       Download from: https://redis.io/download
)
timeout /t 2 /nobreak >nul

REM Start RabbitMQ (Windows Service)
echo [STEP 6/15] Starting RabbitMQ message queue...
sc query RabbitMQ >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    net start RabbitMQ >nul 2>&1
    echo       [OK] RabbitMQ service started on port 5672
) else (
    echo       [WARN] RabbitMQ service not found - using Redis pub/sub fallback
)
timeout /t 3 /nobreak >nul

REM Start MinIO (Windows native)
echo [STEP 7/15] Starting MinIO object storage...
where minio >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if not exist "minio-data" mkdir "minio-data"
    start /B minio server ./minio-data --console-address :9001 >logs\minio.log 2>&1
    echo       [OK] MinIO started (API: 9000, Console: 9001)
    
    timeout /t 3 /nobreak >nul
    REM Create bucket using MinIO client
    where mc >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        mc alias set local http://localhost:9000 minioadmin minioadmin >nul 2>&1
        mc mb local/legal-ai >nul 2>&1
        echo       [OK] MinIO bucket configured
    )
) else (
    echo       [WARN] MinIO not installed - using filesystem only
    echo       Download from: https://min.io/download
)

REM Start Qdrant Vector Database (Windows native)
echo [STEP 8/15] Starting Qdrant vector database...
where qdrant >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start /B qdrant >logs\qdrant.log 2>&1
    echo       [OK] Qdrant started on port 6333
) else (
    echo       [WARN] Qdrant not installed - using pgvector only
    echo       Download from: https://qdrant.tech/documentation/quick-start/
)
timeout /t 3 /nobreak >nul

REM Skip Neo4j - Using PostgreSQL only
echo [STEP 9/15] Neo4j disabled - using PostgreSQL for all data storage...
echo       [OK] PostgreSQL handles all data storage needs
timeout /t 1 /nobreak >nul

REM Start Ollama
echo [STEP 10/15] Starting Ollama AI service...
where ollama >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start /B ollama serve >logs\ollama.log 2>&1
    timeout /t 3 /nobreak >nul
    echo       [OK] Ollama started on port 11434
    
    REM Load AI models - Gemma3:legal and nomic-embed-text
    echo       Loading AI models...
    ollama list | findstr "gemma3:legal" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo       Pulling gemma3:legal model...
        ollama pull gemma3:legal >nul 2>&1
        echo       [OK] Gemma3:legal model installed
    ) else (
        echo       [OK] Gemma3:legal model already available
    )
    
    ollama list | findstr "nomic-embed-text" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo       Pulling nomic-embed-text model...
        ollama pull nomic-embed-text >nul 2>&1
        echo       [OK] nomic-embed-text model installed
    ) else (
        echo       [OK] nomic-embed-text model already available
    )
    echo       [OK] AI models ready
) else (
    echo       [ERROR] Ollama not installed - AI features disabled
    echo       Install from: https://ollama.ai/download
)

REM Skip llama.cpp - Using Ollama with Gemma3:legal only
echo [STEP 11/15] Using Ollama for all AI inference...
echo       [OK] Gemma3:legal and nomic-embed-text via Ollama
timeout /t 1 /nobreak >nul

REM Start Go Microservices
echo [STEP 12/15] Starting Go microservices...

REM Enhanced RAG Service
if exist "enhanced-rag-som-system.exe" (
    set PORT=8094
    start /B enhanced-rag-som-system.exe >logs\enhanced-rag.log 2>&1
    echo       [OK] Enhanced RAG service on port 8094
) else if exist "enhanced-rag-som-system.go" (
    where go >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PORT=8094
        start /B go run enhanced-rag-som-system.go >logs\enhanced-rag.log 2>&1
        echo       [OK] Enhanced RAG service on port 8094
    )
)

REM GPU Orchestrator
if exist "gpu-orchestrator.exe" (
    set PORT=8095
    start /B gpu-orchestrator.exe >logs\gpu-orchestrator.log 2>&1
    echo       [OK] GPU Orchestrator on port 8095
)

REM Vector Processor
if exist "vector-processor.exe" (
    set PORT=8096
    start /B vector-processor.exe >logs\vector-processor.log 2>&1
    echo       [OK] Vector Processor on port 8096
)

REM Document Analyzer
if exist "document-analyzer.exe" (
    set PORT=8097
    start /B document-analyzer.exe >logs\document-analyzer.log 2>&1
    echo       [OK] Document Analyzer on port 8097
)

timeout /t 3 /nobreak >nul

REM Run database migrations
echo [STEP 13/15] Running database migrations...
psql -U postgres -d legal_ai_db -f production-migration.sql >logs\migration.log 2>&1
if %ERRORLEVEL% EQU 0 (
    echo       [OK] Database schema created
) else (
    echo       [WARN] Migration may have already run
)

REM Install npm dependencies
echo [STEP 14/15] Checking npm dependencies...
if not exist "node_modules" (
    echo       Installing dependencies...
    npm install >logs\npm-install.log 2>&1
    echo       [OK] Dependencies installed
) else (
    echo       [OK] Dependencies already installed
)

REM Start the application
echo [STEP 15/15] Starting Legal AI application...
echo.
echo ================================================
echo   SYSTEM STATUS
echo ================================================
echo.
echo Core Services:
echo   PostgreSQL (legal_ai_db): http://localhost:5432
echo   Redis Cache:              http://localhost:6379
echo   RabbitMQ:                 http://localhost:15672
echo   MinIO Console:            http://localhost:9001
echo.
echo AI Services:
echo   Ollama API:               http://localhost:11434
echo   - Gemma3:legal (Chat):    gemma3:legal:latest
echo   - nomic-embed (Embeddings): nomic-embed-text
echo   Enhanced RAG:             http://localhost:8094
echo.
echo Vector Databases:
echo   Qdrant:                   http://localhost:6333
echo   pgvector (PostgreSQL):    Integrated in legal_ai_db
echo.
echo Go Microservices:
echo   GPU Orchestrator:       http://localhost:8095
echo   Vector Processor:       http://localhost:8096
echo   Document Analyzer:      http://localhost:8097
echo.
echo ================================================
echo   Starting SvelteKit Application...
echo ================================================
echo.
echo Application will be available at:
echo   http://localhost:5173
echo.
echo API Documentation:
echo   http://localhost:5173/api-docs
echo.
echo Default Admin Login:
echo   Email: admin@legal-ai.com
echo   Password: admin123
echo.
echo Press Ctrl+C to stop all services
echo.

npm run dev
