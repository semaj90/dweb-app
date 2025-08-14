@echo off
REM ============================================
REM Native Windows Neo4j AI System Launcher
REM No Docker Required - Direct Service Startup
REM ============================================

echo ========================================
echo NATIVE WINDOWS AI SYSTEM LAUNCHER
echo Phase 14 + Filesystem Indexer
echo ========================================
echo.

REM Check Admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Administrator privileges required
    echo Right-click and "Run as Administrator"
    pause
    exit /b 1
)

REM Set environment variables
set NODE_ENV=development
set PORT=8081
set REDIS_HOST=localhost:6379
set NEO4J_URI=neo4j://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password
set OLLAMA_HOST=http://localhost:11434
set POSTGRES_HOST=localhost
set POSTGRES_PORT=5432
set POSTGRES_DB=legal_ai_db
set POSTGRES_USER=postgres
set POSTGRES_PASSWORD=postgres

echo [1/8] Checking Prerequisites...
echo ========================================

REM Check Neo4j Desktop
where neo4j >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Neo4j not in PATH, checking default location...
    if exist "C:\Program Files\Neo4j Desktop\resources\offline\neo4j\bin\neo4j.bat" (
        set NEO4J_PATH=C:\Program Files\Neo4j Desktop\resources\offline\neo4j\bin
        echo [OK] Neo4j Desktop found
    ) else (
        echo [ERROR] Neo4j Desktop not found. Please install from: https://neo4j.com/download/
        pause
        exit /b 1
    )
) else (
    echo [OK] Neo4j found in PATH
)

REM Check PostgreSQL
pg_ctl --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] PostgreSQL not installed or not in PATH
    echo Please install from: https://www.postgresql.org/download/windows/
    pause
    exit /b 1
) else (
    echo [OK] PostgreSQL installed
)

REM Check Redis (Windows port)
where redis-server >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Redis not in PATH, checking common locations...
    if exist "C:\Redis\redis-server.exe" (
        set REDIS_PATH=C:\Redis
        echo [OK] Redis found at C:\Redis
    ) else if exist "%USERPROFILE%\Redis\redis-server.exe" (
        set REDIS_PATH=%USERPROFILE%\Redis
        echo [OK] Redis found
    ) else (
        echo [WARNING] Redis not found. Installing Memurai (Windows Redis)...
        powershell -Command "Invoke-WebRequest -Uri 'https://www.memurai.com/get-memurai' -OutFile memurai-installer.msi"
        msiexec /i memurai-installer.msi /quiet
        del memurai-installer.msi
    )
) else (
    echo [OK] Redis found
)

REM Check Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Ollama not installed
    echo Please install from: https://ollama.com/download/windows
    pause
    exit /b 1
) else (
    echo [OK] Ollama installed
)

REM Check CUDA
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected
    for /f "tokens=2 delims=:" %%a in ('nvidia-smi --query-gpu=name --format=csv^,noheader') do echo     GPU: %%a
    set USE_GPU=true
) else (
    echo [WARNING] No NVIDIA GPU - CPU mode enabled
    set USE_GPU=false
)

echo.
echo [2/8] Starting PostgreSQL...
echo ========================================

REM Check if PostgreSQL is running
pg_isready -h localhost -p 5432 >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting PostgreSQL service...
    net start postgresql-x64-16 2>nul || net start postgresql-x64-15 2>nul || net start postgresql-x64-14 2>nul
    timeout /t 3 /nobreak >nul
    
    REM Verify PostgreSQL started
    pg_isready -h localhost -p 5432 >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to start PostgreSQL
        pause
        exit /b 1
    )
)
echo [OK] PostgreSQL is running

REM Create database if needed
psql -U postgres -h localhost -c "SELECT 1 FROM pg_database WHERE datname = 'legal_ai_db'" | findstr "1" >nul 2>&1
if %errorlevel% neq 0 (
    echo Creating database...
    psql -U postgres -h localhost -c "CREATE DATABASE legal_ai_db;"
    psql -U postgres -h localhost -c "CREATE USER legal_admin WITH PASSWORD 'LegalAI2024!';"
    psql -U postgres -h localhost -c "GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;"
)

REM Install pgvector extension
psql -U postgres -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo [OK] pgvector extension ready

echo.
echo [3/8] Starting Neo4j...
echo ========================================

REM Check if Neo4j is running
curl -s http://localhost:7474 >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Neo4j Desktop database...
    if defined NEO4J_PATH (
        start /B "%NEO4J_PATH%\neo4j.bat" console
    ) else (
        start /B neo4j console
    )
    
    echo Waiting for Neo4j to start...
    :WAIT_NEO4J
    timeout /t 2 /nobreak >nul
    curl -s http://localhost:7474 >nul 2>&1
    if %errorlevel% neq 0 goto WAIT_NEO4J
)
echo [OK] Neo4j is running at http://localhost:7474

echo.
echo [4/8] Starting Redis/Memurai...
echo ========================================

REM Check if Redis is running
redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Redis server...
    if defined REDIS_PATH (
        start /B "%REDIS_PATH%\redis-server.exe"
    ) else (
        start /B redis-server
    )
    timeout /t 2 /nobreak >nul
)
redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Redis is running
) else (
    echo [WARNING] Redis not responding
)

echo.
echo [5/8] Starting Ollama...
echo ========================================

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama service...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
)

REM Pull required models
echo Checking AI models...
ollama list | findstr "nomic-embed-text" >nul 2>&1
if %errorlevel% neq 0 (
    echo Pulling nomic-embed-text model...
    ollama pull nomic-embed-text
)

ollama list | findstr "llama3" >nul 2>&1
if %errorlevel% neq 0 (
    echo Pulling llama3 model...
    ollama pull llama3
)

echo [OK] Ollama ready with models

echo.
echo [6/8] Building Go Microservice...
echo ========================================

cd go-microservice

if not exist "go.mod" (
    echo Initializing Go module...
    go mod init ai-microservice
)

echo Installing dependencies...
go get github.com/gin-gonic/gin
go get github.com/gin-contrib/cors
go get github.com/neo4j/neo4j-go-driver/v5
go get github.com/go-redis/redis/v8
go get github.com/jackc/pgx/v5
go get github.com/pgvector/pgvector-go
go get github.com/minio/simdjson-go
go get github.com/bytedance/sonic
go get github.com/valyala/fastjson
go get github.com/NVIDIA/go-nvml/pkg/nvml
go get github.com/gorilla/websocket

echo Building microservice...
if "%USE_GPU%"=="true" (
    echo Building with GPU support...
    set CGO_ENABLED=1
    set CGO_CFLAGS=-I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include"
    set CGO_LDFLAGS=-L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" -lcudart -lcublas
    go build -o ai-microservice.exe .
) else (
    echo Building CPU-only version...
    go build -tags nogpu -o ai-microservice.exe .
)

if not exist "ai-microservice.exe" (
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo [OK] Microservice built successfully

echo.
echo [7/8] Starting AI Microservice...
echo ========================================

start "AI Microservice" /B ai-microservice.exe

timeout /t 3 /nobreak >nul
curl -f http://localhost:8081/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Microservice failed to start
    pause
    exit /b 1
)
echo [OK] AI Microservice running on port 8081

cd ..

echo.
echo [8/8] Initializing Databases...
echo ========================================

REM Initialize PostgreSQL schema
echo Creating PostgreSQL schema...
psql -U postgres -d legal_ai_db -f sql\init-pgvector.sql 2>nul

REM Initialize Neo4j schema
echo Creating Neo4j indexes...
echo CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE; | cypher-shell -u neo4j -p password >nul 2>&1
echo CREATE INDEX IF NOT EXISTS FOR (f:File) ON (f.type); | cypher-shell -u neo4j -p password >nul 2>&1
echo CREATE INDEX IF NOT EXISTS FOR (t:Type) ON (t.name); | cypher-shell -u neo4j -p password >nul 2>&1
echo CREATE FULLTEXT INDEX fileSearch IF NOT EXISTS FOR (f:File) ON EACH [f.path, f.summary]; | cypher-shell -u neo4j -p password >nul 2>&1

echo [OK] Database schemas initialized

echo.
echo ========================================
echo SYSTEM READY!
echo ========================================
echo.
echo Services Running:
echo   Neo4j Browser:    http://localhost:7474
echo   AI Microservice:  http://localhost:8081
echo   Ollama:           http://localhost:11434
echo   PostgreSQL:       localhost:5432
echo   Redis:            localhost:6379
echo.
echo Available Endpoints:
echo   /health              - System health check
echo   /index               - Index filesystem
echo   /analyze-errors      - Analyze TypeScript errors
echo   /parse/simd          - SIMD JSON parsing
echo   /gpu/compute         - GPU computations
echo   /stream/ws           - WebSocket streaming
echo.
echo To index your project:
echo   curl -X POST http://localhost:8081/index -H "Content-Type: application/json" -d "{\"rootPath\": \"./sveltekit-frontend\"}"
echo.
echo Press Ctrl+C to stop all services
echo ========================================

REM Keep running and monitor
:MONITOR_LOOP
timeout /t 30 /nobreak >nul
curl -s http://localhost:8081/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Microservice not responding, restarting...
    cd go-microservice
    start "AI Microservice" /B ai-microservice.exe
    cd ..
)
goto MONITOR_LOOP
