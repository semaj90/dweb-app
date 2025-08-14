@echo off
REM ============================================
REM Start Neo4j Desktop and AI Microservice
REM Phase 14: Advanced Features & Optimization
REM ============================================

echo ========================================
echo Starting Deeds Web App - Phase 14
echo Advanced AI Microservice with Neo4j
echo ========================================
echo.

REM Set environment variables
set NODE_ENV=development
set PORT=8081
set REDIS_HOST=localhost:6379
set NEO4J_URI=neo4j://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password
set OLLAMA_HOST=http://localhost:11434

REM Check if running as Administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires Administrator privileges.
    echo Please run as Administrator.
    pause
    exit /b 1
)

REM Function to check if a service is running
:CHECK_SERVICE
echo Checking services...
echo.

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH
    echo Please install Docker Desktop for Windows
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to start...
    timeout /t 30 /nobreak >nul
)

REM Check CUDA availability
echo Checking CUDA/GPU availability...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    set USE_GPU=true
) else (
    echo [WARNING] No NVIDIA GPU detected, using CPU mode
    set USE_GPU=false
)

echo.
echo ========================================
echo Step 1: Starting Infrastructure Services
echo ========================================
echo.

REM Start infrastructure with Docker Compose
echo Starting Neo4j, Redis, and other services...
docker-compose -f docker-compose.neo4j.yml up -d neo4j redis postgres qdrant

REM Wait for services to be healthy
echo Waiting for services to be ready...
:WAIT_NEO4J
timeout /t 5 /nobreak >nul
docker exec neo4j-deeds cypher-shell -u neo4j -p password "MATCH () RETURN count(*) LIMIT 1" >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for Neo4j...
    goto WAIT_NEO4J
)
echo [OK] Neo4j is ready

:WAIT_REDIS
docker exec redis-deeds redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for Redis...
    timeout /t 2 /nobreak >nul
    goto WAIT_REDIS
)
echo [OK] Redis is ready

echo.
echo ========================================
echo Step 2: Building Go Microservice
echo ========================================
echo.

cd go-microservice

REM Check if binary exists
if not exist "build\ai-microservice.exe" (
    echo Building AI Microservice...
    if "%USE_GPU%"=="true" (
        echo Building with GPU support...
        make build
    ) else (
        echo Building CPU-only version...
        make build-cpu
    )
) else (
    echo [OK] AI Microservice binary found
)

echo.
echo ========================================
echo Step 3: Starting Ollama Service
echo ========================================
echo.

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    
    REM Pull required models
    echo Pulling AI models...
    ollama pull nomic-embed-text
    ollama pull gemma3-legal 2>nul || ollama pull llama3
) else (
    echo [OK] Ollama is already running
)

echo.
echo ========================================
echo Step 4: Starting AI Microservice
echo ========================================
echo.

REM Start the Go microservice
echo Starting AI Microservice on port 8081...
start "AI Microservice" /B build\ai-microservice.exe

REM Wait for microservice to be ready
timeout /t 3 /nobreak >nul
:WAIT_MICROSERVICE
curl -f http://localhost:8081/health >nul 2>&1
if %errorlevel% neq 0 (
    echo Waiting for AI Microservice...
    timeout /t 2 /nobreak >nul
    goto WAIT_MICROSERVICE
)
echo [OK] AI Microservice is ready

echo.
echo ========================================
echo Step 5: Starting Frontend (Vite)
echo ========================================
echo.

cd ..

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

REM Start Vite dev server
echo Starting Vite development server...
start "Vite Dev Server" /B npm run dev

echo.
echo ========================================
echo Step 6: Initialize Neo4j Schema
echo ========================================
echo.

REM Create initial Neo4j schema
echo Creating Neo4j indexes and constraints...
docker exec -i neo4j-deeds cypher-shell -u neo4j -p password <<EOF
// Create constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;

// Create indexes
CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.created_at);
CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.embedding);
CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type, e.name);
CREATE FULLTEXT INDEX documentSearch IF NOT EXISTS FOR (d:Document) ON EACH [d.title, d.content];
CREATE FULLTEXT INDEX chunkSearch IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];

// Create initial data
MERGE (system:System {id: 'deeds-ai-system', version: '2.0.0', initialized: datetime()});
EOF

echo [OK] Neo4j schema initialized

echo.
echo ========================================
echo Step 7: System Health Check
echo ========================================
echo.

REM Perform comprehensive health check
echo Checking all services...
curl -s http://localhost:8081/health | findstr /C:"healthy" >nul
if %errorlevel% equ 0 (
    echo [OK] AI Microservice: Healthy
) else (
    echo [WARNING] AI Microservice: Not responding correctly
)

docker exec neo4j-deeds cypher-shell -u neo4j -p password "RETURN 1" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Neo4j: Connected
) else (
    echo [WARNING] Neo4j: Connection failed
)

docker exec redis-deeds redis-cli ping >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Redis: Connected
) else (
    echo [WARNING] Redis: Connection failed
)

curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ollama: Running
) else (
    echo [WARNING] Ollama: Not responding
)

echo.
echo ========================================
echo System Started Successfully!
echo ========================================
echo.
echo Services Running:
echo - Neo4j Browser: http://localhost:7474
echo - AI Microservice: http://localhost:8081
echo - Frontend (Vite): http://localhost:5173
echo - Redis: localhost:6379
echo - Ollama: http://localhost:11434
echo.
echo API Endpoints:
echo - Health: http://localhost:8081/health
echo - Metrics: http://localhost:8081/metrics
echo - WebSocket: ws://localhost:8081/stream/ws
echo - SSE: http://localhost:8081/stream/sse
echo.
echo GPU Status:
if "%USE_GPU%"=="true" (
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader
) else (
    echo CPU Mode - No GPU acceleration
)
echo.
echo Press Ctrl+C to stop all services
echo ========================================
echo.

REM Keep the script running
:MONITOR
timeout /t 30 /nobreak >nul
echo [%date% %time%] System health check...
curl -s http://localhost:8081/health >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] AI Microservice not responding, attempting restart...
    taskkill /F /FI "WINDOWTITLE eq AI Microservice" >nul 2>&1
    start "AI Microservice" /B go-microservice\build\ai-microservice.exe
)
goto MONITOR

:CLEANUP
echo.
echo Shutting down services...
taskkill /F /FI "WINDOWTITLE eq AI Microservice" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Vite Dev Server" >nul 2>&1
docker-compose -f docker-compose.neo4j.yml down
echo Services stopped.
exit /b 0
