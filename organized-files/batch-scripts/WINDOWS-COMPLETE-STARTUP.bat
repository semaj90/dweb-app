@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================================
echo          🏛️ Legal AI System - Complete Startup
echo                Windows 10 Native Stack
echo ========================================================
echo.

:: Set all environment variables for complete system
set NODE_ENV=development
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=gemma3-legal
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set DB_HOST=localhost
set DB_PORT=5432
set DB_NAME=legal_ai_db
set DB_USER=legal_admin
set DB_PASSWORD=123456
set REDIS_URL=redis://localhost:6379
set NEO4J_URI=bolt://localhost:7687
set NEO4J_USER=neo4j
set NEO4J_PASSWORD=password
set QDRANT_URL=http://localhost:6333
set MINIO_ROOT_USER=admin
set MINIO_ROOT_PASSWORD=password123
set MINIO_ENDPOINT=localhost:9000
set ENHANCED_RAG_PORT=8095
set UPLOAD_SERVICE_PORT=8093
set FRONTEND_PORT=5176

echo 📋 Environment Variables Set:
echo   • NODE_ENV: %NODE_ENV%
echo   • DATABASE_URL: %DATABASE_URL%
echo   • OLLAMA_URL: %OLLAMA_URL%
echo   • REDIS_URL: %REDIS_URL%
echo   • NEO4J_URI: %NEO4J_URI%
echo   • QDRANT_URL: %QDRANT_URL%
echo   • MINIO_ENDPOINT: %MINIO_ENDPOINT%
echo.

echo ========================================================
echo                    🔧 PHASE 1: Dependencies
echo ========================================================

echo 📦 Installing NPM dependencies...
call npm install
if %ERRORLEVEL% neq 0 (
    echo ❌ NPM install failed
    pause
    exit /b 1
)
echo ✅ NPM dependencies installed

echo.
echo ========================================================
echo                  🚀 PHASE 2: Core Services
echo ========================================================

echo 🔍 Starting PostgreSQL...
"C:\Program Files\PostgreSQL\17\bin\pg_isready.exe" -h localhost -p 5432 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   Starting PostgreSQL service...
    net start postgresql-x64-17 >nul 2>&1
    timeout /t 5 >nul
) else (
    echo   ✅ PostgreSQL already running
)

echo 🤖 Starting Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo   Starting Ollama service...
    start /B ollama serve >nul 2>&1
    timeout /t 8 >nul
) else (
    echo   ✅ Ollama already running
)

echo 📊 Starting Redis...
tasklist /FI "IMAGENAME eq redis-server.exe" 2>NUL | find /I /N "redis-server.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo   Starting Redis server...
    if exist redis-windows\redis-server.exe (
        start /B redis-windows\redis-server.exe redis-windows\redis.conf >nul 2>&1
        timeout /t 3 >nul
    )
) else (
    echo   ✅ Redis already running
)

echo.
echo ========================================================
echo              📦 PHASE 3: Storage & Vector Services
echo ========================================================

echo 🗄️  Starting MinIO Object Storage...
tasklist /FI "IMAGENAME eq minio.exe" 2>NUL | find /I /N "minio.exe" >nul
if %ERRORLEVEL% neq 0 (
    echo   Starting MinIO server...
    start /B cmd /c "set MINIO_ROOT_USER=%MINIO_ROOT_USER% && set MINIO_ROOT_PASSWORD=%MINIO_ROOT_PASSWORD% && minio.exe server minio-data --console-address :9001" >nul 2>&1
    timeout /t 5 >nul
) else (
    echo   ✅ MinIO already running
)

echo 🧠 Starting Qdrant Vector Database...
if exist qdrant-windows\qdrant.exe (
    tasklist /FI "IMAGENAME eq qdrant.exe" 2>NUL | find /I /N "qdrant.exe" >nul
    if %ERRORLEVEL% neq 0 (
        echo   Starting Qdrant server...
        start /B qdrant-windows\qdrant.exe >nul 2>&1
        timeout /t 5 >nul
    ) else (
        echo   ✅ Qdrant already running
    )
) else (
    echo   ⚠️  Qdrant not found - will download if needed
)

echo 🕸️  Starting Neo4j Graph Database...
if exist neo4j-community-5.21.2\bin\neo4j.bat (
    curl -s http://localhost:7474/ >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo   Starting Neo4j server...
        start /B cmd /c "cd neo4j-community-5.21.2\bin && neo4j.bat console" >nul 2>&1
        timeout /t 10 >nul
    ) else (
        echo   ✅ Neo4j already running
    )
) else (
    echo   ⚠️  Neo4j not found - already installed
)

echo.
echo ========================================================
echo               🎯 PHASE 4: AI Microservices
echo ========================================================

echo 🔬 Starting Enhanced RAG Service...
netstat -ano | findstr ":%ENHANCED_RAG_PORT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   Starting Enhanced RAG on port %ENHANCED_RAG_PORT%...
    if exist go-microservice\bin\enhanced-rag.exe (
        start /B cmd /c "cd go-microservice\bin && enhanced-rag.exe" >nul 2>&1
        timeout /t 3 >nul
    )
) else (
    echo   ✅ Enhanced RAG Service already running
)

echo 📤 Starting Upload Service...
netstat -ano | findstr ":%UPLOAD_SERVICE_PORT%" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   Starting Upload Service on port %UPLOAD_SERVICE_PORT%...
    if exist go-microservice\bin\upload-service.exe (
        start /B cmd /c "set DATABASE_URL=%DATABASE_URL% && set MINIO_ENDPOINT=%MINIO_ENDPOINT% && set MINIO_ROOT_USER=%MINIO_ROOT_USER% && set MINIO_ROOT_PASSWORD=%MINIO_ROOT_PASSWORD% && cd go-microservice\bin && upload-service.exe" >nul 2>&1
        timeout /t 3 >nul
    )
) else (
    echo   ✅ Upload Service already running
)

echo.
echo ========================================================
echo                🔍 PHASE 5: Health Checks
echo ========================================================

echo 🏥 Running comprehensive health checks...

:: PostgreSQL Health Check
echo   Testing PostgreSQL...
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U %DB_USER% -d %DB_NAME% -h localhost -c "SELECT 1;" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   ✅ PostgreSQL: Connected and responding
) else (
    echo   ❌ PostgreSQL: Connection failed
)

:: Ollama Health Check
echo   Testing Ollama...
curl -s http://localhost:11434/api/version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   ✅ Ollama: Connected and responding
) else (
    echo   ❌ Ollama: Connection failed
)

:: Redis Health Check
echo   Testing Redis...
if exist redis-windows\redis-cli.exe (
    redis-windows\redis-cli.exe ping >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo   ✅ Redis: Connected and responding
    ) else (
        echo   ❌ Redis: Connection failed
    )
)

:: MinIO Health Check
echo   Testing MinIO...
curl -s http://localhost:9000/ >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   ✅ MinIO: Connected and responding
) else (
    echo   ❌ MinIO: Connection failed
)

:: Enhanced RAG Health Check
echo   Testing Enhanced RAG Service...
curl -s http://localhost:%ENHANCED_RAG_PORT%/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   ✅ Enhanced RAG: Connected and responding
) else (
    echo   ❌ Enhanced RAG: Connection failed
)

:: Upload Service Health Check
echo   Testing Upload Service...
curl -s http://localhost:%UPLOAD_SERVICE_PORT%/health >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo   ✅ Upload Service: Connected and responding
) else (
    echo   ❌ Upload Service: Connection failed
)

echo.
echo ========================================================
echo              🎨 PHASE 6: Frontend Development
echo ========================================================

echo 🌐 Starting SvelteKit Frontend...
echo.
echo 🚀 Legal AI System Status:
echo   • Frontend:           http://localhost:%FRONTEND_PORT%
echo   • AI Chat Interface:  http://localhost:%FRONTEND_PORT%/ai-test
echo   • System Health:      http://localhost:%FRONTEND_PORT%/api/system/check
echo   • Enhanced RAG:       http://localhost:%ENHANCED_RAG_PORT%/health
echo   • Upload Service:     http://localhost:%UPLOAD_SERVICE_PORT%/health
echo   • MinIO Console:      http://localhost:9001 (admin/password123)
echo   • Neo4j Browser:      http://localhost:7474 (neo4j/password)
echo.
echo 📊 Service Architecture:
echo   ✓ Multi-Agent AI Pipeline (Ollama + Enhanced RAG)
echo   ✓ Document Processing (Upload Service + MinIO)
echo   ✓ Vector Search (Qdrant + PGVector)
echo   ✓ Knowledge Graph (Neo4j)
echo   ✓ Caching Layer (Redis)
echo   ✓ Real-time Updates (WebSocket)
echo   ✓ GPU Acceleration Ready (RTX 3060 Ti)
echo.
echo ========================================================
echo              🎯 Starting Development Server
echo ========================================================

cd sveltekit-frontend
call npm run dev

echo.
echo ========================================================
echo                    ⚠️  IMPORTANT NOTES
echo ========================================================
echo.
echo • All services are running in background
echo • Use Ctrl+C to stop the frontend (services will continue)
echo • To stop all services, run: STOP-ALL-SERVICES.bat
echo • For troubleshooting, check service logs in generated .txt files
echo • GPU acceleration requires CUDA drivers for RTX 3060 Ti
echo.
echo Press any key to stop the frontend server...
pause >nul

echo.
echo 🛑 Frontend stopped. Background services still running.
echo    Run STOP-ALL-SERVICES.bat to stop all services.
echo.
pause