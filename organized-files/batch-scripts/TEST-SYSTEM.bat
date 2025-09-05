@echo off
echo 🧪 Testing Evidence Processing System (Windows Native)
echo ======================================================

echo.
echo 🔍 Testing Prerequisites...

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js not found
    exit /b 1
) else (
    echo ✅ Node.js available
)

:: Check if worker dependencies are installed
if not exist "workers\node_modules" (
    echo ❌ Worker dependencies not installed. Run: cd workers && npm install
    exit /b 1
) else (
    echo ✅ Worker dependencies installed
)

echo.
echo 🌐 Testing Service Connectivity...

:: Test Redis
echo 🔴 Testing Redis (localhost:6379)...
netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Redis is listening on port 6379
    
    :: Try to connect to Redis
    echo ping | timeout 3 | telnet localhost 6379 >nul 2>&1
    echo ✅ Redis connectivity test passed
) else (
    echo ❌ Redis not running on port 6379
)

:: Test Qdrant
echo 🔍 Testing Qdrant (localhost:6333)...
netstat -an | findstr ":6333" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Qdrant is listening on port 6333
    
    :: Try HTTP request to Qdrant
    curl -s http://localhost:6333/collections >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Qdrant HTTP API responding
    ) else (
        echo ⚠️ Qdrant port open but API not responding (may still be starting)
    )
) else (
    echo ❌ Qdrant not running on port 6333
)

:: Test PostgreSQL
echo 🗄️ Testing PostgreSQL (localhost:5432)...
netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL is listening on port 5432
    
    :: Try to connect to database
    psql -U postgres -d evidence_processing -c "SELECT 1;" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Database connection successful
        
        :: Check if tables exist
        psql -U postgres -d evidence_processing -c "\dt" | findstr evidence_process >nul 2>&1
        if %errorlevel% == 0 (
            echo ✅ Evidence processing tables exist
        ) else (
            echo ❌ Evidence processing tables not found - run setup-database.bat
        )
    ) else (
        echo ❌ Cannot connect to evidence_processing database
    )
) else (
    echo ❌ PostgreSQL not running on port 5432
)

:: Test Neo4j
echo 🕸️ Testing Neo4j (localhost:7474)...
netstat -an | findstr ":7474" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Neo4j is listening on port 7474
    
    :: Try HTTP request to Neo4j
    curl -s http://localhost:7474/ >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Neo4j HTTP interface responding
    ) else (
        echo ⚠️ Neo4j port open but interface not responding (may still be starting)
    )
) else (
    echo ❌ Neo4j not running on port 7474
)

:: Test MinIO
echo 📦 Testing MinIO (localhost:9000)...
netstat -an | findstr ":9000" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ MinIO is listening on port 9000
    
    :: Try HTTP request to MinIO health
    curl -s http://localhost:9000/minio/health/live >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ MinIO health endpoint responding
    ) else (
        echo ⚠️ MinIO port open but health endpoint not responding
    )
) else (
    echo ❌ MinIO not running on port 9000
)

:: Test RabbitMQ
echo 🐰 Testing RabbitMQ (localhost:5672)...
netstat -an | findstr ":5672" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ RabbitMQ is listening on port 5672
    
    :: Test management interface
    netstat -an | findstr ":15672" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ RabbitMQ management interface available on port 15672
    ) else (
        echo ⚠️ RabbitMQ management interface not available
    )
) else (
    echo ❌ RabbitMQ not running on port 5672
)

:: Test Ollama (optional)
echo 🦙 Testing Ollama (localhost:11434) - Optional...
netstat -an | findstr ":11434" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama is listening on port 11434
    
    :: Try to get Ollama version
    curl -s http://localhost:11434/api/version >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Ollama API responding
    ) else (
        echo ⚠️ Ollama port open but API not responding
    )
) else (
    echo ⚠️ Ollama not running (optional service)
)

echo.
echo 🔧 Testing Worker Health Check...

cd workers
echo Running worker health check...
node health-check.js
set HEALTH_CHECK_RESULT=%errorlevel%
cd ..

if %HEALTH_CHECK_RESULT% == 0 (
    echo ✅ Worker health check passed
) else (
    echo ❌ Worker health check failed
)

echo.
echo 📋 System Test Summary
echo ======================

echo.
echo 📊 Service Status:
echo • Redis: %REDIS_STATUS%
echo • Qdrant: %QDRANT_STATUS%
echo • PostgreSQL: %POSTGRES_STATUS%
echo • Neo4j: %NEO4J_STATUS%
echo • MinIO: %MINIO_STATUS%
echo • RabbitMQ: %RABBITMQ_STATUS%
echo • Ollama: %OLLAMA_STATUS% (optional)

echo.
echo 🌐 Web Interfaces:
echo • Qdrant Dashboard: http://localhost:6333/dashboard
echo • Neo4j Browser: http://localhost:7474
echo • MinIO Console: http://localhost:9001
echo • RabbitMQ Management: http://localhost:15672

echo.
echo 🎯 Next Steps:
if %HEALTH_CHECK_RESULT% == 0 (
    echo ✅ System is ready! You can now:
    echo   1. Run start-worker.bat to start evidence processing
    echo   2. Upload evidence files to test the system
    echo   3. Monitor processing through the web interfaces
) else (
    echo ❌ Some issues detected. Please:
    echo   1. Check that all services are running: start-all-services.bat
    echo   2. Verify database setup: setup-database.bat
    echo   3. Check service logs for specific error messages
    echo   4. Ensure Windows Firewall allows the required ports
)

echo.
echo 🛠️ Troubleshooting:
echo • If services fail to start, check Windows Event Viewer
echo • Ensure no other applications are using the required ports
echo • Try running as Administrator if permission issues occur
echo • Check that all dependencies are properly installed

echo.
pause
