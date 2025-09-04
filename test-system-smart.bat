@echo off
echo 🧪 Testing Evidence Processing System (Smart Detection)
echo =======================================================

echo.
echo 🔍 Running Smart Service Detection...

cd workers
node services/smart-service-detector.js
set DETECTION_RESULT=%errorlevel%
cd ..

if %DETECTION_RESULT% neq 0 (
    echo ❌ Service detection failed
    pause
    exit /b 1
)

echo.
echo 🌐 Testing Service Connectivity with Smart Detection...

:: Set PostgreSQL password for tests
set PGPASSWORD=123456

:: Test PostgreSQL with correct password
echo 🗄️ Testing PostgreSQL (localhost:5432) with password 123456...
netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL is listening on port 5432
    
    :: Test database connection
    psql -U postgres -d evidence_processing -c "SELECT 1;" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Database connection successful
        
        :: Check if tables exist
        psql -U postgres -d evidence_processing -c "\dt" | findstr evidence_process >nul 2>&1
        if %errorlevel% == 0 (
            echo ✅ Evidence processing tables exist
        ) else (
            echo ❌ Evidence processing tables not found - run setup-database-smart.bat
        )
    ) else (
        echo ❌ Cannot connect to evidence_processing database with password 123456
    )
) else (
    echo ❌ PostgreSQL not running on port 5432
)

:: Test Redis (smart detection)
echo 🔴 Testing Redis (localhost:6379)...
netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Redis is listening on port 6379
    
    :: Test Redis connection
    redis-cli ping >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Redis responding to ping
    ) else (
        echo ⚠️ Redis port open but not responding to ping
    )
) else (
    echo ❌ Redis not running on port 6379
    echo 💡 Try starting Redis: sc start Redis (if system service)
    echo 💡 Or run: start-all-services-smart.bat
)

:: Test RabbitMQ (smart detection)
echo 🐰 Testing RabbitMQ (localhost:5672)...
netstat -an | findstr ":5672" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ RabbitMQ is listening on port 5672
    
    :: Test management interface
    netstat -an | findstr ":15672" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ RabbitMQ management interface available on port 15672
    ) else (
        echo ⚠️ RabbitMQ running but management interface not available
    )
) else (
    echo ❌ RabbitMQ not running on port 5672
    echo 💡 Try starting RabbitMQ: sc start RabbitMQ (if system service)
)

:: Test Qdrant (portable)
echo 🔍 Testing Qdrant (localhost:6333)...
netstat -an | findstr ":6333" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Qdrant is listening on port 6333
    
    :: Try HTTP request to Qdrant
    curl -s http://localhost:6333/collections >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Qdrant API responding
    ) else (
        echo ⚠️ Qdrant port open but API not responding (may be starting)
    )
) else (
    echo ❌ Qdrant not running on port 6333
    echo 💡 Check if portable Qdrant is installed in services/ directory
)

:: Test Neo4j (portable)
echo 🕸️ Testing Neo4j (localhost:7474)...
netstat -an | findstr ":7474" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Neo4j is listening on port 7474
    
    :: Try HTTP request to Neo4j
    curl -s http://localhost:7474/ >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Neo4j browser interface responding
    ) else (
        echo ⚠️ Neo4j port open but interface not responding (may be starting)
    )
) else (
    echo ❌ Neo4j not running on port 7474
    echo 💡 Check if portable Neo4j is installed in services/ directory
)

:: Test MinIO (portable)
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
    
    :: Check MinIO console
    netstat -an | findstr ":9001" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ MinIO console available on port 9001
    ) else (
        echo ⚠️ MinIO console not available
    )
) else (
    echo ❌ MinIO not running on port 9000
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
        
        :: Check for models
        curl -s http://localhost:11434/api/tags 2>nul | findstr "nomic-embed" >nul 2>&1
        if %errorlevel% == 0 (
            echo ✅ Embedding model available
        ) else (
            echo ⚠️ No embedding models found - run: ollama pull nomic-embed-text
        )
    ) else (
        echo ⚠️ Ollama port open but API not responding
    )
) else (
    echo ⚠️ Ollama not running (optional service)
)

echo.
echo 🔧 Testing Worker Dependencies...

:: Check if worker dependencies are installed
if not exist "workers\node_modules" (
    echo ❌ Worker dependencies not installed. Installing...
    cd workers
    call npm install
    if %errorlevel% neq 0 (
        echo ❌ Failed to install worker dependencies
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo ✅ Worker dependencies installed
) else (
    echo ✅ Worker dependencies found
)

:: Run worker health check
echo.
echo 🏥 Running Worker Health Check...
cd workers
node health-check.js
set HEALTH_CHECK_RESULT=%errorlevel%
cd ..

echo.
echo 📋 Smart System Test Summary
echo ===========================

if %HEALTH_CHECK_RESULT% == 0 (
    echo.
    echo ✅ SYSTEM READY FOR EVIDENCE PROCESSING!
    echo.
    echo 🎯 Next Steps:
    echo   1. Run: start-worker.bat (to start evidence processing)
    echo   2. Upload evidence files to test the system
    echo   3. Monitor processing through the web interfaces
    echo.
    echo 🌐 Web Interfaces Available:
    netstat -an | findstr ":15672" >nul 2>&1 && echo   • RabbitMQ Management: http://localhost:15672 (guest/guest)
    netstat -an | findstr ":7474" >nul 2>&1 && echo   • Neo4j Browser: http://localhost:7474 (neo4j/neo4j)
    netstat -an | findstr ":9001" >nul 2>&1 && echo   • MinIO Console: http://localhost:9001 (evidence/evidence123)
    netstat -an | findstr ":6333" >nul 2>&1 && echo   • Qdrant Dashboard: http://localhost:6333/dashboard
    
) else (
    echo.
    echo ❌ SYSTEM ISSUES DETECTED
    echo.
    echo 🔧 Troubleshooting Steps:
    echo   1. Check service status: start-all-services-smart.bat
    echo   2. Verify database: setup-database-smart.bat
    echo   3. Check specific error messages above
    echo   4. Ensure Windows Firewall allows required ports
    echo.
    echo 📊 Common Issues:
    echo   • PostgreSQL: Check password is 123456
    echo   • Redis: Start service or check portable installation
    echo   • RabbitMQ: Enable management plugin if installed
    echo   • Portable services: Check services/ directory
)

echo.
echo 💡 Smart Detection Features Used:
echo   • Automatically detected existing PostgreSQL (password: 123456)
echo   • Checked for system Redis and RabbitMQ installations
echo   • Verified portable service installations
echo   • Provided specific startup recommendations
echo.
echo 🛠️ Manual Override Options:
echo   • Start all services: start-all-services-smart.bat
echo   • Stop all services: stop-all-services.bat  
echo   • Database setup: setup-database-smart.bat
echo   • Worker startup: start-worker.bat

pause
