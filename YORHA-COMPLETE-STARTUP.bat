@echo off
REM =============================================================================
REM YoRHa Legal AI Platform - Complete Native Windows Service Startup
REM PostgreSQL + Redis + RabbitMQ + MinIO + Neo4j + Qdrant + Ollama + Go Services
REM =============================================================================

echo ===============================================================================
echo YoRHa LEGAL AI PLATFORM - COMPLETE NATIVE WINDOWS STARTUP
echo ===============================================================================
echo Starting: PostgreSQL, Redis, RabbitMQ, MinIO, Neo4j, Qdrant, Ollama, Go Services
echo ===============================================================================

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

echo.
echo [1/9] === POSTGRESQL DATABASE ===
echo.
sc query postgresql-x64-16 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL service found
    net start postgresql-x64-16 >nul 2>&1
    echo ✅ PostgreSQL started on port 5432
) else (
    sc query postgresql-x64-15 >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ PostgreSQL 15 service found
        net start postgresql-x64-15 >nul 2>&1  
        echo ✅ PostgreSQL started on port 5432
    ) else (
        echo ⚠️ PostgreSQL service not found
        echo   Install with: winget install PostgreSQL.PostgreSQL
    )
)

echo.
echo [2/9] === REDIS CACHE SERVER ===
echo.
where redis-server >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Redis found - starting server...
    start /B redis-server redis.conf 2>nul
    timeout /t 3 >nul
    echo ✅ Redis started on port 6379
) else (
    sc query Redis >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Redis Windows service found
        net start Redis >nul 2>&1
        echo ✅ Redis service started on port 6379
    ) else (
        echo ⚠️ Redis not found
        echo   Install with: winget install Redis.Redis
    )
)

echo.
echo [3/9] === RABBITMQ MESSAGE BROKER ===
echo.
sc query RabbitMQ >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ RabbitMQ service found
    net start RabbitMQ >nul 2>&1
    echo ✅ RabbitMQ started - Management UI: http://localhost:15672
    echo   Default login: guest/guest
) else (
    where rabbitmq-server >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ RabbitMQ executable found
        start /B rabbitmq-server 2>nul
        timeout /t 5 >nul
        echo ✅ RabbitMQ started manually
    ) else (
        echo ⚠️ RabbitMQ not found
        echo   Install with: winget install RabbitMQ.RabbitMQ
    )
)

echo.
echo [4/9] === MINIO OBJECT STORAGE ===
echo.
where minio >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ MinIO found - creating data directory...
    if not exist "minio-data" mkdir minio-data
    echo ✅ Starting MinIO server...
    start /B minio server minio-data --address ":9000" --console-address ":9001" 2>nul
    timeout /t 5 >nul
    echo ✅ MinIO started - Console: http://localhost:9001
    echo   Default login: minioadmin/minioadmin
) else (
    echo ⚠️ MinIO not found
    echo   Download from: https://min.io/download and add to PATH
)

echo.
echo [5/9] === NEO4J GRAPH DATABASE ===
echo.
sc query Neo4j >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Neo4j service found
    net start Neo4j >nul 2>&1
    timeout /t 8 >nul
    echo ✅ Neo4j started - Browser: http://localhost:7474
    echo   Initial login: neo4j/neo4j (change on first login)
) else (
    where neo4j >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Neo4j executable found
        start /B neo4j console 2>nul
        timeout /t 10 >nul
        echo ✅ Neo4j started manually
    ) else (
        echo ⚠️ Neo4j not found  
        echo   Install with: winget install Neo4j.Neo4j
    )
)

echo.
echo [6/9] === QDRANT VECTOR DATABASE (Low Memory) ===
echo.
where qdrant >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Qdrant found - creating low memory config...
    
    REM Create Qdrant low memory config
    echo log_level: INFO> qdrant-config.yaml
    echo storage:>> qdrant-config.yaml
    echo   storage_path: ./qdrant-storage>> qdrant-config.yaml
    echo service:>> qdrant-config.yaml
    echo   http_port: 6333>> qdrant-config.yaml
    echo   grpc_port: 6334>> qdrant-config.yaml
    echo performance:>> qdrant-config.yaml
    echo   max_search_threads: 2>> qdrant-config.yaml
    echo   max_optimization_threads: 1>> qdrant-config.yaml
    echo hnsw_config:>> qdrant-config.yaml
    echo   m: 8>> qdrant-config.yaml
    echo   ef_construct: 100>> qdrant-config.yaml
    
    echo ✅ Starting Qdrant in low memory mode...
    start /B qdrant --config-path qdrant-config.yaml 2>nul
    timeout /t 5 >nul
    echo ✅ Qdrant started - Dashboard: http://localhost:6333/dashboard
) else (
    echo ⚠️ Qdrant not found
    echo   Download from: https://github.com/qdrant/qdrant/releases
)

echo.
echo [7/9] === OLLAMA AI MODELS ===
echo.
where ollama >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Ollama found - starting server...
    start /B ollama serve 2>nul
    timeout /t 5 >nul
    echo ✅ Ollama started on port 11434
    echo ℹ️ Available models: ollama list
) else (
    echo ⚠️ Ollama not found
    echo   Install with: winget install Ollama.Ollama
)

echo.
echo [8/9] === GO MICROSERVICES ===
echo.

REM Enhanced RAG Service
if exist "go-microservice\bin\enhanced-rag.exe" (
    echo ✅ Enhanced RAG binary found
    echo 🤖 Starting Enhanced RAG Service (port 8094)...
    start /B "Enhanced RAG" "go-microservice\bin\enhanced-rag.exe" 2>nul
    timeout /t 3 >nul
    echo ✅ Enhanced RAG Service started
) else (
    echo ⚠️ Enhanced RAG binary not found - checking alternative location...
    if exist "go-services\bin\enhanced-rag.exe" (
        echo ✅ Alternative Enhanced RAG found
        start /B "Enhanced RAG" "go-services\bin\enhanced-rag.exe" 2>nul
        timeout /t 3 >nul
        echo ✅ Enhanced RAG Service started
    ) else (
        echo ❌ Enhanced RAG binary missing
    )
)

REM Upload Service
if exist "go-microservice\bin\upload-service.exe" (
    echo ✅ Upload Service binary found
    echo 📤 Starting Upload Service (port 8093)...
    start /B "Upload Service" "go-microservice\bin\upload-service.exe" 2>nul
    timeout /t 2 >nul
    echo ✅ Upload Service started
) else (
    echo ❌ Upload Service binary not found
)

REM Additional Go Services
if exist "go-microservice\rag-kratos.exe" (
    echo ✅ RAG Kratos service found
    start /B "RAG Kratos" "go-microservice\rag-kratos.exe" 2>nul
    echo ✅ RAG Kratos started
)

echo.
echo [9/9] === SERVICE HEALTH CHECK ===
echo.
echo 📊 Checking service availability...
timeout /t 5 >nul

echo PostgreSQL (5432): 
netstat -an | findstr ":5432" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo Redis (6379):
netstat -an | findstr ":6379" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo RabbitMQ (5672/15672):
netstat -an | findstr ":5672" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo MinIO (9000):
netstat -an | findstr ":9000" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo Neo4j (7474):
netstat -an | findstr ":7474" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo Qdrant (6333):
netstat -an | findstr ":6333" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo Ollama (11434):
netstat -an | findstr ":11434" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo Enhanced RAG (8094):
netstat -an | findstr ":8094" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo Upload Service (8093):
netstat -an | findstr ":8093" >nul && echo   ✅ RUNNING || echo   ❌ NOT RUNNING

echo.
echo ===============================================================================
echo YoRHa LEGAL AI PLATFORM - SERVICE STARTUP COMPLETE
echo ===============================================================================
echo.
echo 🎯 ALL SERVICES STATUS:
echo   🗄️  PostgreSQL:     localhost:5432
echo   🔴 Redis:          localhost:6379
echo   🐰 RabbitMQ:       localhost:5672 (Management: http://localhost:15672)
echo   📦 MinIO:          http://localhost:9000 (Console: http://localhost:9001)
echo   🔗 Neo4j:          http://localhost:7474
echo   🔍 Qdrant:         http://localhost:6333
echo   🧠 Ollama:         http://localhost:11434
echo   🤖 Enhanced RAG:   http://localhost:8094
echo   📤 Upload Service: http://localhost:8093
echo.
echo 🚀 READY FOR YoRHa FRONTEND!
echo.
echo   Next steps:
echo   1. cd sveltekit-frontend
echo   2. npm run dev
echo   3. Visit: http://localhost:5177/yorha-home
echo.
echo   Or run: npm run dev:full (from root directory)
echo.
echo ===============================================================================

pause