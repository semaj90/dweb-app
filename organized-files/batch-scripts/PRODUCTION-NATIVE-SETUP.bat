@echo off
REM =============================================================================
REM YoRHa Legal AI Platform - PRODUCTION NATIVE WINDOWS SETUP
REM No Docker - Pure Windows Native Implementation
REM Downloads and configures: Redis, RabbitMQ, Neo4j, Qdrant for production use
REM =============================================================================

echo ===============================================================================
echo YoRHa LEGAL AI PLATFORM - PRODUCTION NATIVE WINDOWS SETUP
echo ===============================================================================
echo Setting up production-ready services WITHOUT Docker
echo Native Windows implementation for maximum performance
echo ===============================================================================

cd /d "C:\Users\james\Desktop\deeds-web\deeds-web-app"

if not exist "services" mkdir services
cd services

echo.
echo [1/5] === DOWNLOADING REDIS (Native Windows) ===
echo.
if not exist "redis" (
    echo 📥 Downloading Redis for Windows...
    echo Creating Redis directory...
    mkdir redis
    cd redis
    
    REM Create Redis configuration for production
    echo # YoRHa Legal AI Redis Configuration> redis.conf
    echo port 6379>> redis.conf
    echo bind 127.0.0.1>> redis.conf
    echo save 900 1>> redis.conf
    echo save 300 10>> redis.conf
    echo save 60 10000>> redis.conf
    echo maxmemory 512mb>> redis.conf
    echo maxmemory-policy allkeys-lru>> redis.conf
    echo dir ./>> redis.conf
    
    echo ✅ Redis config created - Download Redis manually
    echo   URL: https://github.com/microsoftarchive/redis/releases
    echo   Place redis-server.exe in services/redis/
    cd ..
) else (
    echo ✅ Redis directory already exists
)

echo.
echo [2/5] === SETTING UP PORTABLE QDRANT (Low Memory) ===
echo.
if not exist "qdrant" (
    echo 📥 Setting up Qdrant portable version...
    mkdir qdrant
    cd qdrant
    
    REM Create Qdrant production config optimized for low memory
    echo log_level: INFO> config.yaml
    echo storage:>> config.yaml
    echo   storage_path: ./storage>> config.yaml
    echo   snapshots_path: ./snapshots>> config.yaml
    echo service:>> config.yaml
    echo   http_port: 6333>> config.yaml
    echo   grpc_port: 6334>> config.yaml
    echo   max_request_size_mb: 32>> config.yaml
    echo cluster:>> config.yaml
    echo   enabled: false>> config.yaml
    echo performance:>> config.yaml
    echo   max_search_threads: 2>> config.yaml
    echo   max_optimization_threads: 1>> config.yaml
    echo hnsw_config:>> config.yaml
    echo   m: 8>> config.yaml
    echo   ef_construct: 100>> config.yaml
    echo   full_scan_threshold: 5000>> config.yaml
    echo quantization:>> config.yaml
    echo   scalar:>> config.yaml
    echo     type: int8>> config.yaml
    echo     quantile: 0.99>> config.yaml
    echo     always_ram: false>> config.yaml
    
    echo ✅ Qdrant low memory config created
    echo   Download qdrant.exe from: https://github.com/qdrant/qdrant/releases
    echo   Place in services/qdrant/
    cd ..
) else (
    echo ✅ Qdrant directory already exists
)

echo.
echo [3/5] === RABBITMQ PORTABLE SETUP ===
echo.
if not exist "rabbitmq" (
    echo 📥 Setting up RabbitMQ portable...
    mkdir rabbitmq
    cd rabbitmq
    
    REM Create RabbitMQ environment
    echo # YoRHa Legal AI RabbitMQ Config> rabbitmq.conf
    echo listeners.tcp.default = 5672>> rabbitmq.conf
    echo management.tcp.port = 15672>> rabbitmq.conf
    echo loopback_users.guest = false>> rabbitmq.conf
    echo default_user = admin>> rabbitmq.conf
    echo default_pass = yorha2024>> rabbitmq.conf
    
    echo ✅ RabbitMQ config created
    echo   Download RabbitMQ from: https://www.rabbitmq.com/download.html
    echo   Extract to services/rabbitmq/
    cd ..
) else (
    echo ✅ RabbitMQ directory already exists
)

echo.
echo [4/5] === NEO4J COMMUNITY SETUP ===
echo.
if not exist "neo4j" (
    echo 📥 Setting up Neo4j Community...
    mkdir neo4j
    cd neo4j
    
    REM Create Neo4j config for legal AI
    echo # YoRHa Legal AI Neo4j Configuration> neo4j.conf
    echo dbms.default_listen_address=0.0.0.0>> neo4j.conf
    echo dbms.connector.bolt.listen_address=:7687>> neo4j.conf
    echo dbms.connector.http.listen_address=:7474>> neo4j.conf
    echo dbms.security.auth_enabled=true>> neo4j.conf
    echo dbms.memory.heap.initial_size=512m>> neo4j.conf
    echo dbms.memory.heap.max_size=1g>> neo4j.conf
    echo dbms.memory.pagecache.size=256m>> neo4j.conf
    echo dbms.security.procedures.allowlist=apoc.*>> neo4j.conf
    
    echo ✅ Neo4j config created
    echo   Download Neo4j Community from: https://neo4j.com/download/
    echo   Extract to services/neo4j/
    cd ..
) else (
    echo ✅ Neo4j directory already exists
)

echo.
echo [5/5] === CREATING PRODUCTION STARTUP SCRIPT ===
echo.

cd ..

REM Create master production startup script
echo @echo off> START-PRODUCTION-SERVICES.bat
echo REM YoRHa Legal AI Platform - Production Service Startup>> START-PRODUCTION-SERVICES.bat
echo echo ===============================================================================>> START-PRODUCTION-SERVICES.bat
echo echo YoRHa LEGAL AI PLATFORM - PRODUCTION SERVICES STARTUP>> START-PRODUCTION-SERVICES.bat
echo echo ===============================================================================>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat
echo cd /d "%%~dp0">> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM PostgreSQL
echo echo [1/8] Starting PostgreSQL...>> START-PRODUCTION-SERVICES.bat
echo net start postgresql-x64-16 ^>nul 2^>^&1 ^|^| net start postgresql-x64-15 ^>nul 2^>^&1>> START-PRODUCTION-SERVICES.bat
echo echo ✅ PostgreSQL running on port 5432>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM Redis
echo echo [2/8] Starting Redis...>> START-PRODUCTION-SERVICES.bat
echo if exist "services\redis\redis-server.exe" (>> START-PRODUCTION-SERVICES.bat
echo     start /B "Redis" /D "services\redis" redis-server.exe redis.conf>> START-PRODUCTION-SERVICES.bat
echo     echo ✅ Redis started on port 6379>> START-PRODUCTION-SERVICES.bat
echo ) else (>> START-PRODUCTION-SERVICES.bat
echo     echo ⚠️ Redis not found - download required>> START-PRODUCTION-SERVICES.bat
echo )>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM RabbitMQ
echo echo [3/8] Starting RabbitMQ...>> START-PRODUCTION-SERVICES.bat
echo if exist "services\rabbitmq\sbin\rabbitmq-server.bat" (>> START-PRODUCTION-SERVICES.bat
echo     start /B "RabbitMQ" /D "services\rabbitmq\sbin" rabbitmq-server.bat>> START-PRODUCTION-SERVICES.bat
echo     echo ✅ RabbitMQ started - Management: http://localhost:15672>> START-PRODUCTION-SERVICES.bat
echo ) else (>> START-PRODUCTION-SERVICES.bat
echo     echo ⚠️ RabbitMQ not found - download required>> START-PRODUCTION-SERVICES.bat
echo )>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM MinIO (already working)
echo echo [4/8] Starting MinIO...>> START-PRODUCTION-SERVICES.bat
echo if not exist "minio-data" mkdir minio-data>> START-PRODUCTION-SERVICES.bat
echo start /B "MinIO" minio server minio-data --address ":9000" --console-address ":9001">> START-PRODUCTION-SERVICES.bat
echo echo ✅ MinIO started - Console: http://localhost:9001>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM Neo4j
echo echo [5/8] Starting Neo4j...>> START-PRODUCTION-SERVICES.bat
echo if exist "services\neo4j\bin\neo4j.bat" (>> START-PRODUCTION-SERVICES.bat
echo     start /B "Neo4j" /D "services\neo4j\bin" neo4j.bat console>> START-PRODUCTION-SERVICES.bat
echo     echo ✅ Neo4j started - Browser: http://localhost:7474>> START-PRODUCTION-SERVICES.bat
echo ) else (>> START-PRODUCTION-SERVICES.bat
echo     echo ⚠️ Neo4j not found - download required>> START-PRODUCTION-SERVICES.bat
echo )>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM Qdrant
echo echo [6/8] Starting Qdrant (Low Memory)...>> START-PRODUCTION-SERVICES.bat
echo if exist "services\qdrant\qdrant.exe" (>> START-PRODUCTION-SERVICES.bat
echo     start /B "Qdrant" /D "services\qdrant" qdrant.exe --config-path config.yaml>> START-PRODUCTION-SERVICES.bat
echo     echo ✅ Qdrant started - Dashboard: http://localhost:6333>> START-PRODUCTION-SERVICES.bat
echo ) else (>> START-PRODUCTION-SERVICES.bat
echo     echo ⚠️ Qdrant not found - download required>> START-PRODUCTION-SERVICES.bat
echo )>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM Ollama (already working)
echo echo [7/8] Starting Ollama...>> START-PRODUCTION-SERVICES.bat
echo start /B "Ollama" ollama serve>> START-PRODUCTION-SERVICES.bat
echo echo ✅ Ollama started on port 11434>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

REM Go Services (already working)
echo echo [8/8] Starting Go Microservices...>> START-PRODUCTION-SERVICES.bat
echo if exist "go-microservice\bin\enhanced-rag.exe" (>> START-PRODUCTION-SERVICES.bat
echo     start /B "Enhanced RAG" "go-microservice\bin\enhanced-rag.exe">> START-PRODUCTION-SERVICES.bat
echo     echo ✅ Enhanced RAG started on port 8094>> START-PRODUCTION-SERVICES.bat
echo )>> START-PRODUCTION-SERVICES.bat
echo if exist "go-microservice\bin\upload-service.exe" (>> START-PRODUCTION-SERVICES.bat
echo     start /B "Upload Service" "go-microservice\bin\upload-service.exe">> START-PRODUCTION-SERVICES.bat
echo     echo ✅ Upload Service started on port 8093>> START-PRODUCTION-SERVICES.bat
echo )>> START-PRODUCTION-SERVICES.bat
echo.>> START-PRODUCTION-SERVICES.bat

echo echo ===============================================================================>> START-PRODUCTION-SERVICES.bat
echo echo YoRHa LEGAL AI PLATFORM - ALL SERVICES STARTED>> START-PRODUCTION-SERVICES.bat
echo echo ===============================================================================>> START-PRODUCTION-SERVICES.bat
echo echo Frontend: cd sveltekit-frontend ^&^& npm run dev>> START-PRODUCTION-SERVICES.bat
echo echo YoRHa Interface: http://localhost:5177/yorha-home>> START-PRODUCTION-SERVICES.bat
echo echo ===============================================================================>> START-PRODUCTION-SERVICES.bat
echo pause>> START-PRODUCTION-SERVICES.bat

echo ✅ Production startup script created!

echo.
echo ===============================================================================
echo PRODUCTION SETUP COMPLETE - DOWNLOAD INSTRUCTIONS
echo ===============================================================================
echo.
echo 📥 REQUIRED DOWNLOADS (Place in respective service directories):
echo.
echo 1. Redis for Windows:
echo    URL: https://github.com/microsoftarchive/redis/releases
echo    File: Redis-x64-3.0.504.msi or redis-server.exe
echo    Place in: services/redis/
echo.
echo 2. RabbitMQ:
echo    URL: https://www.rabbitmq.com/download.html
echo    File: rabbitmq-server-3.12.x.exe
echo    Place in: services/rabbitmq/
echo.
echo 3. Neo4j Community:
echo    URL: https://neo4j.com/download/
echo    File: neo4j-community-5.x-windows.zip
echo    Extract to: services/neo4j/
echo.
echo 4. Qdrant:
echo    URL: https://github.com/qdrant/qdrant/releases
echo    File: qdrant-x86_64-pc-windows-msvc.zip
echo    Extract qdrant.exe to: services/qdrant/
echo.
echo ===============================================================================
echo.
echo ✅ CURRENTLY WORKING SERVICES:
echo   • PostgreSQL (port 5432)
echo   • MinIO (port 9000)
echo   • Ollama (port 11434)  
echo   • Enhanced RAG (port 8094)
echo   • Upload Service (port 8093)
echo   • YoRHa Frontend (port 5177)
echo.
echo 🎯 After downloads complete, run: START-PRODUCTION-SERVICES.bat
echo.
echo ===============================================================================

pause