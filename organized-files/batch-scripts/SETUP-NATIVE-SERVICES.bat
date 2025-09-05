@echo off
REM Native Windows Legal AI Microservices Setup
REM No Docker - Pure Windows Native Architecture
REM Downloads services and configures native Windows deployment

echo 🏗️  Setting up Native Windows Legal AI Platform...

REM Create service directories
if not exist "services" mkdir services
if not exist "services\minio" mkdir services\minio
if not exist "services\neo4j" mkdir services\neo4j  
if not exist "services\qdrant" mkdir services\qdrant
if not exist "services\redis" mkdir services\redis
if not exist "services\postgresql" mkdir services\postgresql

echo 📦 Downloading required services...

REM Download MinIO (Object Storage) - Native Windows Binary
if not exist "services\minio\minio.exe" (
    echo ⬇️  Downloading MinIO Server (~108MB)...
    curl -L -o "services\minio\minio.exe" https://dl.min.io/server/minio/release/windows-amd64/minio.exe
    echo ✅ MinIO downloaded
) else (
    echo ✅ MinIO already exists
)

REM Download Neo4j (Graph Database) - Native Windows
if not exist "services\neo4j\neo4j-community.zip" (
    echo ⬇️  Downloading Neo4j Community (~118MB)...
    curl -L -o "services\neo4j\neo4j-community.zip" https://dist.neo4j.org/neo4j-community-5.23.0-windows.zip
    echo ✅ Neo4j downloaded
    
    echo 📂 Extracting Neo4j...
    powershell -Command "Expand-Archive -Path 'services\neo4j\neo4j-community.zip' -DestinationPath 'services\neo4j' -Force"
    echo ✅ Neo4j extracted
) else (
    echo ✅ Neo4j already exists
)

REM Download Qdrant (Vector Database) - Native Windows
if not exist "services\qdrant\qdrant.zip" (
    echo ⬇️  Downloading Qdrant Vector Database (~77MB)...
    curl -L -o "services\qdrant\qdrant.zip" https://github.com/qdrant/qdrant/releases/download/v1.8.4/qdrant-x86_64-pc-windows-msvc.zip
    echo ✅ Qdrant downloaded
    
    echo 📂 Extracting Qdrant...
    powershell -Command "Expand-Archive -Path 'services\qdrant\qdrant.zip' -DestinationPath 'services\qdrant' -Force"
    echo ✅ Qdrant extracted
) else (
    echo ✅ Qdrant already exists
)

REM Download Redis (Caching) - Native Windows Build
if not exist "services\redis\redis.zip" (
    echo ⬇️  Downloading Redis for Windows (~5MB)...
    curl -L -o "services\redis\redis.zip" https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip
    echo ✅ Redis downloaded
    
    echo 📂 Extracting Redis...
    powershell -Command "Expand-Archive -Path 'services\redis\redis.zip' -DestinationPath 'services\redis' -Force"
    echo ✅ Redis extracted
) else (
    echo ✅ Redis already exists
)

echo 🔧 Creating Windows Service Configuration...

REM Create Windows service start scripts
call :create_service_scripts

echo ⚡ Creating Native Microservice Orchestrator...

REM Build Go microservices with Protobuf support
if exist "go-microservice\go.mod" (
    echo 🔨 Building Go microservices...
    cd go-microservice
    go build -o ..\services\enhanced-rag-service.exe cmd\enhanced-rag\main.go
    go build -o ..\services\upload-service.exe cmd\upload-service\main.go
    go build -o ..\services\multi-protocol-gateway.exe cmd\multi-protocol-gateway\main.go
    cd ..
    echo ✅ Go microservices built
)

echo 🎉 Native Windows setup complete!
echo 💡 All services are now available as native Windows binaries
echo 📋 Next steps:
echo    1. Run START-NATIVE-SERVICES.bat to start all services
echo    2. Use STOP-NATIVE-SERVICES.bat to stop all services
echo    3. Check SERVICE-STATUS.bat for health monitoring

pause
goto :eof

:create_service_scripts
echo 📝 Creating service management scripts...

REM Create START-NATIVE-SERVICES.bat
echo @echo off > START-NATIVE-SERVICES.bat
echo REM Start all native Windows services for Legal AI Platform >> START-NATIVE-SERVICES.bat
echo echo 🚀 Starting Native Windows Legal AI Services... >> START-NATIVE-SERVICES.bat
echo. >> START-NATIVE-SERVICES.bat
echo REM Start MinIO Object Storage >> START-NATIVE-SERVICES.bat
echo start "MinIO Server" /min "services\minio\minio.exe" server "services\minio\data" --console-address ":9001" >> START-NATIVE-SERVICES.bat
echo echo ✅ MinIO started on :9000 (Console: :9001^) >> START-NATIVE-SERVICES.bat
echo. >> START-NATIVE-SERVICES.bat
echo REM Start Redis Cache >> START-NATIVE-SERVICES.bat
echo start "Redis Server" /min "services\redis\redis-server.exe" --port 6379 >> START-NATIVE-SERVICES.bat
echo echo ✅ Redis started on :6379 >> START-NATIVE-SERVICES.bat
echo. >> START-NATIVE-SERVICES.bat
echo REM Start Qdrant Vector Database >> START-NATIVE-SERVICES.bat
echo start "Qdrant Server" /min "services\qdrant\qdrant.exe" --uri http://localhost:6333 >> START-NATIVE-SERVICES.bat
echo echo ✅ Qdrant started on :6333 >> START-NATIVE-SERVICES.bat
echo. >> START-NATIVE-SERVICES.bat
echo REM Start Neo4j Graph Database >> START-NATIVE-SERVICES.bat
echo start "Neo4j Server" /min "services\neo4j\neo4j-community-5.23.0\bin\neo4j.bat" console >> START-NATIVE-SERVICES.bat
echo echo ✅ Neo4j started on :7474 (Bolt: :7687^) >> START-NATIVE-SERVICES.bat
echo. >> START-NATIVE-SERVICES.bat
echo REM Start Go Microservices >> START-NATIVE-SERVICES.bat
echo start "Enhanced RAG Service" /min "services\enhanced-rag-service.exe" --port 8094 >> START-NATIVE-SERVICES.bat
echo start "Upload Service" /min "services\upload-service.exe" --port 8093 >> START-NATIVE-SERVICES.bat
echo start "Multi-Protocol Gateway" /min "services\multi-protocol-gateway.exe" --port 8080 >> START-NATIVE-SERVICES.bat
echo echo ✅ Go microservices started >> START-NATIVE-SERVICES.bat
echo. >> START-NATIVE-SERVICES.bat
echo echo 🎉 All services started! Access via: >> START-NATIVE-SERVICES.bat
echo echo    - Frontend: http://localhost:5173 >> START-NATIVE-SERVICES.bat
echo echo    - MinIO Console: http://localhost:9001 >> START-NATIVE-SERVICES.bat
echo echo    - Neo4j Browser: http://localhost:7474 >> START-NATIVE-SERVICES.bat
echo echo    - API Gateway: http://localhost:8080 >> START-NATIVE-SERVICES.bat
echo pause >> START-NATIVE-SERVICES.bat

REM Create STOP-NATIVE-SERVICES.bat
echo @echo off > STOP-NATIVE-SERVICES.bat
echo REM Stop all native Windows services >> STOP-NATIVE-SERVICES.bat
echo echo 🛑 Stopping Native Windows Legal AI Services... >> STOP-NATIVE-SERVICES.bat
echo. >> STOP-NATIVE-SERVICES.bat
echo taskkill /f /im minio.exe 2^>nul >> STOP-NATIVE-SERVICES.bat
echo taskkill /f /im redis-server.exe 2^>nul >> STOP-NATIVE-SERVICES.bat
echo taskkill /f /im qdrant.exe 2^>nul >> STOP-NATIVE-SERVICES.bat
echo taskkill /f /im java.exe /fi "WINDOWTITLE eq Neo4j*" 2^>nul >> STOP-NATIVE-SERVICES.bat
echo taskkill /f /im enhanced-rag-service.exe 2^>nul >> STOP-NATIVE-SERVICES.bat
echo taskkill /f /im upload-service.exe 2^>nul >> STOP-NATIVE-SERVICES.bat
echo taskkill /f /im multi-protocol-gateway.exe 2^>nul >> STOP-NATIVE-SERVICES.bat
echo echo ✅ All services stopped >> STOP-NATIVE-SERVICES.bat
echo pause >> STOP-NATIVE-SERVICES.bat

REM Create SERVICE-STATUS.bat
echo @echo off > SERVICE-STATUS.bat
echo REM Check status of all native services >> SERVICE-STATUS.bat
echo echo 📊 Native Windows Legal AI Service Status... >> SERVICE-STATUS.bat
echo echo. >> SERVICE-STATUS.bat
echo tasklist /fi "imagename eq minio.exe" 2^>nul ^| find "minio.exe" ^>nul ^&^& echo ✅ MinIO: Running ^|^| echo ❌ MinIO: Not Running >> SERVICE-STATUS.bat
echo tasklist /fi "imagename eq redis-server.exe" 2^>nul ^| find "redis-server.exe" ^>nul ^&^& echo ✅ Redis: Running ^|^| echo ❌ Redis: Not Running >> SERVICE-STATUS.bat
echo tasklist /fi "imagename eq qdrant.exe" 2^>nul ^| find "qdrant.exe" ^>nul ^&^& echo ✅ Qdrant: Running ^|^| echo ❌ Qdrant: Not Running >> SERVICE-STATUS.bat
echo tasklist /fi "imagename eq java.exe" 2^>nul ^| find "java.exe" ^>nul ^&^& echo ✅ Neo4j: Running ^|^| echo ❌ Neo4j: Not Running >> SERVICE-STATUS.bat
echo tasklist /fi "imagename eq enhanced-rag-service.exe" 2^>nul ^| find "enhanced-rag-service.exe" ^>nul ^&^& echo ✅ Enhanced RAG: Running ^|^| echo ❌ Enhanced RAG: Not Running >> SERVICE-STATUS.bat
echo tasklist /fi "imagename eq upload-service.exe" 2^>nul ^| find "upload-service.exe" ^>nul ^&^& echo ✅ Upload Service: Running ^|^| echo ❌ Upload Service: Not Running >> SERVICE-STATUS.bat
echo tasklist /fi "imagename eq multi-protocol-gateway.exe" 2^>nul ^| find "multi-protocol-gateway.exe" ^>nul ^&^& echo ✅ Multi-Protocol Gateway: Running ^|^| echo ❌ Multi-Protocol Gateway: Not Running >> SERVICE-STATUS.bat
echo pause >> SERVICE-STATUS.bat

echo ✅ Service management scripts created
goto :eof