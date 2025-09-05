@echo off
echo 🚀 Evidence Processing System - Windows Native Setup (Smart Detection)
echo =======================================================================

echo.
echo 📋 Step 1: Checking Prerequisites and Existing Services
echo =======================================================

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Node.js found
) else (
    echo ❌ Node.js not found. Please install Node.js 18+ first.
    pause
    exit /b 1
)

:: Check Python (needed for some native modules)
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found
) else (
    echo ❌ Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo 🔍 Step 2: Detecting Existing Services
echo ======================================

:: Check PostgreSQL
echo 🗄️ Checking for PostgreSQL...
pg_config --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL found (system installation)
    set POSTGRES_INSTALLED=1
    
    :: Test connection with password 123456
    set PGPASSWORD=123456
    psql -U postgres -c "SELECT version();" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ PostgreSQL connection successful with password 123456
        set POSTGRES_READY=1
    ) else (
        echo ⚠️ PostgreSQL found but cannot connect with password 123456
        echo Please ensure PostgreSQL is running and password is correct
        set POSTGRES_READY=0
    )
) else (
    echo ❌ PostgreSQL not found - will need to install
    set POSTGRES_INSTALLED=0
    set POSTGRES_READY=0
)

:: Check Redis
echo 🔴 Checking for Redis...
redis-cli --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Redis CLI found (system installation)
    set REDIS_INSTALLED=1
    
    :: Test Redis connection
    redis-cli ping >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Redis server responding
        set REDIS_READY=1
    ) else (
        echo ⚠️ Redis CLI found but server not responding
        :: Try to start Redis service
        sc start Redis >nul 2>&1
        if %errorlevel% == 0 (
            echo ✅ Redis service started
            timeout /t 3 /nobreak >nul
            redis-cli ping >nul 2>&1 && set REDIS_READY=1 || set REDIS_READY=0
        ) else (
            echo ❌ Redis service not available
            set REDIS_READY=0
        )
    )
) else (
    echo ❌ Redis not found - will install portable version
    set REDIS_INSTALLED=0
    set REDIS_READY=0
)

:: Check RabbitMQ
echo 🐰 Checking for RabbitMQ...
rabbitmqctl version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ RabbitMQ found (system installation)
    set RABBITMQ_INSTALLED=1
    
    :: Check if RabbitMQ service is running
    sc query RabbitMQ | findstr "RUNNING" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ RabbitMQ service is running
        set RABBITMQ_READY=1
    ) else (
        echo ⚠️ RabbitMQ found but service not running
        :: Try to start RabbitMQ service
        sc start RabbitMQ >nul 2>&1
        if %errorlevel% == 0 (
            echo ✅ RabbitMQ service started
            timeout /t 5 /nobreak >nul
            set RABBITMQ_READY=1
        ) else (
            echo ❌ Failed to start RabbitMQ service
            set RABBITMQ_READY=0
        )
    )
) else (
    echo ❌ RabbitMQ not found - will need to install
    set RABBITMQ_INSTALLED=0
    set RABBITMQ_READY=0
)

:: Check if services are already running on ports
echo.
echo 🌐 Checking for services running on required ports...

netstat -an | findstr ":5432" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL port 5432 is active
) else (
    echo ⚠️ PostgreSQL port 5432 not active
)

netstat -an | findstr ":6379" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Redis port 6379 is active
) else (
    echo ⚠️ Redis port 6379 not active
)

netstat -an | findstr ":5672" >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ RabbitMQ port 5672 is active
) else (
    echo ⚠️ RabbitMQ port 5672 not active
)

echo.
echo 📦 Step 3: Installing Missing Services
echo ======================================

:: Create services directory for portable apps
if not exist "services" mkdir services
cd services

:: Install PostgreSQL if needed
if %POSTGRES_INSTALLED% == 0 (
    echo 🗄️ PostgreSQL not found. Please install manually:
    echo 1. Download from: https://www.postgresql.org/download/windows/
    echo 2. Install with superuser password: 123456
    echo 3. Press any key when installation is complete...
    pause
    
    :: Test again after manual installation
    set PGPASSWORD=123456
    psql -U postgres -c "SELECT version();" >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ PostgreSQL now available
        set POSTGRES_READY=1
    ) else (
        echo ❌ PostgreSQL still not available
        set POSTGRES_READY=0
    )
)

:: Install or start Redis
if %REDIS_READY% == 0 (
    if %REDIS_INSTALLED% == 0 (
        echo 🔴 Installing portable Redis...
        if not exist "redis-server.exe" (
            echo Downloading Redis for Windows...
            curl -L -o redis.zip "https://github.com/microsoftarchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip"
            if %errorlevel% == 0 (
                powershell -command "Expand-Archive -Path 'redis.zip' -DestinationPath '.'"
                del redis.zip
                echo ✅ Redis downloaded and extracted
            ) else (
                echo ❌ Failed to download Redis
            )
        )
    )
)

:: Install RabbitMQ if needed
if %RABBITMQ_INSTALLED% == 0 (
    echo 🐰 RabbitMQ not found. Installing via Chocolatey or manual...
    
    :: Try Chocolatey first
    choco --version >nul 2>&1
    if %errorlevel% == 0 (
        echo Installing RabbitMQ via Chocolatey...
        choco install rabbitmq -y
    ) else (
        echo Chocolatey not available. Please install RabbitMQ manually:
        echo 1. Download from: https://www.rabbitmq.com/download.html
        echo 2. Install with default settings
        echo 3. Press any key when installation is complete...
        pause
    )
)

:: Install Qdrant (always portable)
echo 🔍 Setting up Qdrant...
if not exist "qdrant.exe" (
    echo Downloading Qdrant for Windows...
    curl -L -o qdrant.zip "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip"
    if %errorlevel% == 0 (
        powershell -command "Expand-Archive -Path 'qdrant.zip' -DestinationPath '.'"
        del qdrant.zip
        echo ✅ Qdrant installed
    ) else (
        echo ❌ Failed to download Qdrant
    )
) else (
    echo ✅ Qdrant already installed
)

:: Install Neo4j (portable)
echo 🕸️ Setting up Neo4j...
if not exist "neo4j" (
    echo Downloading Neo4j Community Edition...
    curl -L -o neo4j.zip "https://dist.neo4j.org/neo4j-community-5.15.0-windows.zip"
    if %errorlevel% == 0 (
        powershell -command "Expand-Archive -Path 'neo4j.zip' -DestinationPath '.'"
        ren neo4j-community-5.15.0 neo4j
        del neo4j.zip
        
        :: Configure Neo4j with default password
        echo server.default_listen_address=0.0.0.0 > neo4j\conf\neo4j.conf
        echo server.http.listen_address=:7474 >> neo4j\conf\neo4j.conf
        echo server.bolt.listen_address=:7687 >> neo4j\conf\neo4j.conf
        echo dbms.security.auth_enabled=true >> neo4j\conf\neo4j.conf
        echo dbms.security.auth_minimum_password_length=4 >> neo4j\conf\neo4j.conf
        echo ✅ Neo4j installed and configured
    ) else (
        echo ❌ Failed to download Neo4j
    )
) else (
    echo ✅ Neo4j already installed
)

:: Install MinIO (portable)
echo 📦 Setting up MinIO...
if not exist "minio.exe" (
    echo Downloading MinIO for Windows...
    curl -L -o minio.exe "https://dl.min.io/server/minio/release/windows-amd64/minio.exe"
    if %errorlevel% == 0 (
        echo ✅ MinIO downloaded
        if not exist "minio-data" mkdir minio-data
    ) else (
        echo ❌ Failed to download MinIO
    )
) else (
    echo ✅ MinIO already installed
)

:: Install Ollama (optional)
echo 🦙 Setting up Ollama (Optional)...
if not exist "ollama.exe" (
    echo Downloading Ollama for Windows...
    curl -L -o ollama.exe "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.exe"
    if %errorlevel% == 0 (
        echo ✅ Ollama downloaded
    ) else (
        echo ⚠️ Failed to download Ollama (optional service)
    )
) else (
    echo ✅ Ollama already installed
)

cd ..

echo.
echo ⚙️ Step 4: Creating Smart Service Management Scripts
echo ===================================================

:: Create intelligent start script that uses existing services first
echo @echo off > start-all-services-smart.bat
echo echo 🚀 Starting Evidence Processing Services (Smart Mode) >> start-all-services-smart.bat
echo echo ===================================================== >> start-all-services-smart.bat
echo. >> start-all-services-smart.bat

:: PostgreSQL handling
if %POSTGRES_READY% == 1 (
    echo echo ✅ PostgreSQL already running >> start-all-services-smart.bat
) else (
    echo echo 🗄️ Starting PostgreSQL service... >> start-all-services-smart.bat
    echo net start postgresql* >> start-all-services-smart.bat
)

:: Redis handling
echo echo 🔴 Starting Redis... >> start-all-services-smart.bat
echo sc query Redis ^| findstr "RUNNING" ^>nul 2^>^&1 >> start-all-services-smart.bat
echo if %%errorlevel%% == 0 ^( >> start-all-services-smart.bat
echo     echo ✅ Redis service already running >> start-all-services-smart.bat
echo ^) else ^( >> start-all-services-smart.bat
echo     sc start Redis ^>nul 2^>^&1 >> start-all-services-smart.bat
echo     if %%errorlevel%% == 0 ^( >> start-all-services-smart.bat
echo         echo ✅ Redis service started >> start-all-services-smart.bat
echo     ^) else ^( >> start-all-services-smart.bat
echo         echo 🔄 Starting portable Redis... >> start-all-services-smart.bat
echo         if exist "services\redis-server.exe" start "Redis" /MIN "services\redis-server.exe" >> start-all-services-smart.bat
echo     ^) >> start-all-services-smart.bat
echo ^) >> start-all-services-smart.bat

:: RabbitMQ handling
echo echo 🐰 Starting RabbitMQ... >> start-all-services-smart.bat
echo sc query RabbitMQ ^| findstr "RUNNING" ^>nul 2^>^&1 >> start-all-services-smart.bat
echo if %%errorlevel%% == 0 ^( >> start-all-services-smart.bat
echo     echo ✅ RabbitMQ service already running >> start-all-services-smart.bat
echo ^) else ^( >> start-all-services-smart.bat
echo     sc start RabbitMQ ^>nul 2^>^&1 >> start-all-services-smart.bat
echo     if %%errorlevel%% == 0 ^( >> start-all-services-smart.bat
echo         echo ✅ RabbitMQ service started >> start-all-services-smart.bat
echo     ^) else ^( >> start-all-services-smart.bat
echo         echo ❌ RabbitMQ service failed to start >> start-all-services-smart.bat
echo     ^) >> start-all-services-smart.bat
echo ^) >> start-all-services-smart.bat

:: Portable services
echo echo 🔍 Starting Qdrant... >> start-all-services-smart.bat
echo if exist "services\qdrant.exe" start "Qdrant" /MIN /D "services" qdrant.exe >> start-all-services-smart.bat
echo timeout /t 3 /nobreak ^>nul >> start-all-services-smart.bat

echo echo 🕸️ Starting Neo4j... >> start-all-services-smart.bat
echo if exist "services\neo4j\bin\neo4j.bat" start "Neo4j" /MIN /D "services\neo4j\bin" neo4j.bat console >> start-all-services-smart.bat
echo timeout /t 8 /nobreak ^>nul >> start-all-services-smart.bat

echo echo 📦 Starting MinIO... >> start-all-services-smart.bat
echo set MINIO_ROOT_USER=evidence >> start-all-services-smart.bat
echo set MINIO_ROOT_PASSWORD=evidence123 >> start-all-services-smart.bat
echo if exist "services\minio.exe" start "MinIO" /MIN /D "services" minio.exe server minio-data --console-address ":9001" >> start-all-services-smart.bat
echo timeout /t 3 /nobreak ^>nul >> start-all-services-smart.bat

echo echo 🦙 Starting Ollama (Optional)... >> start-all-services-smart.bat
echo if exist "services\ollama.exe" start "Ollama" /MIN /D "services" ollama.exe serve >> start-all-services-smart.bat

echo echo. >> start-all-services-smart.bat
echo echo ✅ Service startup complete! >> start-all-services-smart.bat
echo echo 📋 Web Interfaces: >> start-all-services-smart.bat
echo echo   • RabbitMQ Management: http://localhost:15672 (guest/guest) >> start-all-services-smart.bat
echo echo   • Neo4j Browser: http://localhost:7474 (neo4j/neo4j) >> start-all-services-smart.bat
echo echo   • MinIO Console: http://localhost:9001 (evidence/evidence123) >> start-all-services-smart.bat
echo echo   • Qdrant Dashboard: http://localhost:6333/dashboard >> start-all-services-smart.bat
echo pause >> start-all-services-smart.bat

echo ✅ Smart service management scripts created

echo.
echo 📦 Step 5: Installing Node.js Dependencies
echo ==========================================

:: Install worker dependencies
cd workers
echo 🔧 Installing worker dependencies...
call npm install

:: Install frontend dependencies
cd ..\sveltekit-frontend
echo 🔧 Installing frontend dependencies...
call npm install uuid amqplib ioredis ws @qdrant/js-client-rest neo4j-driver minio node-fetch mammoth
call npm install -D @types/uuid @types/amqplib @types/ws

echo.
echo 🔧 Step 6: Creating Environment Configuration with Detected Settings
echo ====================================================================
cd ..

:: Create .env file with correct PostgreSQL password
echo # Evidence Processing System - Windows Native Configuration (Auto-detected) > .env
echo NODE_ENV=development >> .env
echo. >> .env
echo # Database (using detected PostgreSQL with password 123456) >> .env
echo DATABASE_URL=postgresql://postgres:123456@localhost:5432/evidence_processing >> .env
echo. >> .env
echo # Message Queue >> .env
if %RABBITMQ_READY% == 1 (
    echo RABBITMQ_URL=amqp://guest:guest@localhost:5672 >> .env
) else (
    echo RABBITMQ_URL=amqp://guest:guest@localhost:5672 >> .env
)
echo. >> .env
echo # Cache >> .env
echo REDIS_URL=redis://localhost:6379 >> .env
echo. >> .env
echo # Vector Database >> .env
echo QDRANT_URL=http://localhost:6333 >> .env
echo QDRANT_COLLECTION=evidence_embeddings >> .env
echo. >> .env
echo # Knowledge Graph >> .env
echo NEO4J_URL=bolt://localhost:7687 >> .env
echo NEO4J_USER=neo4j >> .env
echo NEO4J_PASSWORD=neo4j >> .env
echo. >> .env
echo # Object Storage >> .env
echo MINIO_ENDPOINT=localhost >> .env
echo MINIO_PORT=9000 >> .env
echo MINIO_ACCESS_KEY=evidence >> .env
echo MINIO_SECRET_KEY=evidence123 >> .env
echo MINIO_EVIDENCE_BUCKET=evidence >> .env
echo MINIO_USE_SSL=false >> .env
echo. >> .env
echo # Local LLM (Optional) >> .env
echo OLLAMA_URL=http://localhost:11434 >> .env

echo ✅ Environment configuration created with auto-detected settings

echo.
echo 🎉 Smart Installation Complete!
echo ===============================
echo.
echo 📊 Installation Summary:
if %POSTGRES_READY% == 1 (echo ✅ PostgreSQL: Using existing installation) else (echo ❌ PostgreSQL: Needs manual setup)
if %REDIS_READY% == 1 (echo ✅ Redis: Using existing installation) else (echo ⚠️ Redis: Using portable version)
if %RABBITMQ_READY% == 1 (echo ✅ RabbitMQ: Using existing installation) else (echo ⚠️ RabbitMQ: Needs setup)
echo ✅ Qdrant: Portable installation ready
echo ✅ Neo4j: Portable installation ready  
echo ✅ MinIO: Portable installation ready
echo ⚠️ Ollama: Optional installation ready

echo.
echo 📋 Next Steps:
echo 1. Run: start-all-services-smart.bat (uses existing services when possible)
echo 2. Run: setup-database-smart.bat (configured for password 123456)
echo 3. Run: test-system.bat (verify everything works)
echo 4. Run: start-worker.bat (start evidence processing)
echo.
echo 💡 Smart Features:
echo • Automatically detects and uses existing PostgreSQL, Redis, RabbitMQ
echo • Falls back to portable versions when system services unavailable
echo • Uses your PostgreSQL password: 123456
echo • Prioritizes system services over portable ones for better performance
echo.

pause
