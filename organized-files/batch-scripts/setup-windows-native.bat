@echo off
echo 🚀 Evidence Processing System - Windows Native Setup
echo ====================================================

echo.
echo 📋 Step 1: Checking Prerequisites
echo =================================

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Node.js found
) else (
    echo ❌ Node.js not found. Please install Node.js 18+ first.
    pause
    exit /b 1
)

:: Check if Chocolatey is available
choco --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Chocolatey found
    set CHOCO_AVAILABLE=1
) else (
    echo ⚠️ Chocolatey not found. Will provide manual installation instructions.
    set CHOCO_AVAILABLE=0
)

:: Check Python (needed for some native modules)
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Python found
) else (
    echo ❌ Python not found. Please install Python 3.8+ first.
    if %CHOCO_AVAILABLE%==1 (
        echo   Quick install: choco install python
    )
    pause
    exit /b 1
)

echo.
echo 📦 Step 2: Installing Native Windows Services
echo =============================================

:: Create directories for services
if not exist "services" mkdir services
cd services

echo.
echo 🐰 Installing RabbitMQ (Native Windows)
echo =======================================
if %CHOCO_AVAILABLE%==1 (
    echo Installing RabbitMQ via Chocolatey...
    choco install rabbitmq -y
    echo Enabling RabbitMQ Management Plugin...
    "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.10\sbin\rabbitmq-plugins.exe" enable rabbitmq_management
) else (
    echo Please install RabbitMQ manually:
    echo 1. Download from: https://www.rabbitmq.com/download.html
    echo 2. Install with default settings
    echo 3. Enable management plugin: rabbitmq-plugins enable rabbitmq_management
    echo 4. Press any key when done...
    pause
)

echo.
echo 🔴 Installing Redis (Native Windows)
echo ====================================
if %CHOCO_AVAILABLE%==1 (
    echo Installing Redis via Chocolatey...
    choco install redis-64 -y
) else (
    echo Please install Redis manually:
    echo 1. Download from: https://github.com/microsoftarchive/redis/releases
    echo 2. Extract and run redis-server.exe
    echo 3. Press any key when done...
    pause
)

echo.
echo 🗄️ Installing PostgreSQL + pgvector (Native Windows)
echo =====================================================
if %CHOCO_AVAILABLE%==1 (
    echo Installing PostgreSQL via Chocolatey...
    choco install postgresql --params '/Password:evidence123' -y
) else (
    echo Please install PostgreSQL manually:
    echo 1. Download from: https://www.postgresql.org/download/windows/
    echo 2. Install with password: evidence123
    echo 3. Press any key when done...
    pause
)

echo.
echo 🔍 Setting up Qdrant (Portable Windows Binary)
echo ==============================================
if not exist "qdrant" (
    echo Downloading Qdrant for Windows...
    curl -L -o qdrant-windows.zip "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.zip"
    powershell -command "Expand-Archive -Path 'qdrant-windows.zip' -DestinationPath 'qdrant'"
    del qdrant-windows.zip
    echo ✅ Qdrant installed
) else (
    echo ✅ Qdrant already installed
)

echo.
echo 🕸️ Setting up Neo4j (Community Edition)
echo ========================================
if not exist "neo4j" (
    echo Downloading Neo4j Community Edition...
    curl -L -o neo4j-community.zip "https://dist.neo4j.org/neo4j-community-5.15.0-windows.zip"
    powershell -command "Expand-Archive -Path 'neo4j-community.zip' -DestinationPath '.'"
    ren neo4j-community-5.15.0 neo4j
    del neo4j-community.zip
    
    :: Configure Neo4j
    echo Setting up Neo4j configuration...
    echo dbms.default_database=neo4j > neo4j\conf\neo4j.conf
    echo server.default_listen_address=0.0.0.0 >> neo4j\conf\neo4j.conf
    echo dbms.security.auth_enabled=true >> neo4j\conf\neo4j.conf
    echo server.http.listen_address=:7474 >> neo4j\conf\neo4j.conf
    echo server.bolt.listen_address=:7687 >> neo4j\conf\neo4j.conf
    echo ✅ Neo4j installed and configured
) else (
    echo ✅ Neo4j already installed
)

echo.
echo 📦 Setting up MinIO (Native Windows Binary)
echo ===========================================
if not exist "minio.exe" (
    echo Downloading MinIO for Windows...
    curl -L -o minio.exe "https://dl.min.io/server/minio/release/windows-amd64/minio.exe"
    echo ✅ MinIO downloaded
) else (
    echo ✅ MinIO already installed
)

echo.
echo 🦙 Setting up Ollama (Optional - for local LLM)
echo ===============================================
if not exist "ollama.exe" (
    echo Downloading Ollama for Windows...
    curl -L -o ollama-windows.zip "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
    powershell -command "Expand-Archive -Path 'ollama-windows.zip' -DestinationPath '.'"
    del ollama-windows.zip
    echo ✅ Ollama downloaded
) else (
    echo ✅ Ollama already installed
)

cd ..

echo.
echo ⚙️ Step 3: Creating Service Management Scripts
echo ==============================================

:: Create start-all-services.bat
echo @echo off > start-all-services.bat
echo echo 🚀 Starting All Evidence Processing Services >> start-all-services.bat
echo echo =============================================== >> start-all-services.bat
echo. >> start-all-services.bat
echo echo 🐰 Starting RabbitMQ... >> start-all-services.bat
echo start "RabbitMQ" "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.12.10\sbin\rabbitmq-server.bat" >> start-all-services.bat
echo timeout /t 5 /nobreak >> start-all-services.bat
echo. >> start-all-services.bat
echo echo 🔴 Starting Redis... >> start-all-services.bat
echo start "Redis" "C:\Program Files\Redis\redis-server.exe" >> start-all-services.bat
echo timeout /t 3 /nobreak >> start-all-services.bat
echo. >> start-all-services.bat
echo echo 🗄️ Starting PostgreSQL... >> start-all-services.bat
echo net start postgresql-x64-15 >> start-all-services.bat
echo timeout /t 3 /nobreak >> start-all-services.bat
echo. >> start-all-services.bat
echo echo 🔍 Starting Qdrant... >> start-all-services.bat
echo start "Qdrant" /D "services\qdrant" qdrant.exe >> start-all-services.bat
echo timeout /t 5 /nobreak >> start-all-services.bat
echo. >> start-all-services.bat
echo echo 🕸️ Starting Neo4j... >> start-all-services.bat
echo start "Neo4j" /D "services\neo4j\bin" neo4j.bat console >> start-all-services.bat
echo timeout /t 10 /nobreak >> start-all-services.bat
echo. >> start-all-services.bat
echo echo 📦 Starting MinIO... >> start-all-services.bat
echo set MINIO_ROOT_USER=evidence >> start-all-services.bat
echo set MINIO_ROOT_PASSWORD=evidence123 >> start-all-services.bat
echo start "MinIO" /D "services" minio.exe server data --console-address ":9001" >> start-all-services.bat
echo timeout /t 5 /nobreak >> start-all-services.bat
echo. >> start-all-services.bat
echo echo 🦙 Starting Ollama (Optional)... >> start-all-services.bat
echo start "Ollama" /D "services" ollama.exe serve >> start-all-services.bat
echo. >> start-all-services.bat
echo echo ✅ All services started! >> start-all-services.bat
echo echo Check services at: >> start-all-services.bat
echo echo   • RabbitMQ Management: http://localhost:15672 (guest/guest) >> start-all-services.bat
echo echo   • Neo4j Browser: http://localhost:7474 (neo4j/neo4j) >> start-all-services.bat
echo echo   • MinIO Console: http://localhost:9001 (evidence/evidence123) >> start-all-services.bat
echo echo   • Qdrant Dashboard: http://localhost:6333/dashboard >> start-all-services.bat
echo pause >> start-all-services.bat

:: Create stop-all-services.bat
echo @echo off > stop-all-services.bat
echo echo 🛑 Stopping All Evidence Processing Services >> stop-all-services.bat
echo echo ============================================== >> stop-all-services.bat
echo. >> stop-all-services.bat
echo taskkill /F /IM rabbitmq-server.exe /T 2^>nul >> stop-all-services.bat
echo taskkill /F /IM redis-server.exe /T 2^>nul >> stop-all-services.bat
echo taskkill /F /IM qdrant.exe /T 2^>nul >> stop-all-services.bat
echo taskkill /F /IM neo4j.exe /T 2^>nul >> stop-all-services.bat
echo taskkill /F /IM minio.exe /T 2^>nul >> stop-all-services.bat
echo taskkill /F /IM ollama.exe /T 2^>nul >> stop-all-services.bat
echo net stop postgresql-x64-15 2^>nul >> stop-all-services.bat
echo echo ✅ All services stopped! >> stop-all-services.bat
echo pause >> stop-all-services.bat

echo ✅ Service management scripts created

echo.
echo 📦 Step 4: Installing Node.js Dependencies
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
echo 🗄️ Step 5: Database Setup
echo =========================
cd ..
echo 📊 Setting up PostgreSQL database and pgvector extension...

:: Create setup database script
echo @echo off > setup-database.bat
echo echo Setting up Evidence Processing Database... >> setup-database.bat
echo. >> setup-database.bat
echo :: Create database >> setup-database.bat
echo createdb -U postgres evidence_processing >> setup-database.bat
echo. >> setup-database.bat
echo :: Install pgvector extension >> setup-database.bat
echo psql -U postgres -d evidence_processing -c "CREATE EXTENSION IF NOT EXISTS vector;" >> setup-database.bat
echo. >> setup-database.bat
echo :: Run migration >> setup-database.bat
echo psql -U postgres -d evidence_processing -f migrations/create_evidence_processing_schema.sql >> setup-database.bat
echo. >> setup-database.bat
echo echo ✅ Database setup complete! >> setup-database.bat
echo pause >> setup-database.bat

echo ✅ Database setup script created (run setup-database.bat after starting PostgreSQL)

echo.
echo 🔧 Step 6: Creating Environment Configuration
echo =============================================

:: Create .env file for the project
echo # Evidence Processing System - Windows Native Configuration > .env
echo NODE_ENV=development >> .env
echo. >> .env
echo # Database >> .env
echo DATABASE_URL=postgresql://postgres:evidence123@localhost:5432/evidence_processing >> .env
echo. >> .env
echo # Message Queue >> .env
echo RABBITMQ_URL=amqp://guest:guest@localhost:5672 >> .env
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

echo ✅ Environment configuration created

echo.
echo 📝 Step 7: Creating Quick Commands
echo ==================================

:: Create start-worker.bat
echo @echo off > start-worker.bat
echo echo 🏭 Starting Evidence Processing Worker >> start-worker.bat
echo echo ====================================== >> start-worker.bat
echo cd workers >> start-worker.bat
echo npm start >> start-worker.bat

:: Create test-system.bat
echo @echo off > test-system.bat
echo echo 🧪 Testing Evidence Processing System >> test-system.bat
echo echo ===================================== >> test-system.bat
echo cd workers >> test-system.bat
echo npm run health >> test-system.bat

:: Create setup-ollama-models.bat (optional)
echo @echo off > setup-ollama-models.bat
echo echo 🦙 Setting up Ollama Models (Optional) >> setup-ollama-models.bat
echo echo ==================================== >> setup-ollama-models.bat
echo cd services >> setup-ollama-models.bat
echo ollama.exe pull nomic-embed-text >> setup-ollama-models.bat
echo ollama.exe pull llama3.1:8b >> setup-ollama-models.bat
echo echo ✅ Ollama models installed >> setup-ollama-models.bat
echo pause >> setup-ollama-models.bat

echo ✅ Quick command scripts created

echo.
echo 🎉 Installation Complete!
echo =========================
echo.
echo 📋 Next Steps:
echo 1. Run: start-all-services.bat
echo 2. Wait for all services to start (about 30 seconds)
echo 3. Run: setup-database.bat
echo 4. Run: test-system.bat (to verify everything works)
echo 5. Run: start-worker.bat (to start processing)
echo.
echo 🔧 Optional:
echo • Run: setup-ollama-models.bat (for local LLM support)
echo.
echo 📋 Service Management:
echo • Start all: start-all-services.bat
echo • Stop all: stop-all-services.bat
echo • Test health: test-system.bat
echo • Start worker: start-worker.bat
echo.
echo 🌐 Service URLs (after starting):
echo • RabbitMQ Management: http://localhost:15672 (guest/guest)
echo • Neo4j Browser: http://localhost:7474 (neo4j/neo4j)
echo • MinIO Console: http://localhost:9001 (evidence/evidence123)
echo • Qdrant Dashboard: http://localhost:6333/dashboard
echo.
echo ⚠️ Important Notes:
echo • PostgreSQL service should auto-start with Windows
echo • Change Neo4j password on first login
echo • Ensure Windows Firewall allows these ports
echo • Some antivirus software may flag the executables
echo.

pause
