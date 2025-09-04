@echo off
echo 🚀 Starting All Evidence Processing Services (Windows Native)
echo ===========================================================

echo.
echo 🔍 Checking if services are installed...

:: Check if services directory exists
if not exist "services" (
    echo ❌ Services not installed. Run setup-windows-native.bat first.
    pause
    exit /b 1
)

echo ✅ Services directory found

echo.
echo 🚀 Starting services in order...

:: Set environment variables for MinIO
set MINIO_ROOT_USER=evidence
set MINIO_ROOT_PASSWORD=evidence123

:: Start Redis
echo 🔴 Starting Redis...
if exist "services\redis-server.exe" (
    start "Redis Server" /MIN "services\redis-server.exe"
    timeout /t 3 /nobreak >nul
    echo ✅ Redis started
) else (
    echo ⚠️ Redis not found, trying system installation...
    sc start Redis >nul 2>&1
    if %errorlevel% == 0 (
        echo ✅ Redis service started
    ) else (
        echo ❌ Redis not available
    )
)

:: Start Qdrant
echo 🔍 Starting Qdrant...
if exist "services\qdrant.exe" (
    start "Qdrant" /MIN /D "services" qdrant.exe
    timeout /t 5 /nobreak >nul
    echo ✅ Qdrant started on http://localhost:6333
) else (
    echo ❌ Qdrant not found
)

:: Start PostgreSQL (if not already running)
echo 🗄️ Starting PostgreSQL...
net start postgresql-x64-15 >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ PostgreSQL started
) else (
    echo ℹ️ PostgreSQL already running or not installed as service
)

:: Start Neo4j
echo 🕸️ Starting Neo4j...
if exist "services\neo4j\bin\neo4j.bat" (
    start "Neo4j" /MIN /D "services\neo4j\bin" neo4j.bat console
    timeout /t 10 /nobreak >nul
    echo ✅ Neo4j started on http://localhost:7474
) else (
    echo ❌ Neo4j not found
)

:: Start MinIO
echo 📦 Starting MinIO...
if exist "services\minio.exe" (
    if not exist "services\minio-data" mkdir "services\minio-data"
    start "MinIO" /MIN /D "services" minio.exe server minio-data --console-address ":9001"
    timeout /t 5 /nobreak >nul
    echo ✅ MinIO started on http://localhost:9001
) else (
    echo ❌ MinIO not found
)

:: Start RabbitMQ (system service)
echo 🐰 Starting RabbitMQ...
sc start RabbitMQ >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ RabbitMQ service started on http://localhost:15672
) else (
    echo ℹ️ RabbitMQ already running or not installed as service
)

:: Start Ollama (optional)
echo 🦙 Starting Ollama (Optional)...
if exist "services\ollama.exe" (
    start "Ollama" /MIN /D "services" ollama.exe serve
    timeout /t 3 /nobreak >nul
    echo ✅ Ollama started on http://localhost:11434
) else (
    echo ⚠️ Ollama not found (optional service)
)

echo.
echo ✅ All services started!
echo.
echo 📋 Service Status Check:
echo ========================

:: Check each service
echo 🔴 Redis: 
netstat -an | findstr ":6379" >nul && echo     ✅ Running on port 6379 || echo     ❌ Not running

echo 🔍 Qdrant:
netstat -an | findstr ":6333" >nul && echo     ✅ Running on port 6333 || echo     ❌ Not running

echo 🗄️ PostgreSQL:
netstat -an | findstr ":5432" >nul && echo     ✅ Running on port 5432 || echo     ❌ Not running

echo 🕸️ Neo4j:
netstat -an | findstr ":7474" >nul && echo     ✅ Running on port 7474 || echo     ❌ Not running

echo 📦 MinIO:
netstat -an | findstr ":9000" >nul && echo     ✅ Running on port 9000 || echo     ❌ Not running

echo 🐰 RabbitMQ:
netstat -an | findstr ":5672" >nul && echo     ✅ Running on port 5672 || echo     ❌ Not running

echo 🦙 Ollama:
netstat -an | findstr ":11434" >nul && echo     ✅ Running on port 11434 || echo     ❌ Not running

echo.
echo 🌐 Access URLs:
echo ==============
echo • Qdrant Dashboard: http://localhost:6333/dashboard
echo • Neo4j Browser: http://localhost:7474 (neo4j/neo4j - change password on first login)
echo • MinIO Console: http://localhost:9001 (evidence/evidence123)
echo • RabbitMQ Management: http://localhost:15672 (guest/guest)
echo • PostgreSQL: localhost:5432 (postgres/evidence123)
echo.
echo 💡 Next Steps:
echo 1. Run setup-database.bat (if not done already)
echo 2. Run test-system.bat to verify everything works
echo 3. Run start-worker.bat to start evidence processing
echo.
echo ⚠️ Notes:
echo • Some services may take 30-60 seconds to fully initialize
echo • Check Windows Firewall if you can't access web interfaces
echo • Neo4j requires password change on first login
echo.

pause
