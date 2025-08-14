@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   Legal AI Assistant - Complete Setup (Optimized)
echo =====================================================
echo.
echo This script performs a complete environment setup using
echo the OPTIMIZED configuration with proper volume mounting.
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

:: Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Running without administrator privileges%NC%
    echo %YELLOW%Some operations may require elevated permissions%NC%
    echo.
)

:: Prerequisites check
echo %BLUE%🔍 Checking prerequisites...%NC%

:: Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Docker not found%NC%
    echo Please install Docker Desktop and restart this script.
    pause
    exit /b 1
)
echo %GREEN%✅ Docker found%NC%

:: Check Docker Compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Docker Compose not found%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Docker Compose found%NC%

:: Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Node.js not found - required for frontend development%NC%
    echo You can continue with Docker-only setup or install Node.js later.
    set "NODE_AVAILABLE=false"
) else (
    echo %GREEN%✅ Node.js found%NC%
    set "NODE_AVAILABLE=true"
)

:: Check npm
if "%NODE_AVAILABLE%"=="true" (
    npm --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo %YELLOW%⚠️  npm not found%NC%
        set "NPM_AVAILABLE=false"
    ) else (
        echo %GREEN%✅ npm found%NC%
        set "NPM_AVAILABLE=true"
    )
)

:: Check GPU
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  NVIDIA GPU not detected - using CPU mode%NC%
    set "GPU_AVAILABLE=false"
) else (
    echo %GREEN%✅ NVIDIA GPU detected%NC%
    set "GPU_AVAILABLE=true"
)

:: Setup directories
echo.
echo %BLUE%📁 Creating project directories...%NC%
if not exist "models" (
    mkdir models
    echo %GREEN%✅ Created models directory%NC%
)
if not exist "data" (
    mkdir data
    echo %GREEN%✅ Created data directory%NC%
)
if not exist "data\uploads" (
    mkdir data\uploads
    echo %GREEN%✅ Created uploads directory%NC%
)
if not exist "data\processed" (
    mkdir data\processed
    echo %GREEN%✅ Created processed directory%NC%
)
if not exist "logs" (
    mkdir logs
    echo %GREEN%✅ Created logs directory%NC%
)
if not exist "database" (
    mkdir database
    echo %GREEN%✅ Created database directory%NC%
)
if not exist "database\migrations" (
    mkdir database\migrations
    echo %GREEN%✅ Created migrations directory%NC%
)

:: Create database initialization script
echo.
echo %BLUE%🗄️  Setting up database initialization...%NC%
(
echo -- Initialize legal AI database
echo CREATE EXTENSION IF NOT EXISTS vector;
echo CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
echo.
echo -- Create legal documents table
echo CREATE TABLE IF NOT EXISTS legal_documents ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     title VARCHAR^(255^) NOT NULL,
echo     content TEXT,
echo     document_type VARCHAR^(100^),
echo     case_id VARCHAR^(255^),
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     embedding vector^(384^)
echo ^);
echo.
echo -- Create cases table
echo CREATE TABLE IF NOT EXISTS cases ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     case_number VARCHAR^(255^) UNIQUE NOT NULL,
echo     title VARCHAR^(255^) NOT NULL,
echo     status VARCHAR^(100^) DEFAULT 'active',
echo     priority VARCHAR^(50^) DEFAULT 'medium',
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
echo     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- Create audit log table
echo CREATE TABLE IF NOT EXISTS audit_logs ^(
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4^(^),
echo     action VARCHAR^(255^) NOT NULL,
echo     entity_type VARCHAR^(100^),
echo     entity_id UUID,
echo     user_id VARCHAR^(255^),
echo     ip_address INET,
echo     details JSONB,
echo     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
echo ^);
echo.
echo -- Insert sample data
echo INSERT INTO cases ^(case_number, title, status, priority^) VALUES
echo ^('CASE-2024-001', 'Contract Dispute - Tech Corp vs StartupXYZ', 'active', 'high'^),
echo ^('CASE-2024-002', 'Employment Law - Wrongful Termination', 'active', 'medium'^),
echo ^('CASE-2024-003', 'Intellectual Property - Patent Infringement', 'pending', 'high'^)
echo ON CONFLICT ^(case_number^) DO NOTHING;
) > "database\migrations\001_init.sql"
echo %GREEN%✅ Database initialization script created%NC%

:: Stop any existing services
echo.
echo %BLUE%🛑 Stopping any existing services...%NC%
docker-compose -f docker-compose-optimized.yml down >nul 2>&1

:: Install dependencies if npm available
if "%NPM_AVAILABLE%"=="true" (
    echo.
    echo %BLUE%📦 Installing root dependencies...%NC%
    npm install
    if %errorlevel% neq 0 (
        echo %YELLOW%⚠️  Root npm install failed - continuing with Docker setup%NC%
    ) else (
        echo %GREEN%✅ Root dependencies installed%NC%
    )
    
    if exist "sveltekit-frontend\package.json" (
        echo %BLUE%📦 Installing frontend dependencies...%NC%
        cd sveltekit-frontend
        npm install
        if %errorlevel% neq 0 (
            echo %YELLOW%⚠️  Frontend npm install failed%NC%
        ) else (
            echo %GREEN%✅ Frontend dependencies installed%NC%
        )
        cd ..
    )
)

:: Start infrastructure services
echo.
echo %BLUE%🚀 Starting infrastructure services...%NC%
docker-compose -f docker-compose-optimized.yml up -d postgres redis qdrant rabbitmq
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to start infrastructure services%NC%
    goto :error_exit
)

:: Wait for PostgreSQL
echo %BLUE%⏳ Waiting for PostgreSQL...%NC%
set /a attempts=0
:wait_db
set /a attempts+=1
if %attempts% gtr 30 (
    echo %RED%❌ PostgreSQL timeout%NC%
    goto :error_exit
)
docker exec deeds-postgres pg_isready -U legal_admin -d prosecutor_db >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 2 >nul
    goto :wait_db
)
echo %GREEN%✅ PostgreSQL ready%NC%

:: Initialize database
echo %BLUE%🗄️  Initializing database...%NC%
docker exec -i deeds-postgres psql -U legal_admin -d prosecutor_db < database\migrations\001_init.sql >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Database initialization had warnings - this is normal%NC%
) else (
    echo %GREEN%✅ Database initialized%NC%
)

:: Start Ollama
echo.
echo %BLUE%🤖 Starting Ollama AI service...%NC%
docker-compose -f docker-compose-optimized.yml up -d ollama
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to start Ollama%NC%
    goto :error_exit
)

:: Wait for Ollama
echo %BLUE%⏳ Waiting for Ollama...%NC%
set /a attempts=0
:wait_ollama
set /a attempts+=1
if %attempts% gtr 30 (
    echo %RED%❌ Ollama timeout%NC%
    goto :error_exit
)
curl -f http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    timeout /t 2 >nul
    goto :wait_ollama
)
echo %GREEN%✅ Ollama ready%NC%

:: Pull Gemma3 Legal AI model
echo.
echo %BLUE%📥 Setting up Gemma3 Legal AI model...%NC%
docker exec deeds-ollama-gpu ollama list | findstr "gemma3-legal" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%Pulling Gemma3 Legal model... This may take 10-15 minutes.%NC%
    echo %BLUE%Specialized legal model optimized for legal document analysis%NC%
    docker exec deeds-ollama-gpu ollama pull gemma3-legal
    if %errorlevel% neq 0 (
        echo %YELLOW%⚠️  Gemma3-legal not available, using Gemma2 9B as fallback...%NC%
        docker exec deeds-ollama-gpu ollama pull gemma2:9b
        if %errorlevel% neq 0 (
            echo %RED%❌ Failed to pull any model%NC%
            goto :error_exit
        )
        echo %GREEN%✅ Gemma2 9B model ready (fallback)%NC%
    ) else (
        echo %GREEN%✅ Gemma3 Legal model ready%NC%
    )
) else (
    echo %GREEN%✅ Gemma3 Legal model already available%NC%
)

:: Final status
echo.
echo %GREEN%🎉 Setup completed successfully!%NC%
echo.
echo %BLUE%📊 Service Status:%NC%
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr -E "(deeds-|NAMES)"

echo.
echo %BLUE%🚀 Next Steps:%NC%
echo %YELLOW%1. Start frontend: cd sveltekit-frontend && npm run dev%NC%
echo %YELLOW%2. Test Legal AI: curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma3-legal\",\"prompt\":\"Test legal analysis\",\"stream\":false}"%NC%
echo %YELLOW%3. Access app: http://localhost:5173%NC%
echo %YELLOW%4. Health check: check-setup.bat%NC%

if "%NPM_AVAILABLE%"=="true" (
    echo.
    echo %BLUE%💡 NPM Commands Available:%NC%
    echo %YELLOW%• npm run dev      - Start development environment%NC%
    echo %YELLOW%• npm run health   - Run health checks%NC%
    echo %YELLOW%• npm run monitor  - Monitor resources%NC%
)

echo.
echo %BLUE%Environment is ready for legal AI development!%NC%
pause
exit /b 0

:error_exit
echo.
echo %RED%❌ Setup failed%NC%
echo %YELLOW%Check logs: docker-compose -f docker-compose-optimized.yml logs%NC%
echo %YELLOW%For help: check-setup.bat%NC%
pause
exit /b 1
