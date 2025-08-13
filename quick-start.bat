@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo       Quick MinIO + SvelteKit + RAG Integration Start
echo ================================================================
echo.
echo This script bypasses cache manager and starts services directly.
echo.

:: Color codes
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

:: Set environment variables first
echo %BLUE%🔧 Setting environment variables...%NC%
set MINIO_ROOT_USER=minioadmin
set MINIO_ROOT_PASSWORD=minioadmin
set DATABASE_URL=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set PG_CONN_STRING=postgresql://legal_admin:123456@localhost:5432/legal_ai_db
set MINIO_ENDPOINT=localhost:9000
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin
set MINIO_BUCKET=legal-documents
set UPLOAD_SERVICE_PORT=8094
set RAG_HTTP_PORT=8093
set OLLAMA_BASE_URL=http://localhost:11434
set EMBED_MODEL=nomic-embed-text

:: Check prerequisites
echo %BLUE%🔍 Checking prerequisites...%NC%
where go >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Go not found. Please install Go.%NC%
    pause
    exit /b 1
)

:: Test PostgreSQL connection
echo %BLUE%🗄️ Testing PostgreSQL connection...%NC%
"C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "SELECT version();" >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ PostgreSQL not accessible. Please start PostgreSQL service.%NC%
    pause
    exit /b 1
)

:: Create required directories
echo %BLUE%📁 Setting up directories...%NC%
if not exist "C:\minio-data" mkdir "C:\minio-data"
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs
echo %GREEN%✅ Directories ready%NC%

:: Start Ollama if not running
echo %BLUE%🤖 Starting Ollama service...%NC%
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    echo %YELLOW%Starting Ollama...%NC%
    start /B "Ollama" cmd /c "ollama serve > logs\ollama.log 2>&1"
    timeout /t 3 >nul
    echo %GREEN%✅ Ollama started%NC%
) else (
    echo %GREEN%✅ Ollama already running%NC%
)

:: Start infrastructure services first
echo.
echo %BLUE%🚀 Starting infrastructure services...%NC%
echo %YELLOW%• PostgreSQL with pgvector%NC%
echo %YELLOW%• Redis for caching%NC%
echo %YELLOW%• Qdrant for vector search%NC%

docker-compose -f docker-compose-optimized.yml up -d postgres redis qdrant
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to start infrastructure services%NC%
    pause
    exit /b 1
)

:: Wait for PostgreSQL to be ready
echo.
echo %BLUE%⏳ Waiting for PostgreSQL to be ready...%NC%
set /a attempts=0
:wait_postgres
set /a attempts+=1
if %attempts% gtr 30 (
    echo %RED%❌ PostgreSQL failed to start within 60 seconds%NC%
    goto :error_exit
)

docker exec deeds-postgres pg_isready -U legal_admin -d prosecutor_db >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%.%NC%
    timeout /t 2 >nul
    goto :wait_postgres
)
echo %GREEN%✅ PostgreSQL is ready%NC%

:: Start Ollama with optimized settings
echo.
echo %BLUE%🤖 Starting Ollama AI service...%NC%
echo %YELLOW%• Using volume mounting (no copying required)%NC%
echo %YELLOW%• 6GB memory allocation%NC%
echo %YELLOW%• GPU acceleration enabled%NC%

docker-compose -f docker-compose-optimized.yml up -d ollama
if %errorlevel% neq 0 (
    echo %RED%❌ Failed to start Ollama service%NC%
    goto :error_exit
)

:: Wait for Ollama to be ready
echo.
echo %BLUE%⏳ Waiting for Ollama to be ready...%NC%
set /a attempts=0
:wait_ollama
set /a attempts+=1
if %attempts% gtr 30 (
    echo %RED%❌ Ollama failed to start within 60 seconds%NC%
    goto :error_exit
)

curl -f http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%.%NC%
    timeout /t 2 >nul
    goto :wait_ollama
)
echo %GREEN%✅ Ollama is ready%NC%

:: Check if Gemma3 legal model needs to be pulled
echo.
echo %BLUE%📥 Checking Gemma3 Legal AI model availability...%NC%
docker exec deeds-ollama-gpu ollama list | findstr "gemma3-legal" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Gemma3 Legal model not found. Pulling specialized legal model...%NC%
    echo %BLUE%This may take several minutes depending on your internet connection.%NC%
    echo %BLUE%Note: If gemma3-legal is not available, falling back to gemma2:9b%NC%
    docker exec deeds-ollama-gpu ollama pull gemma3-legal
    if %errorlevel% neq 0 (
        echo %YELLOW%⚠️  Gemma3-legal not available, pulling Gemma2 9B as fallback...%NC%
        docker exec deeds-ollama-gpu ollama pull gemma2:9b
        if %errorlevel% neq 0 (
            echo %RED%❌ Failed to pull any Gemma model%NC%
            goto :error_exit
        )
        echo %GREEN%✅ Gemma2 9B model ready (fallback)%NC%
    ) else (
        echo %GREEN%✅ Gemma3 Legal model ready%NC%
    )
) else (
    echo %GREEN%✅ Gemma3 Legal model already available%NC%
)

:: Show final status
echo.
echo %BLUE%📊 Final service status:%NC%
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | findstr -E "(deeds-|NAMES)"

:: Test AI service with legal model
echo.
echo %BLUE%🧪 Testing Legal AI service...%NC%
curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma3-legal\",\"prompt\":\"Analyze this legal document for compliance issues.\",\"stream\":false}" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Testing with fallback model...%NC%
    curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma2:9b\",\"prompt\":\"Analyze this legal document for compliance issues.\",\"stream\":false}" >nul 2>&1
)
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  AI service test failed - but services are running%NC%
) else (
    echo %GREEN%✅ AI service test successful%NC%
)

echo.
echo %GREEN%🎉 Legal AI Assistant is now running!%NC%
echo.
echo %BLUE%Next steps:%NC%
echo %YELLOW%• Frontend: cd sveltekit-frontend && npm run dev%NC%
echo %YELLOW%• API Test: http://localhost:11434/api/tags%NC%
echo %YELLOW%• Health Check: run check-setup.bat%NC%
echo %YELLOW%• Monitor: docker stats%NC%
echo.
echo %BLUE%To stop services: docker-compose -f docker-compose-optimized.yml down%NC%
echo.
pause
exit /b 0

:error_exit
echo.
echo %RED%❌ Setup failed. Check the logs above for details.%NC%
echo %YELLOW%💡 Try running: docker-compose -f docker-compose-optimized.yml logs%NC%
pause
exit /b 1
