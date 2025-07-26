@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo   Legal AI Assistant - Health Check Diagnostic
echo =====================================================
echo.

:: Color codes for output
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

:: Check if Docker is running
echo %BLUE%🐳 Checking Docker status...%NC%
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Docker not found. Please install Docker Desktop.%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Docker is available%NC%

:: Check Docker Compose
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%❌ Docker Compose not found.%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Docker Compose is available%NC%

:: Check NVIDIA GPU
echo.
echo %BLUE%🎮 Checking NVIDIA GPU support...%NC%
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  NVIDIA GPU not detected - using CPU mode%NC%
    set "GPU_AVAILABLE=false"
) else (
    echo %GREEN%✅ NVIDIA GPU detected%NC%
    set "GPU_AVAILABLE=true"
)

:: Check for optimized Docker Compose file
echo.
echo %BLUE%📄 Checking project structure...%NC%
if not exist "docker-compose-optimized.yml" (
    echo %RED%❌ docker-compose-optimized.yml not found%NC%
    echo Please ensure you have the optimized Docker Compose configuration.
) else (
    echo %GREEN%✅ docker-compose-optimized.yml found%NC%
)

if not exist "sveltekit-frontend" (
    echo %RED%❌ sveltekit-frontend directory not found%NC%
) else (
    echo %GREEN%✅ sveltekit-frontend directory found%NC%
)

if not exist "models" (
    echo %YELLOW%⚠️  models directory not found - creating...%NC%
    mkdir models
    echo %GREEN%✅ models directory created%NC%
) else (
    echo %GREEN%✅ models directory found%NC%
)

:: Check if containers are running
echo.
echo %BLUE%📦 Checking container status...%NC%

:: Check PostgreSQL
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr "deeds-postgres" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  PostgreSQL container not running%NC%
    set "POSTGRES_RUNNING=false"
) else (
    echo %GREEN%✅ PostgreSQL container is running%NC%
    set "POSTGRES_RUNNING=true"
)

:: Check Ollama
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr "deeds-ollama" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%⚠️  Ollama container not running%NC%
    set "OLLAMA_RUNNING=false"
) else (
    echo %GREEN%✅ Ollama container is running%NC%
    set "OLLAMA_RUNNING=true"
)

:: Test service health if containers are running
echo.
echo %BLUE%🏥 Testing service health...%NC%

if "%POSTGRES_RUNNING%"=="true" (
    docker exec deeds-postgres pg_isready -U legal_admin -d prosecutor_db >nul 2>&1
    if %errorlevel% neq 0 (
        echo %RED%❌ PostgreSQL health check failed%NC%
    ) else (
        echo %GREEN%✅ PostgreSQL is healthy%NC%
    )
)

if "%OLLAMA_RUNNING%"=="true" (
    curl -f http://localhost:11434/api/version >nul 2>&1
    if %errorlevel% neq 0 (
        echo %RED%❌ Ollama health check failed%NC%
    ) else (
        echo %GREEN%✅ Ollama is healthy%NC%
    )
)

:: Check memory usage
echo.
echo %BLUE%📊 Resource Usage:%NC%
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | findstr -E "(ollama|postgres|qdrant|redis)"

echo.
echo %BLUE%💡 Recommendations:%NC%

if "%POSTGRES_RUNNING%"=="false" or "%OLLAMA_RUNNING%"=="false" (
    echo %YELLOW%• Run: npm run ai:start   (or docker-compose -f docker-compose-optimized.yml up -d)%NC%
)

if "%GPU_AVAILABLE%"=="false" (
    echo %YELLOW%• Update NVIDIA drivers for GPU acceleration%NC%
    echo %YELLOW%• Enable GPU support in Docker Desktop settings%NC%
)

echo %YELLOW%• For full development: npm run dev%NC%
echo %YELLOW%• For monitoring: npm run monitor%NC%
echo %YELLOW%• For health checks: npm run health%NC%

echo.
echo %BLUE%Health check completed!%NC%
pause
