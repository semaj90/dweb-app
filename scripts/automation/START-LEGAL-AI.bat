@echo off
echo 🚀 Legal AI System - Production Launch
echo =====================================

echo Setting up environment...
set POSTGRES_PASSWORD=LegalAI2024!
set OLLAMA_PARALLEL=2

echo Checking Docker Desktop...
docker version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Stopping any existing containers...
docker-compose -f docker-compose-unified.yml down --remove-orphans

echo Starting Legal AI services with GPU support...
docker-compose -f docker-compose-unified.yml --profile gpu up -d

echo Waiting for services to initialize...
timeout /t 30 /nobreak >nul

echo Checking service health...
docker-compose -f docker-compose-unified.yml ps

echo Creating database tables...
cd sveltekit-frontend
npm run db:migrate 2>nul || echo ⚠️ Database migration pending

echo ✅ Legal AI System Ready!
echo.
echo Services:
echo - Database: localhost:5432
echo - Redis: localhost:6379  
echo - Qdrant: localhost:6333
echo - Ollama: localhost:11434
echo - Frontend: npm run dev (port 5173)
echo.
pause