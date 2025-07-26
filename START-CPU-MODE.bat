@echo off
echo 🛠️ Legal AI CPU Mode (No GPU Required)
echo =====================================

set POSTGRES_PASSWORD=LegalAI2024!

docker version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Desktop not running
    pause
    exit /b 1
)

echo Starting CPU-only mode...
docker-compose -f docker-compose-unified.yml down
docker-compose -f docker-compose-unified.yml --profile cpu-only up -d

timeout /t 20 /nobreak >nul
echo ✅ CPU mode ready at localhost:11434
pause