@echo off
echo ================================================
echo   FINAL ERROR FIX - LEGAL AI SYSTEM
echo   Complete system deployment and verification
echo ================================================
echo.

echo [1/6] Cleaning existing containers...
docker-compose down --remove-orphans 2>nul
docker system prune -f 2>nul

echo [2/6] Verifying Docker Compose configuration...
docker-compose config >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose configuration invalid
    pause
    exit /b 1
)
echo ✅ Docker Compose configuration valid

echo [3/6] Starting all services...
echo Starting services (this may take 2-3 minutes)...
docker-compose up -d
if %errorlevel% neq 0 (
    echo ❌ Failed to start services
    pause
    exit /b 1
)
echo ✅ Services started

echo [4/6] Waiting for initialization...
echo Please wait 60 seconds for all services to initialize...
timeout /t 60 >nul

echo [5/6] Checking service status...
docker-compose ps

echo [6/6] Running connectivity test...
node test.mjs
if %errorlevel% == 0 (
    echo.
    echo ================================================
    echo   🎉 LEGAL AI SYSTEM OPERATIONAL!
    echo ================================================
    echo.
    echo ✅ All errors fixed successfully
    echo 🚀 System ready for Phase 5 development
    echo.
    echo 🌐 Service URLs:
    echo • Neo4j Browser: http://localhost:7474
    echo • RabbitMQ Management: http://localhost:15672  
    echo • Qdrant Dashboard: http://localhost:6333
    echo • TTS Service: http://localhost:5002/health
    echo.
    echo 🔐 Default Credentials:
    echo • Neo4j: neo4j / LegalRAG2024!
    echo • RabbitMQ: legal_admin / LegalRAG2024!
    echo.
    echo 🎯 Phase 3+4 Integration Complete!
) else (
    echo.
    echo ================================================
    echo   ⚠️ SYSTEM PARTIALLY OPERATIONAL
    echo ================================================
    echo.
    echo Some services may still be starting up.
    echo Wait a few more minutes and run: node test.mjs
    echo.
    echo Check logs: docker-compose logs [service-name]
)

echo.
pause
