@echo off
title Docker Service Health Check - Fixed
color 0A

echo ========================================
echo DOCKER SERVICE HEALTH CHECK - FIXED
echo ========================================
echo.

echo Checking Docker Desktop...
docker version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Docker Desktop not running
    echo Please start Docker Desktop first
    pause
    exit /b 1
)
echo ✅ Docker Desktop running

echo.
echo All running containers:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo Testing PostgreSQL (trying multiple containers)...
docker exec deeds-postgres pg_isready -U legal_admin 2>nul && echo ✅ deeds-postgres ready || (
    docker exec legal-postgres-optimized pg_isready -U legal_admin 2>nul && echo ✅ legal-postgres-optimized ready || echo ❌ No PostgreSQL ready
)

echo Testing Redis (trying multiple containers)...
docker exec deeds-redis redis-cli ping 2>nul && echo ✅ deeds-redis ready || (
    docker exec legal-redis-cluster redis-cli ping 2>nul && echo ✅ legal-redis-cluster ready || echo ❌ No Redis ready
)

echo Testing Qdrant (trying multiple containers)...
docker exec deeds-qdrant curl -s http://localhost:6333/health 2>nul && echo ✅ deeds-qdrant ready || (
    docker exec legal-qdrant-optimized curl -s http://localhost:6333/health 2>nul && echo ✅ legal-qdrant-optimized ready || echo ❌ No Qdrant ready
)

echo Testing Ollama (trying multiple containers)...
docker exec deeds-ollama curl -s http://localhost:11434/api/version 2>nul && echo ✅ deeds-ollama ready || (
    docker exec legal-ollama-gpu curl -s http://localhost:11434/api/version 2>nul && echo ✅ legal-ollama-gpu ready || echo ❌ No Ollama ready
)

echo Testing RabbitMQ...
docker exec deeds-rabbitmq rabbitmq-diagnostics ping 2>nul && echo ✅ deeds-rabbitmq ready || echo ❌ RabbitMQ down

echo.
echo Service endpoints:
echo - PostgreSQL: localhost:5432
echo - Redis: localhost:6379  
echo - Qdrant: localhost:6333
echo - Ollama: localhost:11434 or 11435
echo - RabbitMQ: localhost:15672 (management)

pause
