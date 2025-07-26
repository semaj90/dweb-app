@echo off
title Docker Service Health Check
color 0A

echo ========================================
echo DOCKER SERVICE HEALTH CHECK
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

echo Checking services...
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo Testing PostgreSQL...
docker exec legal-ai-postgres pg_isready -U legal_admin 2>nul && echo ✅ PostgreSQL ready || echo ❌ PostgreSQL down

echo Testing Redis...
docker exec legal-ai-redis redis-cli ping 2>nul && echo ✅ Redis ready || echo ❌ Redis down

echo Testing Qdrant...
curl -s http://localhost:6333/health 2>nul && echo ✅ Qdrant ready || echo ❌ Qdrant down

pause
