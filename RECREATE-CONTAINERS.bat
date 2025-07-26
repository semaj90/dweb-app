@echo off
title Recreate Docker Containers
echo Recreating standard Docker containers...

echo [1/4] Starting core services...
docker-compose up -d postgres redis qdrant

echo [2/4] Checking service health...
timeout /t 15 >nul
docker ps --format "table {{.Names}}\t{{.Status}}"

echo [3/4] Testing database...
docker exec deeds-postgres pg_isready -U legal_admin && echo ✅ PostgreSQL ready

echo [4/4] Testing Redis...
docker exec deeds-redis redis-cli ping && echo ✅ Redis ready

echo.
echo Standard containers recreated!
echo GPU containers from docker-compose-gpu.yml are separate.
pause
