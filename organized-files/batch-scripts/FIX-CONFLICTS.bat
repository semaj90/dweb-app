@echo off
title Fix Critical Errors
echo Fixing port conflicts and path issues...

echo [1/3] Stopping conflicting containers...
docker stop legal-redis-cluster legal-qdrant-optimized legal-ollama-gpu legal-postgres-optimized

echo [2/3] Removing orphaned containers...
docker-compose down --remove-orphans

echo [3/3] Starting clean services...
docker-compose up -d

echo ✅ Port conflicts resolved
pause
