@echo off
title RAG System Diagnostics
echo ======================================
echo      RAG System Diagnostics
echo ======================================
echo.

echo [DIAGNOSTIC 1] Port Usage Check
echo Checking ports 5432, 6333, 6379, 11434...
echo.
echo PostgreSQL (5432):
netstat -ano | findstr :5432
echo.
echo Qdrant (6333):
netstat -ano | findstr :6333
echo.
echo Redis (6379):
netstat -ano | findstr :6379
echo.
echo Ollama (11434):
netstat -ano | findstr :11434
echo.

echo [DIAGNOSTIC 2] Docker Container Status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

echo [DIAGNOSTIC 3] Docker Images Available
docker images | findstr -i "ollama\|postgres\|redis\|qdrant"
echo.

echo [DIAGNOSTIC 4] Service Health Checks
echo.
echo Testing PostgreSQL connection...
docker exec deeds-postgres psql -U legal_user -d deeds_legal -c "SELECT 'PostgreSQL OK' as status;" 2>nul
if %errorlevel% neq 0 echo ❌ PostgreSQL connection failed

echo Testing Redis connection...
docker exec deeds-redis redis-cli ping 2>nul
if %errorlevel% neq 0 echo ❌ Redis connection failed

echo Testing Qdrant API...
curl -s http://localhost:6333/health
if %errorlevel% neq 0 echo ❌ Qdrant API not responding

echo Testing Ollama API...
curl -s http://localhost:11434/api/tags
if %errorlevel% neq 0 echo ❌ Ollama API not responding

echo.
echo [DIAGNOSTIC 5] Available Ollama Models
docker exec deeds-ollama-gpu ollama list 2>nul
if %errorlevel% neq 0 echo ❌ Could not list Ollama models

echo.
echo [DIAGNOSTIC 6] System Resources
echo Docker system info:
docker system df

echo.
echo [DIAGNOSTIC 7] Recent Container Logs (last 10 lines each)
echo.
echo === Ollama Logs ===
docker logs --tail 10 deeds-ollama-gpu 2>nul
echo.
echo === PostgreSQL Logs ===
docker logs --tail 10 deeds-postgres 2>nul
echo.
echo === Redis Logs ===
docker logs --tail 10 deeds-redis 2>nul
echo.
echo === Qdrant Logs ===
docker logs --tail 10 deeds-qdrant 2>nul

echo.
echo ======================================
echo      Diagnostic Complete
echo ======================================
pause
