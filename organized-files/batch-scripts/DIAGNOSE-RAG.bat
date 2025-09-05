@echo off
title 🔍 RAG System Diagnostics
color 0B

echo.
echo ================================================================
echo  🔍 COMPREHENSIVE RAG SYSTEM DIAGNOSTICS
echo ================================================================
echo.

echo [PORTS] Checking port usage...
echo Port 11434 (Ollama):
netstat -ano | findstr :11434
echo.
echo Port 5432 (PostgreSQL):
netstat -ano | findstr :5432
echo.
echo Port 6379 (Redis):
netstat -ano | findstr :6379
echo.

echo [PROCESSES] Checking running processes...
echo Native Ollama processes:
tasklist | findstr ollama.exe
echo.

echo [DOCKER] Checking Docker containers...
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

echo [MODELS] Checking available models...
echo Attempting to list models via API:
curl -s http://localhost:11434/api/tags 2>nul || echo "❌ API not accessible"
echo.

echo Attempting to list models via Docker:
docker exec ollama-gpu ollama list 2>nul || echo "❌ Docker container not accessible"
echo.

echo [HEALTH] Testing API endpoints...
echo Testing health endpoint:
curl -s http://localhost:11434/api/version 2>nul || echo "❌ Health check failed"
echo.

echo [GPU] Checking GPU status...
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>nul || echo "❌ GPU not accessible"
echo.

echo [VOLUMES] Checking Docker volumes...
docker volume ls | findstr ollama
echo.

echo ================================================================
echo  🔍 DIAGNOSTIC COMPLETE
echo ================================================================
echo.
echo Press any key to exit...
pause >nul
