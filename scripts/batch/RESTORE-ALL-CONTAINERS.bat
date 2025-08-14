@echo off
title Restore All Containers with GPU
echo Restoring all previous containers + GPU Ollama...

echo [1/5] Stopping current containers...
docker-compose down
docker stop deeds-ollama-gpu monitoring-prometheus-1 monitoring-grafana-1 2>nul

echo [2/5] Starting unified stack...
docker-compose -f docker-compose-unified-gpu.yml up -d

echo [3/5] Waiting for services...
timeout /t 30 >nul

echo [4/5] Health check...
echo PostgreSQL:
docker exec deeds-postgres pg_isready -U legal_admin && echo ✅ Ready

echo Redis:
docker exec deeds-redis redis-cli ping && echo ✅ Ready

echo Qdrant:
curl -s http://localhost:6333/health && echo ✅ Ready

echo RabbitMQ:
docker exec deeds-rabbitmq rabbitmq-diagnostics ping && echo ✅ Ready

echo Ollama GPU:
curl -s http://localhost:11434/api/version && echo ✅ Ready

echo [5/5] Loading your 3GB model...
docker cp Modelfile-LowMem deeds-ollama-gpu:/tmp/
docker cp gemma3Q4_K_M/mo16.gguf deeds-ollama-gpu:/tmp/
docker exec deeds-ollama-gpu ollama create gemma3-legal -f /tmp/Modelfile-LowMem

echo.
echo ✅ ALL CONTAINERS RESTORED:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

pause
