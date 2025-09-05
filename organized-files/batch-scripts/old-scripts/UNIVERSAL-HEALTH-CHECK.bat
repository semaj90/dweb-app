@echo off
title Universal Container Health Check
color 0A

echo Checking Docker Desktop...
docker version >nul 2>&1 || (echo ❌ Docker not available & pause & exit /b 1)
echo ✅ Docker available

echo.
echo All containers:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo Service health checks:

:: PostgreSQL - check all possible containers
for %%c in (deeds-postgres legal-postgres-optimized legal-ai-postgres) do (
    docker exec %%c pg_isready -U legal_admin 2>nul && (
        echo ✅ PostgreSQL: %%c ready
        goto redis_check
    )
)
echo ❌ No PostgreSQL container responding

:redis_check
:: Redis - check all possible containers  
for %%c in (deeds-redis legal-redis-cluster legal-ai-redis) do (
    docker exec %%c redis-cli ping 2>nul && (
        echo ✅ Redis: %%c ready
        goto qdrant_check
    )
)
echo ❌ No Redis container responding

:qdrant_check
:: Qdrant - check all possible containers
for %%c in (deeds-qdrant legal-qdrant-optimized legal-ai-qdrant) do (
    docker exec %%c curl -s http://localhost:6333/health 2>nul && (
        echo ✅ Qdrant: %%c ready
        goto ollama_check
    )
)
echo ❌ No Qdrant container responding

:ollama_check
:: Ollama - check all possible containers
for %%c in (deeds-ollama legal-ollama-gpu legal-ai-ollama) do (
    docker exec %%c curl -s http://localhost:11434/api/version 2>nul && (
        echo ✅ Ollama: %%c ready
        goto end_check
    )
)
echo ❌ No Ollama container responding

:end_check
:: RabbitMQ if present
docker exec deeds-rabbitmq rabbitmq-diagnostics ping 2>nul && echo ✅ RabbitMQ: deeds-rabbitmq ready

echo.
echo Available endpoints:
echo - Database: localhost:5432
echo - Cache: localhost:6379
echo - Vector DB: localhost:6333
echo - AI Model: localhost:11434
pause
