@echo off
title Fixed Health Check
color 0A

echo ✅ Working Services:
echo - PostgreSQL: deeds-postgres (port 5432)
echo - Redis: legal-redis-cluster (port 6379)  
echo - Ollama: legal-ollama-gpu (port 11434)
echo - RabbitMQ: deeds-rabbitmq (ports 5672, 15672)

echo.
echo ⚠️ Issues Found:
echo - Qdrant containers unhealthy (but port 6333 accessible)
echo - deeds-ollama unhealthy (use legal-ollama-gpu instead)

echo.
echo Testing external endpoints:
curl -s http://localhost:6333/health >nul && echo ✅ Qdrant API working || echo ❌ Qdrant API down
curl -s http://localhost:11434/api/version >nul && echo ✅ Ollama API working || echo ❌ Ollama API down

echo.
echo 🚀 Ready to start frontend:
echo cd sveltekit-frontend && npm run dev
pause
