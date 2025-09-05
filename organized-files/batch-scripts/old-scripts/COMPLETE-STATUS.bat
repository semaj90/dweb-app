@echo off
title Complete System Status
color 0A

echo ✅ WORKING SERVICES:
echo - PostgreSQL: deeds-postgres (5432)
echo - Redis: legal-redis-cluster (6379) 
echo - Ollama: legal-ollama-gpu (11434)
echo - RabbitMQ: deeds-rabbitmq (5672, 15672)
echo - Qdrant: legal-qdrant-optimized (6333)
echo - Prometheus: monitoring-prometheus-1 (9090)
echo - Grafana: monitoring-grafana-1 (3001)

echo.
echo 🎯 ACCESS POINTS:
echo - App: http://localhost:5173 (start with npm run dev)
echo - Grafana: http://localhost:3001
echo - Prometheus: http://localhost:9090
echo - RabbitMQ: http://localhost:15672

echo.
echo 🚀 START FRONTEND:
echo cd sveltekit-frontend && npm run dev

pause
