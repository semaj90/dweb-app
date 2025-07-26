@echo off
title System Check
echo Checking GPU, memory, and database...

echo [1/4] Ollama GPU check:
docker exec deeds-ollama ollama list
nvidia-smi 2>nul && echo ✅ GPU available || echo ❌ No GPU detected

echo [2/4] Qdrant memory usage:
docker stats deeds-qdrant --no-stream --format "table {{.Container}}\t{{.MemUsage}}"

echo [3/4] Testing pgvector:
docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "SELECT version(); SELECT extname FROM pg_extension WHERE extname = 'vector';"

echo [4/4] Seeding database:
npm run seed

echo.
echo ✅ System check complete
echo Press any key to exit...
pause >nul
