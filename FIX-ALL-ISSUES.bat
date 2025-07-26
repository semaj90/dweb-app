@echo off
title Fix All Issues
color 0A

echo [1/5] Starting PostgreSQL container...
docker start deeds-postgres
timeout /t 5 >nul

echo [2/5] Creating pgvector extension...
docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo [3/5] Checking Ollama models...
docker exec legal-ollama-gpu ollama list

echo [4/5] Running database setup...
cd sveltekit-frontend
npm run db:migrate
npm run seed

echo [5/5] Running TypeScript check...
npm run check

echo ✅ All fixes complete
pause
