@echo off
title AI Stack Health Check
echo [1/6] Docker services...
docker ps --format "table {{.Names}}\t{{.Status}}"

echo [2/6] Nomic embed model...
docker exec deeds-ollama ollama list | findstr "nomic-embed-text" && echo ✅ Nomic ready || echo ❌ Missing nomic-embed-text

echo [3/6] Gemma3 legal model...
docker exec deeds-ollama ollama list | findstr "gemma3-legal" && echo ✅ Gemma3 ready || echo ❌ Missing gemma3-legal

echo [4/6] pgvector extension...
docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "SELECT extname FROM pg_extension WHERE extname='vector';" | findstr vector && echo ✅ pgvector ready || echo ❌ pgvector missing

echo [5/6] TypeScript check...
cd sveltekit-frontend
npm run check --silent

echo [6/6] Testing AI analysis...
curl -X POST http://localhost:11434/api/generate -d "{\"model\":\"gemma3-legal\",\"prompt\":\"Test legal analysis\",\"stream\":false}" && echo ✅ AI responsive

echo.
echo Health check complete
pause
