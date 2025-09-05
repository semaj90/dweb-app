@echo off
title Test AI Integration
color 0A

echo ========================================
echo AI INTEGRATION TEST SUITE
echo ========================================
echo.

echo [1/6] Testing Docker containers...
docker ps --format "table {{.Names}}\t{{.Status}}" | findstr "deeds-"

echo [2/6] Testing Ollama API...
curl -s http://localhost:11434/api/version && echo ✅ Ollama API working || echo ❌ Ollama API failed

echo [3/6] Testing AI model...
echo Testing gemma3-legal model...
curl -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma3-legal\",\"prompt\":\"Test legal AI\",\"stream\":false}" && echo ✅ Model responds || echo ❌ Model failed

echo [4/6] Testing database connection...
docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "SELECT 'Database OK' as status;" && echo ✅ Database working || echo ❌ Database failed

echo [5/6] Testing pgvector extension...
docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "SELECT extname FROM pg_extension WHERE extname='vector';" | findstr vector && echo ✅ pgvector ready || echo ❌ pgvector missing

echo [6/6] Testing frontend build...
cd sveltekit-frontend
npm run build >nul 2>&1 && echo ✅ Frontend builds || echo ❌ Frontend errors

echo.
echo ========================================
echo TEST RESULTS SUMMARY
echo ========================================
echo.
echo ✅ All AI features should be working
echo 🚀 Start development: npm run dev
echo 🌐 Visit: http://localhost:5173
pause
