@echo off
title RAG System with Alternative Ollama Port
echo ======================================
echo    RAG System (Ollama on Port 11435)
echo ======================================
echo.
echo Native Ollama is running on 11434, using 11435 for Docker Ollama
echo.

echo [STEP 1/6] Cleaning up any existing containers...
docker stop deeds-ollama-gpu-alt 2>nul
docker rm deeds-ollama-gpu-alt 2>nul
docker stop deeds-postgres 2>nul
docker rm deeds-postgres 2>nul
docker stop deeds-redis 2>nul
docker rm deeds-redis 2>nul
docker stop deeds-qdrant 2>nul
docker rm deeds-qdrant 2>nul

echo [STEP 2/6] Starting PostgreSQL with pgvector...
docker run -d ^
  --name deeds-postgres ^
  -e POSTGRES_DB=deeds_legal ^
  -e POSTGRES_USER=legal_user ^
  -e POSTGRES_PASSWORD=legal_pass_2024 ^
  -p 5432:5432 ^
  -v postgres_data:/var/lib/postgresql/data ^
  pgvector/pgvector:pg16

echo [STEP 3/6] Starting Redis for caching...
docker run -d ^
  --name deeds-redis ^
  -p 6379:6379 ^
  -v redis_data:/data ^
  redis:7-alpine

echo [STEP 4/6] Starting Qdrant vector database...
docker run -d ^
  --name deeds-qdrant ^
  -p 6333:6333 ^
  -p 6334:6334 ^
  -v qdrant_data:/qdrant/storage ^
  qdrant/qdrant

echo [STEP 5/6] Starting Ollama with GPU support on port 11435...
docker run -d ^
  --gpus=all ^
  --name deeds-ollama-gpu-alt ^
  -v ollama_data_alt:/root/.ollama ^
  -p 11435:11434 ^
  ollama/ollama

echo [STEP 6/6] Waiting for services to initialize...
timeout /t 15 /nobreak > nul

echo.
echo ======================================
echo      Setting up Ollama Models
echo ======================================

echo Pulling models on port 11435...
docker exec deeds-ollama-gpu-alt ollama pull gemma2:2b
docker exec deeds-ollama-gpu-alt ollama pull nomic-embed-text

echo Creating legal model...
docker exec deeds-ollama-gpu-alt sh -c "cat > /tmp/Modelfile << 'EOF'
FROM gemma2:2b

TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are a specialized legal AI assistant for prosecutors and legal professionals. You excel at evidence analysis, case timeline construction, legal precedent research, witness statement evaluation, strategic prosecution planning, and document review. Provide concise, actionable legal insights with proper citations and maintain professional standards.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_k 30
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 8192
PARAMETER num_predict 1024
EOF"

docker exec deeds-ollama-gpu-alt ollama create gemma2-legal -f /tmp/Modelfile

echo.
echo ======================================
echo      Testing System
echo ======================================

echo Testing services...
echo PostgreSQL:
docker exec deeds-postgres psql -U legal_user -d deeds_legal -c "SELECT 'OK' as status;" 2>nul

echo Redis:
docker exec deeds-redis redis-cli ping 2>nul

echo Qdrant:
curl -s http://localhost:6333/health 2>nul

echo Ollama (port 11435):
curl -s http://localhost:11435/api/tags 2>nul

echo Testing legal model:
docker exec deeds-ollama-gpu-alt ollama run gemma2-legal "What are the elements of probable cause?" | head -5

echo.
echo ======================================
echo      🎉 RAG System Ready!
echo ======================================
echo.
echo ✅ Services Running:
echo   - PostgreSQL: localhost:5432 (DB: deeds_legal)
echo   - Redis: localhost:6379
echo   - Qdrant: localhost:6333
echo   - Ollama: localhost:11435 (Model: gemma2-legal)
echo   - Native Ollama: localhost:11434 (unchanged)
echo.
echo 🔧 Update your .env file:
echo   OLLAMA_URL=http://localhost:11435
echo   DATABASE_URL=postgresql://legal_user:legal_pass_2024@localhost:5432/deeds_legal
echo   REDIS_URL=redis://localhost:6379
echo   QDRANT_URL=http://localhost:6333
echo.
echo 📋 Next Steps:
echo   1. Update .env with new OLLAMA_URL=http://localhost:11435
echo   2. Run: npm run db:migrate
echo   3. Run: npm run db:seed
echo   4. Run: npm run dev
echo   5. Test: http://localhost:5173/ai-test
echo.
pause
