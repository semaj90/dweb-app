@echo off
title Complete RAG Legal AI Setup
echo ======================================
echo   Complete RAG Legal AI System Setup
echo ======================================
echo.

echo [STEP 1/8] Cleaning up existing containers and processes...
docker stop deeds-ollama-gpu 2>nul
docker rm deeds-ollama-gpu 2>nul
docker stop deeds-postgres 2>nul
docker rm deeds-postgres 2>nul
docker stop deeds-redis 2>nul
docker rm deeds-redis 2>nul
docker stop deeds-qdrant 2>nul
docker rm deeds-qdrant 2>nul

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :11434') do (
    taskkill /PID %%a /F 2>nul
)

echo [STEP 2/8] Starting PostgreSQL with pgvector...
docker run -d ^
  --name deeds-postgres ^
  -e POSTGRES_DB=deeds_legal ^
  -e POSTGRES_USER=legal_user ^
  -e POSTGRES_PASSWORD=legal_pass_2024 ^
  -p 5432:5432 ^
  -v postgres_data:/var/lib/postgresql/data ^
  pgvector/pgvector:pg16

echo [STEP 3/8] Starting Redis for caching...
docker run -d ^
  --name deeds-redis ^
  -p 6379:6379 ^
  -v redis_data:/data ^
  redis:7-alpine

echo [STEP 4/8] Starting Qdrant vector database...
docker run -d ^
  --name deeds-qdrant ^
  -p 6333:6333 ^
  -p 6334:6334 ^
  -v qdrant_data:/qdrant/storage ^
  qdrant/qdrant

echo [STEP 5/8] Starting Ollama with GPU support...
docker run -d ^
  --gpus=all ^
  --name deeds-ollama-gpu ^
  -v ollama_data:/root/.ollama ^
  -p 11434:11434 ^
  ollama/ollama

echo [STEP 6/8] Waiting for services to initialize...
timeout /t 15 /nobreak > nul

echo [STEP 7/8] Setting up Ollama models...
docker exec deeds-ollama-gpu ollama pull gemma2:2b
docker exec deeds-ollama-gpu ollama pull nomic-embed-text

echo [STEP 8/8] Creating optimized legal model...
docker exec deeds-ollama-gpu sh -c "cat > /tmp/Modelfile-Legal << 'EOF'
FROM gemma2:2b

TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are a specialized legal AI assistant for prosecutors and legal professionals. You excel at:

**Core Competencies:**
- Evidence analysis and chain of custody review
- Case timeline construction and fact pattern analysis
- Legal precedent research and citation formatting
- Witness statement evaluation and credibility assessment
- Strategic prosecution planning and case theory development
- Document summarization and contract review
- Discovery management and privilege determinations

**Response Guidelines:**
- Provide concise, actionable legal insights
- Cite relevant statutes, cases, and rules when applicable
- Maintain professional legal writing standards
- Respect attorney-client privilege and confidentiality
- Flag potential ethical considerations
- Structure responses with clear headings and bullet points

**Context Awareness:**
- Consider jurisdiction-specific laws and procedures
- Account for case type (criminal, civil, regulatory)
- Adapt complexity to user expertise level
- Prioritize time-sensitive legal deadlines

Always maintain accuracy and encourage verification of legal conclusions with qualified counsel.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_k 30
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.15
PARAMETER stop \"<|im_start|>\"
PARAMETER stop \"<|im_end|>\"
PARAMETER num_ctx 8192
PARAMETER num_predict 1024
PARAMETER num_thread 8
EOF"

docker exec deeds-ollama-gpu ollama create gemma2-legal -f /tmp/Modelfile-Legal

echo.
echo ======================================
echo         Testing Complete System
echo ======================================

echo Testing PostgreSQL...
docker exec deeds-postgres psql -U legal_user -d deeds_legal -c "SELECT version();"

echo Testing Redis...
docker exec deeds-redis redis-cli ping

echo Testing Qdrant...
curl -s http://localhost:6333/health

echo Testing Ollama...
docker exec deeds-ollama-gpu ollama list

echo Testing Legal Model...
docker exec deeds-ollama-gpu ollama run gemma2-legal "What are the key elements of probable cause in criminal law?"

echo.
echo ======================================
echo      RAG System Ready!
echo ======================================
echo.
echo Services Running:
echo - PostgreSQL: localhost:5432 (DB: deeds_legal)
echo - Redis: localhost:6379
echo - Qdrant: localhost:6333
echo - Ollama: localhost:11434 (Model: gemma2-legal)
echo.
echo Next Steps:
echo 1. Run database migrations: npm run db:migrate
echo 2. Start SvelteKit: npm run dev
echo 3. Test at: http://localhost:5173/ai-test
echo.
echo Environment Variables:
echo DATABASE_URL=postgresql://legal_user:legal_pass_2024@localhost:5432/deeds_legal
echo REDIS_URL=redis://localhost:6379
echo QDRANT_URL=http://localhost:6333
echo OLLAMA_URL=http://localhost:11434
echo.
pause
