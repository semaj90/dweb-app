@echo off
title Port 11434 Cleanup and RAG System Fix
echo ======================================
echo      Port 11434 Cleanup & RAG Fix
echo ======================================
echo.

echo [STEP 1/7] Identifying processes using port 11434...
netstat -ano | findstr :11434
echo.

echo [STEP 2/7] Stopping native Ollama processes...
echo Attempting to gracefully stop Ollama...
ollama stop all >nul 2>&1

echo Waiting 5 seconds for graceful shutdown...
timeout /t 5 /nobreak > nul

echo Checking if port is still in use...
netstat -ano | findstr :11434 > port_check.tmp
if %errorlevel% equ 0 (
    echo Port still in use, force killing processes...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :11434') do (
        echo Killing PID %%a
        taskkill /PID %%a /F 2>nul
    )
) else (
    echo Port 11434 is now free!
)
del port_check.tmp 2>nul

echo [STEP 3/7] Cleaning up any stuck Docker containers...
docker stop deeds-ollama-gpu 2>nul
docker rm deeds-ollama-gpu 2>nul
docker stop deeds-postgres 2>nul
docker rm deeds-postgres 2>nul
docker stop deeds-redis 2>nul
docker rm deeds-redis 2>nul
docker stop deeds-qdrant 2>nul
docker rm deeds-qdrant 2>nul

echo [STEP 4/7] Starting PostgreSQL with pgvector...
docker run -d ^
  --name deeds-postgres ^
  -e POSTGRES_DB=deeds_legal ^
  -e POSTGRES_USER=legal_user ^
  -e POSTGRES_PASSWORD=legal_pass_2024 ^
  -p 5432:5432 ^
  -v postgres_data:/var/lib/postgresql/data ^
  pgvector/pgvector:pg16

echo [STEP 5/7] Starting Redis for caching...
docker run -d ^
  --name deeds-redis ^
  -p 6379:6379 ^
  -v redis_data:/data ^
  redis:7-alpine

echo [STEP 6/7] Starting Qdrant vector database...
docker run -d ^
  --name deeds-qdrant ^
  -p 6333:6333 ^
  -p 6334:6334 ^
  -v qdrant_data:/qdrant/storage ^
  qdrant/qdrant

echo [STEP 7/7] Starting Ollama with GPU support (port should be free now)...
docker run -d ^
  --gpus=all ^
  --name deeds-ollama-gpu ^
  -v ollama_data:/root/.ollama ^
  -p 11434:11434 ^
  ollama/ollama

echo.
echo ======================================
echo      Waiting for Services...
echo ======================================
timeout /t 20 /nobreak > nul

echo Testing service connectivity...
echo.

echo [TEST 1] PostgreSQL:
docker exec deeds-postgres psql -U legal_user -d deeds_legal -c "SELECT version();" 2>nul
if %errorlevel% equ 0 (
    echo ✅ PostgreSQL is ready
) else (
    echo ❌ PostgreSQL not ready yet
)

echo [TEST 2] Redis:
docker exec deeds-redis redis-cli ping 2>nul
if %errorlevel% equ 0 (
    echo ✅ Redis is ready
) else (
    echo ❌ Redis not ready yet
)

echo [TEST 3] Qdrant:
curl -s http://localhost:6333/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Qdrant is ready
) else (
    echo ❌ Qdrant not ready yet
)

echo [TEST 4] Ollama:
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Ollama is ready
) else (
    echo ❌ Ollama not ready yet - will configure models next
)

echo.
echo ======================================
echo      Setting up Ollama Models
echo ======================================

echo Pulling base models (this may take several minutes)...
docker exec deeds-ollama-gpu ollama pull gemma2:2b
docker exec deeds-ollama-gpu ollama pull nomic-embed-text

echo Creating legal-optimized model...
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
echo      Final System Test
echo ======================================

echo Testing legal model...
docker exec deeds-ollama-gpu ollama run gemma2-legal "What are the key elements of probable cause?"

echo.
echo ======================================
echo      🎉 RAG System Ready!
echo ======================================
echo.
echo ✅ Services Running:
echo   - PostgreSQL: localhost:5432 (DB: deeds_legal)
echo   - Redis: localhost:6379
echo   - Qdrant: localhost:6333
echo   - Ollama: localhost:11434 (Model: gemma2-legal)
echo.
echo 🔧 Next Steps:
echo   1. Run database migrations: npm run db:migrate
echo   2. Seed with legal knowledge: npm run db:seed
echo   3. Start SvelteKit: npm run dev
echo   4. Test at: http://localhost:5173/ai-test
echo.
echo 📋 Environment Variables (add to .env):
echo   DATABASE_URL=postgresql://legal_user:legal_pass_2024@localhost:5432/deeds_legal
echo   REDIS_URL=redis://localhost:6379
echo   QDRANT_URL=http://localhost:6333
echo   OLLAMA_URL=http://localhost:11434
echo.
pause
