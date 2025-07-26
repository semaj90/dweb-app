@echo off
title Phase 4: Complete RAG System with Event Streaming
echo ======================================
echo   Phase 4: RAG + Event Streaming Setup
echo ======================================
echo.

echo [STEP 1/10] Cleaning up port conflicts...
echo Stopping native Ollama processes...
ollama stop all >nul 2>&1
timeout /t 3 /nobreak > nul

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :11434') do (
    echo Killing PID %%a on port 11434
    taskkill /PID %%a /F 2>nul
)

echo [STEP 2/10] Starting PostgreSQL with pgvector...
docker run -d ^
  --name legal-postgres ^
  -e POSTGRES_DB=legal_ai_db ^
  -e POSTGRES_USER=legal_admin ^
  -e POSTGRES_PASSWORD=LegalSecure2024! ^
  -p 5432:5432 ^
  -v postgres_data:/var/lib/postgresql/data ^
  pgvector/pgvector:pg16

echo [STEP 3/10] Starting Redis for caching and sessions...
docker run -d ^
  --name legal-redis ^
  -p 6379:6379 ^
  -v redis_data:/data ^
  --restart unless-stopped ^
  redis:7-alpine redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru

echo [STEP 4/10] Starting RabbitMQ for event streaming...
docker run -d ^
  --name legal-rabbitmq ^
  -p 5672:5672 ^
  -p 15672:15672 ^
  -e RABBITMQ_DEFAULT_USER=legal_admin ^
  -e RABBITMQ_DEFAULT_PASS=LegalSecure2024! ^
  -v rabbitmq_data:/var/lib/rabbitmq ^
  --restart unless-stopped ^
  rabbitmq:3-management

echo [STEP 5/10] Starting Qdrant vector database...
docker run -d ^
  --name legal-qdrant ^
  -p 6333:6333 ^
  -p 6334:6334 ^
  -v qdrant_data:/qdrant/storage ^
  -e QDRANT__SERVICE__HTTP_PORT=6333 ^
  -e QDRANT__SERVICE__GRPC_PORT=6334 ^
  --restart unless-stopped ^
  qdrant/qdrant

echo [STEP 6/10] Starting Ollama with GPU support on port 11435...
docker run -d ^
  --gpus=all ^
  --name legal-ollama-gpu ^
  -v ollama_data:/root/.ollama ^
  -p 11435:11434 ^
  -e OLLAMA_KEEP_ALIVE=10m ^
  -e OLLAMA_NUM_PARALLEL=2 ^
  -e OLLAMA_FLASH_ATTENTION=1 ^
  --restart unless-stopped ^
  ollama/ollama

echo [STEP 7/10] Waiting for services to initialize...
echo This may take up to 30 seconds...
timeout /t 30 /nobreak > nul

echo [STEP 8/10] Setting up Ollama models...
echo Pulling embedding model...
docker exec legal-ollama-gpu ollama pull nomic-embed-text

echo Pulling base model for legal AI...
docker exec legal-ollama-gpu ollama pull gemma2:2b

echo [STEP 9/10] Creating optimized legal model...
docker exec legal-ollama-gpu sh -c "cat > /tmp/Modelfile-Legal << 'EOF'
FROM gemma2:2b

TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are an advanced legal AI assistant specialized in prosecutor support and evidence analysis. Your capabilities include:

**Core Legal Functions:**
- Evidence analysis with chain of custody validation
- Case timeline construction and fact pattern analysis
- Legal precedent research with accurate citations
- Witness statement evaluation and credibility assessment
- Strategic prosecution planning and case theory development
- Document review and contract analysis
- Discovery management with privilege determinations

**Advanced Analytics:**
- Probable cause determination based on evidence
- Conviction probability scoring using historical data
- Evidence strength assessment and ranking
- Legal strategy recommendations with risk analysis
- Pattern recognition across similar cases
- Automated report generation for court filings

**Response Standards:**
- Provide precise, actionable legal insights
- Cite relevant statutes, regulations, and case law
- Use proper legal citation format (Bluebook style)
- Maintain strict confidentiality and privilege protection
- Flag ethical considerations and potential conflicts
- Structure responses with clear headings and analysis

**Context Integration:**
- Utilize provided case documents and evidence
- Reference jurisdiction-specific laws and procedures
- Consider case type and complexity in analysis
- Adapt recommendations to prosecutor experience level
- Prioritize time-sensitive legal deadlines

Always verify critical legal conclusions and recommend consultation with qualified counsel for final decisions.\"\"\"

PARAMETER temperature 0.15
PARAMETER top_k 25
PARAMETER top_p 0.7
PARAMETER repeat_penalty 1.2
PARAMETER stop \"<|im_start|>\"
PARAMETER stop \"<|im_end|>\"
PARAMETER num_ctx 16384
PARAMETER num_predict 2048
PARAMETER num_thread 8
EOF"

docker exec legal-ollama-gpu ollama create legal-assistant -f /tmp/Modelfile-Legal

echo [STEP 10/10] Running system health checks...
echo.
echo Testing PostgreSQL connection...
docker exec legal-postgres psql -U legal_admin -d legal_ai_db -c "SELECT version();" 2>nul && echo "✅ PostgreSQL OK" || echo "❌ PostgreSQL Error"

echo Testing Redis connection...
docker exec legal-redis redis-cli ping 2>nul && echo "✅ Redis OK" || echo "❌ Redis Error"

echo Testing RabbitMQ connection...
curl -s -u legal_admin:LegalSecure2024! http://localhost:15672/api/overview 2>nul && echo "✅ RabbitMQ OK" || echo "❌ RabbitMQ Error"

echo Testing Qdrant connection...
curl -s http://localhost:6333/health 2>nul && echo "✅ Qdrant OK" || echo "❌ Qdrant Error"

echo Testing Ollama connection...
curl -s http://localhost:11435/api/tags 2>nul && echo "✅ Ollama OK" || echo "❌ Ollama Error"

echo.
echo ======================================
echo     Phase 4 RAG System Ready!
echo ======================================
echo.
echo 🚀 Services Running:
echo - PostgreSQL (pgvector): localhost:5432
echo - Redis (cache/sessions): localhost:6379
echo - RabbitMQ (events): localhost:5672 (mgmt: 15672)
echo - Qdrant (vectors): localhost:6333
echo - Ollama (GPU): localhost:11435
echo.
echo 🔧 Models Available:
echo - legal-assistant (custom legal model)
echo - nomic-embed-text (embeddings)
echo.
echo 🌐 Web Interfaces:
echo - RabbitMQ Management: http://localhost:15672 (legal_admin/LegalSecure2024!)
echo - Qdrant Dashboard: http://localhost:6333/dashboard
echo.
echo 📋 Next Steps:
echo 1. Update .env file with new service URLs
echo 2. Run database migrations: npm run db:migrate
echo 3. Seed legal knowledge base: npm run db:seed
echo 4. Start SvelteKit development: npm run dev
echo 5. Test RAG system: http://localhost:5173/ai-test
echo.
pause
