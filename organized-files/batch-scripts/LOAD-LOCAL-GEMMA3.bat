@echo off
title Enhanced Local AI Stack Setup
echo [1/8] Checking Nomic Embed...
docker exec deeds-ollama ollama list | findstr "nomic-embed-text" >nul
if !errorlevel! neq 0 (
    echo Pulling nomic-embed-text...
    docker exec deeds-ollama ollama pull nomic-embed-text
)

echo [2/8] Creating Gemma3 legal model...
docker exec deeds-ollama ollama create gemma3-legal -f Gemma3-Legal-Enhanced-Modelfile-v2

echo [3/8] Installing LangChain dependencies...
cd sveltekit-frontend
npm install langchain @langchain/ollama @langchain/postgres ioredis

echo [4/8] Creating embedding service...
mkdir src\lib\services 2>nul
> src\lib\services\embedding-service.ts (
echo import { Ollama } from '@langchain/ollama';
echo import { Pool } from 'pg';
echo.
echo export class EmbeddingService {
echo   private embedModel = new Ollama({
echo     baseUrl: 'http://localhost:11434',
echo     model: 'nomic-embed-text'
echo   }^);
echo.
echo   private db = new Pool({
echo     connectionString: process.env.DATABASE_URL
echo   }^);
echo.
echo   async embedText(text: string^): Promise^<number[]^> {
echo     const [embedding] = await this.embedModel.embedDocuments([text]);
echo     return embedding;
echo   }
echo.
echo   async storeEmbedding(content: string, caseId: string, type: string^) {
echo     const embedding = await this.embedText(content^);
echo     await this.db.query(`
echo       INSERT INTO embeddings (content, embedding, case_id, type^)
echo       VALUES ($1, $2, $3, $4^)
echo     `, [content, JSON.stringify(embedding^), caseId, type]);
echo   }
echo.
echo   async vectorSearch(query: string, limit = 5^) {
echo     const queryEmbedding = await this.embedText(query^);
echo     const result = await this.db.query(`
echo       SELECT content, case_id, type,
echo              1 - (embedding::vector ^<^=^> $1::vector^) as similarity
echo       FROM embeddings
echo       ORDER BY embedding::vector ^<^=^> $1::vector
echo       LIMIT $2
echo     `, [JSON.stringify(queryEmbedding^), limit]);
echo     return result.rows;
echo   }
echo }
)

echo [5/8] Creating AI analysis service...
> src\lib\services\ai-analysis.ts (
echo import { Ollama } from '@langchain/ollama';
echo import { EmbeddingService } from './embedding-service.js';
echo.
echo export class AIAnalysisService {
echo   private llm = new Ollama({
echo     baseUrl: 'http://localhost:11434',
echo     model: 'gemma3-legal'
echo   }^);
echo.
echo   private embeddings = new EmbeddingService(^);
echo.
echo   async analyzeEvidence(content: string, caseId: string^) {
echo     // Store embedding
echo     await this.embeddings.storeEmbedding(content, caseId, 'evidence'^);
echo.
echo     // Get analysis
echo     const prompt = `Analyze this evidence for a legal case:\n\n${content}\n\nProvide key findings, relevance, and recommendations.`;
echo     return await this.llm.invoke(prompt^);
echo   }
echo.
echo   async contextualQuery(query: string, caseId?: string^) {
echo     const context = await this.embeddings.vectorSearch(query^);
echo     const contextText = context.map(c =^> `[${c.type}] ${c.content}`^).join('\n\n'^);
echo     const fullPrompt = `Context:\n${contextText}\n\nQuery: ${query}`;
echo     return await this.llm.invoke(fullPrompt^);
echo   }
echo }
)

echo [6/8] Adding embeddings table migration...
cd ..
> database\migrations\002_embeddings.sql (
echo CREATE EXTENSION IF NOT EXISTS vector;
echo.
echo CREATE TABLE IF NOT EXISTS embeddings ^(
echo   id UUID PRIMARY KEY DEFAULT gen_random_uuid(^),
echo   content TEXT NOT NULL,
echo   embedding vector(768^), -- nomic-embed-text dimension
echo   case_id UUID REFERENCES cases(id^),
echo   type VARCHAR(50^) NOT NULL,
echo   created_at TIMESTAMP DEFAULT NOW(^)
echo ^);
echo.
echo CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
echo ON embeddings USING ivfflat (embedding vector_cosine_ops^);
)

echo [7/8] Running migration...
docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -f /tmp/002_embeddings.sql

echo [8/8] Health check...
docker exec deeds-ollama ollama list
docker exec deeds-postgres psql -U legal_admin -d prosecutor_db -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';"

echo ✅ Enhanced AI stack ready
pause
