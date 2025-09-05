# Local Gemma3 + Nomic Embed + LangChain + pgvector Setup

Write-Host "🚀 Setting up local AI stack" -ForegroundColor Cyan

# Install dependencies
Write-Host "📦 Installing dependencies..."
Set-Location sveltekit-frontend
npm install langchain @langchain/ollama @langchain/postgres @types/pg

# Create embedding service
$embeddingService = @"
import { Ollama } from '@langchain/ollama';
import { Pool } from 'pg';

export class EmbeddingService {
  private ollama: Ollama;
  private db: Pool;

  constructor() {
    this.ollama = new Ollama({
      baseUrl: 'http://localhost:11434',
      model: 'nomic-embed-text'
    });
    
    this.db = new Pool({
      connectionString: process.env.DATABASE_URL
    });
  }

  async embedText(text: string): Promise<number[]> {
    const embedding = await this.ollama.embedDocuments([text]);
    return embedding[0];
  }

  async storeEmbedding(text: string, caseId: string, type: string) {
    const embedding = await this.embedText(text);
    
    await this.db.query(`
      INSERT INTO embeddings (content, embedding, case_id, type, created_at)
      VALUES ($1, $2, $3, $4, NOW())
    `, [text, JSON.stringify(embedding), caseId, type]);
  }

  async similaritySearch(query: string, limit = 5) {
    const queryEmbedding = await this.embedText(query);
    
    const result = await this.db.query(`
      SELECT content, case_id, type,
             1 - (embedding::vector <=> $1::vector) as similarity
      FROM embeddings
      ORDER BY embedding::vector <=> $1::vector
      LIMIT $2
    `, [JSON.stringify(queryEmbedding), limit]);
    
    return result.rows;
  }
}
"@

$embeddingService | Out-File -FilePath "src/lib/services/embedding-service.ts" -Encoding UTF8

# Create AI service with local models
$aiService = @"
import { Ollama } from '@langchain/ollama';
import { EmbeddingService } from './embedding-service.js';

export class LocalAIService {
  private llm: Ollama;
  private embeddings: EmbeddingService;

  constructor() {
    this.llm = new Ollama({
      baseUrl: 'http://localhost:11434',
      model: 'gemma3-legal'
    });
    
    this.embeddings = new EmbeddingService();
  }

  async queryWithContext(prompt: string, caseId?: string) {
    // Get relevant context via similarity search
    const context = await this.embeddings.similaritySearch(prompt);
    
    const contextText = context
      .map(c => `[${c.type}] ${c.content}`)
      .join('\n\n');

    const fullPrompt = `Context:\n${contextText}\n\nQuery: ${prompt}`;
    
    const response = await this.llm.invoke(fullPrompt);
    return response;
  }

  async embedDocument(content: string, caseId: string, type: string) {
    await this.embeddings.storeEmbedding(content, caseId, type);
  }
}
"@

$aiService | Out-File -FilePath "src/lib/services/ai-service.ts" -Encoding UTF8

Write-Host "✅ Services created"

# Update database schema
Write-Host "📊 Adding embeddings table..."
Set-Location ..
$migration = @"
-- Add embeddings table for vector storage
CREATE TABLE IF NOT EXISTS embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT NOT NULL,
  embedding vector(768), -- nomic-embed-text dimension
  case_id UUID REFERENCES cases(id),
  type VARCHAR(50) NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create vector similarity index
CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
ON embeddings USING ivfflat (embedding vector_cosine_ops);
"@

$migration | Out-File -FilePath "database/migrations/002_embeddings.sql" -Encoding UTF8

Write-Host "✅ Migration created"

# Create model loading script
$loadModels = @"
@echo off
title Load Local Models
echo Loading local Gemma3 and Nomic models...

echo [1/3] Loading local Gemma3...
docker exec deeds-ollama ollama create gemma3-legal -f Gemma3-Legal-Modelfile

echo [2/3] Pulling Nomic embeddings...
docker exec deeds-ollama ollama pull nomic-embed-text

echo [3/3] Testing models...
docker exec deeds-ollama ollama list

echo ✅ Local AI stack ready
pause
"@

$loadModels | Out-File -FilePath "LOAD-LOCAL-MODELS.bat" -Encoding ASCII

Write-Host "✅ Setup complete"
Write-Host "Run: LOAD-LOCAL-MODELS.bat"
Write-Host "Then: npm run db:migrate"
