# Legal AI API Documentation

## Tech Stack
- **Frontend/Backend**: SvelteKit 2
- **Database**: PostgreSQL with pgvector extension
- **ORM**: Drizzle ORM
- **AI/LLM**: Ollama + llama.cpp
- **AI Bridge**: LangChain.js
- **Vector Store**: PostgreSQL pgvector
- **Response Format**: JSON

## API Endpoints

### Document Analysis
```typescript
// POST /api/documents/analyze
interface AnalyzeRequest {
  title: string;
  content: string;
  documentType?: 'contract' | 'motion' | 'brief' | 'statute' | 'case' | 'general';
  jurisdiction?: 'federal' | 'state' | 'local';
  practiceArea?: 'criminal' | 'civil' | 'corporate' | 'family' | 'immigration' | 'tax';
}

interface AnalyzeResponse {
  success: boolean;
  documentId: string;
  analysis: {
    entities: Array<{ type: string; value: string; confidence: number }>;
    keyTerms: string[];
    risks: Array<{ type: string; severity: 'low' | 'medium' | 'high'; description: string }>;
    sentimentScore: number; // -1 to 1
    complexityScore: number; // 1 to 10
    parties: string[];
    obligations: string[];
    extractedDates: string[];
    extractedAmounts: string[];
    confidenceLevel: number;
    processingTime: number;
    llmModel: string; // e.g., "llama3.2:3b"
  };
  embeddings: {
    contentVector: number[]; // pgvector compatible
    titleVector: number[];
    dimensions: number; // 384 for nomic-embed-text
  };
  processing: {
    status: 'completed' | 'processing' | 'failed';
    processingTime: number;
    chunksProcessed: number;
  };
}
```

### Document Search with pgvector
```typescript
// POST /api/documents/search
interface SearchRequest {
  query: string;
  searchType: 'semantic' | 'full-text' | 'hybrid';
  limit?: number;
  offset?: number;
  filters?: {
    documentType?: string;
    jurisdiction?: string;
    practiceArea?: string;
    dateRange?: { start: Date; end: Date };
    tags?: string[];
  };
  similarityThreshold?: number; // 0.0 to 1.0 for pgvector cosine similarity
}

interface SearchResponse {
  success: boolean;
  results: Array<{
    score: number; // pgvector similarity score
    rank: number;
    id: string;
    title: string;
    excerpt: string;
    documentType: string;
    jurisdiction: string;
    practiceArea: string;
    metadata: Record<string, any>;
    createdAt: string;
    similarity: number; // cosine similarity from pgvector
  }>;
  metadata: {
    query: string;
    searchType: string;
    totalResults: number;
    processingTime: number;
    llmModel: string;
    embeddingModel: string;
    vectorDimensions: number;
  };
  pagination: {
    hasMore: boolean;
    nextOffset: number;
  };
}
```

### AI Analysis with Ollama
```typescript
// POST /api/ai/analyze
interface OllamaAnalysisRequest {
  title: string;
  content: string;
  model?: string; // e.g., "llama3.2:3b", "mistral:7b"
  temperature?: number; // 0.0 to 1.0
  systemPrompt?: string;
  analysisType?: 'comprehensive' | 'quick' | 'focused';
}

interface OllamaAnalysisResponse {
  success: boolean;
  analysis: {
    summary: string;
    keyInsights: string[];
    legalEntities: Array<{ type: string; value: string; confidence: number }>;
    riskAssessment: {
      overallRisk: 'low' | 'medium' | 'high';
      specificRisks: Array<{ area: string; level: string; description: string }>;
    };
    recommendations: string[];
    compliance: {
      issues: string[];
      suggestions: string[];
    };
    citations: Array<{ text: string; type: 'case' | 'statute' | 'regulation' }>;
  };
  llmMetadata: {
    model: string;
    temperature: number;
    tokensUsed: number;
    responseTime: number;
    ollamaVersion: string;
  };
}
```

### Embeddings with pgvector
```typescript
// POST /api/ai/embeddings
interface EmbeddingRequest {
  text: string;
  model?: string; // e.g., "nomic-embed-text"
  documentId?: string;
  chunkIndex?: number;
  metadata?: Record<string, any>;
}

interface EmbeddingResponse {
  success: boolean;
  embedding: number[]; // 384-dimensional vector for nomic-embed-text
  dimensions: number;
  model: string;
  savedRecord?: {
    id: string;
    vectorId: string; // pgvector record ID
    similarity?: number; // if comparing to existing vectors
  };
  processing: {
    time: number;
    method: 'ollama' | 'langchain';
  };
}
```

### Legal Research with LangChain.js
```typescript
// POST /api/ai/legal-research
interface LegalResearchRequest {
  query: string;
  jurisdiction?: string;
  practiceArea?: string;
  documentTypes?: string[];
  limit?: number;
  llmModel?: string;
  embeddingModel?: string;
  useReranking?: boolean;
}

interface LegalResearchResponse {
  success: boolean;
  results: Array<{
    document: {
      id: string;
      title: string;
      content: string;
      type: string;
      jurisdiction: string;
    };
    similarity: number; // pgvector cosine similarity
    relevanceScore: number; // LangChain reranking score
    explanation: string; // LLM-generated relevance explanation
    keyPassages: string[];
    citations: string[];
  }>;
  synthesis: {
    summary: string;
    keyFindings: string[];
    recommendations: string[];
    confidenceLevel: number;
  };
  metadata: {
    vectorSearch: {
      similarityThreshold: number;
      vectorDimensions: number;
      documentsScanned: number;
    };
    llmProcessing: {
      model: string;
      tokensUsed: number;
      responseTime: number;
    };
    langchainVersion: string;
  };
}
```

## Database Schema (Drizzle ORM + pgvector)

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Legal documents with vector embeddings
CREATE TABLE legal_documents (
  id SERIAL PRIMARY KEY,
  title VARCHAR(500) NOT NULL,
  content TEXT NOT NULL,
  document_type VARCHAR(100) DEFAULT 'general',
  jurisdiction VARCHAR(100) DEFAULT 'federal',
  practice_area VARCHAR(100),
  content_embedding vector(384), -- pgvector for semantic search
  title_embedding vector(384),
  metadata JSONB DEFAULT '{}',
  tags TEXT[],
  processing_status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for pgvector similarity search
CREATE INDEX legal_documents_content_embedding_idx 
ON legal_documents USING ivfflat (content_embedding vector_cosine_ops) 
WITH (lists = 100);

CREATE INDEX legal_documents_title_embedding_idx 
ON legal_documents USING ivfflat (title_embedding vector_cosine_ops) 
WITH (lists = 100);

-- Full-text search indexes
CREATE INDEX legal_documents_content_fts_idx 
ON legal_documents USING gin(to_tsvector('english', content));

CREATE INDEX legal_documents_title_fts_idx 
ON legal_documents USING gin(to_tsvector('english', title));
```

## LangChain.js Integration

```typescript
// Vector store using pgvector
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { Ollama } from "@langchain/community/llms/ollama";

// Initialize embeddings with Ollama
const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text",
  baseUrl: "http://localhost:11434"
});

// Initialize LLM with Ollama
const llm = new Ollama({
  model: "llama3.2:3b",
  baseUrl: "http://localhost:11434",
  temperature: 0.1
});

// Initialize vector store with pgvector
const vectorStore = await PGVectorStore.initialize(embeddings, {
  postgresConnectionOptions: {
    host: process.env.DB_HOST,
    port: parseInt(process.env.DB_PORT || "5432"),
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_NAME,
  },
  tableName: "legal_documents_vectors",
  columns: {
    idColumnName: "id",
    vectorColumnName: "embedding",
    contentColumnName: "content",
    metadataColumnName: "metadata",
  },
});
```

## Ollama Model Management

```bash
# Install and run models
ollama pull llama3.2:3b          # Main LLM for analysis
ollama pull mistral:7b           # Alternative LLM
ollama pull nomic-embed-text     # Embeddings model
ollama pull codellama:7b         # For code analysis in contracts

# List available models
ollama list

# Model info
ollama show gemma3 legal-latest 
```

## Search Query Examples

### Semantic Search with pgvector
```sql
-- Find similar documents using cosine similarity
SELECT 
  id, 
  title, 
  1 - (content_embedding <=> $1::vector) as similarity_score
FROM legal_documents 
WHERE 1 - (content_embedding <=> $1::vector) > 0.7
ORDER BY content_embedding <=> $1::vector
LIMIT 10;
```

### Hybrid Search (Full-text + Semantic)
```sql
-- Combine full-text and vector search
WITH semantic_results AS (
  SELECT id, title, content, 
         1 - (content_embedding <=> $1::vector) as semantic_score
  FROM legal_documents 
  WHERE 1 - (content_embedding <=> $1::vector) > 0.6
),
fulltext_results AS (
  SELECT id, title, content,
         ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)) as text_score
  FROM legal_documents 
  WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $2)
)
SELECT DISTINCT 
  COALESCE(s.id, f.id) as id,
  COALESCE(s.title, f.title) as title,
  COALESCE(s.content, f.content) as content,
  COALESCE(s.semantic_score, 0) * 0.7 + COALESCE(f.text_score, 0) * 0.3 as combined_score
FROM semantic_results s
FULL OUTER JOIN fulltext_results f ON s.id = f.id
ORDER BY combined_score DESC
LIMIT 20;
```

## Error Handling

```typescript
interface APIError {
  error: string;
  details?: string;
  code?: string;
  timestamp: string;
  requestId: string;
}

// Ollama-specific errors
interface OllamaError extends APIError {
  ollamaDetails: {
    modelNotFound?: boolean;
    connectionFailed?: boolean;
    modelLoading?: boolean;
    insufficientMemory?: boolean;
  };
}

// pgvector-specific errors
interface VectorError extends APIError {
  vectorDetails: {
    dimensionMismatch?: boolean;
    indexNotFound?: boolean;
    similarityThresholdTooLow?: boolean;
  };
}
```

## Performance Considerations

- **pgvector**: Use IVFFlat index for large datasets (>10k vectors)
- **Ollama**: Keep models loaded in memory for faster inference
- **LangChain.js**: Implement response streaming for long generations
- **Database**: Use connection pooling with Drizzle ORM
- **Chunking**: Optimal chunk size 500-1000 tokens for legal documents

## Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/legal_ai
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=123456
DB_NAME=legal_ai

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=gemma3-legal
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# LangChain
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=legal_ai

# Vector Search
VECTOR_SIMILARITY_THRESHOLD=0.7
VECTOR_DIMENSIONS=384
MAX_SEARCH_RESULTS=50
```
