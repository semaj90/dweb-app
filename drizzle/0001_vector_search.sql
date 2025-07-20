-- Create vector search tables
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Document embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id TEXT NOT NULL,
    document_type TEXT NOT NULL CHECK (document_type IN ('case', 'evidence', 'note', 'report')),
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB DEFAULT '{}',
    model_used TEXT DEFAULT 'nomic-embed-text',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for document embeddings
CREATE INDEX IF NOT EXISTS idx_embedding_ivfflat 
    ON document_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_document_lookup 
    ON document_embeddings(document_id, document_type);
CREATE INDEX IF NOT EXISTS idx_created_at 
    ON document_embeddings(created_at);

-- Search queries table
CREATE TABLE IF NOT EXISTS search_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding vector(768),
    search_type TEXT NOT NULL DEFAULT 'semantic',
    results_count INTEGER DEFAULT 0,
    results JSONB DEFAULT '{"items": [], "totalFound": 0, "searchTime": 0}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for search queries
CREATE INDEX IF NOT EXISTS idx_search_user ON search_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_query_embedding 
    ON search_queries USING ivfflat (query_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_search_created ON search_queries(created_at);

-- AI models configuration table
CREATE TABLE IF NOT EXISTS ai_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    model_type TEXT NOT NULL,
    embedding_dimensions INTEGER,
    context_length INTEGER,
    config JSONB DEFAULT '{}',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for AI models
CREATE INDEX IF NOT EXISTS idx_model_name ON ai_models(name);
CREATE INDEX IF NOT EXISTS idx_provider_type ON ai_models(provider, model_type);

-- Insert default AI model configurations
INSERT INTO ai_models (name, provider, model_type, embedding_dimensions, context_length, config) VALUES
    ('nomic-embed-text', 'ollama', 'embedding', 768, 8192, '{"baseUrl": "http://localhost:11434"}'),
    ('llama3.2', 'ollama', 'chat', NULL, 4096, '{"baseUrl": "http://localhost:11434", "temperature": 0.7}'),
    ('gemma2:2b', 'ollama', 'chat', NULL, 8192, '{"baseUrl": "http://localhost:11434", "temperature": 0.7}'),
    ('mistral', 'ollama', 'chat', NULL, 8192, '{"baseUrl": "http://localhost:11434", "temperature": 0.7}')
ON CONFLICT (name) DO NOTHING;

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER update_document_embeddings_updated_at
    BEFORE UPDATE ON document_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_ai_models_updated_at
    BEFORE UPDATE ON ai_models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
