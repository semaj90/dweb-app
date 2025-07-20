-- Initial schema setup for Legal AI Assistant
-- This creates vector-enabled tables for embeddings

-- Ensure vector extension is loaded
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector search configuration
CREATE TABLE IF NOT EXISTS vector_search_config (
  id SERIAL PRIMARY KEY,
  model_name VARCHAR(100) NOT NULL,
  embedding_dimension INTEGER NOT NULL,
  chunk_size INTEGER DEFAULT 1000,
  chunk_overlap INTEGER DEFAULT 200,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configurations for different embedding models
INSERT INTO vector_search_config (model_name, embedding_dimension) VALUES
  ('nomic-embed-text', 768),
  ('mxbai-embed-large', 1024),
  ('all-minilm', 384),
  ('bge-base', 768)
ON CONFLICT DO NOTHING;

-- Create document embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  document_id VARCHAR(255) NOT NULL,
  document_type VARCHAR(50) NOT NULL, -- 'case', 'evidence', 'note', 'template'
  chunk_index INTEGER NOT NULL,
  chunk_text TEXT NOT NULL,
  embedding vector(768), -- Default to 768 dimensions, can be altered
  metadata JSONB DEFAULT '{}',
  model_used VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT check_embedding_dim CHECK (
    embedding IS NULL OR array_length(embedding::real[], 1) = 768
  )
);

-- Create indexes for document embeddings
CREATE INDEX IF NOT EXISTS idx_document_embeddings_document 
  ON document_embeddings(document_id, document_type);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_created 
  ON document_embeddings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_metadata 
  ON document_embeddings USING gin(metadata);

-- Create semantic search history
CREATE TABLE IF NOT EXISTS search_history (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id VARCHAR(255) NOT NULL,
  query_text TEXT NOT NULL,
  query_embedding vector(768),
  results_count INTEGER,
  results JSONB DEFAULT '[]',
  search_type VARCHAR(50) DEFAULT 'semantic',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to perform semantic search
CREATE OR REPLACE FUNCTION semantic_search(
  query_embedding vector,
  search_type text DEFAULT 'all',
  limit_results integer DEFAULT 10,
  similarity_threshold float DEFAULT 0.7
)
RETURNS TABLE (
  id UUID,
  document_id VARCHAR(255),
  document_type VARCHAR(50),
  chunk_text TEXT,
  similarity float,
  metadata JSONB
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    de.id,
    de.document_id,
    de.document_type,
    de.chunk_text,
    1 - (de.embedding <=> query_embedding) as similarity,
    de.metadata
  FROM document_embeddings de
  WHERE 
    (search_type = 'all' OR de.document_type = search_type)
    AND de.embedding IS NOT NULL
    AND 1 - (de.embedding <=> query_embedding) >= similarity_threshold
  ORDER BY de.embedding <=> query_embedding
  LIMIT limit_results;
END;
$$ LANGUAGE plpgsql;

-- Function to add document embedding
CREATE OR REPLACE FUNCTION add_document_embedding(
  p_document_id VARCHAR(255),
  p_document_type VARCHAR(50),
  p_chunk_index INTEGER,
  p_chunk_text TEXT,
  p_embedding vector,
  p_metadata JSONB DEFAULT '{}',
  p_model_used VARCHAR(100) DEFAULT 'nomic-embed-text'
)
RETURNS UUID AS $$
DECLARE
  v_id UUID;
BEGIN
  INSERT INTO document_embeddings (
    document_id, document_type, chunk_index, 
    chunk_text, embedding, metadata, model_used
  ) VALUES (
    p_document_id, p_document_type, p_chunk_index,
    p_chunk_text, p_embedding, p_metadata, p_model_used
  ) RETURNING id INTO v_id;
  
  RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Create materialized view for frequently accessed embeddings
CREATE MATERIALIZED VIEW IF NOT EXISTS recent_embeddings AS
SELECT 
  id, document_id, document_type, chunk_text, 
  embedding, metadata, created_at
FROM document_embeddings
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_recent_embeddings_vector 
  ON recent_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Auto-refresh materialized view
CREATE OR REPLACE FUNCTION refresh_recent_embeddings()
RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY recent_embeddings;
END;
$$ LANGUAGE plpgsql;

-- Notification
DO $$
BEGIN
  RAISE NOTICE 'Vector search schema initialized successfully';
END
$$;
