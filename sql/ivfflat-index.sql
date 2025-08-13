-- IVFFLAT index template for pgvector
-- Replace placeholders with your actual table/column names or run scripts/create-ivfflat-index.ps1

CREATE EXTENSION IF NOT EXISTS vector;

-- Recommended starting point: lists ~= number_of_rows / 1000 (tune per dataset)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding_ivfflat
  ON documents USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

ANALYZE documents;
