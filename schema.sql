-- PostgreSQL schema with pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id VARCHAR(255) PRIMARY KEY,
    url TEXT,
    content TEXT,
    parsed JSONB,
    summary TEXT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_docs_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_docs_created ON documents(created_at DESC);
CREATE INDEX idx_docs_metadata ON documents USING gin(metadata);
