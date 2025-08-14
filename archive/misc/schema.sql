-- PostgreSQL schema with pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table (generic)
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

-- Legal documents table (for Go service)
CREATE TABLE IF NOT EXISTS legal_documents (
    id VARCHAR(255) PRIMARY KEY,
    case_id VARCHAR(255),
    title TEXT,
    content TEXT,
    summary TEXT,
    entities TEXT,
    risk_score FLOAT DEFAULT 0,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_docs_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_docs_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_docs_metadata ON documents USING gin(metadata);

CREATE INDEX IF NOT EXISTS idx_legal_docs_embedding ON legal_documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_legal_docs_case ON legal_documents(case_id);
CREATE INDEX IF NOT EXISTS idx_legal_docs_created ON legal_documents(created_at DESC);
