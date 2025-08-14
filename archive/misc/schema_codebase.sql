-- PostgreSQL schema with pgvector
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE code_chunks (
    id VARCHAR(255) PRIMARY KEY,
    file_path VARCHAR(500),
    content TEXT,
    chunk_idx INTEGER,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chunks_embedding ON code_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_filepath ON code_chunks(file_path);
CREATE INDEX idx_chunks_metadata ON code_chunks USING gin(metadata);
