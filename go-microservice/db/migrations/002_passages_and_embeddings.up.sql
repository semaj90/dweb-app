-- Migration 002: Passages + Embedding Metadata + Projection Tables
-- Preconditions: pgvector extension installed (see 001 migrations)
-- Idempotent guards where possible

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Passages table: normalized textual units extracted from documents (clauses, sections, paragraphs)
CREATE TABLE IF NOT EXISTS passages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,                        -- order within document
    text TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    section_type VARCHAR(50),                    -- heading, clause, recital, table_cell, etc.
    hash VARCHAR(64) NOT NULL,                   -- SHA256(text canonicalized)
    language VARCHAR(10) DEFAULT 'en',
    -- Embedding (nullable until backfilled). Dimension fixed by EMB_DIM (default 768)
    embedding VECTOR(768),
    pagerank FLOAT,                              -- optional enrichment (set later)
    cluster_id INTEGER,                          -- Louvain/Leiden cluster
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(document_id, seq),
    UNIQUE(hash)
);

-- Guard: ensure dimension correctness if embedding populated
ALTER TABLE passages ADD CONSTRAINT passages_embedding_dim_ck CHECK (
    embedding IS NULL OR vector_dims(embedding) = 768
);

CREATE INDEX IF NOT EXISTS idx_passages_document_seq ON passages(document_id, seq);
CREATE INDEX IF NOT EXISTS idx_passages_section_type ON passages(section_type);
CREATE INDEX IF NOT EXISTS idx_passages_language ON passages(language);
CREATE INDEX IF NOT EXISTS idx_passages_hash ON passages(hash);
-- Defer IVFFLAT/HNSW index until row count threshold (>50000) to avoid slow initial build
-- Example (manual later):
-- CREATE INDEX CONCURRENTLY idx_passages_embedding_cosine ON passages USING ivfflat (embedding vector_cosine_ops) WITH (lists=200);

-- Embedding metadata to track model versioning & schema invariants
CREATE TABLE IF NOT EXISTS embedding_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    dim INTEGER NOT NULL,
    method TEXT,                  -- e.g., 'nomic-embed-text', 'legal-bert', etc.
    quantization TEXT,            -- 'fp16','int8','q4'
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT
);

-- Latest active embedding spec (one row)
CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_metadata_active ON embedding_metadata(model_name, model_version);

-- UMAP / projection coordinates (offline job output)
CREATE TABLE IF NOT EXISTS embedding_projection (
    passage_id UUID PRIMARY KEY REFERENCES passages(id) ON DELETE CASCADE,
    x DOUBLE PRECISION NOT NULL,
    y DOUBLE PRECISION NOT NULL,
    z DOUBLE PRECISION,                 -- optional third axis (temporal/secondary reduction)
    cluster_id INTEGER,
    pagerank FLOAT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embedding_projection_cluster ON embedding_projection(cluster_id);

-- Graph edges for similarity / citation / hierarchy (lightweight initial schema)
CREATE TABLE IF NOT EXISTS graph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    src_id UUID NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    dst_id UUID NOT NULL REFERENCES passages(id) ON DELETE CASCADE,
    edge_type VARCHAR(30) NOT NULL,               -- similarity | citation | hierarchy
    weight FLOAT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(src_id, dst_id, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_src_type ON graph_edges(src_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_dst_type ON graph_edges(dst_id, edge_type);

-- Trigger for updated_at on passages
CREATE OR REPLACE FUNCTION update_passages_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_passages_updated_at
BEFORE UPDATE ON passages
FOR EACH ROW EXECUTE FUNCTION update_passages_updated_at();

-- View to enrich passages with projection data (if available)
CREATE OR REPLACE VIEW v_passages_enriched AS
SELECT p.*, pr.x, pr.y, pr.z, pr.cluster_id AS projection_cluster_id
FROM passages p
LEFT JOIN embedding_projection pr ON pr.passage_id = p.id;
