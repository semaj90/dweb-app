-- Phase 1: Create document_sections table (idempotent / guarded)
-- Creates a per-document section table with a 768-dim pgvector embedding column.

BEGIN;

-- Ensure vector extension exists (safe: will be no-op if already present)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table if it does not exist
CREATE TABLE IF NOT EXISTS document_sections (
	id BIGSERIAL PRIMARY KEY,
	document_id UUID,
	section_index INTEGER NOT NULL DEFAULT 0,
	title TEXT,
	content TEXT NOT NULL,
	embedding VECTOR(768),
	content_tokens INTEGER,
	created_at TIMESTAMPTZ DEFAULT now()
);

-- Basic index for lookups by document
CREATE INDEX IF NOT EXISTS idx_document_sections_document_id ON document_sections(document_id);

-- Create an IVFFLAT vector index only if pgvector extension is available.
DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
		IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_document_sections_embedding') THEN
			EXECUTE 'CREATE INDEX idx_document_sections_embedding ON document_sections USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)';
		END IF;
	END IF;
END$$;

COMMIT;

