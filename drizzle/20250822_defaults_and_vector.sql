-- 20250822_defaults_and_vector.sql
-- Migration: Set safe defaults for JSONB columns and enable vector extension

-- Ensure vector extension is available
CREATE EXTENSION IF NOT EXISTS vector;

-- Set safe defaults for evidence table
ALTER TABLE evidence
  ALTER COLUMN tags SET DEFAULT '[]'::jsonb,
  ALTER COLUMN tags SET NOT NULL;

-- Set safe defaults for reports table  
ALTER TABLE reports
  ALTER COLUMN summary SET DEFAULT '',
  ALTER COLUMN summary SET NOT NULL,
  ALTER COLUMN doc SET DEFAULT '{}'::jsonb,
  ALTER COLUMN doc SET NOT NULL;

-- Set safe defaults for vectors table
ALTER TABLE vectors
  ALTER COLUMN payload SET DEFAULT '{}'::jsonb,
  ALTER COLUMN payload SET NOT NULL;

-- Ensure embedding column exists with proper vector type
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name = 'vectors' 
                 AND column_name = 'embedding' 
                 AND data_type = 'vector') THEN
    ALTER TABLE vectors ADD COLUMN embedding vector(768);
  END IF;
END
$$;

-- Create index on vectors for similarity search if not exists
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'vectors_embedding_idx') THEN
    CREATE INDEX vectors_embedding_idx ON vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
  END IF;
END
$$;