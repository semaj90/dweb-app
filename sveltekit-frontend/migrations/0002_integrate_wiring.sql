ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding vector(1536);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_indexed boolean DEFAULT false;

CREATE TABLE IF NOT EXISTS search_index (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_type varchar(50) NOT NULL,
  entity_id uuid NOT NULL,
  content text NOT NULL,
  embedding vector(1536),
  metadata json,
  created_at timestamp DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS search_index_embedding_idx ON search_index USING ivfflat (embedding vector_cosine_ops);