-- vectors_autocreate_notify.sql
-- Auto-create zero vectors on evidence/reports insert + pg_notify for Redis relay

CREATE EXTENSION IF NOT EXISTS vector;

-- Ensure vector type exists
DO $
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector') THEN
    RAISE NOTICE 'vector type missing; ensure pgvector extension installed';
  END IF;
END $;

-- Function: create default vector for evidence
CREATE OR REPLACE FUNCTION create_vector_for_evidence()
RETURNS TRIGGER AS $
BEGIN
  INSERT INTO vectors (owner_type, owner_id, embedding, payload, created_at, updated_at)
  VALUES (
    'evidence',
    NEW.id,
    (SELECT ARRAY(SELECT 0.0::real FROM generate_series(1,768))::vector),
    jsonb_build_object('filename', NEW.filename, 'caseId', COALESCE(NEW.case_id, NULL)),
    now(),
    now()
  )
  ON CONFLICT (owner_type, owner_id) DO NOTHING;
  
  -- Notify Redis relay for realtime updates
  PERFORM pg_notify('evidence_inserted', 
    json_build_object(
      'id', NEW.id::text, 
      'caseId', NEW.case_id, 
      'filename', NEW.filename,
      'timestamp', extract(epoch from now())
    )::text
  );
  RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Function: create default vector for reports
CREATE OR REPLACE FUNCTION create_vector_for_report()
RETURNS TRIGGER AS $
BEGIN
  INSERT INTO vectors (owner_type, owner_id, embedding, payload, created_at, updated_at)
  VALUES (
    'report',
    NEW.id,
    (SELECT ARRAY(SELECT 0.0::real FROM generate_series(1,768))::vector),
    jsonb_build_object('title', NEW.title, 'caseId', COALESCE(NEW.case_id, NULL)),
    now(),
    now()
  )
  ON CONFLICT (owner_type, owner_id) DO NOTHING;
  
  -- Notify Redis relay
  PERFORM pg_notify('report_inserted', 
    json_build_object(
      'id', NEW.id::text, 
      'caseId', NEW.case_id, 
      'title', NEW.title,
      'timestamp', extract(epoch from now())
    )::text
  );
  RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Function: create default vector for documents  
CREATE OR REPLACE FUNCTION create_vector_for_document()
RETURNS TRIGGER AS $
BEGIN
  INSERT INTO vectors (owner_type, owner_id, embedding, payload, created_at, updated_at)
  VALUES (
    'document',
    NEW.id,
    (SELECT ARRAY(SELECT 0.0::real FROM generate_series(1,768))::vector),
    jsonb_build_object('title', NEW.title, 'caseId', COALESCE(NEW.case_id, NULL)),
    now(),
    now()
  )
  ON CONFLICT (owner_type, owner_id) DO NOTHING;
  
  -- Notify Redis relay
  PERFORM pg_notify('document_inserted', 
    json_build_object(
      'id', NEW.id::text, 
      'caseId', NEW.case_id, 
      'title', NEW.title,
      'timestamp', extract(epoch from now())
    )::text
  );
  RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Function: create default vector for chunks
CREATE OR REPLACE FUNCTION create_vector_for_chunk()
RETURNS TRIGGER AS $
BEGIN
  INSERT INTO vectors (owner_type, owner_id, embedding, payload, created_at, updated_at)
  VALUES (
    'chunk',
    NEW.id,
    (SELECT ARRAY(SELECT 0.0::real FROM generate_series(1,768))::vector),
    jsonb_build_object('docId', NEW.doc_id, 'idx', NEW.idx, 'text_preview', LEFT(NEW.text_excerpt, 100)),
    now(),
    now()
  )
  ON CONFLICT (owner_type, owner_id) DO NOTHING;
  
  -- Notify Redis relay
  PERFORM pg_notify('chunk_inserted', 
    json_build_object(
      'id', NEW.id::text, 
      'docId', NEW.doc_id::text, 
      'idx', NEW.idx,
      'timestamp', extract(epoch from now())
    )::text
  );
  RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Drop existing triggers
DROP TRIGGER IF EXISTS evidence_vector_insert ON evidence;
DROP TRIGGER IF EXISTS report_vector_insert ON reports;
DROP TRIGGER IF EXISTS document_vector_insert ON documents;
DROP TRIGGER IF EXISTS chunk_vector_insert ON chunks;

-- Create triggers
CREATE TRIGGER evidence_vector_insert
AFTER INSERT ON evidence
FOR EACH ROW EXECUTE FUNCTION create_vector_for_evidence();

CREATE TRIGGER report_vector_insert
AFTER INSERT ON reports
FOR EACH ROW EXECUTE FUNCTION create_vector_for_report();

CREATE TRIGGER document_vector_insert  
AFTER INSERT ON documents
FOR EACH ROW EXECUTE FUNCTION create_vector_for_document();

CREATE TRIGGER chunk_vector_insert
AFTER INSERT ON chunks
FOR EACH ROW EXECUTE FUNCTION create_vector_for_chunk();

-- Create unique constraint on vectors table to prevent duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_vectors_owner_unique 
ON vectors (owner_type, owner_id);

-- Create vector_outbox if not exists
CREATE TABLE IF NOT EXISTS vector_outbox (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_type text NOT NULL,
  owner_id uuid NOT NULL,
  event text NOT NULL, -- upsert | delete | reembed | rotate
  vector jsonb NULL,
  payload jsonb NULL,
  attempts integer NOT NULL DEFAULT 0,
  processed_at timestamptz NULL,
  created_at timestamptz DEFAULT now()
);

-- Index for worker polling
CREATE INDEX IF NOT EXISTS idx_vector_outbox_unprocessed 
ON vector_outbox (created_at) 
WHERE processed_at IS NULL;

COMMENT ON TABLE vectors IS 'Vector embeddings with pgvector support - auto-created via triggers';
COMMENT ON TABLE vector_outbox IS 'Outbox pattern for async vector processing via Redis Streams';
COMMENT ON FUNCTION create_vector_for_evidence() IS 'Auto-creates zero vector on evidence insert + pg_notify';
COMMENT ON FUNCTION create_vector_for_report() IS 'Auto-creates zero vector on report insert + pg_notify';
COMMENT ON FUNCTION create_vector_for_document() IS 'Auto-creates zero vector on document insert + pg_notify';
COMMENT ON FUNCTION create_vector_for_chunk() IS 'Auto-creates zero vector on chunk insert + pg_notify';