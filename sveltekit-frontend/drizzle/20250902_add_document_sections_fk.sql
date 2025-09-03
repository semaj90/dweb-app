-- Add FK from document_sections.document_id -> legal_documents.id (idempotent & type-aware)
-- This handles earlier mismatch where document_sections.document_id may have been created as UUID while legal_documents.id is integer.
-- Safe to run multiple times.

BEGIN;

DO $$
DECLARE
  doc_id_type text;
  legal_pk_type text;
  has_fk boolean := false;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='document_sections') THEN
    RAISE NOTICE 'document_sections table missing; skipping FK creation';
    RETURN;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname='legal_documents') THEN
    RAISE NOTICE 'legal_documents table missing; skipping FK creation';
    RETURN;
  END IF;

  SELECT data_type INTO doc_id_type
    FROM information_schema.columns
    WHERE table_name='document_sections' AND column_name='document_id';

  IF doc_id_type IS NULL THEN
    RAISE NOTICE 'document_sections.document_id column missing; skipping';
    RETURN;
  END IF;

  SELECT CASE
           WHEN format_type(a.atttypid,a.atttypmod) LIKE 'integer%' THEN 'integer'
           WHEN format_type(a.atttypid,a.atttypmod) LIKE 'uuid%' THEN 'uuid'
           ELSE format_type(a.atttypid,a.atttypmod)
         END
    INTO legal_pk_type
  FROM pg_attribute a
  JOIN pg_index i ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey)
  WHERE i.indrelid='public.legal_documents'::regclass AND i.indisprimary
  LIMIT 1;

  SELECT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname='document_sections_document_id_fkey'
  ) INTO has_fk;

  IF has_fk THEN
    RAISE NOTICE 'FK document_sections_document_id_fkey already exists; nothing to do';
    RETURN;
  END IF;

  -- If types mismatch (uuid vs integer) and column is empty, coerce to match legal_documents PK type
  IF doc_id_type <> legal_pk_type THEN
    IF (SELECT COUNT(*) FROM document_sections WHERE document_id IS NOT NULL) = 0 THEN
      IF legal_pk_type='integer' THEN
        EXECUTE 'ALTER TABLE document_sections ALTER COLUMN document_id TYPE integer USING (NULL::integer)';
        doc_id_type := 'integer';
        RAISE NOTICE 'Coerced document_sections.document_id to integer to match legal_documents.id';
      ELSIF legal_pk_type='uuid' THEN
        EXECUTE 'ALTER TABLE document_sections ALTER COLUMN document_id TYPE uuid USING (NULL::uuid)';
        doc_id_type := 'uuid';
        RAISE NOTICE 'Coerced document_sections.document_id to uuid to match legal_documents.id';
      ELSE
        RAISE NOTICE 'Unsupported PK type %; skipping FK creation', legal_pk_type;
        RETURN;
      END IF;
    ELSE
      RAISE NOTICE 'Type mismatch (% vs %). Non-empty column; manual reconciliation required; skipping FK', doc_id_type, legal_pk_type;
      RETURN;
    END IF;
  END IF;

  IF doc_id_type = legal_pk_type THEN
    BEGIN
      EXECUTE 'ALTER TABLE document_sections ADD CONSTRAINT document_sections_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.legal_documents(id) ON DELETE CASCADE';
      RAISE NOTICE 'FK document_sections_document_id_fkey created';
    EXCEPTION WHEN duplicate_object THEN
      RAISE NOTICE 'FK already exists (race condition)';
    END;
  ELSE
    RAISE NOTICE 'Final type mismatch remains (% vs %); FK skipped', doc_id_type, legal_pk_type;
  END IF;
END$$;

COMMIT;
