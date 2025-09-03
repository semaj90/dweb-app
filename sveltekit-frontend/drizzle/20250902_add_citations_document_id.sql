
BEGIN;

-- Defensive: if legal_documents.id is integer, create integer column; if uuid, create uuid; otherwise default to text
DO $$
DECLARE
	pk_type text;
BEGIN
	IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'citations') THEN
		RAISE NOTICE 'Table citations not present; skipping column add.';
		RETURN;
	END IF;

	IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'legal_documents') THEN
		SELECT format_type(a.atttypid, a.atttypmod) INTO pk_type
		FROM pg_attribute a
		JOIN pg_index i ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
		WHERE i.indrelid = 'public.legal_documents'::regclass AND i.indisprimary AND a.attnum > 0
		LIMIT 1;
	END IF;

	IF pk_type IS NULL THEN
		pk_type := 'uuid'; -- default to uuid for safety in new deployments
	END IF;

	IF pk_type LIKE 'integer%' THEN
		EXECUTE 'ALTER TABLE citations ADD COLUMN IF NOT EXISTS document_id integer';
	ELSIF pk_type LIKE 'uuid%' THEN
		EXECUTE 'ALTER TABLE citations ADD COLUMN IF NOT EXISTS document_id uuid';
	ELSE
		EXECUTE format('ALTER TABLE citations ADD COLUMN IF NOT EXISTS document_id %s', pk_type);
	END IF;
END$$;

-- Add an index to speed up joins if not exists
CREATE INDEX IF NOT EXISTS idx_citations_document_id ON citations(document_id);

COMMIT;

