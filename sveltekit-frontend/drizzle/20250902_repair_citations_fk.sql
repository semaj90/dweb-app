-- Phase 1: Repair citations foreign key constraints (idempotent/guarded)
-- This migration will ensure citations.document_id exists and is constrained to documents(id)

BEGIN;

-- We'll add a document_id column with a type matching the referenced table's PK (if present).
-- This avoids incompatible-type FK creation (uuid vs integer).

-- Helper: determine referenced table and column types and create/alter accordingly in a single DO block.
DO $$
DECLARE
	ref_table regclass;
	pk_type text;
	citations_exists boolean := EXISTS (SELECT 1 FROM pg_class WHERE relname = 'citations' AND relkind = 'r');
BEGIN
	IF NOT citations_exists THEN
		RAISE NOTICE 'Table citations does not exist; skipping FK repair.';
		RETURN;
	END IF;

	-- try common document table names (legal_documents, documents, case_documents)
	IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'legal_documents') THEN
		ref_table := 'public.legal_documents'::regclass;
	ELSIF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'documents') THEN
		ref_table := 'public.documents'::regclass;
	ELSIF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'case_documents') THEN
		ref_table := 'public.case_documents'::regclass;
	ELSE
		RAISE NOTICE 'No documents table found to reference; skipping FK creation.';
		RETURN;
	END IF;

	-- get primary key column data type for referenced table
	SELECT format_type(a.atttypid, a.atttypmod) INTO pk_type
	FROM pg_attribute a
	JOIN pg_index i ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
	WHERE i.indrelid = ref_table AND i.indisprimary AND a.attnum > 0
	LIMIT 1;

	IF pk_type IS NULL THEN
		RAISE NOTICE 'Unable to determine PK type for %, skipping', ref_table;
		RETURN;
	END IF;

	-- ensure document_id column has compatible type; skip if existing column matches
	IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='citations' AND column_name='document_id') THEN
		-- check existing type
		PERFORM 1 FROM information_schema.columns WHERE table_name='citations' AND column_name='document_id' AND data_type = CASE WHEN pk_type LIKE 'integer%' THEN 'integer' WHEN pk_type LIKE 'uuid%' THEN 'uuid' ELSE pk_type END;
		IF FOUND THEN
			RAISE NOTICE 'citations.document_id already exists with compatible type %', pk_type;
		ELSE
			RAISE NOTICE 'citations.document_id exists but type is incompatible; leaving as-is for manual reconciliation.';
		END IF;
	ELSE
		-- create document_id with matching type
		IF pk_type LIKE 'integer%' THEN
			EXECUTE 'ALTER TABLE citations ADD COLUMN IF NOT EXISTS document_id integer';
		ELSIF pk_type LIKE 'uuid%' THEN
			EXECUTE 'ALTER TABLE citations ADD COLUMN IF NOT EXISTS document_id uuid';
		ELSE
			EXECUTE format('ALTER TABLE citations ADD COLUMN IF NOT EXISTS document_id %s', pk_type);
		END IF;
		RAISE NOTICE 'Added citations.document_id with type %', pk_type;
	END IF;

	-- drop obviously-broken FK constraints referencing non-matching columns (best-effort)
	FOR r IN SELECT con.oid, con.conname
					 FROM pg_constraint con
					 JOIN pg_class cl ON con.conrelid = cl.oid
					 WHERE cl.relname = 'citations' AND con.contype = 'f'
	LOOP
		BEGIN
			EXECUTE format('ALTER TABLE citations DROP CONSTRAINT IF EXISTS %I', r.conname);
		EXCEPTION WHEN others THEN
			-- ignore
		END;
	END LOOP;

	-- Add FK constraint if possible (name guarded)
	BEGIN
		IF pk_type LIKE 'integer%' THEN
			EXECUTE 'ALTER TABLE citations ADD CONSTRAINT IF NOT EXISTS citations_document_id_fkey FOREIGN KEY (document_id) REFERENCES ' || quote_ident(split_part(ref_table::text, '.', 2)) || '(id) ON DELETE CASCADE';
		ELSIF pk_type LIKE 'uuid%' THEN
			EXECUTE 'ALTER TABLE citations ADD CONSTRAINT IF NOT EXISTS citations_document_id_fkey FOREIGN KEY (document_id) REFERENCES ' || quote_ident(split_part(ref_table::text, '.', 2)) || '(id) ON DELETE CASCADE';
		ELSE
			-- best-effort: attempt to add FK using detected pk_type
			EXECUTE 'ALTER TABLE citations ADD CONSTRAINT IF NOT EXISTS citations_document_id_fkey FOREIGN KEY (document_id) REFERENCES ' || quote_ident(split_part(ref_table::text, '.', 2)) || '(id) ON DELETE CASCADE';
		END IF;
	EXCEPTION WHEN others THEN
		RAISE NOTICE 'Could not create FK constraint automatically: %', SQLERRM;
	END;
END$$;

COMMIT;

