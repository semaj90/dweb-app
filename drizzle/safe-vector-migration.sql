-- Enhanced Legal AI - Safe Vector Column Migration
-- This migration handles vector columns with graceful fallbacks

-- First, check if pgvector extension exists and enable it
DO $$
BEGIN
    -- Try to create vector extension
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vector;
        RAISE NOTICE 'pgvector extension enabled successfully';
    EXCEPTION WHEN OTHERS THEN
        RAISE WARNING 'Could not enable pgvector extension: %', SQLERRM;
    END;
END $$;

-- Create helper function for safe vector column creation
CREATE OR REPLACE FUNCTION safe_add_vector_column(
    table_name text,
    column_name text,
    dimensions integer DEFAULT 1536,
    nullable boolean DEFAULT true
) RETURNS void AS $$
DECLARE
    has_vector_type boolean;
    null_clause text := '';
BEGIN
    -- Check if vector type exists
    SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector') INTO has_vector_type;
    
    -- Set null clause
    IF NOT nullable THEN
        null_clause := ' NOT NULL';
    END IF;
    
    IF has_vector_type THEN
        -- Add vector column
        BEGIN
            EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS %I vector(%s)%s', 
                          table_name, column_name, dimensions, null_clause);
            RAISE NOTICE 'Added vector column %s.%s(%s)', table_name, column_name, dimensions;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Failed to add vector column %s.%s: %s', table_name, column_name, SQLERRM;
            -- Fallback to text column
            EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS %I text%s', 
                          table_name, column_name, null_clause);
            RAISE NOTICE 'Added text fallback column %s.%s', table_name, column_name;
        END;
    ELSE
        -- Add text column as fallback
        EXECUTE format('ALTER TABLE %I ADD COLUMN IF NOT EXISTS %I text%s', 
                      table_name, column_name, null_clause);
        RAISE NOTICE 'Added text fallback column %s.%s (pgvector not available)', table_name, column_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to create vector index safely
CREATE OR REPLACE FUNCTION safe_create_vector_index(
    table_name text,
    column_name text,
    index_method text DEFAULT 'ivfflat'
) RETURNS void AS $$
DECLARE
    has_vector_type boolean;
    index_name text;
BEGIN
    SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector') INTO has_vector_type;
    
    IF has_vector_type THEN
        index_name := table_name || '_' || column_name || '_vector_idx';
        BEGIN
            EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I USING %s (%I vector_cosine_ops)', 
                          index_name, table_name, index_method, column_name);
            RAISE NOTICE 'Created vector index %s', index_name;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Failed to create vector index %s: %s', index_name, SQLERRM;
        END;
    ELSE
        RAISE NOTICE 'Skipped vector index for %s.%s (pgvector not available)', table_name, column_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Now apply to existing tables that need vector columns
DO $$
BEGIN
    -- Add vector columns to legal_documents table if it exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'legal_documents') THEN
        PERFORM safe_add_vector_column('legal_documents', 'title_embedding', 1536, true);
        PERFORM safe_add_vector_column('legal_documents', 'content_embedding', 1536, true);
        
        -- Create indexes
        PERFORM safe_create_vector_index('legal_documents', 'title_embedding');
        PERFORM safe_create_vector_index('legal_documents', 'content_embedding');
    END IF;
    
    -- Add vector columns to other tables that need them
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'evidence') THEN
        PERFORM safe_add_vector_column('evidence', 'content_embedding', 1536, true);
        PERFORM safe_create_vector_index('evidence', 'content_embedding');
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'cases') THEN
        PERFORM safe_add_vector_column('cases', 'description_embedding', 1536, true);
        PERFORM safe_create_vector_index('cases', 'description_embedding');
    END IF;
END $$;

-- Cleanup helper functions (optional - keep them for future use)
-- DROP FUNCTION IF EXISTS safe_add_vector_column(text, text, integer, boolean);
-- DROP FUNCTION IF EXISTS safe_create_vector_index(text, text, text);

-- Final status check
DO $$
DECLARE
    vector_available boolean;
    tables_with_vectors text[] := ARRAY['legal_documents', 'evidence', 'cases'];
    table_name text;
BEGIN
    SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector') INTO vector_available;
    
    RAISE NOTICE '=== Enhanced Legal AI Vector Migration Complete ===';
    RAISE NOTICE 'Vector support: %s', CASE WHEN vector_available THEN 'ENABLED' ELSE 'DISABLED (using text fallback)' END;
    
    FOREACH table_name IN ARRAY tables_with_vectors
    LOOP
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = table_name) THEN
            RAISE NOTICE 'Table %s: Vector columns configured', table_name;
        END IF;
    END LOOP;
END $$;
