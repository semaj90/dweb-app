-- Detective Evidence Synthesizer Database Initialization
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create full-text search configuration for legal documents
CREATE TEXT SEARCH CONFIGURATION detective_search (COPY = english);

-- Performance optimizations for vector operations
SET max_parallel_workers_per_gather = 4;
SET shared_preload_libraries = 'pg_stat_statements';

-- Create indexes for performance (will be created after tables exist)
-- These will be applied by Drizzle migrations

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE detective_evidence_db TO detective;
GRANT ALL ON SCHEMA public TO detective;

-- Initialize detective schema metadata
INSERT INTO pg_catalog.pg_description (objoid, classoid, objsubid, description)
VALUES (
 'detective_evidence_db'::regclass::oid,
 'pg_database'::regclass::oid,
 0,
 'Detective Evidence Synthesizer Database - Production Ready'
) ON CONFLICT DO NOTHING;