-- Rollback script for documents and related tables
-- Version 2.0 Enterprise Database Schema Rollback

-- Drop views first
DROP VIEW IF EXISTS active_documents;

-- Drop functions
DROP FUNCTION IF EXISTS cleanup_completed_jobs(INTEGER);
DROP FUNCTION IF EXISTS cleanup_expired_cache();
DROP FUNCTION IF EXISTS update_document_access_stats();
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop triggers
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS user_sessions;
DROP TABLE IF EXISTS cache_entries;
DROP TABLE IF EXISTS system_metrics;
DROP TABLE IF EXISTS processing_jobs;
DROP TABLE IF EXISTS legal_entities;
DROP TABLE IF EXISTS document_relationships;
DROP TABLE IF EXISTS documents;

-- Note: We don't drop the vector extension as it might be used by other applications
-- DROP EXTENSION IF EXISTS vector;
-- DROP EXTENSION IF EXISTS "uuid-ossp";