-- Rollback migration
DROP TRIGGER IF EXISTS update_vector_jobs_updated_at ON vector_jobs;
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP TABLE IF EXISTS vector_jobs;