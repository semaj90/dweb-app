-- Rollback Migration 003
DROP VIEW IF EXISTS v_rag_feedback_stats;
DROP TABLE IF EXISTS rag_feedback;
DROP FUNCTION IF EXISTS increment_search_term(TEXT, TEXT);
DROP TABLE IF EXISTS search_terms;
-- Note: pg_trgm extension retained (not dropped)
