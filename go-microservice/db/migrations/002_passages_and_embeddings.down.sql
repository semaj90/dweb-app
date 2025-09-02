-- Rollback Migration 002: Drop passages & related embedding structures
DROP VIEW IF EXISTS v_passages_enriched;
DROP TABLE IF EXISTS graph_edges;
DROP TABLE IF EXISTS embedding_projection;
DROP TABLE IF EXISTS embedding_metadata;
DROP TABLE IF EXISTS passages;
