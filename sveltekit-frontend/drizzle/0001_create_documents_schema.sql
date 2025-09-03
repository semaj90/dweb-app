-- Migration: Create documents schema with pgvector support
-- Legal AI Platform - Enhanced document management with vector embeddings

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table
CREATE TABLE IF NOT EXISTS "users" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"email" varchar(255) UNIQUE NOT NULL,
	"name" varchar(255),
	"profile_embedding" vector(384),
	"role" varchar(50) DEFAULT 'user',
	"is_active" boolean DEFAULT true NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Cases table  
CREATE TABLE IF NOT EXISTS "cases" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"case_number" varchar(100) UNIQUE,
	"title" varchar(500) NOT NULL,
	"description" text,
	"case_embedding" vector(384),
	"status" varchar(50) DEFAULT 'active',
	"priority" varchar(20) DEFAULT 'medium',
	"case_type" varchar(100),
	"jurisdiction" varchar(200),
	"filed_date" timestamp with time zone,
	"closed_date" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"created_by" uuid REFERENCES "users"("id") ON DELETE SET NULL,
	"assigned_to" uuid REFERENCES "users"("id") ON DELETE SET NULL
);

-- Main documents table
CREATE TABLE IF NOT EXISTS "documents" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" varchar(500) NOT NULL,
	"content" text NOT NULL,
	"embedding" vector(384),
	"title_embedding" vector(384),
	"summary_embedding" vector(384),
	"document_type" varchar(100) DEFAULT 'general' NOT NULL,
	"confidence_level" integer DEFAULT 0,
	"risk_level" varchar(20) DEFAULT 'low',
	"priority" integer DEFAULT 100,
	"ai_summary" text,
	"ai_analysis" jsonb,
	"ai_tags" jsonb,
	"key_entities" jsonb,
	"source_url" varchar(1000),
	"file_path" varchar(1000),
	"file_type" varchar(100),
	"file_size" integer,
	"checksum" varchar(64),
	"case_id" uuid REFERENCES "cases"("id") ON DELETE SET NULL,
	"jurisdiction" varchar(200),
	"practice_area" varchar(200),
	"processing_status" varchar(50) DEFAULT 'pending',
	"embedding_model" varchar(100) DEFAULT 'nomic-embed-text',
	"processed_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"created_by" uuid REFERENCES "users"("id") ON DELETE SET NULL,
	"is_active" boolean DEFAULT true NOT NULL,
	"is_public" boolean DEFAULT false NOT NULL,
	"is_indexed" boolean DEFAULT false NOT NULL
);

-- Document chunks table for enhanced RAG
CREATE TABLE IF NOT EXISTS "document_chunks" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"document_id" uuid REFERENCES "documents"("id") ON DELETE CASCADE NOT NULL,
	"chunk_index" integer NOT NULL,
	"chunk_text" text NOT NULL,
	"chunk_size" integer NOT NULL,
	"embedding" vector(384) NOT NULL,
	"start_position" integer,
	"end_position" integer,
	"page_number" integer,
	"section_title" varchar(500),
	"importance_score" integer DEFAULT 0,
	"chunk_summary" text,
	"key_points" jsonb,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Unified vector storage table
CREATE TABLE IF NOT EXISTS "vectors" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"entity_type" varchar(50) NOT NULL,
	"entity_id" uuid NOT NULL,
	"vector_type" varchar(50) NOT NULL,
	"embedding" vector(384) NOT NULL,
	"model_name" varchar(100) DEFAULT 'nomic-embed-text',
	"model_version" varchar(50),
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Indexes for performance optimization

-- Vector similarity indexes using HNSW (better performance than IVFFlat for most cases)
CREATE INDEX IF NOT EXISTS "documents_embedding_hnsw_idx" 
	ON "documents" 
	USING hnsw ("embedding" vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "documents_title_embedding_hnsw_idx" 
	ON "documents" 
	USING hnsw ("title_embedding" vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "documents_summary_embedding_hnsw_idx" 
	ON "documents" 
	USING hnsw ("summary_embedding" vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "document_chunks_embedding_hnsw_idx" 
	ON "document_chunks" 
	USING hnsw ("embedding" vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "cases_embedding_hnsw_idx" 
	ON "cases" 
	USING hnsw ("case_embedding" vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "users_profile_embedding_hnsw_idx" 
	ON "users" 
	USING hnsw ("profile_embedding" vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "vectors_embedding_hnsw_idx" 
	ON "vectors" 
	USING hnsw ("embedding" vector_cosine_ops)
	WITH (m = 16, ef_construction = 64);

-- Regular indexes for filtering and joins
CREATE INDEX IF NOT EXISTS "documents_case_id_idx" ON "documents" ("case_id");
CREATE INDEX IF NOT EXISTS "documents_created_by_idx" ON "documents" ("created_by");
CREATE INDEX IF NOT EXISTS "documents_document_type_idx" ON "documents" ("document_type");
CREATE INDEX IF NOT EXISTS "documents_risk_level_idx" ON "documents" ("risk_level");
CREATE INDEX IF NOT EXISTS "documents_priority_idx" ON "documents" ("priority");
CREATE INDEX IF NOT EXISTS "documents_created_at_idx" ON "documents" ("created_at");
CREATE INDEX IF NOT EXISTS "documents_is_active_idx" ON "documents" ("is_active");
CREATE INDEX IF NOT EXISTS "documents_processing_status_idx" ON "documents" ("processing_status");

CREATE INDEX IF NOT EXISTS "document_chunks_document_id_idx" ON "document_chunks" ("document_id");
CREATE INDEX IF NOT EXISTS "document_chunks_chunk_index_idx" ON "document_chunks" ("chunk_index");
CREATE INDEX IF NOT EXISTS "document_chunks_importance_score_idx" ON "document_chunks" ("importance_score");

CREATE INDEX IF NOT EXISTS "cases_created_by_idx" ON "cases" ("created_by");
CREATE INDEX IF NOT EXISTS "cases_assigned_to_idx" ON "cases" ("assigned_to");
CREATE INDEX IF NOT EXISTS "cases_status_idx" ON "cases" ("status");
CREATE INDEX IF NOT EXISTS "cases_case_type_idx" ON "cases" ("case_type");
CREATE INDEX IF NOT EXISTS "cases_priority_idx" ON "cases" ("priority");

CREATE INDEX IF NOT EXISTS "vectors_entity_type_idx" ON "vectors" ("entity_type");
CREATE INDEX IF NOT EXISTS "vectors_entity_id_idx" ON "vectors" ("entity_id");
CREATE INDEX IF NOT EXISTS "vectors_vector_type_idx" ON "vectors" ("vector_type");
CREATE INDEX IF NOT EXISTS "vectors_composite_idx" ON "vectors" ("entity_type", "entity_id", "vector_type");

-- Full-text search indexes for content
CREATE INDEX IF NOT EXISTS "documents_content_fts_idx" ON "documents" USING gin(to_tsvector('english', "content"));
CREATE INDEX IF NOT EXISTS "documents_title_fts_idx" ON "documents" USING gin(to_tsvector('english', "title"));
CREATE INDEX IF NOT EXISTS "document_chunks_text_fts_idx" ON "document_chunks" USING gin(to_tsvector('english', "chunk_text"));

-- JSON indexes for metadata
CREATE INDEX IF NOT EXISTS "documents_ai_tags_idx" ON "documents" USING gin("ai_tags");
CREATE INDEX IF NOT EXISTS "documents_key_entities_idx" ON "documents" USING gin("key_entities");
CREATE INDEX IF NOT EXISTS "document_chunks_key_points_idx" ON "document_chunks" USING gin("key_points");

-- Functions for updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers to automatically update updated_at
CREATE OR REPLACE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON "documents"
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER update_cases_updated_at
    BEFORE UPDATE ON "cases"
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER update_users_updated_at
    BEFORE UPDATE ON "users"
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Sample data for testing (optional)
INSERT INTO "users" ("email", "name", "role") VALUES
    ('admin@legalai.com', 'Legal AI Admin', 'admin'),
    ('lawyer@legalai.com', 'Senior Attorney', 'lawyer'),
    ('paralegal@legalai.com', 'Legal Assistant', 'paralegal')
ON CONFLICT ("email") DO NOTHING;

-- Performance analysis function
CREATE OR REPLACE FUNCTION analyze_vector_performance()
RETURNS TABLE (
    table_name text,
    index_name text,
    index_size text,
    rows_estimate bigint
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        indexname as index_name,
        pg_size_pretty(pg_relation_size(schemaname||'.'||indexname)) as index_size,
        reltuples::bigint as rows_estimate
    FROM pg_indexes 
    JOIN pg_class ON pg_class.relname = indexname
    JOIN pg_stat_user_tables ON pg_stat_user_tables.relname = tablename
    WHERE schemaname = 'public' 
    AND (indexname LIKE '%embedding%' OR indexname LIKE '%vector%')
    ORDER BY pg_relation_size(schemaname||'.'||indexname) DESC;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE "documents" IS 'Legal documents with vector embeddings for semantic search';
COMMENT ON TABLE "document_chunks" IS 'Text chunks from large documents for enhanced RAG processing';
COMMENT ON TABLE "cases" IS 'Legal cases with case similarity embeddings';
COMMENT ON TABLE "users" IS 'System users with personalization embeddings';
COMMENT ON TABLE "vectors" IS 'Unified vector storage for cross-entity semantic search';

COMMENT ON COLUMN "documents"."embedding" IS 'Content embedding vector (384 dimensions)';
COMMENT ON COLUMN "documents"."title_embedding" IS 'Title embedding vector for title-based search';
COMMENT ON COLUMN "documents"."summary_embedding" IS 'Summary embedding vector for quick overview search';
COMMENT ON COLUMN "documents"."ai_analysis" IS 'JSON object containing AI analysis results';
COMMENT ON COLUMN "documents"."ai_tags" IS 'Array of AI-extracted tags';
COMMENT ON COLUMN "documents"."key_entities" IS 'JSON object containing named entities';

COMMIT;