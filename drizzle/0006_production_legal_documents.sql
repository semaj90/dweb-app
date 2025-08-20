-- Migration: Production Legal Documents Schema with pgvector
-- This migration sets up the complete legal documents system

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable other necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create legal_documents table
CREATE TABLE IF NOT EXISTS "legal_documents" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" text NOT NULL,
	"content" text NOT NULL,
	"document_type" text NOT NULL,
	"jurisdiction" text DEFAULT 'federal' NOT NULL,
	"practice_area" text,
	"file_hash" text,
	"file_name" text,
	"file_size" integer,
	"mime_type" text,
	"content_embedding" vector(384),
	"title_embedding" vector(384),
	"analysis_results" jsonb,
	"legal_categories" text[],
	"citations" jsonb,
	"processing_status" text DEFAULT 'pending' NOT NULL,
	"is_confidential" boolean DEFAULT false NOT NULL,
	"retention_date" timestamp,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	"created_by" uuid,
	"last_modified_by" uuid,
	CONSTRAINT "legal_documents_file_hash_unique" UNIQUE("file_hash")
);

-- Create legal_cases table
CREATE TABLE IF NOT EXISTS "legal_cases" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"case_number" text NOT NULL,
	"title" text NOT NULL,
	"description" text,
	"client_name" text NOT NULL,
	"opposing_party" text,
	"jurisdiction" text NOT NULL,
	"court_name" text,
	"judge_assigned" text,
	"case_type" text NOT NULL,
	"practice_area" text NOT NULL,
	"priority" text DEFAULT 'medium' NOT NULL,
	"status" text DEFAULT 'active' NOT NULL,
	"filing_date" timestamp,
	"trial_date" timestamp,
	"close_date" timestamp,
	"estimated_value" integer,
	"actual_value" integer,
	"billing_rate" integer,
	"total_billed" integer DEFAULT 0,
	"case_summary" text,
	"legal_strategy" text,
	"key_issues" text[],
	"precedents" jsonb,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	"created_by" uuid NOT NULL,
	"assigned_attorney" uuid,
	CONSTRAINT "legal_cases_case_number_unique" UNIQUE("case_number")
);

-- Create case_documents relationship table
CREATE TABLE IF NOT EXISTS "case_documents" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"case_id" uuid NOT NULL,
	"document_id" uuid NOT NULL,
	"relationship" text NOT NULL,
	"importance" text DEFAULT 'medium' NOT NULL,
	"notes" text,
	"added_at" timestamp DEFAULT now() NOT NULL,
	"added_by" uuid NOT NULL,
	CONSTRAINT "case_documents_unique" UNIQUE("case_id","document_id")
);

-- Create legal_entities table
CREATE TABLE IF NOT EXISTS "legal_entities" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"name" text NOT NULL,
	"entity_type" text NOT NULL,
	"primary_email" text,
	"primary_phone" text,
	"address" jsonb,
	"jurisdiction" text,
	"tax_id" text,
	"incorporation_date" timestamp,
	"parent_entity" uuid,
	"aliases" text[],
	"name_embedding" vector(384),
	"is_active" boolean DEFAULT true NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);

-- Create agent_analysis_cache table
CREATE TABLE IF NOT EXISTS "agent_analysis_cache" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"cache_key" text NOT NULL,
	"document_id" uuid,
	"case_id" uuid,
	"agent_name" text NOT NULL,
	"analysis_type" text NOT NULL,
	"prompt" text NOT NULL,
	"response" text NOT NULL,
	"confidence" integer NOT NULL,
	"processing_time" integer,
	"token_usage" jsonb,
	"expires_at" timestamp NOT NULL,
	"access_count" integer DEFAULT 0 NOT NULL,
	"last_accessed" timestamp DEFAULT now() NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "agent_analysis_cache_cache_key_unique" UNIQUE("cache_key")
);

-- Add foreign key constraints
DO $$ BEGIN
 ALTER TABLE "case_documents" ADD CONSTRAINT "case_documents_case_id_legal_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "legal_cases"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "case_documents" ADD CONSTRAINT "case_documents_document_id_legal_documents_id_fk" FOREIGN KEY ("document_id") REFERENCES "legal_documents"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "legal_entities" ADD CONSTRAINT "legal_entities_parent_entity_legal_entities_id_fk" FOREIGN KEY ("parent_entity") REFERENCES "legal_entities"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "agent_analysis_cache" ADD CONSTRAINT "agent_analysis_cache_document_id_legal_documents_id_fk" FOREIGN KEY ("document_id") REFERENCES "legal_documents"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
 ALTER TABLE "agent_analysis_cache" ADD CONSTRAINT "agent_analysis_cache_case_id_legal_cases_id_fk" FOREIGN KEY ("case_id") REFERENCES "legal_cases"("id") ON DELETE no action ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS "legal_documents_document_type_idx" ON "legal_documents" ("document_type");
CREATE INDEX IF NOT EXISTS "legal_documents_jurisdiction_idx" ON "legal_documents" ("jurisdiction");
CREATE INDEX IF NOT EXISTS "legal_documents_practice_area_idx" ON "legal_documents" ("practice_area");
CREATE INDEX IF NOT EXISTS "legal_documents_status_idx" ON "legal_documents" ("processing_status");
CREATE INDEX IF NOT EXISTS "legal_documents_created_at_idx" ON "legal_documents" ("created_at");
CREATE INDEX IF NOT EXISTS "legal_documents_confidential_idx" ON "legal_documents" ("is_confidential");

-- Create HNSW vector indexes for similarity search
CREATE INDEX IF NOT EXISTS "legal_documents_content_embedding_hnsw_idx" 
ON "legal_documents" USING hnsw ("content_embedding" vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "legal_documents_title_embedding_hnsw_idx" 
ON "legal_documents" USING hnsw ("title_embedding" vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS "legal_entities_name_embedding_hnsw_idx" 
ON "legal_entities" USING hnsw ("name_embedding" vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Case-related indexes
CREATE INDEX IF NOT EXISTS "legal_cases_case_number_idx" ON "legal_cases" ("case_number");
CREATE INDEX IF NOT EXISTS "legal_cases_client_name_idx" ON "legal_cases" ("client_name");
CREATE INDEX IF NOT EXISTS "legal_cases_status_idx" ON "legal_cases" ("status");
CREATE INDEX IF NOT EXISTS "legal_cases_practice_area_idx" ON "legal_cases" ("practice_area");
CREATE INDEX IF NOT EXISTS "legal_cases_priority_idx" ON "legal_cases" ("priority");
CREATE INDEX IF NOT EXISTS "legal_cases_attorney_idx" ON "legal_cases" ("assigned_attorney");

-- Case documents indexes
CREATE INDEX IF NOT EXISTS "case_documents_case_id_idx" ON "case_documents" ("case_id");
CREATE INDEX IF NOT EXISTS "case_documents_document_id_idx" ON "case_documents" ("document_id");
CREATE INDEX IF NOT EXISTS "case_documents_relationship_idx" ON "case_documents" ("relationship");
CREATE INDEX IF NOT EXISTS "case_documents_importance_idx" ON "case_documents" ("importance");

-- Legal entities indexes
CREATE INDEX IF NOT EXISTS "legal_entities_name_idx" ON "legal_entities" ("name");
CREATE INDEX IF NOT EXISTS "legal_entities_type_idx" ON "legal_entities" ("entity_type");
CREATE INDEX IF NOT EXISTS "legal_entities_jurisdiction_idx" ON "legal_entities" ("jurisdiction");
CREATE INDEX IF NOT EXISTS "legal_entities_email_idx" ON "legal_entities" ("primary_email");
CREATE INDEX IF NOT EXISTS "legal_entities_active_idx" ON "legal_entities" ("is_active");

-- Agent analysis cache indexes
CREATE INDEX IF NOT EXISTS "agent_analysis_cache_key_idx" ON "agent_analysis_cache" ("cache_key");
CREATE INDEX IF NOT EXISTS "agent_analysis_cache_document_idx" ON "agent_analysis_cache" ("document_id");
CREATE INDEX IF NOT EXISTS "agent_analysis_cache_case_idx" ON "agent_analysis_cache" ("case_id");
CREATE INDEX IF NOT EXISTS "agent_analysis_cache_agent_idx" ON "agent_analysis_cache" ("agent_name");
CREATE INDEX IF NOT EXISTS "agent_analysis_cache_type_idx" ON "agent_analysis_cache" ("analysis_type");
CREATE INDEX IF NOT EXISTS "agent_analysis_cache_expires_idx" ON "agent_analysis_cache" ("expires_at");

-- Create triggers for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_legal_documents_updated_at BEFORE UPDATE ON legal_documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_legal_cases_updated_at BEFORE UPDATE ON legal_cases FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_legal_entities_updated_at BEFORE UPDATE ON legal_entities FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create GIN indexes for full-text search
CREATE INDEX IF NOT EXISTS "legal_documents_content_gin_idx" ON "legal_documents" USING gin (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS "legal_documents_title_gin_idx" ON "legal_documents" USING gin (to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS "legal_cases_summary_gin_idx" ON "legal_cases" USING gin (to_tsvector('english', case_summary));

-- Create composite indexes for common queries
CREATE INDEX IF NOT EXISTS "legal_documents_type_status_idx" ON "legal_documents" ("document_type", "processing_status");
CREATE INDEX IF NOT EXISTS "legal_cases_status_priority_idx" ON "legal_cases" ("status", "priority");

-- Add check constraints for data validation
ALTER TABLE "legal_documents" ADD CONSTRAINT "legal_documents_document_type_check" 
CHECK (document_type IN ('contract', 'motion', 'evidence', 'correspondence', 'brief', 'regulation', 'case_law'));

ALTER TABLE "legal_documents" ADD CONSTRAINT "legal_documents_processing_status_check" 
CHECK (processing_status IN ('pending', 'processing', 'completed', 'error'));

ALTER TABLE "legal_cases" ADD CONSTRAINT "legal_cases_case_type_check" 
CHECK (case_type IN ('civil', 'criminal', 'administrative', 'appellate', 'arbitration'));

ALTER TABLE "legal_cases" ADD CONSTRAINT "legal_cases_priority_check" 
CHECK (priority IN ('low', 'medium', 'high', 'critical'));

ALTER TABLE "legal_cases" ADD CONSTRAINT "legal_cases_status_check" 
CHECK (status IN ('active', 'pending', 'closed', 'archived', 'on_hold'));

ALTER TABLE "legal_entities" ADD CONSTRAINT "legal_entities_entity_type_check" 
CHECK (entity_type IN ('person', 'corporation', 'partnership', 'llc', 'government', 'nonprofit', 'trust', 'estate'));

-- Create a view for active legal documents with analytics
CREATE OR REPLACE VIEW active_legal_documents AS
SELECT 
    ld.*,
    COUNT(cd.document_id) as case_count,
    MAX(cd.added_at) as last_case_assignment
FROM legal_documents ld
LEFT JOIN case_documents cd ON ld.id = cd.document_id
WHERE ld.processing_status = 'completed'
GROUP BY ld.id;

-- Create a view for case summaries with document counts
CREATE OR REPLACE VIEW case_summaries AS
SELECT 
    lc.*,
    COUNT(cd.document_id) as document_count,
    COUNT(CASE WHEN cd.relationship = 'evidence' THEN 1 END) as evidence_count,
    COUNT(CASE WHEN cd.importance = 'critical' THEN 1 END) as critical_documents
FROM legal_cases lc
LEFT JOIN case_documents cd ON lc.id = cd.case_id
GROUP BY lc.id;

-- Insert sample configuration data
INSERT INTO legal_documents (title, content, document_type, jurisdiction, processing_status) VALUES
('System Configuration Document', 'This document contains system configuration details for the legal AI platform.', 'correspondence', 'federal', 'completed')
ON CONFLICT (file_hash) DO NOTHING;

-- Create a function for vector similarity search
CREATE OR REPLACE FUNCTION search_documents_by_embedding(
    query_embedding vector(384),
    similarity_threshold float DEFAULT 0.7,
    result_limit int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    title text,
    content text,
    document_type text,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ld.id,
        ld.title,
        ld.content,
        ld.document_type,
        (1 - (ld.content_embedding <=> query_embedding)) as similarity
    FROM legal_documents ld
    WHERE 
        ld.content_embedding IS NOT NULL
        AND (1 - (ld.content_embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY ld.content_embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE legal_documents IS 'Legal documents with vector embeddings for semantic search';
COMMENT ON TABLE legal_cases IS 'Legal cases and case management data';
COMMENT ON TABLE case_documents IS 'Relationship between cases and documents';
COMMENT ON TABLE legal_entities IS 'Legal entities (people, organizations) with name embeddings';
COMMENT ON TABLE agent_analysis_cache IS 'Cache for AI agent analysis results';