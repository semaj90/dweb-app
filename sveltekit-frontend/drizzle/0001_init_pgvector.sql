-- Initial migration for PostgreSQL with pgvector extension
-- Date: 2025-09-01

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (core authentication)
CREATE TABLE IF NOT EXISTS "users" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "email" varchar(255) UNIQUE NOT NULL,
  "hashed_password" varchar(255),
  "username" varchar(100),
  "first_name" varchar(100),
  "last_name" varchar(100),
  "role" varchar(50) DEFAULT 'user' NOT NULL,
  "department" varchar(100),
  "jurisdiction" varchar(100),
  "permissions" jsonb DEFAULT '[]' NOT NULL,
  "is_active" boolean DEFAULT true NOT NULL,
  "email_verified" boolean DEFAULT false NOT NULL,
  "avatar_url" varchar(500),
  "last_login_at" timestamp with time zone,
  "practice_areas" jsonb DEFAULT '[]',
  "bar_number" varchar(50),
  "firm_name" varchar(200),
  "profile_embedding" vector(384),
  "metadata" jsonb DEFAULT '{}',
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  "updated_at" timestamp with time zone DEFAULT now() NOT NULL,
  "deleted_at" timestamp with time zone
);

-- Sessions table (Lucia v3 compatible)
CREATE TABLE IF NOT EXISTS "sessions" (
  "id" varchar(255) PRIMARY KEY,
  "user_id" uuid NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
  "expires_at" timestamp with time zone NOT NULL,
  "ip_address" varchar(45),
  "user_agent" text,
  "session_context" jsonb DEFAULT '{}',
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Cases table
CREATE TABLE IF NOT EXISTS "cases" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "case_number" varchar(100),
  "title" varchar(255) NOT NULL,
  "description" text,
  "status" varchar(50) DEFAULT 'open' NOT NULL,
  "priority" varchar(20) DEFAULT 'medium' NOT NULL,
  "assigned_attorney" uuid REFERENCES "users"("id"),
  "created_by" uuid REFERENCES "users"("id"),
  "assigned_to" uuid REFERENCES "users"("id"),
  "metadata" jsonb DEFAULT '{}',
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  "updated_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Evidence table with embeddings
CREATE TABLE IF NOT EXISTS "evidence" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "case_id" uuid REFERENCES "cases"("id") ON DELETE CASCADE,
  "title" varchar(255) NOT NULL,
  "description" text,
  "file_name" varchar(255),
  "original_file_name" varchar(255),
  "file_size" varchar(50),
  "file_type" varchar(100),
  "file_path" varchar(500),
  "evidence_type" varchar(100) NOT NULL,
  "type" varchar(100),
  "created_by" uuid REFERENCES "users"("id"),
  "tags" jsonb DEFAULT '[]',
  "metadata" jsonb DEFAULT '{}',
  "is_public" boolean DEFAULT false,
  "ocr_text" text,
  "content_text" text,
  "embedding" vector(384),
  "uploaded_at" timestamp with time zone DEFAULT now() NOT NULL,
  "processed_at" timestamp with time zone,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  "updated_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Document chunks for RAG
CREATE TABLE IF NOT EXISTS "document_chunks" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "document_id" uuid NOT NULL,
  "document_type" varchar(100) DEFAULT 'evidence' NOT NULL,
  "chunk_index" varchar(50) NOT NULL,
  "content" text NOT NULL,
  "embedding" vector(384),
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Legal documents
CREATE TABLE IF NOT EXISTS "legal_documents" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "title" varchar(255) NOT NULL,
  "document_type" varchar(100) NOT NULL,
  "content" text,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  "updated_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Embedding cache for performance
CREATE TABLE IF NOT EXISTS "embedding_cache" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "content_hash" varchar(255) UNIQUE NOT NULL,
  "embedding" vector(384),
  "model_name" varchar(100) NOT NULL,
  "metadata" jsonb DEFAULT '{}',
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  "expires_at" timestamp with time zone
);

-- Unified vector storage for cross-entity search
CREATE TABLE IF NOT EXISTS "vectors" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "entity_type" varchar(100) NOT NULL, -- 'case'|'evidence'|'chunk'|'user'
  "entity_id" uuid NOT NULL,
  "embedding" vector(384) NOT NULL,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS "users_email_idx" ON "users" ("email");
CREATE INDEX IF NOT EXISTS "users_username_idx" ON "users" ("username");
CREATE INDEX IF NOT EXISTS "users_role_idx" ON "users" ("role");
CREATE INDEX IF NOT EXISTS "users_active_idx" ON "users" ("is_active");
CREATE INDEX IF NOT EXISTS "users_profile_embedding_hnsw_idx" ON "users" USING hnsw ("profile_embedding" vector_cosine_ops);

CREATE INDEX IF NOT EXISTS "sessions_expires_at_idx" ON "sessions" ("expires_at");
CREATE INDEX IF NOT EXISTS "sessions_user_id_idx" ON "sessions" ("user_id");

CREATE INDEX IF NOT EXISTS "evidence_case_id_idx" ON "evidence" ("case_id");
CREATE INDEX IF NOT EXISTS "evidence_file_type_idx" ON "evidence" ("file_type");
CREATE INDEX IF NOT EXISTS "evidence_uploaded_at_idx" ON "evidence" ("uploaded_at");
CREATE INDEX IF NOT EXISTS "evidence_embedding_hnsw_idx" ON "evidence" USING hnsw ("embedding" vector_cosine_ops);
CREATE INDEX IF NOT EXISTS "evidence_tags_gin_idx" ON "evidence" USING gin ("tags");

CREATE INDEX IF NOT EXISTS "document_chunks_document_id_idx" ON "document_chunks" ("document_id");
CREATE INDEX IF NOT EXISTS "document_chunks_embedding_hnsw_idx" ON "document_chunks" USING hnsw ("embedding" vector_cosine_ops);

CREATE INDEX IF NOT EXISTS "embedding_cache_content_hash_idx" ON "embedding_cache" ("content_hash");
CREATE INDEX IF NOT EXISTS "embedding_cache_embedding_hnsw_idx" ON "embedding_cache" USING hnsw ("embedding" vector_cosine_ops);

CREATE INDEX IF NOT EXISTS "vectors_entity_type_idx" ON "vectors" ("entity_type");
CREATE INDEX IF NOT EXISTS "vectors_entity_id_idx" ON "vectors" ("entity_id");
CREATE INDEX IF NOT EXISTS "vectors_embedding_hnsw_idx" ON "vectors" USING hnsw ("embedding" vector_cosine_ops);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS "evidence_content_text_fts_idx" ON "evidence" USING gin(to_tsvector('english', "content_text"));
CREATE INDEX IF NOT EXISTS "document_chunks_content_fts_idx" ON "document_chunks" USING gin(to_tsvector('english', "content"));

-- Metadata indexes for flexible queries
CREATE INDEX IF NOT EXISTS "evidence_metadata_gin_idx" ON "evidence" USING gin ("metadata");
CREATE INDEX IF NOT EXISTS "cases_metadata_gin_idx" ON "cases" USING gin ("metadata");
CREATE INDEX IF NOT EXISTS "users_metadata_gin_idx" ON "users" USING gin ("metadata");

-- Update triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON "users" FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cases_updated_at BEFORE UPDATE ON "cases" FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_evidence_updated_at BEFORE UPDATE ON "evidence" FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_legal_documents_updated_at BEFORE UPDATE ON "legal_documents" FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();