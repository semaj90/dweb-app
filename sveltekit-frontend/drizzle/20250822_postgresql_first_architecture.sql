-- Migration: PostgreSQL-First Architecture Schema Updates
-- Generated: 2025-08-22
-- Purpose: Ensure all tables exist for PostgreSQL-centered legal AI platform

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Ensure users table exists with proper structure
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user' NOT NULL,
    display_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Ensure cases table exists
CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'open' NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium' NOT NULL,
    case_type VARCHAR(100),
    metadata JSONB DEFAULT '{}' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Enhanced evidence table (master record for all evidence)
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Foreign Keys
    case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Core Fields
    title VARCHAR(255) NOT NULL,
    description TEXT,
    evidence_type VARCHAR(50) NOT NULL,
    sub_type VARCHAR(50),
    
    -- File Information
    file_name VARCHAR(255),
    file_size INTEGER,
    mime_type VARCHAR(100),
    hash VARCHAR(128),
    
    -- Storage Information
    storage_path VARCHAR(512), -- MinIO object path
    storage_bucket VARCHAR(255) DEFAULT 'evidence',
    
    -- Collection Details
    collected_at TIMESTAMP WITH TIME ZONE,
    collected_by VARCHAR(255),
    location VARCHAR(255),
    chain_of_custody JSONB DEFAULT '[]' NOT NULL,
    
    -- Classification
    tags JSONB DEFAULT '[]' NOT NULL,
    is_admissible BOOLEAN DEFAULT true,
    confidentiality_level VARCHAR(50) DEFAULT 'internal',
    
    -- AI Analysis (PostgreSQL as single source of truth)
    ai_analysis JSONB DEFAULT '{}',
    ai_tags JSONB DEFAULT '[]',
    ai_summary TEXT,
    summary TEXT,
    summary_type VARCHAR(50),
    
    -- Processing Status
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    ingest_status VARCHAR(50) DEFAULT 'pending', -- pending, ingested, failed
    
    -- Vector Embeddings (stored in PostgreSQL)
    title_embedding vector(384),
    content_embedding vector(384),
    
    -- Board Position (for visual layout)
    board_position JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Document metadata table (for Go ingest-service)
CREATE TABLE IF NOT EXISTS document_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relations
    case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
    evidence_id UUID REFERENCES evidence(id) ON DELETE CASCADE, -- Link to evidence
    
    -- File Information
    filename VARCHAR(512), -- Required by Go service
    object_name VARCHAR(512) UNIQUE, -- Required by Go service (MinIO path)
    original_filename VARCHAR(512) NOT NULL,
    summary TEXT,
    content_type VARCHAR(100),
    file_size INTEGER,
    
    -- Processing Information
    processing_status VARCHAR(50) DEFAULT 'pending',
    extracted_text TEXT, -- Full text content
    document_type VARCHAR(100), -- legal, evidence, case, etc.
    jurisdiction VARCHAR(100), -- US, State, Federal
    priority INTEGER DEFAULT 1, -- Processing priority
    ingest_source VARCHAR(100) DEFAULT 'manual', -- manual, api, batch
    
    -- Metadata
    metadata JSONB DEFAULT '{}' NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Document embeddings table (PostgreSQL as embedding store)
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Relations (flexible - can link to documents OR evidence)
    document_id UUID REFERENCES document_metadata(id) ON DELETE CASCADE,
    evidence_id UUID REFERENCES evidence(id) ON DELETE CASCADE,
    
    -- Content and Embedding
    content TEXT NOT NULL, -- The text that was embedded
    embedding vector(384) NOT NULL, -- pgvector embedding
    
    -- Chunking Information
    chunk_index INTEGER DEFAULT 0,
    chunk_text TEXT, -- Alias for compatibility
    chunk_size INTEGER DEFAULT 0,
    chunk_overlap INTEGER DEFAULT 0,
    parent_chunk_id UUID, -- For hierarchical chunking
    
    -- Model Information
    embedding_model VARCHAR(100) DEFAULT 'nomic-embed-text',
    model_version VARCHAR(50) DEFAULT 'latest',
    
    -- Search Optimization
    similarity REAL, -- Cached similarity scores
    
    -- Metadata
    metadata JSONB DEFAULT '{}' NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Case embeddings for case-level search
CREATE TABLE IF NOT EXISTS case_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    metadata JSONB DEFAULT '{}' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Redis stream events log (track all worker events)
CREATE TABLE IF NOT EXISTS worker_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Event Information
    event_type VARCHAR(50) NOT NULL, -- evidence, document, ingest_complete
    event_action VARCHAR(50), -- tag, mirror, process
    
    -- Target Information
    target_id UUID NOT NULL, -- ID of evidence/document being processed
    case_id UUID REFERENCES cases(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Processing Information
    worker_name VARCHAR(100) DEFAULT 'postgresql-first-worker',
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, processing, completed, failed
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Redis Stream Information
    stream_name VARCHAR(100) DEFAULT 'autotag:requests',
    redis_message_id VARCHAR(100),
    
    -- Performance Metrics
    processing_time_ms INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}' NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    retry_after TIMESTAMP WITH TIME ZONE
);

-- Embedding cache table
CREATE TABLE IF NOT EXISTS embedding_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text_hash VARCHAR(128) NOT NULL UNIQUE, -- SHA-256 hash of text
    embedding vector(384) NOT NULL,
    model VARCHAR(100) DEFAULT 'nomic-embed-text' NOT NULL,
    metadata JSONB DEFAULT '{}',
    access_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS evidence_case_id_idx ON evidence(case_id);
CREATE INDEX IF NOT EXISTS evidence_user_id_idx ON evidence(user_id);
CREATE INDEX IF NOT EXISTS evidence_evidence_type_idx ON evidence(evidence_type);
CREATE INDEX IF NOT EXISTS evidence_processing_status_idx ON evidence(processing_status);
CREATE INDEX IF NOT EXISTS evidence_ingest_status_idx ON evidence(ingest_status);
CREATE INDEX IF NOT EXISTS evidence_created_at_idx ON evidence(created_at);
CREATE INDEX IF NOT EXISTS evidence_hash_idx ON evidence(hash);

-- Vector similarity indexes for evidence
CREATE INDEX IF NOT EXISTS evidence_title_embedding_idx 
ON evidence USING hnsw (title_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS evidence_content_embedding_idx 
ON evidence USING hnsw (content_embedding vector_cosine_ops);

-- Document metadata indexes
CREATE INDEX IF NOT EXISTS doc_metadata_case_id_idx ON document_metadata(case_id);
CREATE INDEX IF NOT EXISTS doc_metadata_evidence_id_idx ON document_metadata(evidence_id);
CREATE INDEX IF NOT EXISTS doc_metadata_filename_idx ON document_metadata(original_filename);
CREATE INDEX IF NOT EXISTS doc_metadata_status_idx ON document_metadata(processing_status);
CREATE INDEX IF NOT EXISTS doc_metadata_type_idx ON document_metadata(document_type);
CREATE INDEX IF NOT EXISTS doc_metadata_priority_idx ON document_metadata(priority);
CREATE INDEX IF NOT EXISTS doc_metadata_object_name_idx ON document_metadata(object_name);

-- Document embeddings indexes
CREATE INDEX IF NOT EXISTS doc_embeddings_document_id_idx ON document_embeddings(document_id);
CREATE INDEX IF NOT EXISTS doc_embeddings_evidence_id_idx ON document_embeddings(evidence_id);
CREATE INDEX IF NOT EXISTS doc_embeddings_chunk_idx_idx ON document_embeddings(chunk_index);
CREATE INDEX IF NOT EXISTS doc_embeddings_model_idx ON document_embeddings(embedding_model);
CREATE INDEX IF NOT EXISTS doc_embeddings_similarity_idx ON document_embeddings(similarity);
CREATE INDEX IF NOT EXISTS doc_embeddings_embedding_idx 
ON document_embeddings USING hnsw (embedding vector_cosine_ops);

-- Case embeddings indexes
CREATE INDEX IF NOT EXISTS case_embeddings_case_id_idx ON case_embeddings(case_id);
CREATE INDEX IF NOT EXISTS case_embeddings_embedding_idx 
ON case_embeddings USING hnsw (embedding vector_cosine_ops);

-- Worker events indexes
CREATE INDEX IF NOT EXISTS worker_events_event_type_idx ON worker_events(event_type);
CREATE INDEX IF NOT EXISTS worker_events_target_id_idx ON worker_events(target_id);
CREATE INDEX IF NOT EXISTS worker_events_case_id_idx ON worker_events(case_id);
CREATE INDEX IF NOT EXISTS worker_events_processing_status_idx ON worker_events(processing_status);
CREATE INDEX IF NOT EXISTS worker_events_created_at_idx ON worker_events(created_at);
CREATE INDEX IF NOT EXISTS worker_events_retry_after_idx ON worker_events(retry_after);

-- Embedding cache indexes
CREATE INDEX IF NOT EXISTS embedding_cache_text_hash_idx ON embedding_cache(text_hash);
CREATE INDEX IF NOT EXISTS embedding_cache_model_idx ON embedding_cache(model);
CREATE INDEX IF NOT EXISTS embedding_cache_last_accessed_idx ON embedding_cache(last_accessed);

-- Add missing columns to existing tables
DO $$ 
BEGIN 
    -- Add storage columns to evidence if they don't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'evidence' AND column_name = 'storage_path'
    ) THEN
        ALTER TABLE evidence ADD COLUMN storage_path VARCHAR(512);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'evidence' AND column_name = 'storage_bucket'
    ) THEN
        ALTER TABLE evidence ADD COLUMN storage_bucket VARCHAR(255) DEFAULT 'evidence';
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'evidence' AND column_name = 'processing_status'
    ) THEN
        ALTER TABLE evidence ADD COLUMN processing_status VARCHAR(50) DEFAULT 'pending';
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'evidence' AND column_name = 'ingest_status'
    ) THEN
        ALTER TABLE evidence ADD COLUMN ingest_status VARCHAR(50) DEFAULT 'pending';
    END IF;
    
    -- Add evidence_id to document_metadata if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_metadata' AND column_name = 'evidence_id'
    ) THEN
        ALTER TABLE document_metadata ADD COLUMN evidence_id UUID REFERENCES evidence(id) ON DELETE CASCADE;
    END IF;
    
    -- Add filename and object_name if they don't exist (required by Go service)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_metadata' AND column_name = 'filename'
    ) THEN
        ALTER TABLE document_metadata ADD COLUMN filename VARCHAR(512);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_metadata' AND column_name = 'object_name'
    ) THEN
        ALTER TABLE document_metadata ADD COLUMN object_name VARCHAR(512) UNIQUE;
    END IF;
    
    -- Add model_version to document_embeddings if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_embeddings' AND column_name = 'model_version'
    ) THEN
        ALTER TABLE document_embeddings ADD COLUMN model_version VARCHAR(50) DEFAULT 'latest';
    END IF;
END $$;

-- PostgreSQL notification function for ingest completion
CREATE OR REPLACE FUNCTION notify_ingest_completion()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify when document processing is completed
    IF NEW.processing_status = 'completed' AND OLD.processing_status != 'completed' THEN
        PERFORM pg_notify('ingest_completed', 
            json_build_object(
                'document_id', NEW.id,
                'case_id', NEW.case_id,
                'evidence_id', NEW.evidence_id,
                'processing_status', NEW.processing_status,
                'timestamp', NOW()
            )::text
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for ingest completion notifications
DROP TRIGGER IF EXISTS trigger_notify_ingest_completion ON document_metadata;
CREATE TRIGGER trigger_notify_ingest_completion
    AFTER UPDATE ON document_metadata
    FOR EACH ROW
    EXECUTE FUNCTION notify_ingest_completion();

-- PostgreSQL notification function for evidence updates
CREATE OR REPLACE FUNCTION notify_evidence_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify when evidence processing status changes
    IF NEW.processing_status != OLD.processing_status OR NEW.ingest_status != OLD.ingest_status THEN
        PERFORM pg_notify('evidence_updated', 
            json_build_object(
                'evidence_id', NEW.id,
                'case_id', NEW.case_id,
                'processing_status', NEW.processing_status,
                'ingest_status', NEW.ingest_status,
                'timestamp', NOW()
            )::text
        );
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for evidence update notifications
DROP TRIGGER IF EXISTS trigger_notify_evidence_update ON evidence;
CREATE TRIGGER trigger_notify_evidence_update
    AFTER UPDATE ON evidence
    FOR EACH ROW
    EXECUTE FUNCTION notify_evidence_update();

-- Update migration tracking
INSERT INTO __drizzle_migrations__ (id, hash, created_at) 
VALUES (
    '20250822_postgresql_first_architecture',
    md5('20250822_postgresql_first_architecture'),
    NOW()
) ON CONFLICT (id) DO NOTHING;

COMMIT;