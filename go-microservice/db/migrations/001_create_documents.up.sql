-- Create documents table with pgvector integration
-- Version 2.0 Enterprise Database Schema

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table for legal document storage with vector embeddings
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA256 hash for deduplication
    
    -- Vector embeddings (768 dimensions for legal-bert, 384 for nomic-embed)
    embedding_768 VECTOR(768), -- Primary embedding for legal documents
    embedding_384 VECTOR(384), -- Secondary embedding for faster search
    
    -- Legal document metadata (JSONB for flexible querying)
    metadata JSONB NOT NULL DEFAULT '{}',
    
    -- Document classification
    document_type VARCHAR(50) NOT NULL DEFAULT 'unknown', -- contract, case, statute, etc.
    jurisdiction VARCHAR(100),
    language VARCHAR(10) NOT NULL DEFAULT 'en',
    
    -- Processing status
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending, processing, completed, error
    processing_error TEXT,
    
    -- Performance and caching
    search_rank INTEGER DEFAULT 0, -- Precomputed search ranking
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    
    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    updated_by UUID
);

-- Create indexes for vector similarity search
CREATE INDEX idx_documents_embedding_768_cosine ON documents 
    USING hnsw (embedding_768 vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_documents_embedding_384_cosine ON documents 
    USING hnsw (embedding_384 vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Additional vector indexes for different distance metrics
CREATE INDEX idx_documents_embedding_768_l2 ON documents 
    USING hnsw (embedding_768 vector_l2_ops) WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_documents_embedding_384_l2 ON documents 
    USING hnsw (embedding_384 vector_l2_ops) WITH (m = 16, ef_construction = 64);

-- JSONB GIN index for metadata queries
CREATE INDEX idx_documents_metadata_gin ON documents USING gin (metadata);

-- Traditional indexes for common queries
CREATE INDEX idx_documents_type ON documents (document_type);
CREATE INDEX idx_documents_jurisdiction ON documents (jurisdiction);
CREATE INDEX idx_documents_status ON documents (processing_status);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);
CREATE INDEX idx_documents_search_rank ON documents (search_rank DESC);
CREATE INDEX idx_documents_access_count ON documents (access_count DESC);

-- Composite indexes for complex queries
CREATE INDEX idx_documents_type_jurisdiction ON documents (document_type, jurisdiction);
CREATE INDEX idx_documents_status_created ON documents (processing_status, created_at DESC);

-- Full-text search index
CREATE INDEX idx_documents_content_fts ON documents USING gin (to_tsvector('english', content));
CREATE INDEX idx_documents_title_fts ON documents USING gin (to_tsvector('english', title));

-- Document relationships table for legal citations and references
CREATE TABLE document_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- cites, references, supersedes, etc.
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    
    UNIQUE(source_document_id, target_document_id, relationship_type)
);

-- Indexes for relationship queries
CREATE INDEX idx_relationships_source ON document_relationships (source_document_id);
CREATE INDEX idx_relationships_target ON document_relationships (target_document_id);
CREATE INDEX idx_relationships_type ON document_relationships (relationship_type);
CREATE INDEX idx_relationships_confidence ON document_relationships (confidence_score DESC);

-- Legal entities extraction table
CREATE TABLE legal_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL, -- person, organization, court, case, etc.
    entity_name TEXT NOT NULL,
    entity_role VARCHAR(50), -- plaintiff, defendant, judge, etc.
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    start_position INTEGER,
    end_position INTEGER,
    context TEXT, -- Surrounding text for context
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for entity queries
CREATE INDEX idx_entities_document ON legal_entities (document_id);
CREATE INDEX idx_entities_type ON legal_entities (entity_type);
CREATE INDEX idx_entities_name ON legal_entities (entity_name);
CREATE INDEX idx_entities_role ON legal_entities (entity_role);
CREATE INDEX idx_entities_confidence ON legal_entities (confidence_score DESC);

-- Processing jobs table for async operations
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL, -- vector_processing, document_analysis, similarity_computation
    status VARCHAR(20) NOT NULL DEFAULT 'queued', -- queued, processing, completed, failed, cancelled
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    
    -- Job payload and configuration
    input_data JSONB NOT NULL,
    output_data JSONB,
    processing_options JSONB DEFAULT '{}',
    
    -- Progress tracking
    progress_percentage FLOAT DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    current_stage TEXT,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Performance metrics
    processing_time_ms FLOAT,
    gpu_memory_used_mb FLOAT,
    
    -- Scheduling
    scheduled_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    
    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    
    -- Resource allocation
    assigned_worker VARCHAR(100),
    resource_requirements JSONB DEFAULT '{}'
);

-- Indexes for job processing
CREATE INDEX idx_jobs_status ON processing_jobs (status);
CREATE INDEX idx_jobs_type ON processing_jobs (job_type);
CREATE INDEX idx_jobs_priority ON processing_jobs (priority DESC);
CREATE INDEX idx_jobs_scheduled ON processing_jobs (scheduled_at) WHERE status = 'queued';
CREATE INDEX idx_jobs_created ON processing_jobs (created_at DESC);
CREATE INDEX idx_jobs_worker ON processing_jobs (assigned_worker);

-- Composite index for job queue processing
CREATE INDEX idx_jobs_queue ON processing_jobs (status, priority DESC, created_at) 
    WHERE status IN ('queued', 'processing');

-- System metrics table for monitoring and observability
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL, -- counter, gauge, histogram
    metric_value FLOAT NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Partitioning hint for time-series data
    recorded_date DATE GENERATED ALWAYS AS (timestamp::date) STORED
);

-- Indexes for metrics queries
CREATE INDEX idx_metrics_name_timestamp ON system_metrics (metric_name, timestamp DESC);
CREATE INDEX idx_metrics_type ON system_metrics (metric_type);
CREATE INDEX idx_metrics_date ON system_metrics (recorded_date);

-- Cache entries table for multi-layer caching
CREATE TABLE cache_entries (
    cache_key VARCHAR(255) PRIMARY KEY,
    cache_namespace VARCHAR(100) NOT NULL,
    cache_value JSONB NOT NULL,
    ttl_seconds INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    size_bytes INTEGER
);

-- Indexes for cache management
CREATE INDEX idx_cache_namespace ON cache_entries (cache_namespace);
CREATE INDEX idx_cache_expires ON cache_entries (expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_cache_access ON cache_entries (last_accessed);

-- User sessions table for authentication integration
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for session management
CREATE INDEX idx_sessions_user ON user_sessions (user_id);
CREATE INDEX idx_sessions_token ON user_sessions (session_token);
CREATE INDEX idx_sessions_expires ON user_sessions (expires_at);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update access statistics
CREATE OR REPLACE FUNCTION update_document_access_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE documents 
    SET access_count = access_count + 1, 
        last_accessed = NOW() 
    WHERE id = NEW.id;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Function for cache cleanup
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM cache_entries 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Function for job cleanup
CREATE OR REPLACE FUNCTION cleanup_completed_jobs(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM processing_jobs 
    WHERE status IN ('completed', 'failed', 'cancelled')
    AND completed_at < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Create a view for active documents with computed fields
CREATE VIEW active_documents AS
SELECT 
    d.*,
    -- Computed similarity search score
    CASE 
        WHEN d.access_count > 0 THEN d.search_rank * LOG(d.access_count + 1)
        ELSE d.search_rank
    END as computed_relevance_score,
    
    -- Document age in days
    EXTRACT(DAYS FROM NOW() - d.created_at) as age_days,
    
    -- Entity count
    (SELECT COUNT(*) FROM legal_entities e WHERE e.document_id = d.id) as entity_count,
    
    -- Relationship count
    (SELECT COUNT(*) FROM document_relationships r 
     WHERE r.source_document_id = d.id OR r.target_document_id = d.id) as relationship_count
     
FROM documents d
WHERE d.processing_status = 'completed';

-- Grant permissions (adjust as needed for your security model)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO legal_ai_service;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO legal_ai_service;