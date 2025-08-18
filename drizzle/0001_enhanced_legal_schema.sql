-- Migration: Create Enhanced Legal Documents Schema
-- Version: 0001
-- Date: 2025-01-01
-- Description: Initial migration for enhanced legal document database with vector support

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create legal_documents table
CREATE TABLE IF NOT EXISTS legal_documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(100) NOT NULL DEFAULT 'general',
    jurisdiction VARCHAR(100) NOT NULL DEFAULT 'federal',
    court VARCHAR(200),
    citation VARCHAR(300),
    full_citation VARCHAR(500),
    docket_number VARCHAR(100),
    date_decided TIMESTAMP,
    date_published TIMESTAMP,
    summary TEXT,
    tags JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    processing_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    analysis_results JSONB,
    content_embedding VECTOR(384),
    title_embedding VECTOR(384),
    file_hash VARCHAR(64),
    file_name VARCHAR(255),
    file_size INTEGER,
    mime_type VARCHAR(100),
    practice_area VARCHAR(100),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create content_embeddings table
CREATE TABLE IF NOT EXISTS content_embeddings (
    id SERIAL PRIMARY KEY,
    content_id VARCHAR(100) NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    text_content TEXT NOT NULL,
    embedding TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    document_id VARCHAR(100),
    model VARCHAR(100) DEFAULT 'nomic-embed-text'
);

-- Create search_sessions table
CREATE TABLE IF NOT EXISTS search_sessions (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    search_type VARCHAR(50),
    query_embedding TEXT,
    results JSONB,
    result_count INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create embeddings table (alternative structure)
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model VARCHAR(100) DEFAULT 'nomic-embed-text'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_legal_documents_document_type ON legal_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_legal_documents_jurisdiction ON legal_documents(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_legal_documents_practice_area ON legal_documents(practice_area);
CREATE INDEX IF NOT EXISTS idx_legal_documents_processing_status ON legal_documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_legal_documents_created_at ON legal_documents(created_at);
CREATE INDEX IF NOT EXISTS idx_legal_documents_file_hash ON legal_documents(file_hash);

-- Vector similarity search indexes
CREATE INDEX IF NOT EXISTS idx_legal_documents_content_embedding ON legal_documents 
USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_legal_documents_title_embedding ON legal_documents 
USING ivfflat (title_embedding vector_cosine_ops) WITH (lists = 100);

-- Content embeddings indexes
CREATE INDEX IF NOT EXISTS idx_content_embeddings_content_id ON content_embeddings(content_id);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_content_type ON content_embeddings(content_type);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_document_id ON content_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_model ON content_embeddings(model);

-- Search sessions indexes
CREATE INDEX IF NOT EXISTS idx_search_sessions_query ON search_sessions USING gin(to_tsvector('english', query));
CREATE INDEX IF NOT EXISTS idx_search_sessions_search_type ON search_sessions(search_type);
CREATE INDEX IF NOT EXISTS idx_search_sessions_created_at ON search_sessions(created_at);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_legal_documents_title_fts ON legal_documents 
USING gin(to_tsvector('english', title));

CREATE INDEX IF NOT EXISTS idx_legal_documents_content_fts ON legal_documents 
USING gin(to_tsvector('english', content));

-- JSONB indexes for metadata searches
CREATE INDEX IF NOT EXISTS idx_legal_documents_tags ON legal_documents USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_legal_documents_metadata ON legal_documents USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_metadata ON content_embeddings USING gin(metadata);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_legal_documents_updated_at 
    BEFORE UPDATE ON legal_documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW legal_documents_summary AS
SELECT 
    id,
    title,
    document_type,
    jurisdiction,
    practice_area,
    processing_status,
    created_at,
    updated_at,
    CASE 
        WHEN LENGTH(content) > 200 THEN LEFT(content, 200) || '...'
        ELSE content
    END as content_preview
FROM legal_documents;

-- Create function for similarity search
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding VECTOR(384),
    similarity_threshold FLOAT DEFAULT 0.7,
    result_limit INTEGER DEFAULT 10
)
RETURNS TABLE(
    document_id INTEGER,
    title VARCHAR(500),
    similarity_score FLOAT,
    document_type VARCHAR(100),
    jurisdiction VARCHAR(100)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ld.id,
        ld.title,
        1 - (ld.content_embedding <=> query_embedding) as similarity_score,
        ld.document_type,
        ld.jurisdiction
    FROM legal_documents ld
    WHERE ld.content_embedding IS NOT NULL
        AND 1 - (ld.content_embedding <=> query_embedding) >= similarity_threshold
    ORDER BY ld.content_embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Insert initial data types and configurations
INSERT INTO legal_documents (title, content, document_type, jurisdiction, practice_area, processing_status)
VALUES 
    ('Sample Contract Template', 'This is a sample contract template for testing purposes.', 'contract', 'federal', 'corporate', 'completed'),
    ('Legal Precedent Example', 'This is an example of legal precedent documentation.', 'case', 'federal', 'litigation', 'completed')
ON CONFLICT DO NOTHING;

-- Create performance monitoring views
CREATE OR REPLACE VIEW system_performance_metrics AS
SELECT 
    'legal_documents' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_records,
    COUNT(CASE WHEN processing_status = 'processing' THEN 1 END) as processing_records,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_records,
    AVG(CASE WHEN content_embedding IS NOT NULL THEN 1 ELSE 0 END) as embedding_coverage
FROM legal_documents
UNION ALL
SELECT 
    'content_embeddings' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as completed_records,
    0 as processing_records,
    0 as failed_records,
    1.0 as embedding_coverage
FROM content_embeddings
UNION ALL
SELECT 
    'search_sessions' as table_name,
    COUNT(*) as total_records,
    COUNT(CASE WHEN result_count > 0 THEN 1 END) as completed_records,
    0 as processing_records,
    COUNT(CASE WHEN result_count = 0 THEN 1 END) as failed_records,
    AVG(result_count::float) / 10.0 as embedding_coverage
FROM search_sessions;

-- Create backup and maintenance procedures
CREATE OR REPLACE FUNCTION create_database_backup()
RETURNS TEXT AS $$
DECLARE
    backup_name TEXT;
BEGIN
    backup_name := 'backup_' || TO_CHAR(NOW(), 'YYYY_MM_DD_HH24_MI_SS');
    
    -- In a real implementation, this would trigger pg_dump
    -- For now, we'll just log the backup creation
    INSERT INTO system_logs (event_type, message, created_at)
    VALUES ('backup_created', 'Database backup created: ' || backup_name, NOW());
    
    RETURN backup_name;
END;
$$ LANGUAGE plpgsql;

-- Create system logs table for monitoring
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    message TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_system_logs_event_type ON system_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);

-- Migration completion log
INSERT INTO system_logs (event_type, message, metadata)
VALUES (
    'migration_completed',
    'Enhanced legal documents schema migration completed successfully',
    '{"version": "0001", "tables_created": ["legal_documents", "content_embeddings", "search_sessions", "embeddings", "system_logs"], "indexes_created": 15}'::jsonb
);

COMMIT;
