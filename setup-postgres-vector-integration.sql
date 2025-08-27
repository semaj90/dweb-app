-- Enhanced PostgreSQL Vector Integration Schema for Legal AI Platform
-- Compatible with pgvector extension and nomic-embed-text (768 dimensions)
-- Optimized for production-grade legal document processing and AI analysis

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Enhanced Users table with AI interaction tracking
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    
    -- AI Usage Analytics
    total_ai_queries INTEGER DEFAULT 0,
    total_tokens_used BIGINT DEFAULT 0,
    preferred_ai_model VARCHAR(100) DEFAULT 'gemma3-legal',
    ai_preferences JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Enhanced Cases table with AI analysis support
CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    priority VARCHAR(20) DEFAULT 'medium',
    case_number VARCHAR(100) UNIQUE,
    
    -- AI Analysis Results
    ai_summary TEXT,
    ai_risk_assessment JSONB,
    ai_recommendations TEXT[],
    sentiment_score REAL,
    complexity_index REAL,
    
    -- Legal Classification
    legal_categories TEXT[],
    jurisdiction VARCHAR(100),
    practice_areas TEXT[],
    
    -- User assignments
    created_by UUID REFERENCES users(id),
    assigned_to UUID REFERENCES users(id),
    
    -- Vector embedding for case similarity
    case_embedding VECTOR(768),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced Documents table with comprehensive AI analysis
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES cases(id),
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255),
    file_type VARCHAR(50),
    file_size INTEGER,
    content TEXT,
    extracted_text TEXT,
    
    -- Vector embeddings for similarity search (nomic-embed-text: 768 dimensions)
    embedding VECTOR(768),
    summary_embedding VECTOR(768),
    
    -- AI Analysis Results
    ai_analysis JSONB,
    semantic_entities JSONB,
    legal_concepts JSONB,
    
    -- Document Classification
    document_type VARCHAR(50),
    confidence_score REAL,
    processing_status VARCHAR(20) DEFAULT 'pending',
    
    -- Metadata and indexing
    metadata JSONB,
    tags TEXT[] DEFAULT '{}',
    is_indexed BOOLEAN DEFAULT false,
    source VARCHAR(100) DEFAULT 'upload',
    
    -- OCR Results (for scanned documents)
    ocr_text TEXT,
    ocr_confidence REAL,
    ocr_language VARCHAR(10),
    
    -- User tracking
    created_by UUID REFERENCES users(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document Chunks for optimized retrieval and chunked processing
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    
    -- Vector embedding for each chunk (768 dimensions for nomic-embed-text)
    embedding VECTOR(768) NOT NULL,
    
    -- Chunk metadata
    start_index INTEGER,
    end_index INTEGER,
    token_count INTEGER,
    metadata JSONB,
    
    -- AI analysis of chunk
    entities JSONB,
    concepts JSONB,
    significance_score REAL,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced Evidence table with chain of custody and AI verification
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID REFERENCES cases(id),
    document_id UUID REFERENCES documents(id),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    evidence_type VARCHAR(50),
    
    -- Digital integrity
    hash VARCHAR(256),
    file_path TEXT,
    
    -- Chain of custody
    chain_of_custody JSONB DEFAULT '[]',
    is_admissible BOOLEAN,
    admissibility_notes TEXT,
    
    -- AI Analysis
    ai_analysis JSONB,
    authenticity_score REAL,
    relevance_score REAL,
    
    -- Vector embedding for similarity search
    embedding VECTOR(768),
    
    -- Metadata
    tags TEXT[] DEFAULT '{}',
    metadata JSONB,
    
    -- User tracking
    created_by UUID REFERENCES users(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI Interactions and Chat History
CREATE TABLE IF NOT EXISTS ai_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    case_id UUID REFERENCES cases(id),
    session_id VARCHAR(255),
    
    -- Conversation content
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    
    -- AI Model information
    model VARCHAR(100),
    temperature REAL,
    max_tokens INTEGER,
    
    -- Performance metrics
    tokens_used INTEGER,
    response_time INTEGER,
    confidence REAL,
    
    -- Context and analysis
    context_embedding VECTOR(768),
    intent_classification VARCHAR(100),
    sentiment REAL,
    
    -- Feedback and improvement
    feedback JSONB,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    
    -- Metadata
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced Search Index with semantic and vector capabilities
CREATE TABLE IF NOT EXISTS search_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    content TEXT NOT NULL,
    
    -- Vector embedding for semantic search (768 dimensions)
    embedding VECTOR(768) NOT NULL,
    
    -- Search optimization
    search_vector TSVECTOR,
    keywords TEXT[],
    
    -- Metadata
    metadata JSONB,
    language VARCHAR(10) DEFAULT 'en',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector Similarity Cache for performance optimization
CREATE TABLE IF NOT EXISTS vector_similarity_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(64) NOT NULL UNIQUE,
    query_text TEXT,
    query_embedding VECTOR(768) NOT NULL,
    
    -- Cached results
    results JSONB NOT NULL,
    result_count INTEGER,
    
    -- Cache management
    hit_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Legal Knowledge Base with semantic embeddings
CREATE TABLE IF NOT EXISTS legal_knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    
    -- Legal classification
    category VARCHAR(100),
    subcategory VARCHAR(100),
    jurisdiction VARCHAR(100),
    legal_area VARCHAR(100),
    
    -- Source information
    source VARCHAR(255),
    source_url TEXT,
    citation_format TEXT,
    publication_date DATE,
    
    -- Vector embedding for knowledge retrieval
    embedding VECTOR(768),
    
    -- Quality control
    is_verified BOOLEAN DEFAULT false,
    verified_by UUID REFERENCES users(id),
    verified_at TIMESTAMP,
    quality_score REAL,
    
    -- Usage analytics
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    
    -- Metadata
    metadata JSONB,
    keywords TEXT[],
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI Processing Jobs for background processing and queue management
CREATE TABLE IF NOT EXISTS ai_processing_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Job classification
    job_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    
    -- Job status and progress
    status VARCHAR(20) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    
    -- Processing details
    input JSONB,
    output JSONB,
    error_message TEXT,
    
    -- Model and configuration
    model VARCHAR(100),
    parameters JSONB,
    
    -- Priority and retry logic
    priority INTEGER DEFAULT 5,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Performance tracking
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embedding Jobs for vector generation and management
CREATE TABLE IF NOT EXISTS embedding_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    
    -- Job configuration
    job_type VARCHAR(50) NOT NULL,
    model VARCHAR(100) DEFAULT 'nomic-embed-text',
    batch_size INTEGER DEFAULT 100,
    
    -- Status and progress
    status VARCHAR(20) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    
    -- Priority and retry logic
    priority INTEGER DEFAULT 5,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB,
    
    -- Performance metrics
    processing_time INTEGER,
    tokens_processed INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System Performance Metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    
    -- Context
    service_name VARCHAR(100),
    endpoint VARCHAR(200),
    user_id UUID REFERENCES users(id),
    session_id VARCHAR(255),
    
    -- Metadata
    metadata JSONB,
    tags TEXT[],
    
    -- Timestamp
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NATS Message Log for real-time event tracking
CREATE TABLE IF NOT EXISTS nats_message_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject VARCHAR(255) NOT NULL,
    message_type VARCHAR(100),
    
    -- Message content
    payload JSONB,
    size_bytes INTEGER,
    
    -- Processing details
    processed BOOLEAN DEFAULT false,
    processed_at TIMESTAMP,
    processing_error TEXT,
    
    -- Context
    user_id UUID REFERENCES users(id),
    session_id VARCHAR(255),
    case_id UUID REFERENCES cases(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- INDEXES FOR OPTIMAL PERFORMANCE
-- ===========================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Cases indexes
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
CREATE INDEX IF NOT EXISTS idx_cases_created_by ON cases(created_by);
CREATE INDEX IF NOT EXISTS idx_cases_assigned_to ON cases(assigned_to);
CREATE INDEX IF NOT EXISTS idx_cases_legal_categories ON cases USING gin(legal_categories);
CREATE INDEX IF NOT EXISTS idx_cases_jurisdiction ON cases(jurisdiction);
-- Vector similarity index for cases
CREATE INDEX IF NOT EXISTS idx_cases_embedding ON cases USING ivfflat (case_embedding vector_cosine_ops) WITH (lists = 100);

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_case_id ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_created_by ON documents(created_by);
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON documents USING gin(to_tsvector('english', coalesce(content, '') || ' ' || coalesce(extracted_text, '')));
-- Vector similarity indexes
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_summary_embedding ON documents USING ivfflat (summary_embedding vector_cosine_ops) WITH (lists = 100);

-- Document Chunks indexes
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_chunk_index ON document_chunks(document_id, chunk_index);
-- Vector similarity index for chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Evidence indexes
CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_document_id ON evidence(document_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence(evidence_type);
CREATE INDEX IF NOT EXISTS idx_evidence_created_by ON evidence(created_by);
CREATE INDEX IF NOT EXISTS idx_evidence_hash ON evidence(hash);
-- Vector similarity index for evidence
CREATE INDEX IF NOT EXISTS idx_evidence_embedding ON evidence USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- AI Interactions indexes
CREATE INDEX IF NOT EXISTS idx_ai_interactions_user_id ON ai_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_case_id ON ai_interactions(case_id);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_session_id ON ai_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_model ON ai_interactions(model);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_created_at ON ai_interactions(created_at);
-- Vector similarity index for conversation context
CREATE INDEX IF NOT EXISTS idx_ai_interactions_context_embedding ON ai_interactions USING ivfflat (context_embedding vector_cosine_ops) WITH (lists = 100);

-- Search Index indexes
CREATE INDEX IF NOT EXISTS idx_search_index_entity ON search_index(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_search_index_content_fts ON search_index USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_search_index_keywords ON search_index USING gin(keywords);
-- Vector similarity index for semantic search
CREATE INDEX IF NOT EXISTS idx_search_index_embedding ON search_index USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Vector Similarity Cache indexes
CREATE INDEX IF NOT EXISTS idx_vector_cache_query_hash ON vector_similarity_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_vector_cache_expires ON vector_similarity_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_vector_cache_last_accessed ON vector_similarity_cache(last_accessed);

-- Legal Knowledge Base indexes
CREATE INDEX IF NOT EXISTS idx_legal_kb_category ON legal_knowledge_base(category, subcategory);
CREATE INDEX IF NOT EXISTS idx_legal_kb_jurisdiction ON legal_knowledge_base(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_legal_kb_verified ON legal_knowledge_base(is_verified);
CREATE INDEX IF NOT EXISTS idx_legal_kb_keywords ON legal_knowledge_base USING gin(keywords);
CREATE INDEX IF NOT EXISTS idx_legal_kb_content_fts ON legal_knowledge_base USING gin(to_tsvector('english', title || ' ' || content));
-- Vector similarity index for knowledge retrieval
CREATE INDEX IF NOT EXISTS idx_legal_kb_embedding ON legal_knowledge_base USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Processing Jobs indexes
CREATE INDEX IF NOT EXISTS idx_ai_jobs_status ON ai_processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_ai_jobs_type ON ai_processing_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_ai_jobs_priority ON ai_processing_jobs(priority, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_jobs_entity ON ai_processing_jobs(entity_type, entity_id);

CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status ON embedding_jobs(status);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_entity ON embedding_jobs(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_embedding_jobs_priority ON embedding_jobs(priority, created_at);

-- System Metrics indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON system_metrics(metric_type, metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_service ON system_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON system_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_user_id ON system_metrics(user_id);

-- NATS Message Log indexes
CREATE INDEX IF NOT EXISTS idx_nats_log_subject ON nats_message_log(subject);
CREATE INDEX IF NOT EXISTS idx_nats_log_processed ON nats_message_log(processed);
CREATE INDEX IF NOT EXISTS idx_nats_log_created_at ON nats_message_log(created_at);
CREATE INDEX IF NOT EXISTS idx_nats_log_user_id ON nats_message_log(user_id);
CREATE INDEX IF NOT EXISTS idx_nats_log_case_id ON nats_message_log(case_id);

-- ===========================================
-- FUNCTIONS AND TRIGGERS
-- ===========================================

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_cases_updated_at BEFORE UPDATE ON cases FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_evidence_updated_at BEFORE UPDATE ON evidence FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_legal_knowledge_base_updated_at BEFORE UPDATE ON legal_knowledge_base FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ai_processing_jobs_updated_at BEFORE UPDATE ON ai_processing_jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_embedding_jobs_updated_at BEFORE UPDATE ON embedding_jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for vector similarity search with legal document context
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding VECTOR(768),
    similarity_threshold REAL DEFAULT 0.7,
    result_limit INTEGER DEFAULT 20,
    case_filter UUID DEFAULT NULL,
    document_types TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    document_id UUID,
    title VARCHAR(255),
    similarity_score REAL,
    case_id UUID,
    document_type VARCHAR(50),
    created_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        1 - (d.embedding <=> query_embedding) as similarity_score,
        d.case_id,
        d.document_type,
        d.created_at
    FROM documents d
    WHERE 
        d.embedding IS NOT NULL
        AND 1 - (d.embedding <=> query_embedding) >= similarity_threshold
        AND (case_filter IS NULL OR d.case_id = case_filter)
        AND (document_types IS NULL OR d.document_type = ANY(document_types))
        AND d.processing_status = 'completed'
    ORDER BY d.embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Function for semantic search across document chunks
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding VECTOR(768),
    similarity_threshold REAL DEFAULT 0.7,
    result_limit INTEGER DEFAULT 50,
    case_filter UUID DEFAULT NULL
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity_score REAL,
    chunk_index INTEGER,
    significance_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id,
        dc.document_id,
        dc.content,
        1 - (dc.embedding <=> query_embedding) as similarity_score,
        dc.chunk_index,
        dc.significance_score
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    WHERE 
        dc.embedding IS NOT NULL
        AND 1 - (dc.embedding <=> query_embedding) >= similarity_threshold
        AND (case_filter IS NULL OR d.case_id = case_filter)
        AND d.processing_status = 'completed'
    ORDER BY dc.embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to update AI usage statistics
CREATE OR REPLACE FUNCTION update_user_ai_usage(
    user_uuid UUID,
    token_count INTEGER,
    model_used VARCHAR(100)
)
RETURNS VOID AS $$
BEGIN
    UPDATE users 
    SET 
        total_ai_queries = total_ai_queries + 1,
        total_tokens_used = total_tokens_used + token_count,
        ai_preferences = COALESCE(ai_preferences, '{}'::jsonb) || 
                        jsonb_build_object('last_model', model_used, 'last_query_time', NOW())
    WHERE id = user_uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_vector_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM vector_similarity_cache 
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- VIEWS FOR COMMON QUERIES
-- ===========================================

-- View for document analytics
CREATE OR REPLACE VIEW document_analytics AS
SELECT 
    d.id,
    d.title,
    d.case_id,
    c.title as case_title,
    d.document_type,
    d.processing_status,
    COALESCE(d.ai_analysis->>'confidence_score', '0')::REAL as ai_confidence,
    array_length(d.tags, 1) as tag_count,
    d.file_size,
    d.created_at,
    CASE 
        WHEN d.embedding IS NOT NULL THEN true 
        ELSE false 
    END as has_embedding,
    CASE 
        WHEN d.ai_analysis IS NOT NULL THEN true 
        ELSE false 
    END as has_ai_analysis
FROM documents d
LEFT JOIN cases c ON d.case_id = c.id;

-- View for AI interaction analytics
CREATE OR REPLACE VIEW ai_interaction_analytics AS
SELECT 
    ai.id,
    ai.user_id,
    u.email as user_email,
    ai.case_id,
    c.title as case_title,
    ai.model,
    ai.tokens_used,
    ai.response_time,
    ai.confidence,
    ai.rating,
    ai.created_at,
    CASE 
        WHEN ai.feedback IS NOT NULL THEN true 
        ELSE false 
    END as has_feedback
FROM ai_interactions ai
LEFT JOIN users u ON ai.user_id = u.id
LEFT JOIN cases c ON ai.case_id = c.id;

-- View for system performance overview
CREATE OR REPLACE VIEW system_performance_overview AS
SELECT 
    sm.service_name,
    sm.metric_type,
    sm.metric_name,
    AVG(sm.metric_value) as avg_value,
    MIN(sm.metric_value) as min_value,
    MAX(sm.metric_value) as max_value,
    COUNT(*) as sample_count,
    MAX(sm.recorded_at) as last_recorded
FROM system_metrics sm
WHERE sm.recorded_at >= (CURRENT_TIMESTAMP - INTERVAL '1 hour')
GROUP BY sm.service_name, sm.metric_type, sm.metric_name;

-- ===========================================
-- SAMPLE DATA FOR TESTING (OPTIONAL)
-- ===========================================

-- Insert sample user for testing
INSERT INTO users (id, email, password_hash, first_name, last_name, role) 
VALUES (
    gen_random_uuid(), 
    'admin@legalai.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj.yB5k9zJx2', -- password: admin123
    'AI', 
    'Administrator', 
    'admin'
) ON CONFLICT (email) DO NOTHING;

-- Insert sample legal knowledge entries
INSERT INTO legal_knowledge_base (title, content, category, jurisdiction, source) VALUES 
('Contract Formation Elements', 'A valid contract requires offer, acceptance, consideration, and mutual assent...', 'Contract Law', 'Federal', 'Legal Reference')
ON CONFLICT DO NOTHING;

INSERT INTO legal_knowledge_base (title, content, category, jurisdiction, source) VALUES 
('Negligence Standard', 'Negligence requires duty of care, breach of duty, causation, and damages...', 'Tort Law', 'Federal', 'Legal Reference')
ON CONFLICT DO NOTHING;

-- ===========================================
-- CONFIGURATION AND MAINTENANCE
-- ===========================================

-- Set appropriate work_mem for vector operations
-- This should be adjusted based on available system memory
-- ALTER SYSTEM SET work_mem = '256MB';
-- SELECT pg_reload_conf();

-- Update table statistics for optimal query planning
ANALYZE users;
ANALYZE cases;
ANALYZE documents;
ANALYZE document_chunks;
ANALYZE evidence;
ANALYZE ai_interactions;
ANALYZE search_index;
ANALYZE legal_knowledge_base;

-- Grant permissions (adjust as needed for your application user)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO legal_ai_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO legal_ai_app;

-- Create maintenance procedures
CREATE OR REPLACE FUNCTION maintenance_cleanup()
RETURNS TEXT AS $$
DECLARE
    cache_cleaned INTEGER;
    old_logs_cleaned INTEGER;
    result_text TEXT;
BEGIN
    -- Clean up expired vector cache
    SELECT cleanup_vector_cache() INTO cache_cleaned;
    
    -- Clean up old NATS message logs (older than 7 days)
    DELETE FROM nats_message_log 
    WHERE created_at < (CURRENT_TIMESTAMP - INTERVAL '7 days')
    AND processed = true;
    GET DIAGNOSTICS old_logs_cleaned = ROW_COUNT;
    
    -- Update statistics
    ANALYZE;
    
    result_text := format(
        'Maintenance completed: %s cache entries cleaned, %s old logs removed',
        cache_cleaned, old_logs_cleaned
    );
    
    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- Success message
SELECT 'PostgreSQL Vector Integration Schema Setup Complete!' as status;