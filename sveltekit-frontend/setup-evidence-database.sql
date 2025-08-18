-- =========================================================
-- EVIDENCE PROCESSING DATABASE SETUP
-- Complete schema for legal evidence management
-- =========================================================

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE evidence_processing'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'evidence_processing');

-- Connect to the evidence_processing database
\c evidence_processing;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =========================================================
-- CORE EVIDENCE TABLES
-- =========================================================

-- Evidence cases table
CREATE TABLE IF NOT EXISTS evidence_cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'closed', 'archived')),
    priority VARCHAR(10) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Evidence items table
CREATE TABLE IF NOT EXISTS evidence_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID REFERENCES evidence_cases(id) ON DELETE CASCADE,
    evidence_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    file_hash VARCHAR(128),
    file_size BIGINT,
    mime_type VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'processed', 'failed')),
    chain_of_custody JSONB DEFAULT '[]',
    extracted_text TEXT,
    embeddings vector(384),
    ocr_confidence DECIMAL(5,2),
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Evidence processing jobs table
CREATE TABLE IF NOT EXISTS evidence_processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id UUID REFERENCES evidence_items(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL, -- 'ocr', 'embedding', 'analysis'
    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    result JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Evidence analysis results table
CREATE TABLE IF NOT EXISTS evidence_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id UUID REFERENCES evidence_items(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL, -- 'legal_classification', 'entity_extraction', 'sentiment'
    confidence_score DECIMAL(5,2),
    results JSONB NOT NULL,
    model_used VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Smart detection logs table
CREATE TABLE IF NOT EXISTS smart_detection_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id UUID REFERENCES evidence_items(id) ON DELETE CASCADE,
    detection_type VARCHAR(50) NOT NULL, -- 'person', 'location', 'organization', 'date', 'legal_entity'
    detected_value TEXT NOT NULL,
    confidence_score DECIMAL(5,2),
    position_start INTEGER,
    position_end INTEGER,
    context TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =========================================================
-- INDEXES FOR PERFORMANCE
-- =========================================================

-- Evidence cases indexes
CREATE INDEX IF NOT EXISTS idx_evidence_cases_status ON evidence_cases(status);
CREATE INDEX IF NOT EXISTS idx_evidence_cases_priority ON evidence_cases(priority);
CREATE INDEX IF NOT EXISTS idx_evidence_cases_created_at ON evidence_cases(created_at);
CREATE INDEX IF NOT EXISTS idx_evidence_cases_case_number ON evidence_cases(case_number);

-- Evidence items indexes
CREATE INDEX IF NOT EXISTS idx_evidence_items_case_id ON evidence_items(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_items_status ON evidence_items(status);
CREATE INDEX IF NOT EXISTS idx_evidence_items_evidence_number ON evidence_items(evidence_number);
CREATE INDEX IF NOT EXISTS idx_evidence_items_file_hash ON evidence_items(file_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_items_created_at ON evidence_items(created_at);

-- Vector similarity index for embeddings
CREATE INDEX IF NOT EXISTS idx_evidence_embeddings_cosine 
ON evidence_items USING ivfflat (embeddings vector_cosine_ops) 
WITH (lists = 100);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_evidence_items_text_search 
ON evidence_items USING gin(to_tsvector('english', title || ' ' || COALESCE(description, '') || ' ' || COALESCE(extracted_text, '')));

-- Processing jobs indexes
CREATE INDEX IF NOT EXISTS idx_processing_jobs_evidence_id ON evidence_processing_jobs(evidence_id);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON evidence_processing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_job_type ON evidence_processing_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_created_at ON evidence_processing_jobs(created_at);

-- Analysis results indexes
CREATE INDEX IF NOT EXISTS idx_evidence_analysis_evidence_id ON evidence_analysis(evidence_id);
CREATE INDEX IF NOT EXISTS idx_evidence_analysis_type ON evidence_analysis(analysis_type);
CREATE INDEX IF NOT EXISTS idx_evidence_analysis_confidence ON evidence_analysis(confidence_score);

-- Smart detection indexes
CREATE INDEX IF NOT EXISTS idx_smart_detection_evidence_id ON smart_detection_logs(evidence_id);
CREATE INDEX IF NOT EXISTS idx_smart_detection_type ON smart_detection_logs(detection_type);
CREATE INDEX IF NOT EXISTS idx_smart_detection_confidence ON smart_detection_logs(confidence_score);

-- =========================================================
-- SAMPLE DATA FOR TESTING
-- =========================================================

-- Insert sample case
INSERT INTO evidence_cases (case_number, title, description, priority, created_by) 
VALUES ('CASE-2025-001', 'Contract Liability Investigation', 'Investigation of contract terms and liability clauses', 'high', 'system_test')
ON CONFLICT (case_number) DO NOTHING;

-- Get the case ID for sample evidence
DO $$
DECLARE
    sample_case_id UUID;
BEGIN
    SELECT id INTO sample_case_id FROM evidence_cases WHERE case_number = 'CASE-2025-001';
    
    -- Insert sample evidence item
    INSERT INTO evidence_items (
        case_id, 
        evidence_number, 
        title, 
        description, 
        status, 
        extracted_text,
        created_by
    ) VALUES (
        sample_case_id,
        'EVD-2025-001',
        'Legal Contract Document',
        'Contract containing liability, indemnification, and dispute resolution clauses',
        'processed',
        'This is a test document for the evidence processing pipeline. It contains legal information about contract terms and conditions. The document includes important clauses about liability, indemnification, and dispute resolution.',
        'system_test'
    ) ON CONFLICT (evidence_number) DO NOTHING;
END $$;

-- =========================================================
-- STORED PROCEDURES FOR EVIDENCE PROCESSING
-- =========================================================

-- Function to update evidence processing status
CREATE OR REPLACE FUNCTION update_evidence_status(
    p_evidence_id UUID,
    p_status VARCHAR(20),
    p_metadata JSONB DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE evidence_items 
    SET 
        status = p_status,
        updated_at = CURRENT_TIMESTAMP,
        metadata = CASE 
            WHEN p_metadata IS NOT NULL THEN metadata || p_metadata 
            ELSE metadata 
        END
    WHERE id = p_evidence_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Function to create processing job
CREATE OR REPLACE FUNCTION create_processing_job(
    p_evidence_id UUID,
    p_job_type VARCHAR(50)
) RETURNS UUID AS $$
DECLARE
    job_id UUID;
BEGIN
    INSERT INTO evidence_processing_jobs (evidence_id, job_type, status)
    VALUES (p_evidence_id, p_job_type, 'queued')
    RETURNING id INTO job_id;
    
    RETURN job_id;
END;
$$ LANGUAGE plpgsql;

-- Function for vector similarity search
CREATE OR REPLACE FUNCTION search_similar_evidence(
    p_query_embedding vector(384),
    p_similarity_threshold DECIMAL(3,2) DEFAULT 0.7,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE (
    evidence_id UUID,
    evidence_number VARCHAR(50),
    title VARCHAR(255),
    similarity_score DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ei.id,
        ei.evidence_number,
        ei.title,
        ROUND((1 - (ei.embeddings <=> p_query_embedding))::decimal, 4) as similarity_score
    FROM evidence_items ei
    WHERE ei.embeddings IS NOT NULL
        AND (1 - (ei.embeddings <=> p_query_embedding)) >= p_similarity_threshold
    ORDER BY ei.embeddings <=> p_query_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- =========================================================
-- VIEWS FOR REPORTING
-- =========================================================

-- Evidence processing status view
CREATE OR REPLACE VIEW evidence_processing_status AS
SELECT 
    ec.case_number,
    ec.title as case_title,
    ei.evidence_number,
    ei.title as evidence_title,
    ei.status,
    ei.created_at,
    ei.updated_at,
    COUNT(epj.id) as processing_jobs_count,
    COUNT(CASE WHEN epj.status = 'completed' THEN 1 END) as completed_jobs,
    COUNT(CASE WHEN epj.status = 'failed' THEN 1 END) as failed_jobs
FROM evidence_cases ec
JOIN evidence_items ei ON ec.id = ei.case_id
LEFT JOIN evidence_processing_jobs epj ON ei.id = epj.evidence_id
GROUP BY ec.case_number, ec.title, ei.evidence_number, ei.title, ei.status, ei.created_at, ei.updated_at;

-- Smart detection summary view
CREATE OR REPLACE VIEW smart_detection_summary AS
SELECT 
    ei.evidence_number,
    ei.title,
    sdl.detection_type,
    COUNT(*) as detection_count,
    AVG(sdl.confidence_score) as avg_confidence,
    MAX(sdl.confidence_score) as max_confidence,
    array_agg(DISTINCT sdl.detected_value ORDER BY sdl.detected_value) as detected_values
FROM evidence_items ei
JOIN smart_detection_logs sdl ON ei.id = sdl.evidence_id
GROUP BY ei.evidence_number, ei.title, sdl.detection_type;

-- =========================================================
-- TRIGGERS FOR AUDIT TRAIL
-- =========================================================

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for evidence_cases
CREATE TRIGGER update_evidence_cases_updated_at 
BEFORE UPDATE ON evidence_cases 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger for evidence_items
CREATE TRIGGER update_evidence_items_updated_at 
BEFORE UPDATE ON evidence_items 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =========================================================
-- PERMISSIONS AND SECURITY
-- =========================================================

-- Create evidence_processor role
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'evidence_processor') THEN
        CREATE ROLE evidence_processor WITH LOGIN PASSWORD 'evidence_secure_2025';
    END IF;
END $$;

-- Grant permissions
GRANT CONNECT ON DATABASE evidence_processing TO evidence_processor;
GRANT USAGE ON SCHEMA public TO evidence_processor;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO evidence_processor;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO evidence_processor;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO evidence_processor;

-- =========================================================
-- COMPLETION MESSAGE
-- =========================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Evidence Processing Database Setup Complete!';
    RAISE NOTICE '📊 Tables created: evidence_cases, evidence_items, evidence_processing_jobs, evidence_analysis, smart_detection_logs';
    RAISE NOTICE '🔍 Indexes created for optimal performance';
    RAISE NOTICE '📝 Sample data inserted for testing';
    RAISE NOTICE '🔧 Stored procedures and views ready';
    RAISE NOTICE '🛡️ Security roles configured';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 Ready for evidence processing pipeline testing!';
END $$;