-- ================================================================================
-- LEGAL AI DATABASE INITIALIZATION SCRIPT
-- ================================================================================
-- PostgreSQL + pgvector + Advanced Legal Document Processing
-- ================================================================================

-- Create legal AI database
DROP DATABASE IF EXISTS legal_ai_db;
CREATE DATABASE legal_ai_db;

-- Connect to the database
\c legal_ai_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create legal documents table with vector embeddings
CREATE TABLE IF NOT EXISTS legal_documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(100),
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(384),
    metadata JSONB,
    legal_category VARCHAR(100),
    confidence_score FLOAT DEFAULT 0.0,
    processed_at TIMESTAMP,
    som_cluster_id INTEGER,
    risk_score FLOAT DEFAULT 0.0,
    compliance_status VARCHAR(50) DEFAULT 'pending'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS gin_content_idx ON legal_documents USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS vector_embedding_idx ON legal_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_legal_category ON legal_documents(legal_category);
CREATE INDEX IF NOT EXISTS idx_document_type ON legal_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_som_cluster ON legal_documents(som_cluster_id);

-- Create legal precedents table
CREATE TABLE IF NOT EXISTS legal_precedents (
    id SERIAL PRIMARY KEY,
    case_name VARCHAR(500) NOT NULL,
    court VARCHAR(200),
    date_decided DATE,
    summary TEXT,
    full_text TEXT,
    embedding vector(384),
    relevance_score FLOAT DEFAULT 0.0,
    citations JSONB,
    legal_area VARCHAR(100),
    jurisdiction VARCHAR(100),
    importance_level INTEGER DEFAULT 1
);

-- Create precedent indexes
CREATE INDEX IF NOT EXISTS precedent_vector_idx ON legal_precedents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS idx_case_name ON legal_precedents(case_name);
CREATE INDEX IF NOT EXISTS idx_legal_area ON legal_precedents(legal_area);
CREATE INDEX IF NOT EXISTS idx_jurisdiction ON legal_precedents(jurisdiction);

-- Create analytics events table
CREATE TABLE IF NOT EXISTS analytics_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    user_session VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data JSONB,
    processing_time_ms INTEGER,
    gpu_used BOOLEAN DEFAULT FALSE,
    som_cluster_used VARCHAR(100),
    workflow_id VARCHAR(100),
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

-- Create analytics indexes
CREATE INDEX IF NOT EXISTS idx_event_type ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS idx_timestamp ON analytics_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_session ON analytics_events(user_session);
CREATE INDEX IF NOT EXISTS idx_workflow_id ON analytics_events(workflow_id);

-- Create SOM clusters table
CREATE TABLE IF NOT EXISTS som_clusters (
    id SERIAL PRIMARY KEY,
    cluster_name VARCHAR(100) NOT NULL UNIQUE,
    cluster_type VARCHAR(50),
    width INTEGER,
    height INTEGER,
    input_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_trained TIMESTAMP,
    accuracy FLOAT DEFAULT 0.0,
    neuron_data JSONB,
    training_iterations INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create workflow states table
CREATE TABLE IF NOT EXISTS workflow_states (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(100) NOT NULL,
    machine_id VARCHAR(100) NOT NULL,
    current_state VARCHAR(100),
    previous_state VARCHAR(100),
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    transition_count INTEGER DEFAULT 0
);

-- Create workflow indexes
CREATE INDEX IF NOT EXISTS idx_workflow_id ON workflow_states(workflow_id);
CREATE INDEX IF NOT EXISTS idx_machine_id ON workflow_states(machine_id);
CREATE INDEX IF NOT EXISTS idx_current_state ON workflow_states(current_state);

-- Create users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(200) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    preferences JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    organization VARCHAR(200)
);

-- Create user indexes
CREATE INDEX IF NOT EXISTS idx_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_role ON users(role);

-- Create legal risk assessments table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES legal_documents(id),
    risk_level VARCHAR(20),
    risk_score FLOAT,
    risk_factors JSONB,
    recommendations JSONB,
    assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assessed_by VARCHAR(100),
    confidence_level FLOAT DEFAULT 0.0
);

-- Create risk assessment indexes
CREATE INDEX IF NOT EXISTS idx_document_risk ON risk_assessments(document_id);
CREATE INDEX IF NOT EXISTS idx_risk_level ON risk_assessments(risk_level);
CREATE INDEX IF NOT EXISTS idx_risk_score ON risk_assessments(risk_score);

-- Create compliance checks table
CREATE TABLE IF NOT EXISTS compliance_checks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES legal_documents(id),
    regulation_type VARCHAR(100),
    compliance_status VARCHAR(50),
    violations JSONB,
    recommendations JSONB,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    next_review_date DATE
);

-- Create compliance indexes
CREATE INDEX IF NOT EXISTS idx_compliance_document ON compliance_checks(document_id);
CREATE INDEX IF NOT EXISTS idx_compliance_status ON compliance_checks(compliance_status);
CREATE INDEX IF NOT EXISTS idx_regulation_type ON compliance_checks(regulation_type);

-- Insert sample data for testing
INSERT INTO legal_documents (title, content, document_type, legal_category, confidence_score, risk_score) VALUES
('Sample Service Agreement', 'This Service Agreement (Agreement) is entered into by and between Company A and Company B. The parties agree to the following terms and conditions regarding the provision of legal consulting services.', 'contract', 'commercial', 0.95, 0.3),
('Employment Contract Template', 'This Employment Agreement outlines the terms of employment including compensation, benefits, confidentiality obligations, and termination procedures.', 'contract', 'employment', 0.92, 0.4),
('Data Privacy Compliance Guide', 'This document outlines the requirements for GDPR compliance including data collection, processing, storage, and user rights management.', 'compliance', 'regulatory', 0.88, 0.6),
('Intellectual Property License', 'License Agreement for the use of patented technology including royalty payments, usage restrictions, and termination clauses.', 'license', 'intellectual_property', 0.91, 0.5),
('Corporate Merger Agreement', 'Comprehensive agreement outlining the terms of corporate merger including valuation, due diligence, regulatory approvals, and closing conditions.', 'merger', 'corporate', 0.97, 0.8);

INSERT INTO legal_precedents (case_name, court, date_decided, summary, legal_area, jurisdiction, importance_level) VALUES
('Contract Interpretation Standards', 'Supreme Court', '2023-03-15', 'Landmark case establishing standards for contract interpretation in commercial disputes.', 'contract_law', 'federal', 5),
('Employment Rights Protection', 'Federal Appeals Court', '2023-01-20', 'Case defining employee rights and employer obligations in termination procedures.', 'employment_law', 'federal', 4),
('Data Privacy Enforcement', 'District Court', '2023-05-10', 'First major enforcement action under updated privacy regulations with significant penalties.', 'privacy_law', 'state', 3),
('IP Licensing Disputes', 'Patent Court', '2023-02-28', 'Important ruling on patent licensing terms and royalty calculation methods.', 'intellectual_property', 'federal', 4),
('Corporate Liability Limits', 'State Supreme Court', '2023-04-12', 'Decision clarifying corporate liability limits in merger and acquisition contexts.', 'corporate_law', 'state', 4);

INSERT INTO som_clusters (cluster_name, cluster_type, width, height, input_size, accuracy, training_iterations, is_active) VALUES
('contract_analysis', 'legal_documents', 20, 20, 384, 0.87, 1000, true),
('legal_precedent', 'case_law', 16, 16, 384, 0.84, 800, true),
('compliance_docs', 'regulatory', 12, 12, 384, 0.89, 600, true),
('risk_assessment', 'analytics', 15, 15, 384, 0.82, 750, true),
('employment_contracts', 'specialized', 10, 10, 384, 0.91, 500, true);

INSERT INTO analytics_events (event_type, user_session, data, processing_time_ms, gpu_used, som_cluster_used, success) VALUES
('document_upload', 'session_001', '{"document_type": "contract", "size_mb": 2.3}', 1200, false, null, true),
('gpu_tensor_parsing', 'session_001', '{"tensors_processed": 1500, "legal_weight_avg": 1.6}', 8, true, null, true),
('som_classification', 'session_001', '{"cluster_id": "contract_analysis", "confidence": 0.87}', 25, false, 'contract_analysis', true),
('risk_assessment', 'session_002', '{"risk_score": 0.65, "factors": ["liability", "jurisdiction"]}', 450, true, null, true),
('chat_interaction', 'session_003', '{"query": "contract law", "response_length": 180}', 750, false, null, true);

-- Create functions for common operations
CREATE OR REPLACE FUNCTION search_similar_documents(query_embedding vector(384), similarity_threshold float DEFAULT 0.7, max_results int DEFAULT 10)
RETURNS TABLE(id int, title varchar, content text, similarity float) AS $$
BEGIN
    RETURN QUERY
    SELECT d.id, d.title, d.content, 
           1 - (d.embedding <=> query_embedding) as similarity
    FROM legal_documents d
    WHERE d.embedding IS NOT NULL
      AND 1 - (d.embedding <=> query_embedding) > similarity_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_document_risk_summary(doc_id int)
RETURNS TABLE(risk_level varchar, risk_score float, factors jsonb, recommendations jsonb) AS $$
BEGIN
    RETURN QUERY
    SELECT ra.risk_level, ra.risk_score, ra.risk_factors, ra.recommendations
    FROM risk_assessments ra
    WHERE ra.document_id = doc_id
    ORDER BY ra.assessed_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION update_som_training_stats(cluster_name varchar, new_accuracy float, iterations int)
RETURNS void AS $$
BEGIN
    UPDATE som_clusters 
    SET accuracy = new_accuracy,
        training_iterations = training_iterations + iterations,
        last_trained = CURRENT_TIMESTAMP
    WHERE cluster_name = cluster_name;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_workflow_states_modtime
    BEFORE UPDATE ON workflow_states
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- Grant permissions (adjust username as needed)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'legal_admin') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO legal_admin;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO legal_admin;
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO legal_admin;
    END IF;
END
$$;

-- Create indexes for performance optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_full_text ON legal_documents USING gin(to_tsvector('english', title || ' ' || content));
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_precedents_full_text ON legal_precedents USING gin(to_tsvector('english', case_name || ' ' || summary));

-- Analyze tables for query optimization
ANALYZE legal_documents;
ANALYZE legal_precedents;
ANALYZE analytics_events;
ANALYZE som_clusters;
ANALYZE workflow_states;

-- Display summary
SELECT 'Database initialization completed successfully!' as status;
SELECT 'Legal documents: ' || count(*) as documents_count FROM legal_documents;
SELECT 'Legal precedents: ' || count(*) as precedents_count FROM legal_precedents;
SELECT 'SOM clusters: ' || count(*) as clusters_count FROM som_clusters;
SELECT 'Analytics events: ' || count(*) as events_count FROM analytics_events;

COMMIT;
