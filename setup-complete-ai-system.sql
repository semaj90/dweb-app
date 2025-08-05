-- Complete AI System Setup for Legal AI Application
-- PostgreSQL 17 + pgvector + Full Schema Creation

-- ===== EXTENSIONS =====
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ===== DATABASE SETUP =====
-- Create database (if not exists)
-- CREATE DATABASE legal_ai_db;

-- ===== SAMPLE DATA SEEDING =====

-- Create test users
INSERT INTO users (id, email, name, first_name, last_name, role, hashed_password) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'prosecutor@legal.ai', 'John Prosecutor', 'John', 'Prosecutor', 'prosecutor', '$2a$12$example_hashed_password'),
('550e8400-e29b-41d4-a716-446655440001', 'detective@legal.ai', 'Jane Detective', 'Jane', 'Detective', 'detective', '$2a$12$example_hashed_password'),
('550e8400-e29b-41d4-a716-446655440002', 'admin@legal.ai', 'Admin User', 'Admin', 'User', 'admin', '$2a$12$example_hashed_password')
ON CONFLICT (email) DO NOTHING;

-- Create test cases
INSERT INTO cases (id, case_number, title, description, priority, status, created_by) VALUES
('650e8400-e29b-41d4-a716-446655440000', 'CASE-2024-001', 'Financial Fraud Investigation', 'Complex financial fraud case involving multiple entities and cryptocurrency transactions.', 'high', 'open', '550e8400-e29b-41d4-a716-446655440000'),
('650e8400-e29b-41d4-a716-446655440001', 'CASE-2024-002', 'Cybercrime Investigation', 'Data breach and identity theft case with international connections.', 'medium', 'open', '550e8400-e29b-41d4-a716-446655440001'),
('650e8400-e29b-41d4-a716-446655440002', 'CASE-2024-003', 'White Collar Crime', 'Corporate embezzlement case with extensive document evidence.', 'high', 'open', '550e8400-e29b-41d4-a716-446655440000')
ON CONFLICT (case_number) DO NOTHING;

-- Create test evidence
INSERT INTO evidence (id, case_id, title, description, evidence_type, uploaded_by) VALUES
('750e8400-e29b-41d4-a716-446655440000', '650e8400-e29b-41d4-a716-446655440000', 'Bank Transaction Records', 'Suspicious transaction patterns showing money laundering activity.', 'financial_document', '550e8400-e29b-41d4-a716-446655440000'),
('750e8400-e29b-41d4-a716-446655440001', '650e8400-e29b-41d4-a716-446655440000', 'Email Communications', 'Email chain between suspects discussing fraudulent activities.', 'digital_evidence', '550e8400-e29b-41d4-a716-446655440001'),
('750e8400-e29b-41d4-a716-446655440002', '650e8400-e29b-41d4-a716-446655440001', 'Server Logs', 'Access logs showing unauthorized data access and exfiltration.', 'digital_evidence', '550e8400-e29b-41d4-a716-446655440001'),
('750e8400-e29b-41d4-a716-446655440003', '650e8400-e29b-41d4-a716-446655440002', 'Financial Statements', 'Falsified company financial statements with discrepancies.', 'financial_document', '550e8400-e29b-41d4-a716-446655440000')
ON CONFLICT (id) DO NOTHING;

-- Create sample AI queries for testing
INSERT INTO user_ai_queries (id, user_id, case_id, query, response, model, query_type, confidence, tokens_used, processing_time) VALUES
('850e8400-e29b-41d4-a716-446655440000', '550e8400-e29b-41d4-a716-446655440000', '650e8400-e29b-41d4-a716-446655440000', 'Analyze the financial transaction patterns in this case', 'Based on the bank transaction records, I identified 15 suspicious transactions totaling $2.3M that follow classic money laundering patterns. The transactions occur in round amounts, frequently just under reporting thresholds, and involve multiple jurisdictions. I recommend focusing on transactions between accounts ending in 4457 and 8821 as they show the strongest evidence of structuring.', 'gemma3-legal', 'evidence_analysis', 0.92, 1245, 3420),
('850e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', '650e8400-e29b-41d4-a716-446655440001', 'What cybercrime statutes apply to this data breach case?', 'This case likely involves violations of 18 U.S.C. § 1030 (Computer Fraud and Abuse Act), particularly subsections (a)(2) for unauthorized access to obtain information, (a)(4) for intent to defraud, and (a)(7) for damage to computers. Additionally, consider 18 U.S.C. § 1028A for aggravated identity theft if personal identifying information was compromised. State laws may also apply depending on jurisdiction.', 'gemma3-legal', 'legal_research', 0.88, 892, 2150)
ON CONFLICT (id) DO NOTHING;

-- Create sample auto-tags
INSERT INTO auto_tags (id, entity_id, entity_type, tag, confidence, source, model) VALUES
('950e8400-e29b-41d4-a716-446655440000', '750e8400-e29b-41d4-a716-446655440000', 'evidence', 'money_laundering', 0.95, 'ai_analysis', 'gemma3-legal'),
('950e8400-e29b-41d4-a716-446655440001', '750e8400-e29b-41d4-a716-446655440000', 'evidence', 'financial_fraud', 0.91, 'ai_analysis', 'gemma3-legal'),
('950e8400-e29b-41d4-a716-446655440002', '750e8400-e29b-41d4-a716-446655440001', 'evidence', 'digital_evidence', 0.99, 'ai_analysis', 'gemma3-legal'),
('950e8400-e29b-41d4-a716-446655440003', '750e8400-e29b-41d4-a716-446655440002', 'evidence', 'data_breach', 0.87, 'ai_analysis', 'gemma3-legal'),
('950e8400-e29b-41d4-a716-446655440004', '650e8400-e29b-41d4-a716-446655440000', 'case', 'high_priority', 0.93, 'ai_analysis', 'gemma3-legal'),
('950e8400-e29b-41d4-a716-446655440005', '650e8400-e29b-41d4-a716-446655440001', 'case', 'cybercrime', 0.96, 'ai_analysis', 'gemma3-legal')
ON CONFLICT (id) DO NOTHING;

-- ===== INDEXES FOR PERFORMANCE =====

-- Vector similarity indexes
CREATE INDEX IF NOT EXISTS idx_legal_documents_embedding ON legal_documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_embedding ON embedding_cache USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_user_ai_queries_embedding ON user_ai_queries USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence(evidence_type);
CREATE INDEX IF NOT EXISTS idx_user_ai_queries_user_id ON user_ai_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_user_ai_queries_case_id ON user_ai_queries(case_id);
CREATE INDEX IF NOT EXISTS idx_auto_tags_entity ON auto_tags(entity_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_auto_tags_confidence ON auto_tags(confidence DESC);

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_cases_title_search ON cases USING GIN (to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_evidence_description_search ON evidence USING GIN (to_tsvector('english', description));
CREATE INDEX IF NOT EXISTS idx_user_ai_queries_search ON user_ai_queries USING GIN (to_tsvector('english', query || ' ' || response));

-- ===== FUNCTIONS FOR AI OPERATIONS =====

-- Function to find similar queries
CREATE OR REPLACE FUNCTION find_similar_queries(
    query_embedding vector(768),
    similarity_threshold float DEFAULT 0.8,
    max_results int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    query text,
    response text,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        uaq.id,
        uaq.query,
        uaq.response,
        1 - (uaq.embedding <=> query_embedding) as similarity
    FROM user_ai_queries uaq
    WHERE uaq.embedding IS NOT NULL
    AND 1 - (uaq.embedding <=> query_embedding) > similarity_threshold
    ORDER BY uaq.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar documents
CREATE OR REPLACE FUNCTION find_similar_documents(
    query_embedding vector(768),
    doc_type text DEFAULT NULL,
    similarity_threshold float DEFAULT 0.7,
    max_results int DEFAULT 20
)
RETURNS TABLE (
    id uuid,
    content text,
    document_type varchar,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id,
        dc.content,
        dc.document_type,
        1 - (dc.embedding <=> query_embedding) as similarity
    FROM document_chunks dc
    WHERE (doc_type IS NULL OR dc.document_type = doc_type)
    AND 1 - (dc.embedding <=> query_embedding) > similarity_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- ===== VERIFICATION =====

-- Verify pgvector is working
SELECT vector_dims('[1,2,3]'::vector) as vector_test;

-- Count records
SELECT 'users' as table_name, COUNT(*) as count FROM users
UNION ALL
SELECT 'cases', COUNT(*) FROM cases
UNION ALL
SELECT 'evidence', COUNT(*) FROM evidence
UNION ALL
SELECT 'user_ai_queries', COUNT(*) FROM user_ai_queries
UNION ALL
SELECT 'auto_tags', COUNT(*) FROM auto_tags;

-- Show database info
SELECT 
    current_database() as database_name,
    version() as postgresql_version,
    (SELECT extversion FROM pg_extension WHERE extname = 'vector') as pgvector_version;

VACUUM ANALYZE;

-- Setup complete!
SELECT 'Legal AI Database Setup Complete!' as status;