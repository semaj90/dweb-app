-- Legal AI Database Setup Script
-- Run this script with: psql -U postgres -h localhost -f setup-legal-ai-db.sql

-- Create legal_ai_db database
CREATE DATABASE legal_ai_db;

-- Connect to the new database
\c legal_ai_db;

-- Create legal_admin user if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'legal_admin') THEN
        CREATE USER legal_admin WITH PASSWORD 'LegalSecure2024!';
    END IF;
END
$$;

-- Grant privileges to legal_admin
GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;
GRANT ALL PRIVILEGES ON SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO legal_admin;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL PRIVILEGES ON TABLES TO legal_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL PRIVILEGES ON SEQUENCES TO legal_admin;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL PRIVILEGES ON FUNCTIONS TO legal_admin;

-- Try to create vector extension (will fail if not installed, but that's OK)
DO $$ 
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
    RAISE NOTICE 'pgvector extension created successfully!';
EXCEPTION WHEN others THEN
    RAISE NOTICE 'pgvector extension not available - this is OK, the database will work without it';
    RAISE NOTICE 'Vector similarity search features will be disabled until pgvector is installed';
END
$$;

-- Create core tables for Legal AI system
CREATE TABLE IF NOT EXISTS cases (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    priority VARCHAR(20) DEFAULT 'medium',
    created_by INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evidence (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES cases(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100),
    file_path VARCHAR(500),
    file_url VARCHAR(500),
    content TEXT,
    metadata JSONB,
    ai_analysis JSONB,
    tags TEXT[],
    created_by INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Conditional vector column (only if pgvector is available)
DO $$ 
BEGIN
    -- Try to add vector column for embeddings
    ALTER TABLE evidence ADD COLUMN embedding vector(384);
    RAISE NOTICE 'Added embedding vector column to evidence table';
EXCEPTION WHEN undefined_object THEN
    RAISE NOTICE 'Skipped vector column - pgvector not installed (this is OK)';
END
$$;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES cases(id) ON DELETE CASCADE,
    evidence_id INTEGER REFERENCES evidence(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    extracted_text TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Conditional vector column for documents
DO $$ 
BEGIN
    ALTER TABLE documents ADD COLUMN embeddings vector(384);
    RAISE NOTICE 'Added embeddings vector column to documents table';
EXCEPTION WHEN undefined_object THEN
    RAISE NOTICE 'Skipped vector column - pgvector not installed (this is OK)';
END
$$;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_history (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES cases(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    response TEXT,
    model VARCHAR(100),
    tokens_used INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at);
CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence(type);
CREATE INDEX IF NOT EXISTS idx_evidence_created_at ON evidence(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_case_id ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_documents_evidence_id ON documents(evidence_id);
CREATE INDEX IF NOT EXISTS idx_ai_history_case_id ON ai_history(case_id);

-- Conditional vector indexes (only if pgvector is available)
DO $$ 
BEGIN
    CREATE INDEX IF NOT EXISTS idx_evidence_embedding ON evidence USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_documents_embeddings ON documents USING ivfflat (embeddings vector_cosine_ops) WITH (lists = 100);
    RAISE NOTICE 'Created vector indexes for similarity search';
EXCEPTION WHEN undefined_object THEN
    RAISE NOTICE 'Skipped vector indexes - pgvector not installed (this is OK)';
END
$$;

-- Insert sample data
INSERT INTO users (username, email, role) VALUES 
('admin', 'admin@legal-ai.com', 'admin'),
('prosecutor', 'prosecutor@legal-ai.com', 'prosecutor'),
('detective', 'detective@legal-ai.com', 'detective')
ON CONFLICT (username) DO NOTHING;

INSERT INTO cases (title, description, status, priority, created_by) VALUES 
('Sample Case 001', 'This is a sample legal case for testing purposes', 'active', 'high', 1),
('Evidence Review Case', 'Case requiring thorough evidence analysis', 'active', 'medium', 2)
ON CONFLICT DO NOTHING;

INSERT INTO evidence (case_id, title, description, type, content, tags) VALUES 
(1, 'Document Evidence A', 'Key document for case 001', 'document', 'Sample document content for legal analysis', ARRAY['important', 'document']),
(1, 'Witness Statement', 'Statement from key witness', 'statement', 'Witness testimony content', ARRAY['witness', 'testimony']),
(2, 'Digital Evidence', 'Digital forensics evidence', 'digital', 'Digital evidence content', ARRAY['digital', 'forensics'])
ON CONFLICT DO NOTHING;

-- Final status report
SELECT 
    'Database Setup Complete!' as status,
    COUNT(*) as total_cases 
FROM cases;

SELECT 
    'Evidence Records Created' as status,
    COUNT(*) as total_evidence 
FROM evidence;

-- Check if pgvector is available
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE NOTICE '✅ pgvector extension is installed and ready!';
        RAISE NOTICE '🚀 AI similarity search features are enabled';
    ELSE
        RAISE NOTICE '⚠️  pgvector extension not installed - database is functional but AI features limited';
        RAISE NOTICE '📋 To enable full AI features, install pgvector extension later';
    END IF;
END
$$;

RAISE NOTICE '🎉 Legal AI Database setup completed successfully!';
RAISE NOTICE '🔗 Connection string: postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_db';