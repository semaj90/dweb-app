-- Phase 14 Evidence Processing - Database Schema Consolidation
-- Resolves merger conflicts and consolidates schemas

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- CONSOLIDATED USER MANAGEMENT
-- =====================================================

-- Drop existing conflicting tables if they exist
DROP TABLE IF EXISTS user_sessions CASCADE;
DROP TABLE IF EXISTS users_old CASCADE;

-- Unified users table
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  first_name VARCHAR(100),
  last_name VARCHAR(100),
  role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('admin', 'lawyer', 'paralegal', 'user')),
  is_active BOOLEAN DEFAULT true,
  email_verified BOOLEAN DEFAULT false,
  last_login TIMESTAMPTZ,
  failed_login_attempts INTEGER DEFAULT 0,
  locked_until TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User sessions for authentication
CREATE TABLE IF NOT EXISTS user_sessions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  session_token VARCHAR(255) UNIQUE NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- CONSOLIDATED DOCUMENT MANAGEMENT
-- =====================================================

-- Cases table
CREATE TABLE IF NOT EXISTS cases (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  case_number VARCHAR(100) UNIQUE NOT NULL,
  title VARCHAR(500) NOT NULL,
  description TEXT,
  status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'closed', 'pending', 'archived')),
  client_name VARCHAR(200),
  assigned_lawyer_id UUID REFERENCES users(id),
  created_by_id UUID NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  closed_at TIMESTAMPTZ
);

-- Unified documents table (consolidates all document schemas)
CREATE TABLE IF NOT EXISTS legal_documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
  original_name VARCHAR(500) NOT NULL,
  file_name VARCHAR(500) NOT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT,
  mime_type VARCHAR(100),
  document_type VARCHAR(100) DEFAULT 'general' CHECK (
    document_type IN ('contract', 'evidence', 'pleading', 'correspondence', 'research', 'general')
  ),
  content TEXT, -- Extracted text content
  metadata JSONB DEFAULT '{}',
  
  -- Phase 14: Unified embedding dimensions (384)
  embedding VECTOR(384),
  
  -- Processing status
  processing_status VARCHAR(50) DEFAULT 'pending' CHECK (
    processing_status IN ('pending', 'processing', 'completed', 'failed', 'archived')
  ),
  processing_error TEXT,
  
  -- OCR and extraction details
  ocr_confidence DECIMAL(5,4),
  text_extracted_at TIMESTAMPTZ,
  embedding_generated_at TIMESTAMPTZ,
  
  -- Audit fields
  uploaded_by_id UUID NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Search and indexing
  search_vector tsvector GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(original_name, '') || ' ' || coalesce(content, ''))
  ) STORED
);

-- =====================================================
-- EVIDENCE MANAGEMENT (Phase 14 Focus)
-- =====================================================

-- Evidence items linked to cases and documents
CREATE TABLE IF NOT EXISTS evidence_items (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  document_id UUID REFERENCES legal_documents(id) ON DELETE SET NULL,
  
  -- Evidence details
  evidence_number VARCHAR(100) NOT NULL,
  evidence_type VARCHAR(100) NOT NULL CHECK (
    evidence_type IN ('physical', 'digital', 'testimony', 'document', 'photograph', 'video', 'audio')
  ),
  title VARCHAR(500) NOT NULL,
  description TEXT,
  
  -- Chain of custody
  chain_of_custody JSONB DEFAULT '[]',
  current_location VARCHAR(200),
  custodian_id UUID REFERENCES users(id),
  
  -- Evidence metadata
  collection_date DATE,
  collection_location VARCHAR(200),
  collected_by VARCHAR(200),
  
  -- Legal status
  admissible BOOLEAN,
  privilege_claim BOOLEAN DEFAULT false,
  privilege_type VARCHAR(100),
  
  -- Processing for RAG
  content_summary TEXT,
  key_findings JSONB DEFAULT '[]',
  related_evidence UUID[] DEFAULT '{}',
  
  -- Audit trail
  created_by_id UUID NOT NULL REFERENCES users(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(case_id, evidence_number)
);

-- =====================================================
-- RAG PIPELINE TABLES
-- =====================================================

-- Document chunks for RAG processing
CREATE TABLE IF NOT EXISTS document_chunks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  document_id UUID NOT NULL REFERENCES legal_documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  start_offset INTEGER,
  end_offset INTEGER,
  token_count INTEGER,
  
  -- Phase 14: Consistent 384-dimensional embeddings
  embedding VECTOR(384),
  
  -- Chunk metadata
  chunk_type VARCHAR(50) DEFAULT 'text',
  relevance_score DECIMAL(5,4),
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(document_id, chunk_index)
);

-- RAG queries and responses for analytics
CREATE TABLE IF NOT EXISTS rag_queries (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES users(id),
  case_id UUID REFERENCES cases(id),
  
  -- Query details
  query_text TEXT NOT NULL,
  query_embedding VECTOR(384),
  
  -- Response details
  response_text TEXT,
  confidence_score DECIMAL(5,4),
  processing_time_ms INTEGER,
  
  -- Retrieved documents
  retrieved_documents JSONB DEFAULT '[]',
  reranked_results JSONB DEFAULT '[]',
  
  -- Feedback
  user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
  user_feedback TEXT,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- INDEXING AND PERFORMANCE
-- =====================================================

-- Primary indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_cases_number ON cases(case_number);
CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_lawyer ON cases(assigned_lawyer_id);

CREATE INDEX IF NOT EXISTS idx_documents_case ON legal_documents(case_id);
CREATE INDEX IF NOT EXISTS idx_documents_type ON legal_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_status ON legal_documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_search ON legal_documents USING GIN(search_vector);

CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence_items(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence_items(evidence_type);
CREATE INDEX IF NOT EXISTS idx_evidence_number ON evidence_items(evidence_number);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_index ON document_chunks(chunk_index);

CREATE INDEX IF NOT EXISTS idx_rag_user ON rag_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_rag_case ON rag_queries(case_id);
CREATE INDEX IF NOT EXISTS idx_rag_created ON rag_queries(created_at);

-- Vector similarity indexes (Phase 14 Performance)
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON legal_documents 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_rag_embedding ON rag_queries 
USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);

-- =====================================================
-- PHASE 14 MIGRATION DATA
-- =====================================================

-- Insert default admin user if not exists
INSERT INTO users (email, password_hash, first_name, last_name, role, is_active, email_verified)
SELECT 'admin@legal-ai.com', '$2b$10$example_hash', 'System', 'Administrator', 'admin', true, true
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'admin@legal-ai.com');

-- Create default case for testing
INSERT INTO cases (case_number, title, description, created_by_id)
SELECT 'CASE-2025-001', 'Phase 14 Evidence Processing Test Case', 'Test case for Phase 14 evidence processing merger', u.id
FROM users u WHERE u.email = 'admin@legal-ai.com'
AND NOT EXISTS (SELECT 1 FROM cases WHERE case_number = 'CASE-2025-001');

-- =====================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMPS
-- =====================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_cases_updated_at ON cases;
CREATE TRIGGER update_cases_updated_at BEFORE UPDATE ON cases 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_documents_updated_at ON legal_documents;
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON legal_documents 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_evidence_updated_at ON evidence_items;
CREATE TRIGGER update_evidence_updated_at BEFORE UPDATE ON evidence_items 
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Evidence dashboard view
CREATE OR REPLACE VIEW evidence_dashboard AS
SELECT 
  e.id,
  e.evidence_number,
  e.evidence_type,
  e.title,
  e.description,
  e.admissible,
  c.case_number,
  c.title as case_title,
  u.first_name || ' ' || u.last_name as created_by,
  e.created_at,
  d.original_name as document_name,
  d.processing_status as document_status
FROM evidence_items e
JOIN cases c ON e.case_id = c.id
JOIN users u ON e.created_by_id = u.id
LEFT JOIN legal_documents d ON e.document_id = d.id
ORDER BY e.created_at DESC;

-- Document processing status view
CREATE OR REPLACE VIEW document_processing_status AS
SELECT 
  d.id,
  d.original_name,
  d.file_name,
  d.document_type,
  d.processing_status,
  d.processing_error,
  c.case_number,
  u.first_name || ' ' || u.last_name as uploaded_by,
  d.created_at,
  d.text_extracted_at,
  d.embedding_generated_at,
  CASE 
    WHEN d.embedding IS NOT NULL THEN 'Yes'
    ELSE 'No'
  END as has_embedding
FROM legal_documents d
JOIN cases c ON d.case_id = c.id
JOIN users u ON d.uploaded_by_id = u.id
ORDER BY d.created_at DESC;

COMMENT ON SCHEMA public IS 'Phase 14 Evidence Processing - Consolidated Database Schema';