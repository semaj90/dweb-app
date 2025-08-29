-- Fix Vector Dimensions Migration
-- Run this to fix the "Cannot read properties of undefined (reading 'dimensions')" error

-- Connect to legal_ai_db
\c legal_ai_db;

-- Enable vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop and recreate vector columns with proper dimensions
-- Note: This will remove existing vector data, but fixes the schema

-- Fix users table
ALTER TABLE users DROP COLUMN IF EXISTS profile_embedding;
ALTER TABLE users ADD COLUMN profile_embedding vector(384);

-- Fix evidence table  
ALTER TABLE evidence DROP COLUMN IF EXISTS embedding;
ALTER TABLE evidence ADD COLUMN embedding vector(384);

-- Fix legal_documents table
ALTER TABLE legal_documents DROP COLUMN IF EXISTS embedding;
ALTER TABLE legal_documents ADD COLUMN embedding vector(384);

-- Fix vectors table
ALTER TABLE vectors DROP COLUMN IF EXISTS embedding;
ALTER TABLE vectors ADD COLUMN embedding vector(384);

-- Create indexes for vector similarity search
CREATE INDEX IF NOT EXISTS users_profile_embedding_idx ON users USING hnsw (profile_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS evidence_embedding_idx ON evidence USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS legal_documents_embedding_idx ON legal_documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS vectors_embedding_idx ON vectors USING hnsw (embedding vector_cosine_ops);

-- Test vector operations
SELECT 'Vector dimensions fixed successfully' as status;

-- Show table info
\d evidence;
\d legal_documents;
\d vectors;