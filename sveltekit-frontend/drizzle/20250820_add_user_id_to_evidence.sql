-- Migration: Add user_id column to evidence table
-- Generated: 2025-08-20
-- Purpose: Fix missing user_id column causing API errors

BEGIN;

-- Add user_id column to evidence table if it doesn't exist
DO $$ 
BEGIN 
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'evidence' 
        AND column_name = 'user_id'
    ) THEN
        ALTER TABLE evidence 
        ADD COLUMN user_id UUID;
        
        -- Add foreign key constraint
        ALTER TABLE evidence 
        ADD CONSTRAINT evidence_user_id_fkey 
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
        
        -- Add index for performance
        CREATE INDEX IF NOT EXISTS evidence_user_id_idx ON evidence(user_id);
        
        -- Update existing records with a default user if needed
        -- This assumes you have at least one user in the users table
        UPDATE evidence 
        SET user_id = (SELECT id FROM users LIMIT 1)
        WHERE user_id IS NULL;
        
        -- Make the column NOT NULL after updating existing records
        ALTER TABLE evidence 
        ALTER COLUMN user_id SET NOT NULL;
    END IF;
END $$;

-- Ensure pgvector extension is installed
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector columns if they don't exist
DO $$ 
BEGIN 
    -- Add title_embedding column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'evidence' 
        AND column_name = 'title_embedding'
    ) THEN
        ALTER TABLE evidence 
        ADD COLUMN title_embedding vector(384);
        
        -- Add vector index
        CREATE INDEX IF NOT EXISTS evidence_title_embedding_idx 
        ON evidence USING hnsw (title_embedding vector_cosine_ops);
    END IF;
    
    -- Add content_embedding column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.columns 
        WHERE table_name = 'evidence' 
        AND column_name = 'content_embedding'
    ) THEN
        ALTER TABLE evidence 
        ADD COLUMN content_embedding vector(384);
        
        -- Add vector index
        CREATE INDEX IF NOT EXISTS evidence_content_embedding_idx 
        ON evidence USING hnsw (content_embedding vector_cosine_ops);
    END IF;
END $$;

-- Create chat-related tables if they don't exist
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    message_id UUID NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    recommendation_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    feedback VARCHAR(20) CHECK (feedback IN ('helpful', 'not_helpful', 'irrelevant')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS chat_sessions_user_id_idx ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS chat_messages_session_id_idx ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS chat_messages_role_idx ON chat_messages(role);
CREATE INDEX IF NOT EXISTS chat_messages_embedding_idx ON chat_messages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chat_recommendations_user_id_idx ON chat_recommendations(user_id);
CREATE INDEX IF NOT EXISTS chat_recommendations_message_id_idx ON chat_recommendations(message_id);
CREATE INDEX IF NOT EXISTS chat_recommendations_confidence_idx ON chat_recommendations(confidence);

-- Update the migration tracking table
INSERT INTO __drizzle_migrations__ (id, hash, created_at) 
VALUES (
    '20250820_add_user_id_to_evidence',
    md5('20250820_add_user_id_to_evidence'),
    NOW()
) ON CONFLICT (id) DO NOTHING;

COMMIT;