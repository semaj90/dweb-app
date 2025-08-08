-- GPU-Enhanced Legal AI Database Schema Migration
-- Fixes embedding dimension mismatch and adds GPU processing tables

-- 1. Update existing tables for GPU compatibility
ALTER TABLE legal_documents ALTER COLUMN embedding TYPE vector(768);

-- 2. Add GPU processing tables
CREATE TABLE IF NOT EXISTS indexed_files (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    file_path VARCHAR(1000) NOT NULL UNIQUE,
    content TEXT,
    embedding vector(768),
    summary TEXT,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_method VARCHAR(50) DEFAULT 'gpu',
    gpu_processing_time_ms INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- 3. User activity tracking for recommendations
CREATE TABLE IF NOT EXISTS user_activities (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    query TEXT,
    results JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feedback VARCHAR(50),
    processing_time_ms INTEGER
);

-- 4. Job queue table for BullMQ integration
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    payload JSONB,
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0
);

-- 5. Recommendation system tables
CREATE TABLE IF NOT EXISTS recommendation_models (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    model_data BYTEA, -- Serialized SOM weights
    training_iterations INTEGER DEFAULT 0,
    last_trained TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    performance_metrics JSONB DEFAULT '{}'
);

-- 6. Performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_indexed_files_path ON indexed_files(file_path);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_indexed_files_embedding ON indexed_files USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_activities_user_timestamp ON user_activities(user_id, timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processing_jobs_status ON processing_jobs(status, created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recommendation_models_user ON recommendation_models(user_id);

-- 7. Update existing data to new format (if needed)
UPDATE legal_documents SET embedding = NULL WHERE array_length(embedding::float[], 1) != 768;

-- 8. Add GPU processing triggers
CREATE OR REPLACE FUNCTION update_processing_stats()
RETURNS TRIGGER AS $$
BEGIN
    NEW.indexed_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_indexed_files_timestamp
    BEFORE UPDATE ON indexed_files
    FOR EACH ROW
    EXECUTE FUNCTION update_processing_stats();
