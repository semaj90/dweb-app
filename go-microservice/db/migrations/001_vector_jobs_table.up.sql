-- Vector jobs table with JSONB optimization for legal metadata
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Vector processing jobs with advanced indexing
CREATE TABLE vector_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(50) NOT NULL,
    owner_type VARCHAR(50) NOT NULL,
    owner_id VARCHAR(255) NOT NULL,
    
    -- JSONB for flexible metadata storage
    metadata JSONB NOT NULL DEFAULT '{}',
    
    -- Vector data with pgvector extension
    input_vector vector(768),
    output_vector vector(768),
    
    -- Processing details
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    
    -- GPU metrics
    gpu_name VARCHAR(255),
    processing_time_ms INTEGER,
    memory_usage_mb INTEGER,
    
    -- Legal-specific fields
    legal_domain VARCHAR(100),
    confidence_score REAL,
    precedent_strength REAL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scheduled_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Error tracking
    error_message TEXT,
    error_details JSONB
);

-- Indexes for high-performance queries
CREATE INDEX idx_vector_jobs_status ON vector_jobs(status);
CREATE INDEX idx_vector_jobs_owner ON vector_jobs(owner_type, owner_id);
CREATE INDEX idx_vector_jobs_type_status ON vector_jobs(job_type, status);
CREATE INDEX idx_vector_jobs_created_at ON vector_jobs(created_at DESC);
CREATE INDEX idx_vector_jobs_priority_created ON vector_jobs(priority DESC, created_at ASC) WHERE status = 'pending';

-- JSONB GIN index for complex metadata queries
CREATE INDEX idx_vector_jobs_metadata_gin ON vector_jobs USING gin (metadata);

-- Vector similarity index
CREATE INDEX idx_vector_jobs_input_vector ON vector_jobs USING hnsw (input_vector vector_cosine_ops);
CREATE INDEX idx_vector_jobs_output_vector ON vector_jobs USING hnsw (output_vector vector_cosine_ops);

-- Legal domain optimized index
CREATE INDEX idx_vector_jobs_legal_domain ON vector_jobs(legal_domain) WHERE legal_domain IS NOT NULL;

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_vector_jobs_updated_at
BEFORE UPDATE ON vector_jobs
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();