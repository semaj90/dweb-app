-- migrations/create_evidence_processing_schema.sql
-- Database migration for evidence processing system

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable uuid extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create evidence_process table
CREATE TABLE IF NOT EXISTS evidence_process (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    steps JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error TEXT
);

-- Create indexes for evidence_process
CREATE INDEX IF NOT EXISTS evidence_process_evidence_id_idx ON evidence_process(evidence_id);
CREATE INDEX IF NOT EXISTS evidence_process_status_idx ON evidence_process(status);
CREATE INDEX IF NOT EXISTS evidence_process_requested_by_idx ON evidence_process(requested_by);
CREATE INDEX IF NOT EXISTS evidence_process_created_at_idx ON evidence_process(created_at);

-- Create evidence_ocr table
CREATE TABLE IF NOT EXISTS evidence_ocr (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id TEXT NOT NULL,
    text TEXT NOT NULL,
    confidence DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for evidence_ocr
CREATE INDEX IF NOT EXISTS evidence_ocr_evidence_id_idx ON evidence_ocr(evidence_id);
CREATE INDEX IF NOT EXISTS evidence_ocr_created_at_idx ON evidence_ocr(created_at);

-- Create evidence_embeddings table (metadata only)
CREATE TABLE IF NOT EXISTS evidence_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for evidence_embeddings
CREATE INDEX IF NOT EXISTS evidence_embeddings_evidence_id_idx ON evidence_embeddings(evidence_id);
CREATE INDEX IF NOT EXISTS evidence_embeddings_model_idx ON evidence_embeddings(model);

-- Create evidence_vectors table with pgvector
CREATE TABLE IF NOT EXISTS evidence_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector VECTOR,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for evidence_vectors
CREATE INDEX IF NOT EXISTS evidence_vectors_evidence_id_model_idx ON evidence_vectors(evidence_id, model);
-- IVFFLAT index for vector similarity search (adjust lists parameter based on data size)
CREATE INDEX IF NOT EXISTS evidence_vectors_vector_idx ON evidence_vectors USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

-- Create unique constraint to prevent duplicate embeddings
CREATE UNIQUE INDEX IF NOT EXISTS evidence_vectors_unique_idx ON evidence_vectors(evidence_id, model);

-- Create evidence_analysis table
CREATE TABLE IF NOT EXISTS evidence_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    confidence DECIMAL(5,4),
    snippets JSONB,
    relevant_docs JSONB,
    entities JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for evidence_analysis
CREATE INDEX IF NOT EXISTS evidence_analysis_evidence_id_idx ON evidence_analysis(evidence_id);
CREATE INDEX IF NOT EXISTS evidence_analysis_created_at_idx ON evidence_analysis(created_at);

-- Create system health monitoring table
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('healthy', 'degraded', 'down')),
    metrics JSONB,
    last_check TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for system_health
CREATE INDEX IF NOT EXISTS system_health_service_idx ON system_health(service);
CREATE INDEX IF NOT EXISTS system_health_last_check_idx ON system_health(last_check);

-- Create queue statistics table
CREATE TABLE IF NOT EXISTS queue_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_name TEXT NOT NULL,
    messages_pending INTEGER DEFAULT 0,
    messages_processing INTEGER DEFAULT 0,
    messages_completed INTEGER DEFAULT 0,
    messages_failed INTEGER DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for queue_stats
CREATE INDEX IF NOT EXISTS queue_stats_queue_name_idx ON queue_stats(queue_name);
CREATE INDEX IF NOT EXISTS queue_stats_last_updated_idx ON queue_stats(last_updated);

-- Create unique constraint for queue_stats
CREATE UNIQUE INDEX IF NOT EXISTS queue_stats_unique_queue_idx ON queue_stats(queue_name);

-- Create trigger to update updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply the trigger to relevant tables
DROP TRIGGER IF EXISTS update_evidence_process_updated_at ON evidence_process;
CREATE TRIGGER update_evidence_process_updated_at 
    BEFORE UPDATE ON evidence_process 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_evidence_vectors_updated_at ON evidence_vectors;
CREATE TRIGGER update_evidence_vectors_updated_at 
    BEFORE UPDATE ON evidence_vectors 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for vector similarity search
CREATE OR REPLACE FUNCTION find_similar_evidence(
    query_vector VECTOR,
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    evidence_id TEXT,
    similarity FLOAT,
    model TEXT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ev.evidence_id,
        1 - (ev.vector <=> query_vector) as similarity,
        ev.model,
        ev.metadata
    FROM evidence_vectors ev
    WHERE 1 - (ev.vector <=> query_vector) > similarity_threshold
    ORDER BY ev.vector <=> query_vector
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Insert initial queue stats
INSERT INTO queue_stats (queue_name, messages_pending, messages_processing, messages_completed, messages_failed)
VALUES 
    ('evidence.process.queue', 0, 0, 0, 0),
    ('evidence.process.control', 0, 0, 0, 0),
    ('evidence.ocr.queue', 0, 0, 0, 0),
    ('evidence.embedding.queue', 0, 0, 0, 0),
    ('evidence.rag.queue', 0, 0, 0, 0)
ON CONFLICT (queue_name) DO NOTHING;

COMMENT ON TABLE evidence_process IS 'Tracks evidence processing jobs through the pipeline';
COMMENT ON TABLE evidence_ocr IS 'Stores OCR results for evidence files';
COMMENT ON TABLE evidence_embeddings IS 'Metadata for generated embeddings';
COMMENT ON TABLE evidence_vectors IS 'pgvector storage for similarity search';
COMMENT ON TABLE evidence_analysis IS 'RAG analysis results and extracted insights';
COMMENT ON TABLE system_health IS 'System component health monitoring';
COMMENT ON TABLE queue_stats IS 'RabbitMQ queue statistics';
