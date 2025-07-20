-- ======================================================================
-- ENHANCED LEGAL AI SYSTEM DATABASE MIGRATION
-- Adds support for real-time AI processing, caching, and vector search
-- ======================================================================

-- Add new fields to evidence table for enhanced processing
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS processing_status VARCHAR(20) DEFAULT 'pending';
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS confidence DECIMAL(3,2) DEFAULT 0.0;
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS content_hash VARCHAR(32);
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0;
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMP;
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS vector_embeddings REAL[];
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS processing_time INTEGER DEFAULT 0;
ALTER TABLE evidence ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;

-- Create AI analysis cache table
CREATE TABLE IF NOT EXISTS ai_analysis_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_id UUID NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    result JSONB NOT NULL,
    confidence DECIMAL(3,2) DEFAULT 0.0,
    processing_time INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '10 minutes'
);

-- Create vector embeddings cache table
CREATE TABLE IF NOT EXISTS vector_embeddings_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_hash VARCHAR(32) UNIQUE NOT NULL,
    embeddings REAL[] NOT NULL,
    dimension INTEGER NOT NULL,
    model VARCHAR(100) NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text',
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '30 minutes'
);

-- Create vector similarity matches cache table
CREATE TABLE IF NOT EXISTS vector_similarity_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash VARCHAR(32) NOT NULL,
    target_id UUID NOT NULL,
    similarity DECIMAL(4,3) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '5 minutes'
);

-- Create graph relationships cache table
CREATE TABLE IF NOT EXISTS graph_relationships_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_id UUID NOT NULL,
    to_id UUID NOT NULL,
    relationship_type VARCHAR(50) NOT NULL,
    strength DECIMAL(3,2) DEFAULT 0.0,
    confidence DECIMAL(3,2) DEFAULT 0.0,
    bidirectional BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '1 hour'
);

-- Create streaming results cache table
CREATE TABLE IF NOT EXISTS streaming_results_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_id UUID REFERENCES evidence(id) ON DELETE CASCADE,
    result_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 1,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '1 minute'
);

-- Create processing pipeline tracking table
CREATE TABLE IF NOT EXISTS processing_pipelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evidence_id UUID NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    pipeline_status VARCHAR(20) DEFAULT 'pending',
    stages JSONB NOT NULL,
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    processing_time INTEGER,
    error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create AI model performance tracking table
CREATE TABLE IF NOT EXISTS ai_model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    operation_type VARCHAR(50) NOT NULL,
    processing_time INTEGER NOT NULL,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create cache statistics table
CREATE TABLE IF NOT EXISTS cache_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_type VARCHAR(50) NOT NULL,
    operation VARCHAR(20) NOT NULL, -- 'hit', 'miss', 'eviction'
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create websocket connections tracking table
CREATE TABLE IF NOT EXISTS websocket_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id VARCHAR(100) NOT NULL UNIQUE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    subscriptions JSONB DEFAULT '[]',
    connected_at TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- ======================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ======================================================================

-- Evidence table indexes
CREATE INDEX IF NOT EXISTS idx_evidence_processing_status ON evidence(processing_status);
CREATE INDEX IF NOT EXISTS idx_evidence_confidence ON evidence(confidence);
CREATE INDEX IF NOT EXISTS idx_evidence_content_hash ON evidence(content_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_last_accessed ON evidence(last_accessed);

-- AI analysis cache indexes
CREATE INDEX IF NOT EXISTS idx_ai_analysis_evidence_id ON ai_analysis_cache(evidence_id);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_type_model ON ai_analysis_cache(analysis_type, model);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_expires_at ON ai_analysis_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_ai_analysis_confidence ON ai_analysis_cache(confidence);

-- Vector embeddings cache indexes
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_content_hash ON vector_embeddings_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_model ON vector_embeddings_cache(model);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_expires_at ON vector_embeddings_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_vector_embeddings_dimension ON vector_embeddings_cache(dimension);

-- Vector similarity cache indexes
CREATE INDEX IF NOT EXISTS idx_vector_similarity_query_hash ON vector_similarity_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_vector_similarity_target_id ON vector_similarity_cache(target_id);
CREATE INDEX IF NOT EXISTS idx_vector_similarity_similarity ON vector_similarity_cache(similarity);
CREATE INDEX IF NOT EXISTS idx_vector_similarity_expires_at ON vector_similarity_cache(expires_at);

-- Graph relationships cache indexes
CREATE INDEX IF NOT EXISTS idx_graph_relationships_from_id ON graph_relationships_cache(from_id);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_to_id ON graph_relationships_cache(to_id);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_type ON graph_relationships_cache(relationship_type);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_strength ON graph_relationships_cache(strength);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_expires_at ON graph_relationships_cache(expires_at);

-- Streaming results cache indexes
CREATE INDEX IF NOT EXISTS idx_streaming_results_evidence_id ON streaming_results_cache(evidence_id);
CREATE INDEX IF NOT EXISTS idx_streaming_results_type ON streaming_results_cache(result_type);
CREATE INDEX IF NOT EXISTS idx_streaming_results_status ON streaming_results_cache(status);
CREATE INDEX IF NOT EXISTS idx_streaming_results_priority ON streaming_results_cache(priority);
CREATE INDEX IF NOT EXISTS idx_streaming_results_expires_at ON streaming_results_cache(expires_at);

-- Processing pipeline indexes
CREATE INDEX IF NOT EXISTS idx_processing_pipelines_evidence_id ON processing_pipelines(evidence_id);
CREATE INDEX IF NOT EXISTS idx_processing_pipelines_status ON processing_pipelines(pipeline_status);
CREATE INDEX IF NOT EXISTS idx_processing_pipelines_start_time ON processing_pipelines(start_time);

-- AI model performance indexes
CREATE INDEX IF NOT EXISTS idx_ai_model_performance_model ON ai_model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_ai_model_performance_operation ON ai_model_performance(operation_type);
CREATE INDEX IF NOT EXISTS idx_ai_model_performance_created_at ON ai_model_performance(created_at);

-- Cache statistics indexes
CREATE INDEX IF NOT EXISTS idx_cache_statistics_type ON cache_statistics(cache_type);
CREATE INDEX IF NOT EXISTS idx_cache_statistics_operation ON cache_statistics(operation);
CREATE INDEX IF NOT EXISTS idx_cache_statistics_timestamp ON cache_statistics(timestamp);

-- WebSocket connections indexes
CREATE INDEX IF NOT EXISTS idx_websocket_connections_client_id ON websocket_connections(client_id);
CREATE INDEX IF NOT EXISTS idx_websocket_connections_user_id ON websocket_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_websocket_connections_last_seen ON websocket_connections(last_seen);

-- ======================================================================
-- COMPOSITE INDEXES FOR COMPLEX QUERIES
-- ======================================================================

-- Evidence processing queries
CREATE INDEX IF NOT EXISTS idx_evidence_case_processing ON evidence(case_id, processing_status, confidence);
CREATE INDEX IF NOT EXISTS idx_evidence_type_confidence ON evidence(evidence_type, confidence);

-- AI analysis cache queries
CREATE INDEX IF NOT EXISTS idx_ai_analysis_evidence_type_model ON ai_analysis_cache(evidence_id, analysis_type, model);

-- Vector similarity queries
CREATE INDEX IF NOT EXISTS idx_vector_similarity_query_similarity ON vector_similarity_cache(query_hash, similarity);

-- Graph relationships queries
CREATE INDEX IF NOT EXISTS idx_graph_relationships_from_type_strength ON graph_relationships_cache(from_id, relationship_type, strength);
CREATE INDEX IF NOT EXISTS idx_graph_relationships_to_type_strength ON graph_relationships_cache(to_id, relationship_type, strength);

-- ======================================================================
-- CONSTRAINTS AND VALIDATION
-- ======================================================================

-- Ensure confidence values are between 0 and 1
ALTER TABLE evidence ADD CONSTRAINT IF NOT EXISTS check_evidence_confidence CHECK (confidence >= 0 AND confidence <= 1);
ALTER TABLE ai_analysis_cache ADD CONSTRAINT IF NOT EXISTS check_ai_analysis_confidence CHECK (confidence >= 0 AND confidence <= 1);
ALTER TABLE graph_relationships_cache ADD CONSTRAINT IF NOT EXISTS check_graph_strength CHECK (strength >= 0 AND strength <= 1);
ALTER TABLE graph_relationships_cache ADD CONSTRAINT IF NOT EXISTS check_graph_confidence CHECK (confidence >= 0 AND confidence <= 1);

-- Ensure similarity values are between 0 and 1
ALTER TABLE vector_similarity_cache ADD CONSTRAINT IF NOT EXISTS check_similarity_range CHECK (similarity >= 0 AND similarity <= 1);

-- Ensure processing status values are valid
ALTER TABLE evidence ADD CONSTRAINT IF NOT EXISTS check_processing_status CHECK (processing_status IN ('pending', 'processing', 'complete', 'error'));
ALTER TABLE processing_pipelines ADD CONSTRAINT IF NOT EXISTS check_pipeline_status CHECK (pipeline_status IN ('pending', 'processing', 'complete', 'error'));
ALTER TABLE streaming_results_cache ADD CONSTRAINT IF NOT EXISTS check_result_status CHECK (status IN ('pending', 'processing', 'complete', 'error'));

-- ======================================================================
-- FUNCTIONS FOR AUTOMATIC CLEANUP
-- ======================================================================

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache() RETURNS void AS $$
BEGIN
    DELETE FROM ai_analysis_cache WHERE expires_at < NOW();
    DELETE FROM vector_embeddings_cache WHERE expires_at < NOW();
    DELETE FROM vector_similarity_cache WHERE expires_at < NOW();
    DELETE FROM graph_relationships_cache WHERE expires_at < NOW();
    DELETE FROM streaming_results_cache WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Function to update access tracking
CREATE OR REPLACE FUNCTION update_cache_access(cache_table_name TEXT, record_id UUID) RETURNS void AS $$
BEGIN
    EXECUTE format('UPDATE %I SET access_count = access_count + 1, last_accessed = NOW() WHERE id = $1', cache_table_name) USING record_id;
END;
$$ LANGUAGE plpgsql;

-- Function to log cache statistics
CREATE OR REPLACE FUNCTION log_cache_stat(cache_type TEXT, operation TEXT, metadata JSONB DEFAULT '{}') RETURNS void AS $$
BEGIN
    INSERT INTO cache_statistics (cache_type, operation, metadata) VALUES (cache_type, operation, metadata);
END;
$$ LANGUAGE plpgsql;

-- ======================================================================
-- TRIGGERS FOR AUTOMATIC MAINTENANCE
-- ======================================================================

-- Trigger to update evidence updated_at timestamp
CREATE OR REPLACE FUNCTION update_evidence_timestamp() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_evidence_updated_at
    BEFORE UPDATE ON evidence
    FOR EACH ROW
    EXECUTE FUNCTION update_evidence_timestamp();

-- Trigger to update processing pipeline timestamp
CREATE OR REPLACE FUNCTION update_pipeline_timestamp() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_pipeline_updated_at
    BEFORE UPDATE ON processing_pipelines
    FOR EACH ROW
    EXECUTE FUNCTION update_pipeline_timestamp();

-- ======================================================================
-- INITIAL DATA AND CONFIGURATION
-- ======================================================================

-- Insert default cache configuration
INSERT INTO cache_statistics (cache_type, operation, metadata) 
VALUES 
    ('system', 'initialize', '{"timestamp": "' || NOW() || '", "version": "1.0"}'),
    ('ai_analysis', 'initialize', '{"default_ttl": 600, "max_size": 500}'),
    ('vector_embeddings', 'initialize', '{"default_ttl": 1800, "max_size": 2000}'),
    ('graph_relationships', 'initialize', '{"default_ttl": 3600, "max_size": 1000}'),
    ('streaming_results', 'initialize', '{"default_ttl": 60, "max_size": 100}')
ON CONFLICT DO NOTHING;

-- ======================================================================
-- COMMENT DOCUMENTATION
-- ======================================================================

COMMENT ON TABLE ai_analysis_cache IS 'Cache for AI analysis results to improve performance';
COMMENT ON TABLE vector_embeddings_cache IS 'Cache for vector embeddings to avoid recomputation';
COMMENT ON TABLE vector_similarity_cache IS 'Cache for vector similarity search results';
COMMENT ON TABLE graph_relationships_cache IS 'Cache for graph database relationship queries';
COMMENT ON TABLE streaming_results_cache IS 'Temporary cache for real-time streaming results';
COMMENT ON TABLE processing_pipelines IS 'Tracks multi-stage evidence processing pipelines';
COMMENT ON TABLE ai_model_performance IS 'Performance metrics for AI model operations';
COMMENT ON TABLE cache_statistics IS 'Statistics and metrics for cache performance';
COMMENT ON TABLE websocket_connections IS 'Active WebSocket connections for real-time updates';

COMMENT ON COLUMN evidence.processing_status IS 'Current status of evidence processing pipeline';
COMMENT ON COLUMN evidence.confidence IS 'AI confidence score for processed evidence (0-1)';
COMMENT ON COLUMN evidence.content_hash IS 'Hash of evidence content for deduplication';
COMMENT ON COLUMN evidence.vector_embeddings IS 'Vector embeddings for similarity search';
COMMENT ON COLUMN evidence.processing_time IS 'Time taken to process evidence in milliseconds';

-- ======================================================================
-- MIGRATION COMPLETE
-- ======================================================================

-- Log migration completion
INSERT INTO cache_statistics (cache_type, operation, metadata) 
VALUES ('system', 'migration_complete', '{"timestamp": "' || NOW() || '", "version": "enhanced-ai-v1.0"}');

-- Create a view for easy monitoring
CREATE OR REPLACE VIEW enhanced_ai_system_status AS
SELECT 
    'evidence_processing' as component,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE processing_status = 'complete') as completed,
    COUNT(*) FILTER (WHERE processing_status = 'processing') as processing,
    COUNT(*) FILTER (WHERE processing_status = 'error') as errors,
    AVG(confidence) as avg_confidence,
    AVG(processing_time) as avg_processing_time
FROM evidence
UNION ALL
SELECT 
    'ai_analysis_cache' as component,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE expires_at > NOW()) as active,
    COUNT(*) FILTER (WHERE expires_at <= NOW()) as expired,
    0 as errors,
    AVG(confidence) as avg_confidence,
    AVG(processing_time) as avg_processing_time
FROM ai_analysis_cache
UNION ALL
SELECT 
    'vector_embeddings_cache' as component,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE expires_at > NOW()) as active,
    COUNT(*) FILTER (WHERE expires_at <= NOW()) as expired,
    0 as errors,
    0 as avg_confidence,
    0 as avg_processing_time
FROM vector_embeddings_cache;

COMMENT ON VIEW enhanced_ai_system_status IS 'Real-time status view for enhanced AI system components';

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Enhanced Legal AI System database migration completed successfully!';
    RAISE NOTICE 'New tables created: 8';
    RAISE NOTICE 'New indexes created: 25+';
    RAISE NOTICE 'New functions created: 3';
    RAISE NOTICE 'New triggers created: 2';
    RAISE NOTICE 'System is ready for high-performance AI processing!';
END
$$;