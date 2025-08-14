-- Database Schema for File Merge System
-- Native Windows PostgreSQL Implementation

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- File metadata table with JSONB tags
CREATE TABLE IF NOT EXISTS file_metadata (
    id VARCHAR(255) PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    original_path VARCHAR(1000) NOT NULL,
    size BIGINT NOT NULL,
    mime_type VARCHAR(200) NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags JSONB DEFAULT '{}',
    case_id VARCHAR(255),
    user_id VARCHAR(255) NOT NULL,
    embedding JSONB, -- Store as JSON array for flexibility
    vector_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Document embeddings table for pgVector
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) REFERENCES file_metadata(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536), -- Adjust dimension based on your embedding model
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(file_id)
);

-- Merge operations tracking
CREATE TABLE IF NOT EXISTS merge_operations (
    id VARCHAR(255) PRIMARY KEY,
    source_files TEXT[] NOT NULL, -- Array of file IDs
    target_filename VARCHAR(500) NOT NULL,
    merge_type VARCHAR(50) NOT NULL CHECK (merge_type IN ('concatenate', 'overlay', 'archive', 'legal-discovery')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    user_id VARCHAR(255) NOT NULL,
    case_id VARCHAR(255),
    result_file_id VARCHAR(255) REFERENCES file_metadata(id),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- File access logs for audit trail
CREATE TABLE IF NOT EXISTS file_access_logs (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) REFERENCES file_metadata(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL, -- 'upload', 'download', 'delete', 'view', 'merge'
    ip_address INET,
    user_agent TEXT,
    additional_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Legal cases table
CREATE TABLE IF NOT EXISTS legal_cases (
    id VARCHAR(255) PRIMARY KEY,
    case_number VARCHAR(255) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Similarity search cache for performance
CREATE TABLE IF NOT EXISTS similarity_search_cache (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL, -- SHA256 of query + filters
    query_text TEXT NOT NULL,
    filters JSONB DEFAULT '{}',
    results JSONB NOT NULL, -- Cached search results
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_file_metadata_case_id ON file_metadata(case_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_user_id ON file_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_mime_type ON file_metadata(mime_type);
CREATE INDEX IF NOT EXISTS idx_file_metadata_uploaded_at ON file_metadata(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_file_metadata_tags ON file_metadata USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_file_metadata_checksum ON file_metadata(checksum);

-- pgVector indexes for similarity search
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_embeddings_file_id ON document_embeddings(file_id);

-- Merge operations indexes
CREATE INDEX IF NOT EXISTS idx_merge_operations_user_id ON merge_operations(user_id);
CREATE INDEX IF NOT EXISTS idx_merge_operations_case_id ON merge_operations(case_id);
CREATE INDEX IF NOT EXISTS idx_merge_operations_status ON merge_operations(status);
CREATE INDEX IF NOT EXISTS idx_merge_operations_created_at ON merge_operations(created_at DESC);

-- Access logs indexes
CREATE INDEX IF NOT EXISTS idx_access_logs_file_id ON file_access_logs(file_id);
CREATE INDEX IF NOT EXISTS idx_access_logs_user_id ON file_access_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_access_logs_action ON file_access_logs(action);
CREATE INDEX IF NOT EXISTS idx_access_logs_created_at ON file_access_logs(created_at DESC);

-- Similarity search cache indexes
CREATE INDEX IF NOT EXISTS idx_similarity_cache_hash ON similarity_search_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_similarity_cache_expires ON similarity_search_cache(expires_at);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_file_metadata_updated_at BEFORE UPDATE ON file_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_legal_cases_updated_at BEFORE UPDATE ON legal_cases
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically log file access
CREATE OR REPLACE FUNCTION log_file_access()
RETURNS TRIGGER AS $$
BEGIN
    -- Log file uploads
    IF TG_OP = 'INSERT' THEN
        INSERT INTO file_access_logs (file_id, user_id, action, additional_data)
        VALUES (NEW.id, NEW.user_id, 'upload', jsonb_build_object('filename', NEW.filename, 'size', NEW.size));
        RETURN NEW;
    END IF;
    
    -- Log file deletions
    IF TG_OP = 'DELETE' THEN
        INSERT INTO file_access_logs (file_id, user_id, action, additional_data)
        VALUES (OLD.id, OLD.user_id, 'delete', jsonb_build_object('filename', OLD.filename));
        RETURN OLD;
    END IF;
    
    RETURN NULL;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_log_file_access
    AFTER INSERT OR DELETE ON file_metadata
    FOR EACH ROW EXECUTE FUNCTION log_file_access();

-- Function for similarity search with caching
CREATE OR REPLACE FUNCTION cached_similarity_search(
    query_embedding vector(1536),
    search_limit INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7,
    case_filter VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    file_id VARCHAR,
    filename VARCHAR,
    similarity FLOAT,
    content_excerpt TEXT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fm.id as file_id,
        fm.filename,
        (1 - (de.embedding <-> query_embedding))::FLOAT as similarity,
        SUBSTRING(de.content, 1, 500) as content_excerpt,
        fm.tags as metadata
    FROM file_metadata fm
    JOIN document_embeddings de ON fm.id = de.file_id
    WHERE (1 - (de.embedding <-> query_embedding)) > similarity_threshold
    AND (case_filter IS NULL OR fm.case_id = case_filter)
    ORDER BY similarity DESC
    LIMIT search_limit;
END;
$$ language 'plpgsql';

-- Function to cleanup expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM similarity_search_cache WHERE expires_at < NOW();
END;
$$ language 'plpgsql';

-- Function to get file statistics
CREATE OR REPLACE FUNCTION get_file_statistics(case_filter VARCHAR DEFAULT NULL)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total_files', COUNT(*),
        'total_size_bytes', COALESCE(SUM(size), 0),
        'file_types', jsonb_object_agg(mime_type, type_count),
        'upload_timeline', jsonb_object_agg(
            DATE_TRUNC('day', uploaded_at)::date, 
            day_count
        )
    ) INTO result
    FROM (
        SELECT 
            mime_type,
            COUNT(*) as type_count,
            size,
            uploaded_at
        FROM file_metadata
        WHERE (case_filter IS NULL OR case_id = case_filter)
        GROUP BY mime_type, size, uploaded_at
    ) stats,
    (
        SELECT 
            DATE_TRUNC('day', uploaded_at) as day,
            COUNT(*) as day_count
        FROM file_metadata
        WHERE (case_filter IS NULL OR case_id = case_filter)
        GROUP BY DATE_TRUNC('day', uploaded_at)
    ) timeline;
    
    RETURN COALESCE(result, '{}'::jsonb);
END;
$$ language 'plpgsql';

-- Function to merge file metadata during operations
CREATE OR REPLACE FUNCTION create_merge_operation(
    operation_id VARCHAR,
    source_file_ids TEXT[],
    target_name VARCHAR,
    merge_type_val VARCHAR,
    user_id_val VARCHAR,
    case_id_val VARCHAR DEFAULT NULL
)
RETURNS JSONB AS $$
DECLARE
    source_metadata JSONB;
    merge_metadata JSONB;
BEGIN
    -- Gather source file metadata
    SELECT jsonb_agg(
        jsonb_build_object(
            'id', id,
            'filename', filename,
            'size', size,
            'checksum', checksum,
            'uploaded_at', uploaded_at,
            'tags', tags
        )
    ) INTO source_metadata
    FROM file_metadata
    WHERE id = ANY(source_file_ids);
    
    -- Create merge metadata
    merge_metadata := jsonb_build_object(
        'operation_id', operation_id,
        'source_files', source_metadata,
        'merge_type', merge_type_val,
        'target_filename', target_name,
        'created_by', user_id_val,
        'case_id', case_id_val,
        'created_at', NOW()
    );
    
    -- Insert merge operation
    INSERT INTO merge_operations (
        id, source_files, target_filename, merge_type, 
        user_id, case_id, metadata
    ) VALUES (
        operation_id, source_file_ids, target_name, merge_type_val,
        user_id_val, case_id_val, merge_metadata
    );
    
    RETURN merge_metadata;
END;
$$ language 'plpgsql';

-- Views for common queries
CREATE OR REPLACE VIEW file_metadata_with_stats AS
SELECT 
    fm.*,
    CASE 
        WHEN de.id IS NOT NULL THEN true 
        ELSE false 
    END as has_embeddings,
    al.access_count,
    al.last_accessed
FROM file_metadata fm
LEFT JOIN document_embeddings de ON fm.id = de.file_id
LEFT JOIN (
    SELECT 
        file_id,
        COUNT(*) as access_count,
        MAX(created_at) as last_accessed
    FROM file_access_logs
    GROUP BY file_id
) al ON fm.id = al.file_id;

CREATE OR REPLACE VIEW merge_operations_with_details AS
SELECT 
    mo.*,
    array_length(mo.source_files, 1) as source_file_count,
    rf.filename as result_filename,
    rf.size as result_size,
    EXTRACT(EPOCH FROM (COALESCE(mo.completed_at, NOW()) - mo.created_at)) as duration_seconds
FROM merge_operations mo
LEFT JOIN file_metadata rf ON mo.result_file_id = rf.id;

-- Cleanup procedure to be run periodically
CREATE OR REPLACE FUNCTION cleanup_system()
RETURNS void AS $$
BEGIN
    -- Cleanup expired cache
    PERFORM cleanup_expired_cache();
    
    -- Cleanup old access logs (keep last 90 days)
    DELETE FROM file_access_logs 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    -- Cleanup failed merge operations older than 7 days
    DELETE FROM merge_operations 
    WHERE status = 'failed' 
    AND created_at < NOW() - INTERVAL '7 days';
END;
$$ language 'plpgsql';