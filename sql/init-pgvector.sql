-- PostgreSQL + pgvector Schema for Filesystem Indexer
-- Database: legal_ai_db

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Drop existing tables if needed (careful in production)
DROP TABLE IF EXISTS analysis_results CASCADE;
DROP TABLE IF EXISTS indexed_files CASCADE;
DROP TABLE IF EXISTS error_patterns CASCADE;
DROP TABLE IF EXISTS fix_recommendations CASCADE;

-- Indexed files with embeddings
CREATE TABLE indexed_files (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    relative_path TEXT NOT NULL,
    file_type VARCHAR(20),
    content_hash VARCHAR(64),
    size BIGINT,
    modified_at TIMESTAMP,
    exports INTEGER DEFAULT 0,
    imports INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    embedding vector(384),  -- For nomic-embed-text
    summary TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_files_path ON indexed_files USING gin(file_path gin_trgm_ops);
CREATE INDEX idx_files_type ON indexed_files(file_type);
CREATE INDEX idx_files_modified ON indexed_files(modified_at DESC);
CREATE INDEX idx_files_embedding ON indexed_files USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_files_metadata ON indexed_files USING gin(metadata);

-- Analysis results table
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    total_files INTEGER,
    total_errors INTEGER,
    analysis_data JSONB,
    recommendations JSONB,
    fix_strategy JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Error patterns tracking
CREATE TABLE error_patterns (
    id SERIAL PRIMARY KEY,
    pattern VARCHAR(100) UNIQUE,
    error_code VARCHAR(20),
    count INTEGER DEFAULT 1,
    severity VARCHAR(20),
    category VARCHAR(50),
    description TEXT,
    affected_files TEXT[],
    first_seen TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW()
);

-- Fix recommendations
CREATE TABLE fix_recommendations (
    id SERIAL PRIMARY KEY,
    priority INTEGER,
    title VARCHAR(255),
    description TEXT,
    affected_files TEXT[],
    commands TEXT[],
    code_changes JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Type definitions registry
CREATE TABLE type_definitions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    kind VARCHAR(50), -- interface, type, enum, class
    file_path TEXT,
    line_number INTEGER,
    definition TEXT,
    properties JSONB,
    extends TEXT[],
    implements TEXT[],
    generics TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Import/Export relationships
CREATE TABLE module_dependencies (
    id SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    target_module TEXT NOT NULL,
    import_type VARCHAR(50), -- named, default, namespace, type
    specifiers TEXT[],
    is_external BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_file, target_module)
);

-- Component registry (for Svelte/React components)
CREATE TABLE components (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    framework VARCHAR(20), -- svelte, react, vue
    props JSONB,
    events TEXT[],
    slots TEXT[],
    stores TEXT[],
    dependencies TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(name, file_path)
);

-- AI processing queue
CREATE TABLE ai_processing_queue (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    processing_type VARCHAR(50), -- embed, summarize, analyze, fix
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    payload JSONB,
    result JSONB,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create functions for vector similarity search
CREATE OR REPLACE FUNCTION search_similar_files(
    query_embedding vector(384),
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE(
    file_path TEXT,
    similarity FLOAT,
    summary TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.file_path,
        1 - (f.embedding <=> query_embedding) AS similarity,
        f.summary
    FROM indexed_files f
    WHERE f.embedding IS NOT NULL
    ORDER BY f.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Function to find files with specific error patterns
CREATE OR REPLACE FUNCTION find_files_with_errors(
    error_code VARCHAR(20)
)
RETURNS TABLE(
    file_path TEXT,
    error_count INTEGER,
    last_modified TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.file_path,
        f.errors,
        f.modified_at
    FROM indexed_files f
    WHERE f.metadata @> jsonb_build_object('errors', jsonb_build_array(jsonb_build_object('code', error_code)))
    ORDER BY f.errors DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get dependency graph
CREATE OR REPLACE FUNCTION get_dependency_graph(
    root_file TEXT
)
RETURNS TABLE(
    level INTEGER,
    file_path TEXT,
    imports TEXT[]
) AS $$
WITH RECURSIVE dep_tree AS (
    -- Base case
    SELECT 
        0 AS level,
        source_file AS file_path,
        ARRAY_AGG(target_module) AS imports
    FROM module_dependencies
    WHERE source_file = root_file
    GROUP BY source_file
    
    UNION ALL
    
    -- Recursive case
    SELECT 
        dt.level + 1,
        md.source_file,
        ARRAY_AGG(md.target_module)
    FROM dep_tree dt
    CROSS JOIN LATERAL UNNEST(dt.imports) AS imp(module)
    JOIN module_dependencies md ON md.source_file = imp.module
    WHERE dt.level < 5  -- Limit recursion depth
    GROUP BY dt.level, md.source_file
)
SELECT * FROM dep_tree;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_indexed_files_updated_at
    BEFORE UPDATE ON indexed_files
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_type_definitions_updated_at
    BEFORE UPDATE ON type_definitions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Create materialized view for quick stats
CREATE MATERIALIZED VIEW file_statistics AS
SELECT 
    file_type,
    COUNT(*) AS file_count,
    SUM(size) AS total_size,
    AVG(exports) AS avg_exports,
    AVG(imports) AS avg_imports,
    SUM(errors) AS total_errors,
    MAX(modified_at) AS last_modified
FROM indexed_files
GROUP BY file_type;

-- Create index on materialized view
CREATE INDEX idx_file_stats_type ON file_statistics(file_type);

-- Sample data for testing
INSERT INTO indexed_files (file_path, relative_path, file_type, size, exports, imports, errors, summary)
VALUES 
    ('/app/src/index.ts', 'src/index.ts', '.ts', 1024, 5, 3, 0, 'Main application entry point'),
    ('/app/src/components/Button.svelte', 'src/components/Button.svelte', '.svelte', 512, 1, 2, 2, 'Button component with props'),
    ('/app/src/lib/utils.ts', 'src/lib/utils.ts', '.ts', 256, 10, 1, 0, 'Utility functions')
ON CONFLICT (file_path) DO NOTHING;

-- Grant permissions
GRANT ALL ON ALL TABLES IN SCHEMA public TO legal_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO legal_admin;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO legal_admin;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW file_statistics;
