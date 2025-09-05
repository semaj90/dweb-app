@echo off
REM PostgreSQL pgvector Extension Setup for Vector Consumer Service
REM Configures PostgreSQL with pgvector extension for enterprise vector operations

echo Setting up PostgreSQL with pgvector extension...
echo.

REM Check if PostgreSQL is running
echo [1/6] Checking PostgreSQL installation...
pg_isready -h localhost -p 5432 >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: PostgreSQL is not running or not accessible on port 5432
    echo Please ensure PostgreSQL is installed and running
    echo Install from: https://www.postgresql.org/download/windows/
    pause
    exit /b 1
)

echo PostgreSQL is running and accessible
echo.

REM Download and install pgvector
echo [2/6] Setting up pgvector extension...

REM Check if pgvector is already installed
psql -U postgres -d postgres -c "SELECT * FROM pg_extension WHERE extname='vector';" | findstr "vector" >nul 2>&1
if %errorlevel% == 0 (
    echo pgvector extension is already installed
) else (
    echo Installing pgvector extension...
    echo.
    echo Manual installation required:
    echo 1. Download pgvector from: https://github.com/pgvector/pgvector/releases
    echo 2. Extract to PostgreSQL installation directory
    echo 3. Or use: CREATE EXTENSION vector; in psql
    echo.
    pause
)

REM Create vector database
echo [3/6] Creating vector database...
psql -U postgres -c "CREATE DATABASE IF NOT EXISTS vector_db;" 2>nul
if %errorlevel% neq 0 (
    echo Creating vector_db database...
    psql -U postgres -c "CREATE DATABASE vector_db;"
)

REM Enable pgvector extension
echo [4/6] Enabling pgvector extension...
psql -U postgres -d vector_db -c "CREATE EXTENSION IF NOT EXISTS vector;"

REM Enable uuid extension
echo [5/6] Enabling UUID extension...
psql -U postgres -d vector_db -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

REM Create enterprise schema
echo [6/6] Creating enterprise vector schema...
psql -U postgres -d vector_db << EOF
-- Vector Consumer Service Schema
CREATE TABLE IF NOT EXISTS vector_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_vector_documents_embedding ON vector_documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_vector_documents_document_id ON vector_documents (document_id);
CREATE INDEX IF NOT EXISTS idx_vector_documents_metadata ON vector_documents USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_vector_documents_created_at ON vector_documents (created_at);

-- Create similarity search function
CREATE OR REPLACE FUNCTION find_similar_documents(
    query_embedding vector(768),
    similarity_threshold float8 DEFAULT 0.7,
    max_results integer DEFAULT 10
)
RETURNS TABLE(
    id UUID,
    document_id VARCHAR(255),
    content TEXT,
    metadata JSONB,
    similarity float8
)
LANGUAGE sql
STABLE
AS \$\$
    SELECT 
        d.id,
        d.document_id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity
    FROM vector_documents d
    WHERE 1 - (d.embedding <=> query_embedding) > similarity_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT max_results;
\$\$;

-- Create batch insert function
CREATE OR REPLACE FUNCTION insert_vector_document(
    p_document_id VARCHAR(255),
    p_content TEXT,
    p_metadata JSONB,
    p_embedding vector(768)
)
RETURNS UUID
LANGUAGE plpgsql
AS \$\$
DECLARE
    new_id UUID;
BEGIN
    INSERT INTO vector_documents (document_id, content, metadata, embedding)
    VALUES (p_document_id, p_content, p_metadata, p_embedding)
    ON CONFLICT (document_id) 
    DO UPDATE SET 
        content = EXCLUDED.content,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        updated_at = NOW()
    RETURNING id INTO new_id;
    
    RETURN new_id;
END;
\$\$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE vector_db TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Display configuration
SELECT 'PostgreSQL Configuration' as status;
SELECT version() as postgresql_version;
SELECT * FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp');
SELECT 'Schema created successfully' as result;
EOF

echo.
echo ===================================================
echo PostgreSQL + pgvector Setup Complete!
echo ===================================================
echo.
echo Database Information:
echo - Database: vector_db
echo - Host: localhost
echo - Port: 5432
echo - User: postgres
echo.
echo Tables Created:
echo - vector_documents (with embedding vector(768))
echo - Indexes: HNSW, GIN, B-tree for optimal performance
echo.
echo Functions Available:
echo - find_similar_documents(embedding, threshold, limit)
echo - insert_vector_document(id, content, metadata, embedding)
echo.
echo Connection String:
echo postgres://postgres:PASSWORD@localhost:5432/vector_db?sslmode=disable
echo.
echo Next: Run setup-redis.bat
pause