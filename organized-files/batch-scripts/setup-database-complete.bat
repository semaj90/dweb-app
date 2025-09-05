@echo off
REM =============================================
REM COMPLETE DATABASE SETUP SCRIPT
REM Sets up PostgreSQL with proper authentication
REM =============================================

cls
echo ================================================
echo   LEGAL AI DATABASE SETUP
echo ================================================
echo.

REM Set PostgreSQL password environment variable
set PGPASSWORD=123456

echo [1/5] Testing PostgreSQL connection...
psql -U postgres -h localhost -p 5432 -c "SELECT version();" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo       [OK] PostgreSQL is running
) else (
    echo       [ERROR] PostgreSQL connection failed
    echo       Please ensure PostgreSQL is running and password is correct
    pause
    exit /b 1
)

echo.
echo [2/5] Creating legal_ai database if not exists...
psql -U postgres -h localhost -p 5432 -c "CREATE DATABASE legal_ai_db;" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo       [OK] Database created
) else (
    echo       [INFO] Database already exists or error occurred
)

echo.
echo [3/5] Installing pgvector extension...
psql -U postgres -h localhost -p 5432 -d legal_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo       [OK] pgvector extension installed
) else (
    echo       [ERROR] Failed to install pgvector extension
)

echo.
echo [4/5] Running database migrations...
if exist "production-migration.sql" (
    psql -U postgres -h localhost -p 5432 -d legal_ai_db -f production-migration.sql
    if %ERRORLEVEL% EQU 0 (
        echo       [OK] Migrations completed successfully
    ) else (
        echo       [ERROR] Migration failed
    )
) else (
    echo       [INFO] No migration file found, creating basic schema...
    psql -U postgres -h localhost -p 5432 -d legal_ai_db -c "
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS cases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR(255) NOT NULL,
            description TEXT,
            status VARCHAR(50) DEFAULT 'draft',
            priority VARCHAR(20) DEFAULT 'medium',
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS evidence (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL,
            file_type VARCHAR(20) NOT NULL,
            file_path TEXT,
            file_size INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            embedding vector(384)
        );
        
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID REFERENCES evidence(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(384),
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_cases_user_id ON cases(user_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_embedding ON evidence USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
    "
    echo       [OK] Basic schema created
)

echo.
echo [5/5] Verifying database setup...
psql -U postgres -h localhost -p 5432 -d legal_ai_db -c "
    SELECT 
        schemaname,
        tablename 
    FROM pg_tables 
    WHERE schemaname = 'public'
    ORDER BY tablename;
"

echo.
echo ================================================
echo   DATABASE SETUP COMPLETE
echo ================================================
echo.
echo Database: legal_ai_db
echo Host: localhost:5432
echo User: postgres
echo.
echo Connection string: 
echo postgresql://postgres:123456@localhost:5432/legal_ai_db
echo.
echo ✅ Database ready for legal AI application
echo.
pause