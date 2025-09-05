@echo off
REM =========================================================
REM EVIDENCE PROCESSING DATABASE SETUP SCRIPT
REM Smart Detection System Database Configuration
REM =========================================================

echo.
echo ========================================================
echo    EVIDENCE PROCESSING DATABASE SETUP
echo    Smart Detection System - Database Configuration
echo ========================================================
echo.

REM Set PostgreSQL environment
set PGPASSWORD=123456
set PGHOST=localhost
set PGPORT=5432
set PGUSER=postgres

echo [1/5] Testing PostgreSQL connection...
psql -c "SELECT version();" >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Cannot connect to PostgreSQL. Please ensure it's running.
    pause
    exit /b 1
)
echo [OK] PostgreSQL connection successful

echo.
echo [2/5] Creating evidence_processing database...
psql -c "CREATE DATABASE evidence_processing;" 2>nul
if %errorLevel% eq 0 (
    echo [OK] Database evidence_processing created
) else (
    echo [INFO] Database evidence_processing already exists or creation failed
)

echo.
echo [3/5] Enabling required extensions...
psql -d evidence_processing -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
psql -d evidence_processing -c "CREATE EXTENSION IF NOT EXISTS \"vector\";"
psql -d evidence_processing -c "CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";"
echo [OK] Extensions enabled

echo.
echo [4/5] Creating evidence processing tables...

REM Create evidence_cases table
psql -d evidence_processing -c "
CREATE TABLE IF NOT EXISTS evidence_cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'active',
    priority VARCHAR(10) DEFAULT 'medium',
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);"

REM Create evidence_items table
psql -d evidence_processing -c "
CREATE TABLE IF NOT EXISTS evidence_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID,
    evidence_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    file_hash VARCHAR(128),
    file_size BIGINT,
    mime_type VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    chain_of_custody JSONB DEFAULT '[]',
    extracted_text TEXT,
    embeddings vector(384),
    ocr_confidence DECIMAL(5,2),
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);"

REM Create evidence_processing_jobs table
psql -d evidence_processing -c "
CREATE TABLE IF NOT EXISTS evidence_processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evidence_id UUID,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    result JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);"

echo [OK] Evidence processing tables created

echo.
echo [5/5] Creating indexes and sample data...

REM Create indexes
psql -d evidence_processing -c "CREATE INDEX IF NOT EXISTS idx_evidence_cases_status ON evidence_cases(status);"
psql -d evidence_processing -c "CREATE INDEX IF NOT EXISTS idx_evidence_items_case_id ON evidence_items(case_id);"
psql -d evidence_processing -c "CREATE INDEX IF NOT EXISTS idx_evidence_items_status ON evidence_items(status);"

REM Insert sample data
psql -d evidence_processing -c "
INSERT INTO evidence_cases (case_number, title, description, priority, created_by) 
VALUES ('CASE-2025-001', 'Contract Liability Investigation', 'Investigation of contract terms and liability clauses', 'high', 'system_test')
ON CONFLICT (case_number) DO NOTHING;"

echo [OK] Database setup complete

echo.
echo ========================================================
echo    EVIDENCE PROCESSING DATABASE READY!
echo ========================================================
echo.
echo Database: evidence_processing
echo Tables: evidence_cases, evidence_items, evidence_processing_jobs
echo Sample Data: CASE-2025-001 (Contract Liability Investigation)
echo.
echo Available for testing:
echo - Smart detection algorithms
echo - Evidence processing pipeline
echo - Vector similarity search
echo - Chain of custody tracking
echo.
echo ========================================================
echo.

REM Test database connection
echo Testing database connectivity...
psql -d evidence_processing -c "SELECT COUNT(*) as total_cases FROM evidence_cases;" -t
if %errorLevel% eq 0 (
    echo [SUCCESS] Evidence processing database is operational!
) else (
    echo [WARNING] Database created but may need additional configuration
)

echo.
echo Press any key to continue...
pause >nul