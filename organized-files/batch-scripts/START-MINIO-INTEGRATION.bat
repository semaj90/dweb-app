@echo off
echo ========================================
echo MinIO Integration Setup and Startup
echo ========================================

REM Set environment variables
set MINIO_ENDPOINT=localhost:9000
set MINIO_ACCESS_KEY=minioadmin
set MINIO_SECRET_KEY=minioadmin
set MINIO_BUCKET=legal-documents
set DATABASE_URL=postgresql://postgres:password@localhost:5432/deeds_web_app?sslmode=disable
set QDRANT_URL=http://localhost:6333
set RAG_SERVICE_URL=http://localhost:8092
set UPLOAD_SERVICE_PORT=8093

echo.
echo [1/7] Fixing Go module imports...
powershell -ExecutionPolicy Bypass -File fix-minio-imports.ps1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to fix imports
    pause
    exit /b 1
)

echo.
echo [2/7] Starting PostgreSQL...
net start postgresql-x64-14 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo PostgreSQL might already be running or not installed
)

echo.
echo [3/7] Creating database schema...
echo CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; > temp_schema.sql
echo CREATE EXTENSION IF NOT EXISTS vector; >> temp_schema.sql
echo. >> temp_schema.sql
echo CREATE TABLE IF NOT EXISTS document_metadata ( >> temp_schema.sql
echo     id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), >> temp_schema.sql
echo     case_id VARCHAR(255) NOT NULL, >> temp_schema.sql
echo     filename VARCHAR(500) NOT NULL, >> temp_schema.sql
echo     object_name VARCHAR(1000) NOT NULL UNIQUE, >> temp_schema.sql
echo     content_type VARCHAR(100), >> temp_schema.sql
echo     size_bytes BIGINT, >> temp_schema.sql
echo     upload_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(), >> temp_schema.sql
echo     document_type VARCHAR(100), >> temp_schema.sql
echo     tags JSONB, >> temp_schema.sql
echo     metadata JSONB, >> temp_schema.sql
echo     processing_status VARCHAR(50) DEFAULT 'uploaded', >> temp_schema.sql
echo     embedding vector(384), >> temp_schema.sql
echo     extracted_text TEXT, >> temp_schema.sql
echo     qdrant_point_id UUID, >> temp_schema.sql
echo     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(), >> temp_schema.sql
echo     updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() >> temp_schema.sql
echo ); >> temp_schema.sql

psql -U postgres -d deeds_web_app -f temp_schema.sql 2>nul
del temp_schema.sql

echo.
echo [4/7] Starting MinIO...
if not exist "C:\minio-data" mkdir "C:\minio-data"
start /B minio.exe server C:\minio-data --console-address :9001 >nul 2>&1
timeout /t 3 /nobreak >nul

echo.
echo [5/7] Starting Qdrant...
start /B qdrant.exe >nul 2>&1
timeout /t 3 /nobreak >nul

echo.
echo [6/7] Building Go services...
cd go-microservice
go build -o bin\upload-service.exe cmd\upload-service\main.go 2>nul
go build -o bin\summarizer-service.exe cmd\summarizer-service\main.go 2>nul
cd ..

echo.
echo [7/7] Starting services...
start /B go-microservice\bin\upload-service.exe >nul 2>&1
start /B go-microservice\bin\summarizer-service.exe >nul 2>&1

echo.
echo ========================================
echo MinIO Integration Started Successfully!
echo ========================================
echo.
echo Services running at:
echo - MinIO:          http://localhost:9000 (Console: http://localhost:9001)
echo - PostgreSQL:     localhost:5432
echo - Qdrant:         http://localhost:6333
echo - Upload Service: http://localhost:8093
echo - RAG Service:    http://localhost:8092
echo.
echo Test upload with:
echo curl -X POST http://localhost:8093/upload -F "file=@test.pdf" -F "caseId=CASE-001" -F "documentType=evidence"
echo.
pause
