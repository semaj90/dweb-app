-- Setup PostgreSQL database for Legal AI application
-- Make sure to run this as postgres superuser

-- Create database
CREATE DATABASE legal_ai_db;

-- Create user with password
CREATE USER legal_admin WITH ENCRYPTED PASSWORD 'LegalAI2024!';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;

-- Connect to the legal_ai_db database
\c legal_ai_db;

-- Create uuid extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Try to create vector extension (pgvector)
-- This will fail if pgvector is not installed
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO legal_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO legal_admin;

-- Alternative grant for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO legal_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO legal_admin;