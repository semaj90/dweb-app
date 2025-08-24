-- Quick database setup for testing
-- Run with: psql -h localhost -p 5432 -U postgres -f quick-db-setup.sql

-- Create database if not exists
SELECT 'CREATE DATABASE legal_ai_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'legal_ai_db')\gexec

-- Connect to the database
\c legal_ai_db;

-- Create user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'legal_admin') THEN
        CREATE USER legal_admin WITH PASSWORD '123456';
    END IF;
END
$$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;
GRANT ALL ON SCHEMA public TO legal_admin;

-- Create pgvector extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    name TEXT NOT NULL,
    first_name TEXT DEFAULT '',
    last_name TEXT DEFAULT '',
    role TEXT DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    avatar_url TEXT,
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create lucia sessions table
CREATE TABLE IF NOT EXISTS user_session (
    id TEXT PRIMARY KEY,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ip_address TEXT,
    user_agent TEXT,
    device_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert demo user (using bcrypt hash for demo123456)
INSERT INTO users (
    id, 
    email, 
    hashed_password, 
    name, 
    first_name, 
    last_name, 
    role, 
    is_active, 
    email_verified,
    created_at,
    updated_at
) VALUES (
    'demo-user-001',
    'demo@legalai.gov',
    '$2a$10$rqZl.uP8xO/FK.NE8zjTkOGKGbGdQuW.6HI9j2B8QC/7rlHxu5/gW', -- bcrypt hash of 'demo123456'
    'Demo Prosecutor',
    'Demo',
    'Prosecutor',
    'prosecutor',
    true,
    true,
    NOW(),
    NOW()
) ON CONFLICT (email) DO UPDATE SET 
    hashed_password = EXCLUDED.hashed_password,
    updated_at = NOW();

-- Verify setup
SELECT 'Database setup completed!' as message;
SELECT 'Demo user: ' || email || ' (ID: ' || id || ')' as demo_user FROM users WHERE email = 'demo@legalai.gov';