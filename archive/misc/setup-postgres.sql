-- PostgreSQL setup script for Legal AI application
-- Run this script as postgres user

-- Create database
CREATE DATABASE legal_ai_db;

-- Create user
CREATE USER legal_admin WITH ENCRYPTED PASSWORD 'LegalAI2024!';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;

-- Connect to the new database
\c legal_ai_db;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO legal_admin;
GRANT CREATE ON SCHEMA public TO legal_admin;