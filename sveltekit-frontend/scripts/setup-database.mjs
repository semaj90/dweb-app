#!/usr/bin/env node
// Database setup and initialization script for Legal AI System

import { db, initializeDatabase, testDatabaseConnection } from '../src/lib/server/database.js';
import postgres from 'postgres';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Database configuration
const config = {
  host: process.env.POSTGRES_HOST || 'localhost',
  port: parseInt(process.env.POSTGRES_PORT || '5432'),
  database: process.env.POSTGRES_DB || 'legal_ai',
  user: process.env.POSTGRES_USER || 'postgres',
  password: process.env.POSTGRES_PASSWORD || '123456'
};

console.log('🚀 Legal AI Database Setup');
console.log('============================');

async function setupDatabase() {
  let sql;
  
  try {
    // Connect to postgres database first to create our database
    console.log('📡 Connecting to PostgreSQL...');
    sql = postgres({
      ...config,
      database: 'postgres' // Connect to default postgres database first
    });

    // Test connection
    await sql`SELECT 1`;
    console.log('✅ Connected to PostgreSQL server');

    // Create database if it doesn't exist
    console.log(`📄 Creating database '${config.database}'...`);
    try {
      await sql`CREATE DATABASE ${sql(config.database)}`;
      console.log(`✅ Database '${config.database}' created`);
    } catch (error) {
      if (error.message.includes('already exists')) {
        console.log(`ℹ️  Database '${config.database}' already exists`);
      } else {
        throw error;
      }
    }

    await sql.end();

    // Connect to our specific database
    console.log(`📡 Connecting to '${config.database}' database...`);
    sql = postgres(config);

    // Install extensions
    console.log('🔧 Installing PostgreSQL extensions...');
    
    try {
      await sql`CREATE EXTENSION IF NOT EXISTS vector`;
      console.log('✅ pgvector extension installed');
    } catch (error) {
      console.error('❌ Failed to install pgvector extension:', error.message);
      console.log('ℹ️  Please install pgvector manually: https://github.com/pgvector/pgvector');
    }

    try {
      await sql`CREATE EXTENSION IF NOT EXISTS pg_trgm`;
      console.log('✅ pg_trgm extension installed');
    } catch (error) {
      console.warn('⚠️  pg_trgm extension not available:', error.message);
    }

    try {
      await sql`CREATE EXTENSION IF NOT EXISTS "uuid-ossp"`;
      console.log('✅ uuid-ossp extension installed');
    } catch (error) {
      console.warn('⚠️  uuid-ossp extension not available:', error.message);
    }

    // Create tables
    console.log('📋 Creating database tables...');

    // Documents table
    await sql`
      CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        original_content TEXT,
        metadata JSONB,
        confidence REAL,
        legal_analysis JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      )
    `;
    console.log('✅ Documents table created');

    // Legal embeddings table
    await sql`
      CREATE TABLE IF NOT EXISTS legal_embeddings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        embedding VECTOR(768),
        metadata JSONB,
        model TEXT DEFAULT 'nomic-embed-text',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      )
    `;
    console.log('✅ Legal embeddings table created');

    // Search sessions table
    await sql`
      CREATE TABLE IF NOT EXISTS search_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        query TEXT NOT NULL,
        query_embedding VECTOR(768),
        results JSONB,
        search_type TEXT DEFAULT 'hybrid',
        result_count INTEGER DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      )
    `;
    console.log('✅ Search sessions table created');

    // Create indexes
    console.log('🔍 Creating database indexes...');

    // Vector similarity index (requires pgvector)
    try {
      await sql`
        CREATE INDEX IF NOT EXISTS legal_embeddings_embedding_idx 
        ON legal_embeddings USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 100)
      `;
      console.log('✅ Vector similarity index created');
    } catch (error) {
      console.warn('⚠️  Could not create vector index (pgvector may not be installed)');
    }

    // Full-text search index
    try {
      await sql`
        CREATE INDEX IF NOT EXISTS documents_content_fts_idx 
        ON documents USING gin(to_tsvector('english', content))
      `;
      console.log('✅ Full-text search index created');
    } catch (error) {
      console.warn('⚠️  Could not create full-text index:', error.message);
    }

    // Metadata indexes
    await sql`CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents USING gin(metadata)`;
    await sql`CREATE INDEX IF NOT EXISTS documents_created_at_idx ON documents(created_at)`;
    await sql`CREATE INDEX IF NOT EXISTS embeddings_document_id_idx ON legal_embeddings(document_id)`;
    console.log('✅ Metadata indexes created');

    // Insert sample data
    console.log('📝 Inserting sample legal documents...');
    
    const sampleDocuments = [
      {
        filename: 'sample_contract.txt',
        content: 'This is a sample legal contract between Party A and Party B. The contract outlines the terms and conditions for the provision of legal services. Party A agrees to provide legal consultation services to Party B in exchange for compensation as outlined in Schedule A.',
        metadata: { documentType: 'contract', jurisdiction: 'federal', sample: true }
      },
      {
        filename: 'property_deed.txt', 
        content: 'Property Deed for 123 Main Street. This deed transfers ownership of the property located at 123 Main Street from John Doe to Jane Smith. The transfer is subject to all existing liens and encumbrances.',
        metadata: { documentType: 'deed', jurisdiction: 'state', sample: true }
      },
      {
        filename: 'legal_memo.txt',
        content: 'Legal Memorandum regarding negligence in tort law. This memo outlines the four elements of negligence: duty, breach, causation, and damages. Each element must be proven for a successful negligence claim.',
        metadata: { documentType: 'memo', jurisdiction: 'general', sample: true }
      }
    ];

    for (const doc of sampleDocuments) {
      try {
        const result = await sql`
          INSERT INTO documents (filename, content, metadata)
          VALUES (${doc.filename}, ${doc.content}, ${JSON.stringify(doc.metadata)})
          ON CONFLICT DO NOTHING
          RETURNING id
        `;
        
        if (result.length > 0) {
          console.log(`✅ Sample document inserted: ${doc.filename}`);
        }
      } catch (error) {
        console.warn(`⚠️  Could not insert sample document ${doc.filename}:`, error.message);
      }
    }

    // Database statistics
    console.log('\n📊 Database Statistics:');
    const docCount = await sql`SELECT COUNT(*) as count FROM documents`;
    console.log(`📄 Documents: ${docCount[0].count}`);
    
    const embCount = await sql`SELECT COUNT(*) as count FROM legal_embeddings`;
    console.log(`🧠 Embeddings: ${embCount[0].count}`);

    await sql.end();

    console.log('\n🎉 Database setup completed successfully!');
    console.log('\n📋 Next steps:');
    console.log('1. Start Redis: redis-server');
    console.log('2. Start Ollama: ollama serve');
    console.log('3. Pull embedding model: ollama pull nomic-embed-text');
    console.log('4. Start the application: npm run dev');
    console.log('\n🔗 Test the system at: http://localhost:5173/ai-upload-demo');

  } catch (error) {
    console.error('❌ Database setup failed:', error);
    process.exit(1);
  }
}

// Run the setup
setupDatabase().catch(console.error);
