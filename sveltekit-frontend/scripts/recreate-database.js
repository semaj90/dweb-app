// Database cleanup and recreation script
import postgres from 'postgres';

const connectionString = process.env.DATABASE_URL ||
  'postgresql://postgres:123456@localhost:5432/legal_ai_db';

const sql = postgres(connectionString, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10,
});

async function recreateDatabase() {
  try {
    console.log('Recreating database tables...');

    // Drop existing tables that might have wrong schema
    await sql`DROP TABLE IF EXISTS legal_embeddings CASCADE`;
    await sql`DROP TABLE IF EXISTS search_sessions CASCADE`;
    await sql`DROP TABLE IF EXISTS documents CASCADE`;
    console.log('✓ Cleaned up old tables');

    // Enable vector extension
    await sql`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log('✓ Vector extension enabled');

    // Create documents table with proper UUID
    await sql`
      CREATE TABLE documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        filename TEXT NOT NULL,
        content TEXT NOT NULL,
        original_content TEXT,
        metadata JSONB,
        confidence REAL,
        legal_analysis JSONB,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
      )
    `;
    console.log('✓ Documents table created');

    // Create embeddings table
    await sql`
      CREATE TABLE legal_embeddings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        embedding VECTOR(768),
        metadata JSONB,
        model TEXT DEFAULT 'nomic-embed-text',
        created_at TIMESTAMP DEFAULT NOW()
      )
    `;
    console.log('✓ Embeddings table created');

    // Create search sessions table
    await sql`
      CREATE TABLE search_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        query TEXT NOT NULL,
        query_embedding VECTOR(768),
        results JSONB,
        search_type TEXT DEFAULT 'hybrid',
        result_count INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `;
    console.log('✓ Search sessions table created');

    // Create indexes for better performance
    try {
      await sql`
        CREATE INDEX idx_embeddings_vector
        ON legal_embeddings USING hnsw (embedding vector_cosine_ops)
      `;
      console.log('✓ Vector indexes created');
    } catch (indexError) {
      console.log('⚠ Vector index creation skipped (may require restart)');
    }

    await sql`
      CREATE INDEX idx_documents_content
      ON documents USING gin(to_tsvector('english', content))
    `;
    console.log('✓ Text search indexes created');

    console.log('✅ Database recreation complete!');

  } catch (error) {
    console.error('❌ Database recreation failed:', error);
    throw error;
  } finally {
    await sql.end();
  }
}

// Run recreation
recreateDatabase()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
