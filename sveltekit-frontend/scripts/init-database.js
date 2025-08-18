// Database initialization script for PostgreSQL with vector extension
import postgres from 'postgres';

const connectionString = process.env.DATABASE_URL ||
  'postgresql://postgres:123456@localhost:5432/legal_ai_db';

const sql = postgres(connectionString, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10,
});

async function initializeDatabase() {
  try {
    console.log('Initializing database...');

    // Enable vector extension
    await sql`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log('✓ Vector extension enabled');

    // Create tables if they don't exist
    await sql`
      CREATE TABLE IF NOT EXISTS documents (
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
    console.log('✓ Documents table ready');

    await sql`
      CREATE TABLE IF NOT EXISTS legal_embeddings (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID REFERENCES documents(id),
        content TEXT NOT NULL,
        embedding VECTOR(768),
        metadata JSONB,
        model TEXT DEFAULT 'nomic-embed-text',
        created_at TIMESTAMP DEFAULT NOW()
      )
    `;
    console.log('✓ Embeddings table ready');

    await sql`
      CREATE TABLE IF NOT EXISTS search_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        query TEXT NOT NULL,
        query_embedding VECTOR(768),
        results JSONB,
        search_type TEXT DEFAULT 'hybrid',
        result_count INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
      )
    `;
    console.log('✓ Search sessions table ready');

    // Create indexes for better performance
    try {
      await sql`
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector
        ON legal_embeddings USING hnsw (embedding vector_cosine_ops)
      `;
      console.log('✓ Vector indexes created');
    } catch (indexError) {
      console.log('⚠ Vector index creation skipped (may require restart)');
    }

    await sql`
      CREATE INDEX IF NOT EXISTS idx_documents_content
      ON documents USING gin(to_tsvector('english', content))
    `;
    console.log('✓ Text search indexes created');

    console.log('✅ Database initialization complete!');

  } catch (error) {
    console.error('❌ Database initialization failed:', error);
    throw error;
  } finally {
    await sql.end();
  }
}

// Run initialization
initializeDatabase()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
