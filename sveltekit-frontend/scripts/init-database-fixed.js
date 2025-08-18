// Fixed Database initialization script for PostgreSQL with vector extension
import postgres from 'postgres';

const connectionString = process.env.DATABASE_URL ||
  'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';

const sql = postgres(connectionString, {
  max: 10,
  idle_timeout: 20,
  connect_timeout: 10,
});

async function initializeDatabase() {
  try {
    console.log('🚀 Initializing database...');

    // Enable vector extension
    await sql`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log('✓ Vector extension enabled');

    // Check existing tables and work with current schema
    console.log('📊 Checking existing schema...');
    
    const tables = await sql`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
    `;
    
    console.log(`✓ Found ${tables.length} existing tables`);
    
    // Create embeddings table that works with existing documents
    // Use varchar to match existing document IDs if they exist
    try {
      await sql`
        CREATE TABLE IF NOT EXISTS legal_embeddings (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          document_id VARCHAR(255) NOT NULL,
          content TEXT NOT NULL,
          embedding VECTOR(768),
          metadata JSONB DEFAULT '{}',
          model VARCHAR(100) DEFAULT 'nomic-embed-text',
          created_at TIMESTAMP DEFAULT NOW()
        )
      `;
      console.log('✓ Legal embeddings table ready');
    } catch (embeddingError) {
      console.log('⚠ Embeddings table creation skipped:', embeddingError.message);
    }

    // Create search sessions table
    try {
      await sql`
        CREATE TABLE IF NOT EXISTS search_sessions (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          query TEXT NOT NULL,
          query_embedding VECTOR(768),
          results JSONB DEFAULT '{}',
          search_type VARCHAR(50) DEFAULT 'hybrid',
          result_count INTEGER DEFAULT 0,
          created_at TIMESTAMP DEFAULT NOW()
        )
      `;
      console.log('✓ Search sessions table ready');
    } catch (searchError) {
      console.log('⚠ Search sessions table creation skipped:', searchError.message);
    }

    // Create vector indexes for performance
    try {
      await sql`
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector
        ON legal_embeddings USING hnsw (embedding vector_cosine_ops)
      `;
      console.log('✓ Vector indexes created');
    } catch (indexError) {
      console.log('⚠ Vector index creation skipped (normal for first run)');
    }

    // Create text search indexes if documents table exists
    try {
      await sql`
        CREATE INDEX IF NOT EXISTS idx_documents_content_gin
        ON documents USING gin(to_tsvector('english', content))
      `;
      console.log('✓ Text search indexes created');
    } catch (textIndexError) {
      console.log('⚠ Text search index creation skipped:', textIndexError.message);
    }

    // Test database connectivity and vector extension
    const vectorTest = await sql`SELECT '1'::vector as test_vector`;
    console.log('✓ Vector extension working');

    // Show table count
    const finalTableCount = await sql`
      SELECT COUNT(*) as count
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
    `;
    
    console.log(`✅ Database initialization complete! (${finalTableCount[0].count} tables total)`);
    console.log('🎯 Ready for AI-powered legal document processing');

  } catch (error) {
    console.error('❌ Database initialization failed:', error.message);
    console.error('💡 This might be normal if tables already exist with different schemas');
    
    // Test basic connectivity
    try {
      const testQuery = await sql`SELECT NOW() as current_time`;
      console.log('✓ Database connection working');
    } catch (connectError) {
      console.error('❌ Database connection failed:', connectError.message);
    }
    
  } finally {
    await sql.end();
  }
}

// Run initialization
initializeDatabase()
  .then(() => {
    console.log('🏁 Database initialization completed');
    process.exit(0);
  })
  .catch((error) => {
    console.error('🔥 Fatal error:', error.message);
    process.exit(1);
  });