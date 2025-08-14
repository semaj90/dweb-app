import { Client } from 'pg';

const pgClient = new Client({
  connectionString: 'postgresql://postgres:123456@localhost:5432/legal_ai_db'
});

async function checkSchema() {
  try {
    await pgClient.connect();

    console.log('🔍 Checking legal_documents table schema...');

    // Check if table exists and get columns
    const result = await pgClient.query(`
      SELECT column_name, data_type, is_nullable
      FROM information_schema.columns
      WHERE table_name = 'legal_documents'
        AND table_schema = 'public'
      ORDER BY ordinal_position;
    `);

    if (result.rows.length === 0) {
      console.log('❌ Table legal_documents does not exist');
    } else {
      console.log('📋 Current table structure:');
      result.rows.forEach(row => {
        console.log(`   ${row.column_name}: ${row.data_type} (nullable: ${row.is_nullable})`);
      });
    }

    console.log('\n🔧 Creating/updating legal_documents table...');
    await pgClient.query(`
      DROP TABLE IF EXISTS legal_documents CASCADE;
      CREATE TABLE legal_documents (
        id SERIAL PRIMARY KEY,
        filename VARCHAR(255) NOT NULL,
        original_path TEXT,
        s3_bucket VARCHAR(100),
        s3_key TEXT,
        file_size BIGINT,
        mime_type VARCHAR(100),
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        document_type VARCHAR(50),
        title TEXT,
        content_preview TEXT,
        full_text TEXT,
        metadata JSONB,
        processing_status VARCHAR(20) DEFAULT 'uploaded',
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
    `);

    console.log('✅ Table created successfully');

  } catch (error) {
    console.error('❌ Schema check failed:', error.message);
  } finally {
    await pgClient.end();
  }
}

checkSchema();
