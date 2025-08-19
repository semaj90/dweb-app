// Test database connection
import pkg from 'pg';
const { Pool } = pkg;

const pool = new Pool({
  user: 'legal_admin',
  host: 'localhost',
  database: 'legal_ai_db',
  password: '123456',
  port: 5432,
});

async function testConnection() {
  try {
    const client = await pool.connect();
    console.log('✅ Connected to PostgreSQL');
    
    // Test basic query
    const result = await client.query('SELECT version()');
    console.log('📊 PostgreSQL version:', result.rows[0].version);
    
    // Check if legal_ai_db exists
    const dbCheck = await client.query(
      "SELECT 1 FROM pg_database WHERE datname = 'legal_ai_db'"
    );
    console.log('🗄️ legal_ai_db exists:', dbCheck.rows.length > 0);
    
    // Check if pgvector extension is available
    const extensionCheck = await client.query(
      "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
    );
    console.log('🔧 pgvector available:', extensionCheck.rows.length > 0);
    
    client.release();
  } catch (err) {
    console.error('❌ Database connection error:', err.message);
  } finally {
    await pool.end();
  }
}

testConnection();