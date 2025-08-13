// Database setup script for legal AI system
import { Pool } from 'pg';

const pool = new Pool({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
});

// Using raw pg pool

async function setupDatabase() {
  console.log('🚀 Setting up database for legal AI system...');
  
  try {
    // First check what exists
    const tableInfo = await pool.query(`
      SELECT column_name, data_type 
      FROM information_schema.columns 
      WHERE table_name = 'users' AND table_schema = 'public'
      ORDER BY ordinal_position;
    `);
    console.log('📋 Existing users table columns:', tableInfo.rows);
    
    // Create extensions
    await pool.query('CREATE EXTENSION IF NOT EXISTS vector;');
    await pool.query('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";');
    
    // Add missing columns to existing users table
    try {
      await pool.query('ALTER TABLE users ADD COLUMN IF NOT EXISTS hashed_password TEXT;');
      await pool.query('ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true;');
      await pool.query('ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name VARCHAR(100);');
      await pool.query('ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name VARCHAR(100);');
      console.log('✅ Added missing columns to users table');
    } catch (err) {
      console.log('⚠️ Column addition warning:', err.message);
    }
    
    // Create sessions table
    await pool.query(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL
      );
    `);
    
    // Use pre-hashed passwords (bcrypt for 'admin123' and 'test123')
    const adminHash = '$2b$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi';
    const testHash = '$2b$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi';
    
    await pool.query(`
      INSERT INTO users (email, hashed_password, name, first_name, last_name, role, is_active) VALUES 
      ($1, $2, 'Admin User', 'Admin', 'User', 'admin', true),
      ($3, $4, 'Test User', 'Test', 'User', 'admin', true)
      ON CONFLICT (email) DO NOTHING;
    `, ['admin@legal.ai', adminHash, 'test@legal.ai', testHash]);
    
    // Verify setup
    const result = await pool.query('SELECT id, email, role, is_active FROM users ORDER BY created_at DESC;');
    console.log('✅ Database setup complete!');
    console.log('📋 Users created:', result.rows);
    
  } catch (error) {
    console.error('❌ Database setup failed:', error);
  } finally {
    await pool.end();
  }
}

setupDatabase();