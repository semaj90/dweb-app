// Fix login system completely
import { Pool } from 'pg';
import bcrypt from 'bcrypt';

const pool = new Pool({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
});

async function fixLoginSystem() {
  console.log('🔧 Fixing login system completely...');
  
  try {
    // Generate proper bcrypt hashes
    const adminHash = await bcrypt.hash('admin123', 12);
    const testHash = await bcrypt.hash('test123', 12);
    
    console.log('🔑 Generated hashes:');
    console.log('Admin hash:', adminHash);
    console.log('Test hash:', testHash);
    
    // Clear existing users and sessions
    await pool.query('DELETE FROM sessions');
    await pool.query('DELETE FROM users');
    
    console.log('🗑️ Cleared existing users and sessions');
    
    // Insert users with proper hashes
    const adminResult = await pool.query(`
      INSERT INTO users (email, hashed_password, name, first_name, last_name, role, is_active) 
      VALUES ($1, $2, $3, $4, $5, $6, $7) 
      RETURNING id, email, role
    `, ['admin@legal.ai', adminHash, 'Admin User', 'Admin', 'User', 'admin', true]);
    
    const testResult = await pool.query(`
      INSERT INTO users (email, hashed_password, name, first_name, last_name, role, is_active) 
      VALUES ($1, $2, $3, $4, $5, $6, $7) 
      RETURNING id, email, role
    `, ['test@legal.ai', testHash, 'Test User', 'Test', 'User', 'user', true]);
    
    console.log('✅ Created users:');
    console.log('Admin:', adminResult.rows[0]);
    console.log('Test:', testResult.rows[0]);
    
    // Test password verification
    const adminUser = adminResult.rows[0];
    const storedHash = await pool.query('SELECT hashed_password FROM users WHERE email = $1', ['admin@legal.ai']);
    const isValidPassword = await bcrypt.compare('admin123', storedHash.rows[0].hashed_password);
    
    console.log('🔐 Password verification test:', isValidPassword ? 'PASS ✅' : 'FAIL ❌');
    
    // Verify table structure
    const columns = await pool.query(`
      SELECT column_name, data_type, is_nullable 
      FROM information_schema.columns 
      WHERE table_name = 'users' AND table_schema = 'public'
      ORDER BY ordinal_position
    `);
    
    console.log('📋 Users table structure:');
    columns.rows.forEach(col => {
      console.log(`  ${col.column_name}: ${col.data_type} (nullable: ${col.is_nullable})`);
    });
    
  } catch (error) {
    console.error('❌ Failed to fix login system:', error);
  } finally {
    await pool.end();
  }
}

fixLoginSystem();