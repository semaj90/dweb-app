// Create demo users with proper bcrypt hashes
const bcrypt = require('bcryptjs');
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
});

async function createDemoUsers() {
  console.log('🔧 Creating demo users with proper bcrypt hashes...');
  
  try {
    // Generate proper bcrypt hashes
    const adminHash = await bcrypt.hash('admin123', 12);
    const testHash = await bcrypt.hash('test123', 12);
    
    console.log('🔑 Generated hashes for passwords:');
    console.log('admin123 hash length:', adminHash.length);
    console.log('test123 hash length:', testHash.length);
    
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
    const storedHash = await pool.query('SELECT hashed_password FROM users WHERE email = $1', ['admin@legal.ai']);
    const isValidPassword = await bcrypt.compare('admin123', storedHash.rows[0].hashed_password);
    
    console.log('🔐 Password verification test:', isValidPassword ? 'PASS ✅' : 'FAIL ❌');
    
    if (isValidPassword) {
      console.log('🎉 Demo users created successfully! You can now login with:');
      console.log('   admin@legal.ai / admin123');
      console.log('   test@legal.ai / test123');
    }
    
  } catch (error) {
    console.error('❌ Failed to create demo users:', error);
  } finally {
    await pool.end();
  }
}

createDemoUsers();