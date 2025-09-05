// Fix password hash for admin user
import { Pool } from 'pg';

const pool = new Pool({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
});

async function fixPassword() {
  try {
    // The correct bcrypt hash for 'admin123'
    const correctHash = '$2b$12$LQv3c1yqBWVHxkd0LHAkCOEjrGn0YLs4VfcsKI8KN9uoEjZDVnhom';
    
    await pool.query(
      'UPDATE users SET hashed_password = $1 WHERE email = $2',
      [correctHash, 'admin@legal.ai']
    );
    
    // Also update test user
    await pool.query(
      'UPDATE users SET hashed_password = $1 WHERE email = $2', 
      [correctHash, 'test@legal.ai']
    );
    
    console.log('✅ Password hashes updated successfully');
    
    // Verify users
    const result = await pool.query('SELECT email, hashed_password FROM users WHERE email IN ($1, $2)', 
      ['admin@legal.ai', 'test@legal.ai']);
    console.log('📋 Updated users:', result.rows);
    
  } catch (error) {
    console.error('❌ Failed to fix passwords:', error);
  } finally {
    await pool.end();
  }
}

fixPassword();