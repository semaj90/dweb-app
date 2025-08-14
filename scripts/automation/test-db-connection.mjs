import pg from 'pg';
const { Pool } = pg;

const pool = new Pool({
  connectionString: 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db'
});

console.log('🧪 Testing Legal AI Database Connection...\n');

try {
  // Test basic connection
  const version = await pool.query('SELECT version()');
  console.log('✅ Database connection successful');
  console.log('📊 PostgreSQL version:', version.rows[0].version.slice(0, 50) + '...\n');
  
  // Test users table
  const users = await pool.query('SELECT COUNT(*) as count FROM users');
  console.log(`👥 Users table: ${users.rows[0].count} records`);
  
  // Test cases table
  const cases = await pool.query('SELECT COUNT(*) as count FROM cases');
  console.log(`📁 Cases table: ${cases.rows[0].count} records`);
  
  // Test evidence table
  const evidence = await pool.query('SELECT COUNT(*) as count FROM evidence');
  console.log(`🗂️  Evidence table: ${evidence.rows[0].count} records`);
  
  console.log('\n🎉 All database tests passed! Your Legal AI system is ready.');
  
} catch (error) {
  console.log('❌ Database connection failed:', error.message);
} finally {
  await pool.end();
}