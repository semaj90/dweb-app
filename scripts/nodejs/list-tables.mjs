import pg from 'pg';
const { Pool } = pg;

const pool = new Pool({
  connectionString: 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db'
});

try {
  const result = await pool.query('SELECT tablename FROM pg_tables WHERE schemaname = $1 ORDER BY tablename', ['public']);
  
  console.log('📊 Tables in legal_ai_db:');
  result.rows.forEach(row => console.log('  -', row.tablename));
  console.log(`\n✅ Total tables: ${result.rows.length}`);
  
} catch (error) {
  console.log('❌ Error:', error.message);
} finally {
  await pool.end();
}