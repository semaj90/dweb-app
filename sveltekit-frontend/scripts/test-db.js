import postgres from 'postgres';

const sql = postgres('postgresql://legal_admin:123456@localhost:5432/legal_ai_db');

try {
  console.log('🔍 Testing database connection...');
  
  const tables = await sql`
    SELECT COUNT(*) as count 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
  `;
  
  console.log(`✅ Database ready with ${tables[0].count} tables`);
  
  const vectorTest = await sql`SELECT 1 as test`;
  console.log('✅ Basic queries working');
  
} catch (error) {
  console.error('❌ Database test failed:', error.message);
} finally {
  await sql.end();
  console.log('🏁 Database test complete');
}