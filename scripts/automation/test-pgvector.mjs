import pg from 'pg';
const { Pool } = pg;

const pool = new Pool({
  connectionString: 'postgresql://postgres:123456@localhost:5432/legal_ai_db'
});

console.log('🧪 Testing pgvector Installation...\n');

try {
  // Check if extension is installed
  const extResult = await pool.query("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'");
  
  if (extResult.rows.length > 0) {
    console.log(`✅ pgvector v${extResult.rows[0].extversion} is installed!`);
    
    // Test vector creation
    const vectorTest = await pool.query("SELECT '[1,2,3]'::vector as test_vector");
    console.log('✅ Vector creation test:', vectorTest.rows[0].test_vector);
    
    // Test vector operations
    const distanceTest = await pool.query("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as distance");
    console.log('✅ Vector distance test:', distanceTest.rows[0].distance);
    
    console.log('\n🎉 pgvector is fully functional!');
    
  } else {
    console.log('❌ pgvector extension not found');
  }
  
} catch (error) {
  console.log('❌ pgvector test failed:', error.message);
} finally {
  await pool.end();
}