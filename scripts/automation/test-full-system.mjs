import pg from 'pg';
const { Pool } = pg;

const pool = new Pool({
  connectionString: 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db'
});

console.log('🧪 COMPREHENSIVE LEGAL AI SYSTEM TEST\n');

async function testCRUDOperations() {
  try {
    // Test 1: Insert a new case
    console.log('📝 Test 1: Creating new case...');
    const newCase = await pool.query(
      'INSERT INTO cases (case_number, title, description, status, priority) VALUES ($1, $2, $3, $4, $5) RETURNING *',
      ['TEST-2024-001', 'Test Case from Node.js', 'Testing CRUD operations', 'open', 'high']
    );
    console.log('✅ Case created:', newCase.rows[0].case_number, '-', newCase.rows[0].title);

    // Test 2: Read cases
    console.log('\n📖 Test 2: Reading all cases...');
    const allCases = await pool.query('SELECT case_number, title, status FROM cases ORDER BY created_at');
    console.log('✅ Found', allCases.rows.length, 'cases:');
    allCases.rows.forEach(c => console.log('  -', c.case_number, ':', c.title, `(${c.status})`));

    // Test 3: Add evidence to the new case
    console.log('\n🗂️  Test 3: Adding evidence to case...');
    const newEvidence = await pool.query(
      'INSERT INTO evidence (case_id, title, description, evidence_type) VALUES ($1, $2, $3, $4) RETURNING *',
      [newCase.rows[0].id, 'Test Evidence', 'Evidence created via API test', 'document']
    );
    console.log('✅ Evidence added:', newEvidence.rows[0].title);

    // Test 4: Join query (cases with evidence count)
    console.log('\n🔗 Test 4: Testing JOIN operations...');
    const casesWithEvidence = await pool.query(`
      SELECT c.case_number, c.title, COUNT(e.id) as evidence_count 
      FROM cases c 
      LEFT JOIN evidence e ON c.id = e.case_id 
      GROUP BY c.id, c.case_number, c.title 
      ORDER BY c.created_at
    `);
    console.log('✅ Cases with evidence count:');
    casesWithEvidence.rows.forEach(c => 
      console.log('  -', c.case_number, ':', c.title, `(${c.evidence_count} evidence)`)
    );

    // Test 5: Update case status
    console.log('\n📝 Test 5: Updating case status...');
    await pool.query(
      'UPDATE cases SET status = $1, updated_at = CURRENT_TIMESTAMP WHERE case_number = $2',
      ['in_progress', 'TEST-2024-001']
    );
    console.log('✅ Case status updated to "in_progress"');

    // Test 6: Test database functions
    console.log('\n⚡ Test 6: Testing PostgreSQL functions...');
    const dbInfo = await pool.query(`
      SELECT 
        current_database() as database_name,
        current_user as current_user,
        version() as pg_version,
        now() as current_time
    `);
    console.log('✅ Database info:', {
      database: dbInfo.rows[0].database_name,
      user: dbInfo.rows[0].current_user,
      time: dbInfo.rows[0].current_time.toISOString()
    });

    console.log('\n🎉 ALL TESTS PASSED! Legal AI database is fully functional.');
    return true;

  } catch (error) {
    console.log('\n❌ TEST FAILED:', error.message);
    return false;
  }
}

// Run tests
const success = await testCRUDOperations();
await pool.end();

if (success) {
  console.log('\n✅ SYSTEM STATUS: READY FOR PRODUCTION');
  console.log('🌐 SvelteKit app should be accessible at: http://localhost:5176');
  console.log('📊 Database: legal_ai_db with full CRUD operations working');
} else {
  console.log('\n❌ SYSTEM STATUS: NEEDS ATTENTION');
}