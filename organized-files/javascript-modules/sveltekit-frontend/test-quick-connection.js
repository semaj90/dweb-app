// Quick PostgreSQL Connection Test
const PASSWORD = '123456';  // Ensure this is a string

// Test with both libraries
async function testWithPG() {
    const { Client } = require('pg');
    
    const client = new Client({
        host: 'localhost',
        port: 5432,
        database: 'legal_ai_db',
        user: 'legal_admin',
        password: PASSWORD  // Must be a string
    });
    
    try {
        await client.connect();
        const res = await client.query('SELECT NOW()');
        console.log('✅ PG library works:', res.rows[0].now);
        await client.end();
        return true;
    } catch (err) {
        console.error('❌ PG library failed:', err.message);
        return false;
    }
}

async function testWithPostgres() {
    try {
        const postgres = await import('postgres');
        const sql = postgres.default(`postgresql://legal_admin:${PASSWORD}@localhost:5432/legal_ai_db`);
        
        const result = await sql`SELECT NOW() as time`;
        console.log('✅ Postgres.js works:', result[0].time);
        await sql.end();
        return true;
    } catch (err) {
        console.error('❌ Postgres.js failed:', err.message);
        return false;
    }
}

async function runTests() {
    console.log('🔍 Testing PostgreSQL Connections\n');
    console.log('Password type:', typeof PASSWORD);
    console.log('Password value:', PASSWORD);
    console.log('-'.repeat(40));
    
    const pgWorks = await testWithPG();
    const postgresWorks = await testWithPostgres();
    
    console.log('-'.repeat(40));
    if (pgWorks || postgresWorks) {
        console.log('✅ At least one connection method works!');
    } else {
        console.log('❌ Both connection methods failed');
        console.log('\nTroubleshooting:');
        console.log('1. Check PostgreSQL is running: Get-Service postgresql*');
        console.log('2. Verify password: Run FIX-POSTGRES-ADMIN.bat');
        console.log('3. Check database exists: psql -U postgres -l');
    }
}

runTests();
