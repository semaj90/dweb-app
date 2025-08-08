// Fixed Database Connection Test
const { Client } = require('pg');

async function testConnection() {
    console.log('🔍 Testing PostgreSQL Connection (pg library)...\n');
    
    // Configuration with password as string
    const config = {
        host: 'localhost',
        port: 5432,
        database: 'legal_ai_db',
        user: 'legal_admin',
        password: '123456', // Ensure this is a string
        connectionTimeoutMillis: 5000
    };
    
    const client = new Client(config);
    
    try {
        console.log('Connecting to PostgreSQL...');
        await client.connect();
        
        console.log('✅ Connected successfully!\n');
        
        // Test queries
        const versionResult = await client.query('SELECT version()');
        console.log('PostgreSQL version:', versionResult.rows[0].version.split(',')[0]);
        
        const tablesResult = await client.query(`
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            LIMIT 5
        `);
        
        console.log('\nExisting tables:');
        tablesResult.rows.forEach(row => {
            console.log(`  - ${row.tablename}`);
        });
        
        // Test vector extension
        const vectorResult = await client.query(`
            SELECT extname 
            FROM pg_extension 
            WHERE extname = 'vector'
        `);
        
        if (vectorResult.rows.length > 0) {
            console.log('\n✅ Vector extension is installed');
        } else {
            console.log('\n⚠️  Vector extension not found');
        }
        
        await client.end();
        console.log('\n✅ Connection test successful!');
        
    } catch (error) {
        console.error('❌ Connection failed:', error.message);
        
        if (error.code === '28P01') {
            console.log('\n📝 Authentication error. To fix:');
            console.log('   1. Run FIX-POSTGRES-ADMIN.bat as Administrator');
            console.log('   2. Or reset password manually in PostgreSQL');
        } else if (error.code === 'ECONNREFUSED') {
            console.log('\n📝 Connection refused. Check if PostgreSQL is running:');
            console.log('   Get-Service postgresql*');
            console.log('   Start-Service postgresql-x64-17');
        }
        
        process.exit(1);
    }
}

testConnection();
