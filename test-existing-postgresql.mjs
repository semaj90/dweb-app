import pg from 'pg';

const connectionStrings = [
    'postgresql://postgres:password@localhost:5432/postgres',
    'postgresql://postgres:postgres@localhost:5432/postgres',
    'postgresql://postgres@localhost:5432/postgres'
];

async function testConnection(connString) {
    const client = new pg.Client(connString);
    try {
        console.log(`Testing: ${connString}`);
        await client.connect();
        
        // Test basic functionality
        const result = await client.query('SELECT version() as version');
        console.log('✅ Connection successful!');
        console.log('📊 PostgreSQL version:', result.rows[0].version.split(' ')[0] + ' ' + result.rows[0].version.split(' ')[1]);
        
        // Test if pgvector is available
        try {
            await client.query("CREATE EXTENSION IF NOT EXISTS vector");
            console.log('✅ pgvector extension available or enabled');
        } catch (err) {
            console.log('⚠️ pgvector extension not available:', err.message);
        }
        
        // Create database if needed
        try {
            await client.query("CREATE DATABASE legal_ai_db");
            console.log('✅ Created legal_ai_db database');
        } catch (err) {
            if (err.message.includes('already exists')) {
                console.log('ℹ️ legal_ai_db database already exists');
            } else {
                console.log('⚠️ Could not create database:', err.message);
            }
        }
        
        await client.end();
        return true;
        
    } catch (error) {
        console.log('❌ Connection failed:', error.message);
        await client.end().catch(() => {});
        return false;
    }
}

// Test each connection string
for (const connString of connectionStrings) {
    const success = await testConnection(connString);
    if (success) {
        console.log('\n🎉 PostgreSQL is ready for Enhanced RAG System!');
        console.log('📋 Use this connection string in your .env file:');
        console.log(`DATABASE_URL=${connString.replace('/postgres', '/legal_ai_db')}`);
        process.exit(0);
    }
    console.log('');
}

console.log('❌ Could not connect to PostgreSQL with any common credentials');
console.log('💡 Please check your PostgreSQL installation and credentials');
process.exit(1);