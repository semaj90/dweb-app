#!/usr/bin/env node

import pg from 'pg';

console.log('🔌 Testing YoRHa Database Connection...');

const client = new pg.Client({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
});

async function testConnection() {
  try {
    console.log('⏳ Connecting to PostgreSQL...');
    await client.connect();
    console.log('✅ Connected successfully!');

    const result = await client.query('SELECT version(), current_database(), current_user;');
    console.log('📊 Database Info:');
    console.log(`   Version: ${result.rows[0].version.split(' ')[0]}`);
    console.log(`   Database: ${result.rows[0].current_database}`);
    console.log(`   User: ${result.rows[0].current_user}`);

    // Test table creation
    console.log('🔨 Testing table creation...');
    await client.query(`
      CREATE TABLE IF NOT EXISTS yorha_test (
        id SERIAL PRIMARY KEY,
        message TEXT,
        created_at TIMESTAMP DEFAULT NOW()
      );
    `);

    await client.query(`
      INSERT INTO yorha_test (message) VALUES ('YoRHa database connection test successful');
    `);

    const testResult = await client.query('SELECT * FROM yorha_test ORDER BY id DESC LIMIT 1;');
    console.log('✅ Test record:', testResult.rows[0]);

    // Clean up test table
    await client.query('DROP TABLE IF EXISTS yorha_test;');
    console.log('🧹 Cleaned up test table');

    console.log('🎯 Database connection test PASSED!');

  } catch (error) {
    console.error('❌ Database connection test FAILED:', error.message);
    if (error.code === 'ECONNREFUSED') {
      console.error('💡 PostgreSQL server is not running or not accessible');
    }
    if (error.code === '28P01') {
      console.error('💡 Authentication failed - check username/password');
    }
    if (error.code === '3D000') {
      console.error('💡 Database "legal_ai_db" does not exist');
    }
  } finally {
    await client.end();
    console.log('🔌 Connection closed');
  }
}

testConnection();