#!/usr/bin/env node

import pkg from 'pg';
const { Pool } = pkg;

const dbConfigs = [
  'postgresql://postgres:123456@localhost:5432/legal_ai_db',
  'postgresql://postgres:postgres@localhost:5432/legal_ai_db',
  'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
];

async function testConnection() {
  console.log('🔍 Testing database connections...');
  
  for (const config of dbConfigs) {
    try {
      console.log(`\n📡 Testing: ${config.replace(/:[^:@]*@/, ':****@')}`);
      
      const pool = new Pool({ 
        connectionString: config,
        connectionTimeoutMillis: 5000,
        idleTimeoutMillis: 5000
      });
      
      const result = await pool.query('SELECT version() as version, current_database() as database;');
      console.log('✅ Connected successfully!');
      console.log('📊 Database:', result.rows[0].database);
      console.log('🗂️  Version:', result.rows[0].version.split(' ')[0]);
      
      // Test if users table exists
      const tableCheck = await pool.query(`
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'users';
      `);
      
      if (tableCheck.rows.length > 0) {
        console.log('✅ Users table exists');
        
        // Count existing users
        const userCount = await pool.query('SELECT COUNT(*) as count FROM users;');
        console.log(`👥 Existing users: ${userCount.rows[0].count}`);
      } else {
        console.log('⚠️  Users table not found - run migrations first');
      }
      
      await pool.end();
      return config;
    } catch (error) {
      console.log(`❌ Failed: ${error.message}`);
    }
  }
  
  throw new Error('No database connection succeeded');
}

testConnection()
  .then((config) => {
    console.log('\n🎉 Database test completed successfully!');
    console.log(`✅ Working connection: ${config.replace(/:[^:@]*@/, ':****@')}`);
  })
  .catch((error) => {
    console.error('\n💥 All database connections failed:', error.message);
    process.exit(1);
  });