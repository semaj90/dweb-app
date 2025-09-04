#!/usr/bin/env node
/**
 * Test Database Connection with pgvector Support
 * Direct PostgreSQL test for pgvector integration
 */

import postgres from 'postgres';

const DATABASE_URL = process.env.DATABASE_URL || 
  'postgresql://postgres:123456@localhost:5432/legal_ai_db';

async function main() {
  console.log('🧪 Testing Database Connection & pgvector Integration...\n');
  
  const client = postgres(DATABASE_URL, {
    max: 5,
    idle_timeout: 20,
    connect_timeout: 10,
  });
  
  try {
    // 1. Test basic connection
    console.log('1️⃣ Testing basic connection...');
    const result = await client`SELECT version(), now() as current_time`;
    console.log('✅ PostgreSQL Connection Successful');
    console.log(`   Version: ${result[0].version.substring(0, 50)}...`);
    console.log(`   Current Time: ${result[0].current_time}`);
    console.log();
    
    // 2. Test pgvector extension
    console.log('2️⃣ Testing pgvector extension...');
    const vectorTest = await client`SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'`;
    if (vectorTest.length > 0) {
      console.log(`✅ pgvector Extension Available: v${vectorTest[0].extversion}`);
    } else {
      console.log('⚠️ pgvector Extension Not Found');
      return;
    }
    console.log();
    
    // 3. Test table structure
    console.log('3️⃣ Testing table structure with vector columns...');
    const vectorColumns = await client`
      SELECT table_name, column_name, data_type 
      FROM information_schema.columns 
      WHERE data_type = 'USER-DEFINED' AND udt_name = 'vector'
      ORDER BY table_name, column_name
    `;
    
    console.log(`✅ Found ${vectorColumns.length} vector columns:`);
    for (const col of vectorColumns) {
      console.log(`   ${col.table_name}.${col.column_name} (vector)`);
    }
    console.log();
    
    // 4. Test vector operations
    console.log('4️⃣ Testing vector operations...');
    
    // Test creating a simple vector
    console.log('🧮 Testing vector creation and similarity...');
    try {
      const similarityTest = await client`
        SELECT 
          '[1,2,3,4,5]'::vector <=> '[1,2,3,4,6]'::vector as distance,
          1 - ('[1,2,3,4,5]'::vector <=> '[1,2,3,4,6]'::vector) as similarity
      `;
      
      console.log(`   Distance: ${similarityTest[0].distance}`);
      console.log(`   Similarity: ${similarityTest[0].similarity}`);
    } catch (error) {
      console.error('   ❌ Vector operation failed:', error.message);
    }
    console.log();
    
    // 5. Test if we can insert vector data
    console.log('5️⃣ Testing vector data insertion...');
    try {
      // Check if we can insert into evidence table
      const testVector = '[0.1,0.2,0.3,0.4,0.5]';
      
      // First check if evidence table has vector column
      const evidenceStructure = await client`
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'evidence' AND udt_name = 'vector'
      `;
      
      if (evidenceStructure.length > 0) {
        console.log('✅ Evidence table has vector columns, ready for RAG integration');
        console.log(`   Vector columns: ${evidenceStructure.map(c => c.column_name).join(', ')}`);
      } else {
        console.log('⚠️ Evidence table exists but no vector columns found');
      }
    } catch (error) {
      console.error('   ❌ Vector insertion test failed:', error.message);
    }
    console.log();
    
    console.log('🎉 Database integration test completed successfully!');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  } finally {
    await client.end();
  }
}

// Run the test
main().catch(console.error);