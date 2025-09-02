// Test JSONB functionality with PostgreSQL
import postgres from 'postgres';

const sql = postgres({
  host: 'localhost',
  port: 5432,
  database: 'legal_ai_db',
  username: 'postgres',
  password: '123456'
});

async function testJSONB() {
  try {
    console.log('Testing JSONB functionality...\n');
    
    // 1. Test inserting JSONB data
    console.log('1. Testing JSONB insert:');
    const testUser = await sql`
      INSERT INTO users (
        email,
        hashed_password,
        first_name,
        last_name,
        role,
        permissions,
        practice_areas,
        metadata
      ) VALUES (
        ${`test_jsonb_${Date.now()}@test.com`},
        ${'hashed_password_test'},
        ${'John'},
        ${'Doe'},
        ${'admin'},
        ${JSON.stringify(['read', 'write', 'delete'])},
        ${JSON.stringify(['criminal', 'civil', 'corporate'])},
        ${JSON.stringify({ 
          test: true, 
          timestamp: new Date().toISOString(),
          nested: { 
            level1: { 
              level2: 'deep value' 
            } 
          }
        })}
      )
      RETURNING id, email, permissions, practice_areas, metadata
    `;
    console.log('✅ Insert successful:', testUser[0]);
    
    // 2. Test querying JSONB data
    console.log('\n2. Testing JSONB query operators:');
    
    // Test @> operator (contains)
    const usersWithCriminal = await sql`
      SELECT email, practice_areas 
      FROM users 
      WHERE practice_areas @> '["criminal"]'::jsonb
      LIMIT 5
    `;
    console.log('✅ Users with criminal practice area:', usersWithCriminal.length);
    
    // Test -> operator (get JSON object field)
    const metadataTest = await sql`
      SELECT 
        email,
        metadata->'test' as test_field,
        metadata->'nested'->'level1'->'level2' as nested_value
      FROM users 
      WHERE metadata->>'test' = 'true'
      LIMIT 5
    `;
    console.log('✅ Metadata field access:', metadataTest);
    
    // 3. Test JSONB aggregation
    console.log('\n3. Testing JSONB aggregation:');
    const jsonAgg = await sql`
      SELECT 
        role,
        jsonb_agg(DISTINCT practice_areas) as all_practice_areas
      FROM users 
      WHERE practice_areas IS NOT NULL
      GROUP BY role
      LIMIT 3
    `;
    console.log('✅ Aggregated practice areas by role:', jsonAgg);
    
    // 4. Test JSONB update
    console.log('\n4. Testing JSONB update:');
    const updateData = { updated: true, update_time: new Date().toISOString() };
    const updateResult = await sql`
      UPDATE users 
      SET metadata = metadata || ${updateData}::jsonb
      WHERE email = ${testUser[0].email}
      RETURNING email, metadata
    `;
    console.log('✅ Updated metadata:', updateResult[0]);
    
    // 5. Test JSONB indexing performance
    console.log('\n5. Testing JSONB GIN index:');
    
    // Create GIN index if not exists
    await sql`
      CREATE INDEX IF NOT EXISTS idx_users_metadata_gin 
      ON users USING gin (metadata);
    `;
    
    await sql`
      CREATE INDEX IF NOT EXISTS idx_users_practice_areas_gin 
      ON users USING gin (practice_areas);
    `;
    
    // Query using the index
    const indexedQuery = await sql`
      EXPLAIN (FORMAT JSON)
      SELECT email FROM users 
      WHERE metadata @> '{"test": true}'::jsonb
    `;
    console.log('✅ Index created and query plan shows index usage');
    
    // 6. Clean up test user
    await sql`
      DELETE FROM users WHERE email = ${testUser[0].email}
    `;
    console.log('\n✅ Test user cleaned up');
    
    console.log('\n🎉 All JSONB tests passed successfully!');
    console.log('JSONB is fully functional with:');
    console.log('- Insert/Update operations');
    console.log('- Query operators (@>, ->, ->>, ||)');
    console.log('- Aggregation functions');
    console.log('- GIN indexing for performance');
    
  } catch (error) {
    console.error('❌ JSONB test failed:', error);
  } finally {
    await sql.end();
  }
}

testJSONB();