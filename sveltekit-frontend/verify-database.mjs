import postgres from 'postgres';

console.log('🔍 Testing database connection...');

async function verifyDatabase() {
  try {
    // Test basic connection
    const connectionString = process.env.DATABASE_URL || 'postgresql://legal_admin:LegalSecure2024!@localhost:5432/legal_ai_v3';
    const client = postgres(connectionString);
    
    await client`SELECT 1 as test`;
    console.log('✅ Database connection successful!');
    
    // Check pgvector extension
    const vectorCheck = await client`
      SELECT EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
      ) as has_vector
    `;
    
    if (vectorCheck[0].has_vector) {
      console.log('✅ pgvector extension: INSTALLED');
    } else {
      console.log('⚠️ pgvector extension: NOT FOUND');
    }
    
    // Test schema tables
    const tables = await client`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
      ORDER BY table_name
    `;
    
    console.log('📋 Available tables:');
    if (tables.length === 0) {
      console.log('  ⚠️ No tables found - run migrations first');
    } else {
      tables.forEach((table, index) => {
        console.log(`  ${index + 1}. ${table.table_name}`);
      });
    }
    
    await client.end();
    console.log('🎯 Database verification complete!');
    return true;
    
  } catch (error) {
    console.log('❌ Database verification failed:', error.message);
    console.log('🔧 Check:');
    console.log('  • PostgreSQL is running');
    console.log('  • DATABASE_URL in .env is correct'); 
    console.log('  • Database exists and is accessible');
    return false;
  }
}

verifyDatabase();