#!/usr/bin/env node

import pg from 'pg';

const client = new pg.Client({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
});

async function checkCasesStructure() {
  try {
    await client.connect();
    console.log('✅ Connected to database');

    console.log('⚖️ Cases table structure:');
    const columns = await client.query(`
      SELECT column_name, data_type, is_nullable, column_default
      FROM information_schema.columns 
      WHERE table_name = 'cases' 
      ORDER BY ordinal_position;
    `);
    columns.rows.forEach(col => {
      console.log(`   - ${col.column_name}: ${col.data_type} (nullable: ${col.is_nullable})`);
    });

    console.log('\n🔍 Evidence table structure:');
    const evidenceColumns = await client.query(`
      SELECT column_name, data_type, is_nullable, column_default
      FROM information_schema.columns 
      WHERE table_name = 'evidence' 
      ORDER BY ordinal_position;
    `);
    evidenceColumns.rows.forEach(col => {
      console.log(`   - ${col.column_name}: ${col.data_type} (nullable: ${col.is_nullable})`);
    });

  } catch (error) {
    console.error('❌ Database check failed:', error);
  } finally {
    await client.end();
  }
}

checkCasesStructure();