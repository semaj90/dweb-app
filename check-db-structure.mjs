#!/usr/bin/env node

import pg from 'pg';

const client = new pg.Client({
  connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
});

async function checkDatabase() {
  try {
    await client.connect();
    console.log('✅ Connected to database');

    // List all tables
    const tables = await client.query(`
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = 'public'
      ORDER BY table_name;
    `);

    console.log('📊 Existing tables:');
    tables.rows.forEach(row => {
      console.log(`   - ${row.table_name}`);
    });

    // Check legal_documents structure if it exists
    if (tables.rows.some(row => row.table_name === 'legal_documents')) {
      console.log('\n📄 legal_documents table structure:');
      const columns = await client.query(`
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = 'legal_documents'
        ORDER BY ordinal_position;
      `);
      columns.rows.forEach(col => {
        console.log(`   - ${col.column_name}: ${col.data_type} (nullable: ${col.is_nullable})`);
      });
    }

  } catch (error) {
    console.error('❌ Database check failed:', error);
  } finally {
    await client.end();
  }
}

checkDatabase();