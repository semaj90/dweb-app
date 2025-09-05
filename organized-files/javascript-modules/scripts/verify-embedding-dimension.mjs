#!/usr/bin/env node
// Verifies embedding vector dimensions in PostgreSQL catalog vs expected (384)
import postgres from 'postgres';

const EXPECTED = parseInt(process.env.EXPECTED_EMBED_DIM || '384', 10);
const DATABASE_URL = process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db';
const sql = postgres(DATABASE_URL, { max: 1 });

(async () => {
  try {
    const rows = await sql`
      SELECT attrelid::regclass AS table_name, attname AS column_name, atttypid::regtype AS data_type
      FROM pg_attribute
      WHERE atttypid::regtype::text LIKE 'vector%' AND attnum > 0 AND NOT attisdropped;`;

    let ok = true;
    for (const r of rows) {
      const match = /(vector)\((\d+)\)/.exec(r.data_type);
      const dim = match ? parseInt(match[2], 10) : null;
      if (dim !== EXPECTED) {
        ok = false;
        console.log(`❌ ${r.table_name}.${r.column_name} dimension=${dim} (expected ${EXPECTED})`);
      } else {
        console.log(`✅ ${r.table_name}.${r.column_name} dimension=${dim}`);
      }
    }
    if (rows.length === 0) console.log('⚠️ No vector columns found');
    if (!ok) process.exit(2);
    console.log('\nAll vector columns match expected dimension.');
  } catch (e) {
    console.error('Verification error:', e.message);
    process.exit(1);
  } finally {
    await sql.end({ timeout: 5 });
  }
})();
